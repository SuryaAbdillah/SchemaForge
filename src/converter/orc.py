import logging
import json
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
from pathlib import Path
from typing import Optional
from src.schema_reader import SchemaReader, FileSchema
from src.json_loader import load_json_chunks
from src.converter.utils import prepare_dataframe_chunk
from src.converter.memory_manager import (
    calculate_chunk_size,
    MemoryMonitor,
    DEFAULT_CHUNK_SIZE
)

logger = logging.getLogger(__name__)

def convert_to_orc(filepath: Path, output_dir: Path, schema_reader: SchemaReader, 
                  schema: Optional[FileSchema] = None,
                  chunk_size: Optional[int] = None) -> bool:
    """
    Convert a JSON file to ORC format with chunked processing.
    
    Args:
        filepath: Path to JSON file to convert
        output_dir: Output directory for ORC file
        schema_reader: SchemaReader instance
        schema: Optional pre-loaded FileSchema
        chunk_size: Optional chunk size (auto-calculated if not provided)
    
    Returns:
        True if conversion succeeded, False otherwise
    """
    logger.info(f"Converting {filepath.name} to ORC...")
    
    with MemoryMonitor(f"ORC conversion: {filepath.name}"):
        try:
            # Load schema if not provided
            if schema is None:
                schema = schema_reader.infer_schema(filepath)
                if schema is None:
                    logger.error(f"Failed to infer schema for {filepath.name}")
                    return False
            
            # Calculate optimal chunk size if not provided
            if chunk_size is None:
                chunk_size = calculate_chunk_size(filepath)
            
            # Generate output filename
            output_filename = filepath.stem + ".orc"
            output_path = output_dir / output_filename
            
            # Determine column order from schema for consistency across chunks
            schema_fields = sorted(schema.fields.keys())
            
            # Filter out null-type fields before processing
            # ORC/Arrow doesn't support pure null types
            valid_columns = []
            for col_name in schema_fields:
                field = schema.fields.get(col_name)
                if field:
                    field_type = field.field_type
                    # Skip null-type fields
                    if isinstance(field_type, str) and field_type == "null":
                        logger.warning(f"Skipping null-type column '{col_name}' in {filepath.name} for ORC")
                        continue
                valid_columns.append(col_name)
            
            if not valid_columns:
                logger.error(f"No valid columns for ORC conversion in {filepath.name}")
                return False
            
            # Process file in chunks
            writer = None
            arrow_schema = None
            chunk_count = 0
            total_records = 0
            
            try:
                for chunk_records in load_json_chunks(filepath, chunk_size=chunk_size):
                    if not chunk_records:
                        continue
                    
                    chunk_count += 1
                    chunk_len = len(chunk_records)
                    total_records += chunk_len
                    
                    logger.debug(f"Processing chunk {chunk_count} ({chunk_len} records)")
                    
                    # Prepare DataFrame for this chunk
                    df = prepare_dataframe_chunk(chunk_records, schema, column_order=valid_columns)
                    
                    if df.empty:
                        logger.warning(f"Empty DataFrame in chunk {chunk_count}")
                        continue
                    
                    # Convert all array/list columns to JSON strings (like we do for Avro)
                    # This prevents Arrow type conversion errors
                    for col_name in df.columns:
                        original_field = schema.fields.get(col_name)
                        if original_field:
                            original_type = original_field.field_type
                            # Check if this is an array type
                            if isinstance(original_type, str):
                                if original_type == "array" or original_type.startswith("array<"):
                                    # Convert all list values to JSON strings
                                    df[col_name] = df[col_name].apply(
                                        lambda x: json.dumps(x) if isinstance(x, list) else (str(x) if x is not None else "")
                                    )
                    
                    # Replace None values with empty strings for columns that are all None
                    # This prevents PyArrow from inferring null type
                    for col in df.columns:
                        if df[col].isna().all():
                            logger.warning(f"Column '{col}' is all None/NaN, filling with empty strings for ORC compatibility")
                            df[col] = ""
                    
                    # Build explicit Arrow schema on first chunk
                    if arrow_schema is None:
                        arrow_fields = []
                        for col in df.columns:
                            dtype = df[col].dtype
                            if pd.api.types.is_integer_dtype(dtype):
                                arrow_type = pa.int64()
                            elif pd.api.types.is_float_dtype(dtype):
                                arrow_type = pa.float64()
                            elif pd.api.types.is_bool_dtype(dtype):
                                arrow_type = pa.bool_()
                            else:
                                # Use string for everything else to avoid null types
                                arrow_type = pa.string()
                            arrow_fields.append(pa.field(col, arrow_type))
                        
                        arrow_schema = pa.schema(arrow_fields)
                        logger.debug(f"Created Arrow schema with {len(arrow_fields)} fields")
                    
                    # Convert to PyArrow table with explicit schema
                    table = pa.Table.from_pandas(df, schema=arrow_schema)
                    
                    # Write chunk to ORC file
                    if writer is None:
                        # First chunk - create writer with schema
                        writer = orc.ORCWriter(output_path)
                    
                    writer.write(table)
                    logger.debug(f"Wrote chunk {chunk_count} to {output_path}")
                
                # Close the writer
                if writer is not None:
                    writer.close()
                    logger.info(f"Successfully converted {filepath.name} to {output_path} "
                              f"({total_records} records in {chunk_count} chunks)")
                    return True
                else:
                    logger.warning(f"No data to convert in {filepath.name}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error during chunked processing: {e}")
                if writer is not None:
                    writer.close()
                raise
        
        except Exception as e:
            logger.error(f"Failed to convert {filepath.name} to ORC: {e}")
            return False
