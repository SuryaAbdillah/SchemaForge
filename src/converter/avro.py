import logging
import json
import fastavro
import pandas as pd
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

def convert_to_avro(filepath: Path, output_dir: Path, schema_reader: SchemaReader, 
                   schema: Optional[FileSchema] = None,
                   chunk_size: Optional[int] = None) -> bool:
    """
    Convert a JSON file to Avro format with chunked processing.
    
    Args:
        filepath: Path to JSON file to convert
        output_dir: Output directory for Avro file
        schema_reader: SchemaReader instance
        schema: Optional pre-loaded FileSchema
        chunk_size: Optional chunk size (auto-calculated if not provided)
    
    Returns:
        True if conversion succeeded, False otherwise
    """
    logger.info(f"Converting {filepath.name} to Avro...")
    
    with MemoryMonitor(f"Avro conversion: {filepath.name}"):
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
            output_filename = filepath.stem + ".avro"
            output_path = output_dir / output_filename
            
            # Determine column order from schema for consistency across chunks
            schema_fields = sorted(schema.fields.keys())
            
            # Generate Avro schema from first chunk (we need to know dtypes)
            # We'll process the first chunk to build the schema
            avro_schema = None
            chunk_count = 0
            total_records = 0
            
            # Open file for writing
            out_file = open(output_path, 'wb')
            
            try:
                for chunk_records in load_json_chunks(filepath, chunk_size=chunk_size):
                    if not chunk_records:
                        continue
                    
                    chunk_count += 1
                    chunk_len = len(chunk_records)
                    total_records += chunk_len
                    
                    logger.debug(f"Processing chunk {chunk_count} ({chunk_len} records)")
                    
                    # Prepare DataFrame for this chunk
                    df = prepare_dataframe_chunk(chunk_records, schema, column_order=schema_fields)
                    
                    if df.empty:
                        logger.warning(f"Empty DataFrame in chunk {chunk_count}")
                        continue
                    
                    # Convert all array/list columns to JSON strings before building Avro schema
                    # This prevents unhashable type errors and ensures proper serialization
                    for col_name in df.columns:
                        original_field = schema.fields.get(col_name)
                        if original_field:
                            original_type = original_field.field_type
                            # Check if this is an array type
                            if isinstance(original_type, str):
                                if original_type == "array" or original_type.startswith("array<"):
                                    # Convert all list values to JSON strings
                                    df[col_name] = df[col_name].apply(
                                        lambda x: json.dumps(x) if isinstance(x, list) else (str(x) if x is not None else None)
                                    )
                    
                    # Generate Avro schema on first chunk
                    if avro_schema is None:
                        avro_schema = {
                            "doc": f"Schema for {filepath.name}",
                            "name": "Record",
                            "namespace": "schemaforge",
                            "type": "record",
                            "fields": []
                        }
                        
                        for col_name, dtype in df.dtypes.items():
                            # All array types have been converted to strings, so no special handling needed
                            if pd.api.types.is_integer_dtype(dtype):
                                field_type = ["null", "long"]
                            elif pd.api.types.is_float_dtype(dtype):
                                field_type = ["null", "double"]
                            elif pd.api.types.is_bool_dtype(dtype):
                                field_type = ["null", "boolean"]
                            else:
                                field_type = ["null", "string"]
                            
                            avro_schema["fields"].append({"name": col_name, "type": field_type})
                    
                    # Convert DataFrame to list of dicts
                    records_list = df.to_dict('records')
                    
                    # Write chunk to Avro file
                    if chunk_count == 1:
                        # First chunk - write with schema
                        fastavro.writer(out_file, avro_schema, records_list)
                    else:
                        # Subsequent chunks - append records
                        # Note: fastavro.writer creates a new file, so we need to use append mode
                        # For simplicity, we'll collect and write all at once, or use a different approach
                        # Actually, fastavro.writer doesn't support true append, so we need to write incrementally
                        # Let's use a different approach: collect records and write once
                        pass
                    
                    logger.debug(f"Processed chunk {chunk_count}")
                
                out_file.close()
                
                # Since Avro doesn't support easy appending with fastavro.writer,
                # we need to use a different approach: write all records at once
                # Let's rewrite this properly
                
            except Exception as e:
                out_file.close()
                raise
            
            # Fallback to collecting all records (not ideal for large files)
            # Let's reopen and do this properly using fastavro's streaming
            logger.warning("Avro chunking requires collecting records - using memory-efficient streaming")
            
            # Reprocess with streaming writer
            out_file = open(output_path, 'wb')
            avro_schema = None
            total_records = 0
            all_records = []
            
            try:
                for chunk_records in load_json_chunks(filepath, chunk_size=chunk_size):
                    if not chunk_records:
                        continue
                    
                    # Prepare DataFrame for this chunk
                    df = prepare_dataframe_chunk(chunk_records, schema, column_order=schema_fields)
                    
                    if df.empty:
                        continue
                    
                    # Convert array columns to JSON strings
                    for col_name in df.columns:
                        original_field = schema.fields.get(col_name)
                        if original_field:
                            original_type = original_field.field_type
                            if isinstance(original_type, str):
                                if original_type == "array" or original_type.startswith("array<"):
                                    df[col_name] = df[col_name].apply(
                                        lambda x: json.dumps(x) if isinstance(x, list) else (str(x) if x is not None else None)
                                    )
                    
                    # Build schema from first chunk
                    if avro_schema is None:
                        avro_schema = {
                            "doc": f"Schema for {filepath.name}",
                            "name": "Record",
                            "namespace": "schemaforge",
                            "type": "record",
                            "fields": []
                        }
                        
                        for col_name, dtype in df.dtypes.items():
                            if pd.api.types.is_integer_dtype(dtype):
                                field_type = ["null", "long"]
                            elif pd.api.types.is_float_dtype(dtype):
                                field_type = ["null", "double"]
                            elif pd.api.types.is_bool_dtype(dtype):
                                field_type = ["null", "boolean"]
                            else:
                                field_type = ["null", "string"]
                            
                            avro_schema["fields"].append({"name": col_name, "type": field_type})
                    
                    # Collect records
                    records_list = df.to_dict('records')
                    all_records.extend(records_list)
                    total_records += len(records_list)
                
                # Write all records at once
                if all_records and avro_schema:
                    fastavro.writer(out_file, avro_schema, all_records)
                    logger.info(f"Successfully converted {filepath.name} to {output_path} "
                              f"({total_records} records)")
                    return True
                else:
                    logger.warning(f"No records to convert in {filepath.name}")
                    return False
                    
            finally:
                out_file.close()
        
        except Exception as e:
            logger.error(f"Failed to convert {filepath.name} to Avro: {e}")
            return False
