import logging
import pyarrow as pa
import pyarrow.parquet as pq
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

def convert_to_parquet(filepath: Path, output_dir: Path, schema_reader: SchemaReader, 
                      schema: Optional[FileSchema] = None, 
                      chunk_size: Optional[int] = None) -> bool:
    """
    Convert a JSON file to Parquet format with chunked processing.
    
    Args:
        filepath: Path to JSON file to convert
        output_dir: Output directory for parquet file
        schema_reader: SchemaReader instance
        schema: Optional pre-loaded FileSchema
        chunk_size: Optional chunk size (auto-calculated if not provided)
    
    Returns:
        True if conversion succeeded, False otherwise
    """
    logger.info(f"Converting {filepath.name} to Parquet...")
    
    with MemoryMonitor(f"Parquet conversion: {filepath.name}"):
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
            output_filename = filepath.stem + ".parquet"
            output_path = output_dir / output_filename
            
            # Determine column order from schema for consistency across chunks
            schema_fields = sorted(schema.fields.keys())
            
            # Process file in chunks
            writer = None
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
                    df = prepare_dataframe_chunk(chunk_records, schema, column_order=schema_fields)
                    
                    if df.empty:
                        logger.warning(f"Empty DataFrame in chunk {chunk_count}")
                        continue
                    
                    # Convert to PyArrow table
                    table = pa.Table.from_pandas(df)
                    
                    # Write chunk to parquet file
                    if writer is None:
                        # First chunk - create writer with schema
                        writer = pq.ParquetWriter(output_path, table.schema)
                    
                    writer.write_table(table)
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
            logger.error(f"Failed to convert {filepath.name} to Parquet: {e}")
            return False
