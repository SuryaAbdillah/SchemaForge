import logging
import pyarrow as pa
import pyarrow.feather as feather
from pathlib import Path
from typing import Optional
from src.schema_reader import SchemaReader, FileSchema
from src.json_loader import load_json_chunks
from src.converter.utils import prepare_dataframe_chunk
from src.converter.memory_manager import (
    calculate_chunk_size,
    MemoryMonitor,
    DEFAULT_CHUNK_SIZE,
    check_memory_limit
)

logger = logging.getLogger(__name__)

def convert_to_feather(filepath: Path, output_dir: Path, schema_reader: SchemaReader, 
                      schema: Optional[FileSchema] = None,
                      chunk_size: Optional[int] = None) -> bool:
    """
    Convert a JSON file to Feather format with chunked processing.
    
    Note: Feather format doesn't support true append mode, so we accumulate
    chunks in memory up to the memory limit before writing. For very large files
    that exceed memory limits, consider using Parquet instead.
    
    Args:
        filepath: Path to JSON file to convert
        output_dir: Output directory for Feather file
        schema_reader: SchemaReader instance
        schema: Optional pre-loaded FileSchema
        chunk_size: Optional chunk size (auto-calculated if not provided)
    
    Returns:
        True if conversion succeeded, False otherwise
    """
    logger.info(f"Converting {filepath.name} to Feather...")
    
    with MemoryMonitor(f"Feather conversion: {filepath.name}") as monitor:
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
            output_filename = filepath.stem + ".feather"
            output_path = output_dir / output_filename
            
            # Determine column order from schema for consistency across chunks
            schema_fields = sorted(schema.fields.keys())
            
            # Feather doesn't support append, so we'll accumulate chunks
            # and periodically check memory limits
            all_dataframes = []
            chunk_count = 0
            total_records = 0
            
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
                
                all_dataframes.append(df)
                
                # Check memory usage periodically
                monitor.check()
                within_limit, current_percent = check_memory_limit()
                
                if not within_limit:
                    logger.warning(f"Memory limit reached during Feather conversion at chunk {chunk_count}. "
                                 f"Consider using Parquet format for very large files.")
            
            # Concatenate all dataframes
            if all_dataframes:
                logger.info(f"Concatenating {len(all_dataframes)} chunks...")
                combined_df = pa.concat_tables([pa.Table.from_pandas(df) for df in all_dataframes])
                
                # Write to Feather file
                feather.write_feather(combined_df, output_path)
                
                logger.info(f"Successfully converted {filepath.name} to {output_path} "
                          f"({total_records} records in {chunk_count} chunks)")
                return True
            else:
                logger.warning(f"No records to convert in {filepath.name}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to convert {filepath.name} to Feather: {e}")
            return False
