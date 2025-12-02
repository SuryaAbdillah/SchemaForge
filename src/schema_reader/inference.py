import json
import logging
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from src.json_loader import load_json_file
from src.schema_reader.types import SchemaField, FileSchema
from src.schema_reader.utils import (
    looks_like_timestamp, looks_like_url, looks_like_email, looks_like_uuid,
    looks_like_ip_address, looks_like_numeric_string, looks_like_json_string,
    flatten_dict
)

logger = logging.getLogger(__name__)

def infer_type(value: Any) -> str:
    """Infer the type of a Python value."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, int):
        return "integer"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, str):
        # Try to detect special string types
        if looks_like_timestamp(value):
            return "timestamp"
        elif looks_like_url(value):
            return "url"
        elif looks_like_email(value):
            return "email"
        elif looks_like_uuid(value):
            return "uuid"
        elif looks_like_ip_address(value):
            return "ip_address"
        elif looks_like_numeric_string(value):
            # Could be a numeric string that should be parsed
            return "numeric_string"
        elif looks_like_json_string(value):
            # Embedded JSON string
            return "json_string"
        return "string"
    elif isinstance(value, list):
        return "array"
    elif isinstance(value, dict):
        return "object"
    else:
        return "unknown"

def analyze_field(values: List[Any], field_name: str) -> SchemaField:
    """Analyze a list of values for a field and infer its schema."""
    if not values:
        return SchemaField(field_name, "unknown", nullable=True)
    
    # Collect all non-null values
    non_null_values = [v for v in values if v is not None]
    nullable = len(non_null_values) < len(values)
    
    if not non_null_values:
        return SchemaField(field_name, "null", nullable=True)
    
    # Infer types for all values
    types = set()
    example_value = None
    
    # Statistics collection
    numeric_values = []
    string_lengths = []
    distinct_values_set = set()
    min_val = None
    max_val = None
    
    for value in non_null_values:
        inferred_type = infer_type(value)
        types.add(inferred_type)
        
        if example_value is None:
            example_value = value
        
        # Collect distinct values for enum detection
        try:
            # Use hashable representation for distinct values
            if isinstance(value, (str, int, float, bool)):
                distinct_values_set.add(value)
            elif isinstance(value, list):
                distinct_values_set.add(tuple(value) if len(value) <= 10 else str(value))
            elif isinstance(value, dict):
                distinct_values_set.add(str(sorted(value.items())) if len(value) <= 10 else str(value))
            else:
                distinct_values_set.add(str(value))
        except (TypeError, ValueError):
            pass
        
        # Collect numeric statistics
        if isinstance(value, (int, float)):
            numeric_values.append(value)
            if min_val is None or value < min_val:
                min_val = value
            if max_val is None or value > max_val:
                max_val = value
        elif isinstance(value, str) and looks_like_numeric_string(value):
            try:
                num_val = float(value)
                numeric_values.append(num_val)
                if min_val is None or num_val < min_val:
                    min_val = num_val
                if max_val is None or num_val > max_val:
                    max_val = num_val
            except (ValueError, TypeError):
                pass
        
        # Collect string length statistics
        if isinstance(value, str):
            string_lengths.append(len(value))
        
        # Handle nested objects
        if isinstance(value, dict):
            nested_fields = {}
            for nested_key, nested_val in value.items():
                nested_field = analyze_field([nested_val], f"{field_name}.{nested_key}")
                nested_fields[nested_key] = nested_field
            return SchemaField(
                field_name,
                "object",
                nullable=nullable,
                example_value=example_value,
                is_nested=True,
                nested_fields=nested_fields
            )
        
        # Handle arrays
        if isinstance(value, list) and len(value) > 0:
            array_types = set()
            for item in value[:10]:  # Sample first 10 items
                array_types.add(infer_type(item))
            
            if len(array_types) == 1:
                array_type = list(array_types)[0]
                return SchemaField(
                    field_name,
                    f"array<{array_type}>",
                    nullable=nullable,
                    example_value=example_value
                )
            else:
                return SchemaField(
                    field_name,
                    f"array<mixed>",
                    nullable=nullable,
                    example_value=example_value
                )
    
    # Determine final type
    if len(types) == 1:
        final_type = list(types)[0]
    else:
        final_type = types  # Mixed types
    
    # Calculate statistics
    min_value = min_val if numeric_values else None
    max_value = max_val if numeric_values else None
    min_length = min(string_lengths) if string_lengths else None
    max_length = max(string_lengths) if string_lengths else None
    avg_length = sum(string_lengths) / len(string_lengths) if string_lengths else None
    
    # Detect enum (if distinct values are limited and represent a small percentage)
    # Consider it an enum if there are <= 20 distinct values and they represent > 50% of non-null values
    distinct_count = len(distinct_values_set)
    enum_threshold = min(20, len(non_null_values) // 2)
    distinct_values = distinct_values_set if distinct_count <= enum_threshold else set()
    
    return SchemaField(
        field_name,
        final_type,
        nullable=nullable,
        example_value=example_value,
        distinct_values=distinct_values,
        min_value=min_value,
        max_value=max_value,
        min_length=min_length,
        max_length=max_length,
        avg_length=avg_length
    )

def sample_records(records: List[Dict[str, Any]], max_sample_size: Optional[int], sampling_strategy: str) -> List[Dict[str, Any]]:
    """Sample records based on the sampling strategy."""
    size = max_sample_size
    
    if size is None or len(records) <= size:
        return records
    
    if sampling_strategy == "random":
        import random
        return random.sample(records, size)
    else:  # default to "first"
        return records[:size]

def reservoir_sample(iterator, sample_size: int):
    """
    Reservoir sampling algorithm for random sampling from a stream.
    This allows us to randomly sample without loading all records into memory.
    """
    import random
    reservoir = []
    count = 0
    
    for item in iterator:
        count += 1
        if len(reservoir) < sample_size:
            reservoir.append(item)
        else:
            # Randomly replace an element in the reservoir
            j = random.randint(0, count - 1)
            if j < sample_size:
                reservoir[j] = item
    
    return reservoir, count

def infer_schema(filepath: Path, max_sample_size: Optional[int] = None, sampling_strategy: str = "first") -> Optional[FileSchema]:
    """Infer schema for a single JSON file."""
    logger.info(f"Processing file: {filepath.name}")
    
    # Check file size to determine if we should use streaming
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    use_streaming = file_size_mb > 10  # Use streaming for files > 10MB
    
    # Default to a reasonable sample size if not specified to avoid "stuck" analysis on huge files
    effective_sample_size = max_sample_size
    if effective_sample_size is None:
        effective_sample_size = 10000
        logger.info(f"No max_sample_size set. Defaulting to {effective_sample_size} for performance.")

    try:
        # Always use streaming for large files or when sampling strategy allows it
        # For random sampling, we'll use reservoir sampling which doesn't require loading everything
        stream = use_streaming or sampling_strategy == "first"
        records_iter = load_json_file(filepath, stream=stream)
        
        actual_record_count = -1  # Unknown initially
        sampled_records = []
        
        if stream:
            if sampling_strategy == "first" and effective_sample_size:
                # For first N records, just take the first N from the stream
                import itertools
                sampled_records = list(itertools.islice(records_iter, effective_sample_size))
                actual_record_count = len(sampled_records)  # We only know the sampled count
                logger.info(f"Streaming first {len(sampled_records)} records from {filepath.name}")
            elif sampling_strategy == "random" and effective_sample_size:
                # Use reservoir sampling for random sampling without loading everything
                sampled_records, actual_record_count = reservoir_sample(records_iter, effective_sample_size)
                logger.info(f"Reservoir sampled {len(sampled_records)} of {actual_record_count} records from {filepath.name}")
            else:
                # Need all records - still stream but collect them
                sampled_records = list(records_iter)
                actual_record_count = len(sampled_records)
        else:
            # Small file - load into memory
            records = records_iter
            actual_record_count = len(records)
            
            # Sample records if needed
            if sampling_strategy == "random" and effective_sample_size and len(records) > effective_sample_size:
                sampled_records = sample_records(records, effective_sample_size, sampling_strategy)
            elif sampling_strategy == "first" and effective_sample_size and len(records) > effective_sample_size:
                sampled_records = records[:effective_sample_size]
            else:
                sampled_records = records
        
        if not sampled_records:
            logger.warning(f"No records found in {filepath.name}")
            return None
        
        sampled_count = len(sampled_records)
        logger.info(f"Analyzing {sampled_count} of {actual_record_count} records from {filepath.name}")
        
        # Collect all field values
        field_values: Dict[str, List[Any]] = defaultdict(list)
        
        for record in sampled_records:
            if not isinstance(record, dict):
                # Should be handled by json_loader normalization, but extra safety
                logger.warning(f"Skipping non-dict record in {filepath.name}: {type(record)}")
                continue
                
            # Flatten nested structures
            flat_record = flatten_dict(record)
            for key, value in flat_record.items():
                # Try to parse embedded JSON strings
                if isinstance(value, str) and looks_like_json_string(value):
                    try:
                        parsed = json.loads(value)
                        # If it's a dict or list, we might want to flatten it further
                        # For now, keep both the string and parsed version for analysis
                        field_values[key].append(value)  # Keep original
                        # Also add parsed version with a suffix
                        if isinstance(parsed, dict):
                            for nested_key, nested_val in parsed.items():
                                field_values[f"{key}.parsed.{nested_key}"].append(nested_val)
                    except (json.JSONDecodeError, ValueError):
                        field_values[key].append(value)
                else:
                    field_values[key].append(value)
        
        # Analyze each field
        fields = {}
        for field_name, values in field_values.items():
            field = analyze_field(values, field_name)
            fields[field_name] = field
        
        schema = FileSchema(filepath.name, actual_record_count, fields)
        logger.info(f"Successfully inferred schema for {filepath.name}: {len(fields)} fields")
        
        return schema
    
    except Exception as e:
        logger.error(f"Failed to infer schema for {filepath.name}: {e}")
        return None

def scan_directory(data_dir: Path, max_sample_size: Optional[int] = None, sampling_strategy: str = "first") -> Dict[str, FileSchema]:
    """Scan the data directory and infer schemas for all JSON files."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    json_files = list(data_dir.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {data_dir}")
        return {}
    
    logger.info(f"Found {len(json_files)} JSON file(s) in {data_dir}")
    
    schemas = {}
    
    # Use ProcessPoolExecutor for parallel processing
    max_workers = min(len(json_files), 4)  # Cap at 4 workers or number of files
    if max_workers < 1: max_workers = 1
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # We need to pass arguments to infer_schema
        future_to_file = {
            executor.submit(infer_schema, json_file, max_sample_size, sampling_strategy): json_file 
            for json_file in json_files
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            json_file = future_to_file[future]
            try:
                schema = future.result()
                if schema:
                    schemas[json_file.name] = schema
            except Exception as e:
                logger.error(f"File {json_file.name} generated an exception: {e}")
    
    return schemas
