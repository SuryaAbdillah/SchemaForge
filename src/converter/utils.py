import json
import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from src.schema_reader import FileSchema

logger = logging.getLogger(__name__)

def flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Handle arrays
            if len(v) > 0 and isinstance(v[0], dict):
                # Array of objects - convert to JSON string
                items.append((new_key, json.dumps(v)))
            else:
                # Simple array - keep as is or convert to string
                items.append((new_key, v))
        else:
            items.append((new_key, v))
    return dict(items)

def coerce_type(value: Any, field_type: Any) -> Any:
    """Coerce a value to the expected type based on schema."""
    if value is None:
        return None
    
    # Handle mixed types
    if isinstance(field_type, set):
        # For mixed types, try to preserve the original value
        return value
    
    # Handle array types
    if isinstance(field_type, str) and field_type.startswith("array<"):
        return value  # Keep arrays as-is or as JSON strings
    
    # Type coercion
    type_str = field_type if isinstance(field_type, str) else str(field_type)
    
    try:
        if type_str == "integer":
            if isinstance(value, (int, float)):
                return int(value)
            elif isinstance(value, str):
                # Try to parse numeric strings
                try:
                    return int(float(value))  # Use float first to handle "123.0"
                except (ValueError, TypeError):
                    return value
            return value
        elif type_str == "float":
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return value
            return value
        elif type_str == "numeric_string":
            # Keep as string but ensure it's numeric
            if isinstance(value, (int, float)):
                return str(value)
            return str(value)
        elif type_str == "boolean":
            if isinstance(value, bool):
                return value
            str_val = str(value).lower().strip()
            return str_val in ('true', '1', 'yes', 'on', 'y', 't')
        elif type_str in ("string", "timestamp", "url", "email", "uuid", "ip_address"):
            return str(value)
        elif type_str == "json_string":
            # If it's already a JSON string, keep it; otherwise stringify
            if isinstance(value, str):
                return value
            elif isinstance(value, (dict, list)):
                return json.dumps(value)
            return str(value)
        else:
            return value
    except (ValueError, TypeError) as e:
        logger.warning(f"Type coercion failed for value {value} to type {type_str}: {e}")
        return value

def extract_columns_from_metadata(metadata: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Extract column definitions from metadata (e.g., Socrata/OpenData format)."""
    # Try various metadata paths
    metadata_paths = [
        ['meta', 'view', 'columns'],
        ['view', 'columns'],
        ['columns'],
        ['schema', 'fields'],
        ['fields'],
        ['header']
    ]
    
    for path in metadata_paths:
        current = metadata
        found = True
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                found = False
                break
        
        if found and isinstance(current, list) and len(current) > 0:
            # Check if it looks like column definitions
            if isinstance(current[0], dict) and ('name' in current[0] or 'fieldName' in current[0]):
                logger.info(f"Found column definitions in metadata path: {' -> '.join(path)}")
                return current
    
    return None

def convert_array_row_to_object(row: List[Any], columns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert an array-based row to an object using column definitions."""
    record = {}
    
    for col_def in columns:
        # Get column name - try multiple field names
        col_name = None
        for name_field in ['fieldName', 'name', 'id', 'key']:
            if name_field in col_def:
                col_name = str(col_def[name_field])
                break
        
        if col_name is None:
            # Fallback to position-based name
            position = col_def.get('position', len(record))
            col_name = f"column_{position}"
        
        # Skip hidden/meta columns if flagged
        if col_def.get('flags') and 'hidden' in col_def.get('flags', []):
            continue
        
        # Skip meta_data type columns by default (can be overridden)
        if col_def.get('dataTypeName') == 'meta_data' and 'hidden' in col_def.get('flags', []):
            continue
        
        # Get value from row using position
        position = col_def.get('position', -1)
        
        # If position is valid and within row bounds
        if position >= 0 and position < len(row):
            value = row[position]
        else:
            # Fallback: try to use index in columns list
            col_idx = columns.index(col_def) if col_def in columns else -1
            if col_idx >= 0 and col_idx < len(row):
                value = row[col_idx]
            else:
                continue  # Skip if we can't map this column
        
        # Clean column name (remove : prefix if present)
        clean_name = col_name.lstrip(':')
        record[clean_name] = value
    
    return record

def prepare_dataframe(records: List[Dict[str, Any]], schema: FileSchema) -> pd.DataFrame:
    """Prepare a pandas DataFrame from records using the schema."""
    if not records:
        return pd.DataFrame()
    
    # Flatten all records
    flattened_records = []
    for record in records:
        flattened = flatten_dict(record)
        flattened_records.append(flattened)
    
    # Create DataFrame
    df = pd.DataFrame(flattened_records)
    
    # Ensure all schema fields are present (fill missing with None)
    for field_name in schema.fields.keys():
        if field_name not in df.columns:
            df[field_name] = None
    
    # Reorder columns to match schema order
    schema_fields = sorted(schema.fields.keys())
    existing_fields = [f for f in schema_fields if f in df.columns]
    df = df[existing_fields]
    
    # Apply type coercion based on schema
    for field_name, field in schema.fields.items():
        if field_name in df.columns:
            df[field_name] = df[field_name].apply(
                lambda x: coerce_type(x, field.field_type)
            )
    
    return df

def prepare_dataframe_chunk(records: List[Dict[str, Any]], schema: FileSchema, 
                           column_order: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Prepare a pandas DataFrame chunk from records using the schema.
    
    This is optimized for chunked processing where we may need consistent
    column ordering across multiple chunks.
    
    Args:
        records: List of record dictionaries
        schema: FileSchema object with field definitions
        column_order: Optional list of column names for consistent ordering
    
    Returns:
        DataFrame with records prepared according to schema
    """
    if not records:
        return pd.DataFrame()
    
    # Flatten all records
    flattened_records = []
    for record in records:
        flattened = flatten_dict(record)
        flattened_records.append(flattened)
    
    # Create DataFrame
    df = pd.DataFrame(flattened_records)
    
    # Determine column order (use provided or derive from schema)
    if column_order is None:
        schema_fields = sorted(schema.fields.keys())
        column_order = [f for f in schema_fields if f in df.columns]
    
    # Ensure all required columns are present (fill missing with None)
    for field_name in column_order:
        if field_name not in df.columns:
            df[field_name] = None
    
    # Select and reorder columns
    existing_cols = [col for col in column_order if col in df.columns]
    df = df[existing_cols]
    
    # Apply type coercion based on schema
    for field_name, field in schema.fields.items():
        if field_name in df.columns:
            df[field_name] = df[field_name].apply(
                lambda x: coerce_type(x, field.field_type)
            )
    
    return df

