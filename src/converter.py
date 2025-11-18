"""
Converter Module

This module provides functionality to convert JSON files to Parquet and CSV formats
using the schemas inferred by the schema reader.
"""

import json
import ast
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.schema_reader import SchemaReader, FileSchema

logger = logging.getLogger(__name__)


class Converter:
    """Main class for converting JSON files to other formats."""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "output",
                 schema_reader: Optional[SchemaReader] = None,
                 schema_report_path: Optional[str] = None):
        """
        Initialize the Converter.
        
        Args:
            data_dir: Directory containing JSON files
            output_dir: Directory for output files
            schema_reader: Optional SchemaReader instance (will create one if not provided)
            schema_report_path: Path to schema report JSON file (required for conversion)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.schema_report_path = schema_report_path
        
        if schema_reader is None:
            self.schema_reader = SchemaReader(data_dir=data_dir)
        else:
            self.schema_reader = schema_reader
    
    def _flatten_dict(self, d: Dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
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
    
    def _coerce_type(self, value: Any, field_type: Any) -> Any:
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
    
    def _extract_columns_from_metadata(self, metadata: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
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
    
    def _convert_array_row_to_object(self, row: List[Any], columns: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    
    def _load_json_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load JSON data from a file, handling multiple formats:
        
        - Standard JSON array of objects
        - NDJSON (newline-delimited JSON)
        - Wrapper objects with data arrays (data, results, items, records, rows, entries)
        - Array-based tabular data (arrays of arrays) with column metadata
        - GeoJSON format
        - Single JSON object (treated as single record)
        - Python literal format (dict/list literals with single quotes)
        """
        records = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                if not content:
                    logger.warning(f"File {filepath.name} is empty")
                    return records
                
                # Try to parse as JSON first
                try:
                    data = json.loads(content)
                    
                    # Handle different top-level structures
                    if isinstance(data, list):
                        # Check if it's an array of arrays (tabular format)
                        if len(data) > 0 and isinstance(data[0], list):
                            logger.info(f"Detected array-based tabular format in {filepath.name}")
                            # Try to find column metadata by looking for a wrapper structure
                            # This format usually comes from APIs, so we'll create generic column names
                            if len(data) > 0:
                                max_cols = max(len(row) for row in data if isinstance(row, list))
                                columns = [{'name': f'column_{i}', 'position': i} for i in range(max_cols)]
                                records = [self._convert_array_row_to_object(row, columns) for row in data]
                            return records
                        else:
                            # Regular array of objects
                            records = data
                    
                    elif isinstance(data, dict):
                        # Check for array-based tabular data with metadata
                        data_fields = ['data', 'results', 'items', 'records', 'rows', 'entries']
                        extracted_data = None
                        data_field_name = None
                        
                        for field_name in data_fields:
                            if field_name in data and isinstance(data[field_name], list):
                                extracted_data = data[field_name]
                                data_field_name = field_name
                                break
                        
                        if extracted_data is not None:
                            # Check if it's array-based tabular format
                            if len(extracted_data) > 0 and isinstance(extracted_data[0], list):
                                logger.info(f"Detected array-based tabular format in '{data_field_name}' field")
                                # Try to extract column definitions from metadata
                                columns = self._extract_columns_from_metadata(data)
                                
                                if columns:
                                    # Use all columns but the conversion function will skip hidden ones
                                    # Position in column definitions matches array index
                                    records = [
                                        self._convert_array_row_to_object(row, columns) 
                                        for row in extracted_data
                                    ]
                                else:
                                    # No column metadata found, create generic column names
                                    if len(extracted_data) > 0:
                                        max_cols = max(len(row) for row in extracted_data if isinstance(row, list))
                                        columns = [{'name': f'column_{i}', 'position': i} for i in range(max_cols)]
                                        records = [
                                            self._convert_array_row_to_object(row, columns) 
                                            for row in extracted_data
                                        ]
                            else:
                                # Regular array of objects
                                records = extracted_data
                                logger.info(f"Found {len(records)} records in '{data_field_name}' field")
                        else:
                            # Check for GeoJSON format
                            if data.get('type') == 'FeatureCollection' and 'features' in data:
                                logger.info("Detected GeoJSON FeatureCollection format")
                                records = [feature.get('properties', {}) for feature in data.get('features', [])]
                            elif data.get('type') == 'Feature':
                                logger.info("Detected GeoJSON Feature format")
                                records = [data.get('properties', {})]
                            else:
                                # Treat the dict itself as a single record
                                records = [data]
                    else:
                        logger.warning(f"Unexpected JSON structure in {filepath.name}: {type(data)}")
                        return records
                
                except json.JSONDecodeError:
                    # Try Python literal format for entire file
                    try:
                        logger.info(f"Trying Python literal format for {filepath.name}")
                        data = ast.literal_eval(content)
                        
                        if isinstance(data, list):
                            # List of records
                            if len(data) > 0 and isinstance(data[0], dict):
                                records = data
                            elif len(data) > 0 and isinstance(data[0], list):
                                # Array of arrays
                                max_cols = max(len(row) for row in data if isinstance(row, list))
                                columns = [{'name': f'column_{i}', 'position': i} for i in range(max_cols)]
                                records = [self._convert_array_row_to_object(row, columns) for row in data]
                        elif isinstance(data, dict):
                            # Single dict or wrapper
                            data_fields = ['data', 'results', 'items', 'records', 'rows', 'entries']
                            extracted_data = None
                            
                            for field_name in data_fields:
                                if field_name in data and isinstance(data[field_name], list):
                                    extracted_data = data[field_name]
                                    break
                            
                            if extracted_data is not None:
                                records = extracted_data
                            else:
                                records = [data]
                        else:
                            logger.warning(f"Unexpected Python literal structure in {filepath.name}: {type(data)}")
                            return records
                    except (ValueError, SyntaxError):
                        # Try NDJSON format (one JSON object per line)
                        logger.info(f"Trying NDJSON format for {filepath.name}")
                        f.seek(0)
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                                if isinstance(record, dict):
                                    # Check if this line is a wrapper with data array
                                    data_fields = ['data', 'results', 'items', 'records', 'rows', 'entries']
                                    extracted_records = None
                                    
                                    for field_name in data_fields:
                                        if field_name in record and isinstance(record[field_name], list):
                                            extracted_records = record[field_name]
                                            break
                                    
                                    if extracted_records is not None:
                                        records.extend(extracted_records)
                                    else:
                                        records.append(record)
                                elif isinstance(record, list):
                                    # Array in NDJSON line
                                    records.extend(record)
                            except json.JSONDecodeError:
                                # Try Python literal format (e.g., {'key': 'value'})
                                try:
                                    record = ast.literal_eval(line)
                                    if isinstance(record, dict):
                                        records.append(record)
                                    elif isinstance(record, list):
                                        records.extend(record)
                                except (ValueError, SyntaxError) as e:
                                    logger.warning(f"Failed to parse line {line_num} in {filepath.name}: {e}")
                                    continue
        
        except Exception as e:
            logger.error(f"Error reading file {filepath.name}: {e}")
            raise
        
        # Ensure all records are dictionaries
        valid_records = []
        for record in records:
            if isinstance(record, dict):
                valid_records.append(record)
            elif isinstance(record, list):
                # Convert array to object with indexed keys
                valid_records.append({f'field_{i}': val for i, val in enumerate(record)})
        
        logger.info(f"Loaded {len(valid_records)} records from {filepath.name}")
        return valid_records
    
    def _prepare_dataframe(self, records: List[Dict[str, Any]], schema: FileSchema) -> pd.DataFrame:
        """Prepare a pandas DataFrame from records using the schema."""
        if not records:
            return pd.DataFrame()
        
        # Flatten all records
        flattened_records = []
        for record in records:
            flattened = self._flatten_dict(record)
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
                    lambda x: self._coerce_type(x, field.field_type)
                )
        
        return df
    
    def convert_to_parquet(self, filepath: Path, schema: Optional[FileSchema] = None) -> bool:
        """Convert a JSON file to Parquet format."""
        logger.info(f"Converting {filepath.name} to Parquet...")
        
        try:
            # Load schema if not provided
            if schema is None:
                schema = self.schema_reader.infer_schema(filepath)
                if schema is None:
                    logger.error(f"Failed to infer schema for {filepath.name}")
                    return False
            
            # Load JSON data
            records = self._load_json_file(filepath)
            
            if not records:
                logger.warning(f"No records to convert in {filepath.name}")
                return False
            
            # Prepare DataFrame
            df = self._prepare_dataframe(records, schema)
            
            if df.empty:
                logger.warning(f"Empty DataFrame created for {filepath.name}")
                return False
            
            # Generate output filename
            output_filename = filepath.stem + ".parquet"
            output_path = self.output_dir / output_filename
            
            # Convert to PyArrow table and write
            table = pa.Table.from_pandas(df)
            pq.write_table(table, output_path)
            
            logger.info(f"Successfully converted {filepath.name} to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to convert {filepath.name} to Parquet: {e}")
            return False
    
    def convert_to_csv(self, filepath: Path, schema: Optional[FileSchema] = None) -> bool:
        """Convert a JSON file to CSV format."""
        logger.info(f"Converting {filepath.name} to CSV...")
        
        try:
            # Load schema if not provided
            if schema is None:
                schema = self.schema_reader.infer_schema(filepath)
                if schema is None:
                    logger.error(f"Failed to infer schema for {filepath.name}")
                    return False
            
            # Load JSON data
            records = self._load_json_file(filepath)
            
            if not records:
                logger.warning(f"No records to convert in {filepath.name}")
                return False
            
            # Prepare DataFrame
            df = self._prepare_dataframe(records, schema)
            
            if df.empty:
                logger.warning(f"Empty DataFrame created for {filepath.name}")
                return False
            
            # Generate output filename
            output_filename = filepath.stem + ".csv"
            output_path = self.output_dir / output_filename
            
            # Write to CSV
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            logger.info(f"Successfully converted {filepath.name} to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to convert {filepath.name} to CSV: {e}")
            return False
    
    def convert_all(self, format_type: str) -> Dict[str, bool]:
        """Convert all JSON files in the data directory to the specified format.
        
        Requires a schema report to be generated first using scan-schemas command.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        json_files = list(self.data_dir.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {self.data_dir}")
            return {}
        
        # Load schemas from schema report JSON file
        if not self.schema_report_path:
            raise ValueError(
                "Schema report path is required. Please run 'scan-schemas' command first "
                "to generate a schema report, then provide the path using --schema-report option."
            )
        
        logger.info(f"Loading schemas from schema report: {self.schema_report_path}")
        try:
            schemas = SchemaReader.load_schemas_from_json(self.schema_report_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Schema report not found: {self.schema_report_path}. "
                "Please run 'scan-schemas' command first to generate a schema report."
            ) from e
        
        if not schemas:
            raise ValueError("No schemas found in the schema report. Please regenerate the schema report.")
        
        results = {}
        
        for json_file in json_files:
            schema = schemas.get(json_file.name)
            
            if schema is None:
                logger.warning(
                    f"No schema found for {json_file.name} in the schema report. "
                    "Skipping this file. Please regenerate the schema report."
                )
                results[json_file.name] = False
                continue
            
            if format_type.lower() == "parquet":
                success = self.convert_to_parquet(json_file, schema)
            elif format_type.lower() == "csv":
                success = self.convert_to_csv(json_file, schema)
            else:
                logger.error(f"Unsupported format: {format_type}")
                success = False
            
            results[json_file.name] = success
        
        return results

