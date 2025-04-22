"""
Data persistence for the SpectroNeph system.

This module handles data storage and retrieval, including:
- Saving measurements to disk
- Loading measurements from disk
- Managing data formats
- Data indexing and searching
"""

import os
import json
import csv
import time
import datetime
import sqlite3
import numpy as np
import yaml
from typing import Dict, List, Any, Optional, Tuple, Union, BinaryIO
from pathlib import Path

from config import settings
from utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

class DataStorage:
    """
    Data storage class for SpectroNeph data.
    
    This class provides methods for saving and loading measurements,
    managing data formats, and searching stored data.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the data storage.
        
        Args:
            data_dir: Directory for data storage, defaults to settings.DATA_DIR
        """
        self.data_dir = Path(data_dir) if data_dir else Path(settings.get("DATA_DIR", "data"))
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database (if applicable)
        self.db_initialized = False
        
        logger.info(f"Initialized data storage in {self.data_dir}")
    
    def save_session(self, session_data: Dict[str, Any], 
                    format: str = "json", 
                    filename: Optional[str] = None) -> str:
        """
        Save a full acquisition session to disk.
        
        Args:
            session_data: Session data to save
            format: File format ('json', 'csv', 'yaml')
            filename: Optional filename, will be generated if not provided
            
        Returns:
            str: Path to the saved file
        """
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = session_data.get("metadata", {}).get("session_id", "session")
            filename = f"{session_id}_{timestamp}"
        
        # Ensure filename has the correct extension
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"
        
        # Create the full path
        filepath = self.data_dir / filename
        
        # Save in the specified format
        if format == "json":
            return self._save_json(session_data, filepath)
        elif format == "csv":
            return self._save_csv(session_data, filepath)
        elif format == "yaml":
            return self._save_yaml(session_data, filepath)
        else:
            logger.error(f"Unsupported storage format: {format}")
            raise ValueError(f"Unsupported storage format: {format}")
    
    def load_session(self, filename: str) -> Dict[str, Any]:
        """
        Load a session from disk.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Dict[str, Any]: Loaded session data
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Determine format from extension
        extension = filepath.suffix.lower()
        
        if extension == ".json":
            return self._load_json(filepath)
        elif extension == ".csv":
            return self._load_csv(filepath)
        elif extension in [".yaml", ".yml"]:
            return self._load_yaml(filepath)
        else:
            logger.error(f"Unsupported file format: {extension}")
            raise ValueError(f"Unsupported file format: {extension}")
    
    def save_measurements(self, measurements: List[Dict[str, Any]], 
                         format: str = "json", 
                         filename: Optional[str] = None) -> str:
        """
        Save a list of measurements to disk.
        
        Args:
            measurements: List of measurements to save
            format: File format ('json', 'csv', 'yaml')
            filename: Optional filename, will be generated if not provided
            
        Returns:
            str: Path to the saved file
        """
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"measurements_{timestamp}"
        
        # Ensure filename has the correct extension
        if not filename.endswith(f".{format}"):
            filename = f"{filename}.{format}"
        
        # Create the full path
        filepath = self.data_dir / filename
        
        # Create session structure for measurements
        session_data = {
            "metadata": {
                "session_id": f"generated_{int(time.time())}",
                "timestamp": time.time(),
                "count": len(measurements)
            },
            "measurements": measurements
        }
        
        # Save in the specified format
        if format == "json":
            return self._save_json(session_data, filepath)
        elif format == "csv":
            return self._save_csv(session_data, filepath)
        elif format == "yaml":
            return self._save_yaml(session_data, filepath)
        else:
            logger.error(f"Unsupported storage format: {format}")
            raise ValueError(f"Unsupported storage format: {format}")
    
    def list_saved_sessions(self) -> List[Dict[str, Any]]:
        """
        List all saved sessions.
        
        Returns:
            List[Dict[str, Any]]: List of session metadata
        """
        results = []
        
        # Check all supported file types
        for extension in [".json", ".csv", ".yaml", ".yml"]:
            for file in self.data_dir.glob(f"*{extension}"):
                # Extract basic file information
                file_info = {
                    "filename": file.name,
                    "path": str(file),
                    "size": file.stat().st_size,
                    "modified": datetime.datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                    "format": extension[1:]  # Remove leading dot
                }
                
                # Try to extract more metadata if possible
                try:
                    if extension == ".json":
                        with open(file, 'r') as f:
                            # Read just the first 1000 bytes to look for metadata
                            header = f.read(1000)
                            if '"metadata"' in header:
                                # Reopen and load properly
                                f.seek(0)
                                data = json.load(f)
                                if "metadata" in data:
                                    file_info["metadata"] = data["metadata"]
                except Exception as e:
                    logger.warning(f"Could not extract metadata from {file}: {str(e)}")
                
                results.append(file_info)
        
        return results
    
    def initialize_database(self) -> bool:
        """
        Initialize the SQLite database for advanced searching.
        
        Returns:
            bool: True if initialization was successful
        """
        if self.db_initialized:
            return True
        
        db_path = self.data_dir / "measurements.db"
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time REAL,
                end_time REAL,
                config TEXT,
                metadata TEXT
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp REAL,
                raw_data TEXT,
                processed_data TEXT,
                ratios TEXT,
                features TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
            ''')
            
            conn.commit()
            conn.close()
            
            self.db_initialized = True
            logger.info(f"Initialized database at {db_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            return False
    
    def index_file(self, filename: str) -> bool:
        """
        Index a data file into the database for faster searching.
        
        Args:
            filename: Name of the file to index
            
        Returns:
            bool: True if indexing was successful
        """
        # Ensure database is initialized
        if not self.db_initialized and not self.initialize_database():
            return False
        
        try:
            # Load the session data
            session_data = self.load_session(filename)
            
            if not session_data:
                logger.warning(f"No data found in {filename}")
                return False
            
            # Extract metadata and measurements
            metadata = session_data.get("metadata", {})
            measurements = session_data.get("measurements", [])
            
            if not metadata or not measurements:
                logger.warning(f"Missing metadata or measurements in {filename}")
                return False
            
            # Connect to database
            db_path = self.data_dir / "measurements.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Insert session
            session_id = metadata.get("session_id", f"file_{filename}")
            start_time = metadata.get("start_time", 0)
            end_time = metadata.get("end_time", 0)
            config = json.dumps(metadata.get("config", {}))
            metadata_json = json.dumps(metadata)
            
            cursor.execute(
                '''INSERT OR REPLACE INTO sessions 
                   (session_id, start_time, end_time, config, metadata) 
                   VALUES (?, ?, ?, ?, ?)''',
                (session_id, start_time, end_time, config, metadata_json)
            )
            
            # Insert measurements
            for measurement in measurements:
                timestamp = measurement.get("timestamp", 0)
                raw_data = json.dumps(measurement.get("raw", {}))
                processed_data = json.dumps(measurement.get("processed", {}))
                ratios = json.dumps(measurement.get("ratios", {}))
                features = json.dumps(measurement.get("features", {}))
                
                cursor.execute(
                    '''INSERT INTO measurements 
                       (session_id, timestamp, raw_data, processed_data, ratios, features) 
                       VALUES (?, ?, ?, ?, ?, ?)''',
                    (session_id, timestamp, raw_data, processed_data, ratios, features)
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Indexed {len(measurements)} measurements from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing file {filename}: {str(e)}")
            return False
    
    def search_measurements(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search indexed measurements based on criteria.
        
        Args:
            criteria: Search criteria
                - session_id: Session ID to filter by
                - time_range: Tuple of (start_time, end_time)
                - channels: Dict of channel name to (min, max) value range
                - ratios: Dict of ratio name to (min, max) value range
                
        Returns:
            List[Dict[str, Any]]: Matching measurements
        """
        # Ensure database is initialized
        if not self.db_initialized and not self.initialize_database():
            return []
        
        try:
            # Connect to database
            db_path = self.data_dir / "measurements.db"
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()
            
            # Build the query
            query = "SELECT * FROM measurements WHERE 1=1"
            params = []
            
            # Filter by session_id
            if "session_id" in criteria:
                query += " AND session_id = ?"
                params.append(criteria["session_id"])
            
            # Filter by time range
            if "time_range" in criteria:
                start_time, end_time = criteria["time_range"]
                query += " AND timestamp BETWEEN ? AND ?"
                params.extend([start_time, end_time])
            
            # Execute query
            cursor.execute(query, params)
            results = []
            
            # Process results
            for row in cursor.fetchall():
                # Convert row to dict
                measurement = dict(row)
                
                # Parse JSON fields
                measurement["raw"] = json.loads(measurement.pop("raw_data"))
                measurement["processed"] = json.loads(measurement.pop("processed_data"))
                measurement["ratios"] = json.loads(measurement.pop("ratios"))
                measurement["features"] = json.loads(measurement.pop("features"))
                
                # Apply channel filters
                if "channels" in criteria:
                    passes_channel_filter = True
                    for channel, (min_val, max_val) in criteria["channels"].items():
                        if channel in measurement["raw"]:
                            value = measurement["raw"][channel]
                            if value < min_val or value > max_val:
                                passes_channel_filter = False
                                break
                    
                    if not passes_channel_filter:
                        continue
                
                # Apply ratio filters
                if "ratios" in criteria:
                    passes_ratio_filter = True
                    for ratio, (min_val, max_val) in criteria["ratios"].items():
                        if ratio in measurement["ratios"]:
                            value = measurement["ratios"][ratio]
                            if value < min_val or value > max_val:
                                passes_ratio_filter = False
                                break
                    
                    if not passes_ratio_filter:
                        continue
                
                results.append(measurement)
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error searching measurements: {str(e)}")
            return []
    
    def export_to_csv(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], 
                     filepath: Union[str, Path],
                     format_type: str = "measurements") -> bool:
        """
        Export data to CSV format.
        
        Args:
            data: Data to export (session or list of measurements)
            filepath: Path to save the CSV file
            format_type: Type of formatting to use ('measurements', 'timeseries', 'channels')
            
        Returns:
            bool: True if export was successful
        """
        try:
            filepath = Path(filepath)
            
            # Create parent directories if they don't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Extract measurements from session if needed
            measurements = []
            
            if isinstance(data, dict) and "measurements" in data:
                # This is a session structure
                measurements = data["measurements"]
            elif isinstance(data, list):
                # This is a list of measurements
                measurements = data
            else:
                logger.error("Invalid data format for CSV export")
                return False
            
            if not measurements:
                logger.warning("No measurements to export")
                return False
            
            # Select export format
            if format_type == "measurements":
                # One row per measurement with all channels as columns
                return self._export_measurements_csv(measurements, filepath)
            elif format_type == "timeseries":
                # One row per timestamp with channels as columns
                return self._export_timeseries_csv(measurements, filepath)
            elif format_type == "channels":
                # One file per channel with timestamps as rows
                return self._export_channels_csv(measurements, filepath)
            else:
                logger.error(f"Unknown export format: {format_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            return False
    
    def _convert_numpy_types(self, value: Any) -> Any:
        """
        Convert numpy types to native Python types.
        
        Args:
            value: Value to convert
            
        Returns:
            Converted value
        """
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                return value.item()
            except:
                try:
                    return float(value)
                except:
                    return str(value)
        elif isinstance(value, dict):
            return {k: self._convert_numpy_types(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._convert_numpy_types(v) for v in value]
        else:
            return value
    
    def _flatten_measurement(self, measurement: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten a measurement's nested structure for CSV export.
        
        Args:
            measurement: Measurement to flatten
            
        Returns:
            Dict[str, Any]: Flattened measurement
        """
        flat_m = {
            "timestamp": measurement.get("timestamp", ""),
            "session_id": measurement.get("session_id", "")
        }
        
        # Add raw channels
        if "raw" in measurement:
            for channel, value in measurement["raw"].items():
                flat_m[f"raw_{channel}"] = self._convert_numpy_types(value)
        
        # Add processed channels
        if "processed" in measurement:
            for category, data in measurement["processed"].items():
                # If it's a nested dictionary (like filtered or normalized)
                if isinstance(data, dict):
                    for channel, value in data.items():
                        # Skip nested structures within the data
                        if not isinstance(value, (dict, list)):
                            flat_m[f"processed_{category}_{channel}"] = self._convert_numpy_types(value)
                else:
                    # Direct value (not a dictionary)
                    # Skip if it's a complex object
                    if not isinstance(data, (dict, list)):
                        flat_m[f"processed_{category}"] = self._convert_numpy_types(data)
        
        # Add ratios
        if "ratios" in measurement:
            for ratio, value in measurement["ratios"].items():
                flat_m[f"ratio_{ratio}"] = self._convert_numpy_types(value)
        
        return flat_m
    
    def _save_json(self, data: Dict[str, Any], filepath: Path) -> str:
        """
        Save data to JSON format.
        
        Args:
            data: Data to save
            filepath: Path to save the file
            
        Returns:
            str: Path to the saved file
        """
        try:
            # Create parent directories if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Use a custom serializer to handle numpy arrays and other non-serializable types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, datetime.datetime):
                        return obj.isoformat()
                    return super().default(obj)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, cls=NumpyEncoder, indent=2)
            
            logger.info(f"Saved JSON data to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving JSON data: {str(e)}")
            raise
    
    def _load_json(self, filepath: Path) -> Dict[str, Any]:
        """
        Load data from JSON format.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Dict[str, Any]: Loaded data
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded JSON data from {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading JSON data: {str(e)}")
            raise
    
    def _save_csv(self, session_data: Dict[str, Any], filepath: Path) -> str:
        """
        Save session data to CSV format.
        
        Args:
            session_data: Session data to save
            filepath: Path to save the file
            
        Returns:
            str: Path to the saved file
        """
        try:
            # Extract measurements
            measurements = session_data.get("measurements", [])
            if not measurements:
                logger.warning("No measurements to save to CSV")
                return str(filepath)
            
            # Save metadata to separate file
            metadata_filepath = filepath.with_name(f"{filepath.stem}_metadata.json")
            with open(metadata_filepath, 'w') as f:
                # Use the JSON encoder from _save_json
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        if isinstance(obj, np.integer):
                            return int(obj)
                        if isinstance(obj, np.floating):
                            return float(obj)
                        if isinstance(obj, datetime.datetime):
                            return obj.isoformat()
                        return super().default(obj)
                
                json.dump(session_data.get("metadata", {}), f, cls=NumpyEncoder, indent=2)
            
            # Flatten measurements and export
            return self._export_measurements_csv(measurements, filepath)
            
        except Exception as e:
            logger.error(f"Error saving CSV data: {str(e)}")
            raise
    
    def _load_csv(self, filepath: Path) -> Dict[str, Any]:
        """
        Load session data from CSV format.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Dict[str, Any]: Loaded session data
        """
        try:
            # Check for metadata file
            metadata_filepath = filepath.with_name(f"{filepath.stem}_metadata.json")
            metadata = {}
            if metadata_filepath.exists():
                with open(metadata_filepath, 'r') as f:
                    metadata = json.load(f)
            
            # Read CSV file
            measurements = []
            with open(filepath, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                for row in reader:
                    measurement = {
                        "raw": {},
                        "processed": {},
                        "ratios": {}
                    }
                    
                    # Parse values
                    for i, field in enumerate(header):
                        if i < len(row):
                            value = row[i]
                            
                            # Try to convert to number if possible
                            try:
                                if '.' in value:
                                    value = float(value)
                                else:
                                    value = int(value)
                            except (ValueError, TypeError):
                                pass
                            
                            if field == "timestamp":
                                measurement["timestamp"] = value
                            elif field == "session_id":
                                measurement["session_id"] = value
                            elif field.startswith("raw_"):
                                channel = field[4:]
                                measurement["raw"][channel] = value
                            elif field.startswith("processed_"):
                                # Handle flattened processed fields (processed_category_channel)
                                parts = field[10:].split('_', 1)
                                if len(parts) > 1:
                                    category, channel = parts
                                    if category not in measurement["processed"]:
                                        measurement["processed"][category] = {}
                                    measurement["processed"][category][channel] = value
                                else:
                                    # Direct processed value
                                    measurement["processed"][parts[0]] = value
                            elif field.startswith("ratio_"):
                                ratio = field[6:]
                                measurement["ratios"][ratio] = value
                    
                    measurements.append(measurement)
            
            # Build session structure
            session = {
                "metadata": metadata,
                "measurements": measurements
            }
            
            logger.info(f"Loaded CSV data from {filepath}")
            return session
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {str(e)}")
            raise
    
    def _save_yaml(self, data: Dict[str, Any], filepath: Path) -> str:
        """
        Save data to YAML format.
        
        Args:
            data: Data to save
            filepath: Path to save the file
            
        Returns:
            str: Path to the saved file
        """
        try:
            # Create parent directories if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Define a custom representer for numpy arrays
            def numpy_array_representer(dumper, data):
                return dumper.represent_list(data.tolist())
            
            # Add the representer to PyYAML
            yaml.add_representer(np.ndarray, numpy_array_representer)
            
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            logger.info(f"Saved YAML data to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving YAML data: {str(e)}")
            raise
    
    def _load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """
        Load data from YAML format.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Dict[str, Any]: Loaded data
        """
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            
            logger.info(f"Loaded YAML data from {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading YAML data: {str(e)}")
            raise
    
    def _export_measurements_csv(self, measurements: List[Dict[str, Any]], filepath: Path) -> str:
        """
        Export measurements to CSV with completely flattened structure.
        
        Args:
            measurements: List of measurements to export
            filepath: Path to save the CSV file
            
        Returns:
            str: Path to the saved file
        """
        try:
            # Create parent directories if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Build a flattened version of all measurements
            flattened_measurements = []
            
            for m in measurements:
                flat_m = self._flatten_measurement(m)
                flattened_measurements.append(flat_m)
            
            # Get all possible column names from all measurements
            all_columns = set()
            for flat_m in flattened_measurements:
                all_columns.update(flat_m.keys())
            
            # Create a sorted list of columns with a specific order
            # First the basic fields, then raw, then processed, then ratios
            columns = ["timestamp", "session_id"]
            
            # Add raw channels in sorted order
            raw_columns = sorted([col for col in all_columns if col.startswith("raw_")])
            columns.extend(raw_columns)
            
            # Add processed channels in sorted order
            processed_columns = sorted([col for col in all_columns if col.startswith("processed_")])
            columns.extend(processed_columns)
            
            # Add ratio columns in sorted order
            ratio_columns = sorted([col for col in all_columns if col.startswith("ratio_")])
            columns.extend(ratio_columns)
            
            # Make sure we haven't missed any
            remaining_columns = sorted([col for col in all_columns if col not in columns])
            columns.extend(remaining_columns)
            
            # Write CSV file
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(flattened_measurements)
            
            logger.info(f"Exported {len(measurements)} measurements to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting measurements to CSV: {str(e)}", exc_info=True)
            return str(filepath)
    
    def _export_timeseries_csv(self, measurements: List[Dict[str, Any]], filepath: Path) -> bool:
        """
        Export measurements to CSV as a time series.
        
        Args:
            measurements: List of measurements to export
            filepath: Path to save the CSV file
            
        Returns:
            bool: True if export was successful
        """
        try:
            # Sort measurements by timestamp
            sorted_measurements = sorted(measurements, key=lambda m: m.get("timestamp", 0))
            
            # Use standard measurements export - already sorted
            return self._export_measurements_csv(sorted_measurements, filepath)
            
        except Exception as e:
            logger.error(f"Error exporting timeseries to CSV: {str(e)}")
            return False
    
    def _export_channels_csv(self, measurements: List[Dict[str, Any]], filepath: Path) -> bool:
        """
        Export measurements to multiple CSV files, one per channel.
        
        Args:
            measurements: List of measurements to export
            filepath: Path to use as base for the CSV files
            
        Returns:
            bool: True if export was successful
        """
        try:
            # Sort measurements by timestamp
            sorted_measurements = sorted(measurements, key=lambda m: m.get("timestamp", 0))
            
            # Get all unique channels
            raw_channels = set()
            for m in sorted_measurements:
                if "raw" in m:
                    raw_channels.update(m["raw"].keys())
            
            # Create a directory for the channel files
            directory = filepath.parent / filepath.stem
            directory.mkdir(parents=True, exist_ok=True)
            
            # Export each channel to a separate file
            for channel in sorted(raw_channels):
                channel_filepath = directory / f"{channel}.csv"
                
                with open(channel_filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", channel])
                    
                    for m in sorted_measurements:
                        if "raw" in m and channel in m["raw"]:
                            writer.writerow([m.get("timestamp", ""), m["raw"][channel]])
            
            # Export ratios to separate files as well
            ratio_names = set()
            for m in sorted_measurements:
                if "ratios" in m:
                    ratio_names.update(m["ratios"].keys())
            
            for ratio in sorted(ratio_names):
                ratio_filepath = directory / f"ratio_{ratio}.csv"
                
                with open(ratio_filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", ratio])
                    
                    for m in sorted_measurements:
                        if "ratios" in m and ratio in m["ratios"]:
                            writer.writerow([m.get("timestamp", ""), m["ratios"][ratio]])
            
            logger.info(f"Exported {len(raw_channels)} channels to {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting channels to CSV: {str(e)}")
            return False


# Create a global instance for convenience
data_storage = DataStorage()