"""
Data acquisition management for the SpectroNeph system.

This module handles all data acquisition functionality, including:
- Configuration of acquisition parameters
- Management of acquisition sessions
- Handling of real-time and batch acquisition
- Buffering of acquired data
"""

import time
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Deque
from collections import deque
import uuid
import json
import numpy as np

from config import settings
from utils.logging import get_logger
from core.device import DeviceManager
from hardware.nephelometer import Nephelometer, MeasurementMode, AgglutinationState

# Initialize logger
logger = get_logger(__name__)

class AcquisitionSession:
    """
    Manages a single data acquisition session.
    
    This class handles the configuration, execution, and data management
    for a single acquisition session, which may consist of multiple measurements.
    """
    
    def __init__(self, nephelometer: Nephelometer, session_id: Optional[str] = None):
        """
        Initialize a new acquisition session.
        
        Args:
            nephelometer: The nephelometer instance to use for data acquisition
            session_id: Optional session ID, will be generated if not provided
        """
        self.nephelometer = nephelometer
        self.session_id = session_id or str(uuid.uuid4())
        self.start_time = time.time()
        self.end_time = None
        self.measurements = []
        self.metadata = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "config": {}
        }
        self.data_callbacks = []
        self.is_running = False
        self._current_mode = None
        self._lock = threading.Lock()
        
        # Create a buffer for real-time data
        self.data_buffer = deque(maxlen=settings.get("BUFFER_SIZE", 1000))
        
        logger.info(f"Created acquisition session {self.session_id}")
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the acquisition session.
        
        Args:
            config: Configuration parameters
                - mode: Acquisition mode (single, continuous, kinetic)
                - duration: Duration of acquisition in seconds (for kinetic mode)
                - interval: Interval between measurements in seconds (for continuous mode)
                - samples: Number of samples to acquire (for kinetic mode)
                - sensor_config: Configuration for the nephelometer
                
        Returns:
            bool: True if configuration was successful
        """
        with self._lock:
            try:
                # Update metadata
                self.metadata["config"] = config.copy()
                
                # Apply sensor configuration if provided
                if "sensor_config" in config:
                    self.nephelometer.configure(config["sensor_config"])
                
                # Store the acquisition mode
                if "mode" in config:
                    mode = config["mode"].lower()
                    if mode in ["single", "continuous", "kinetic"]:
                        self._current_mode = mode
                    else:
                        logger.warning(f"Invalid acquisition mode: {mode}")
                        return False
                
                logger.info(f"Configured acquisition session {self.session_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error configuring acquisition session: {str(e)}", exc_info=True)
                return False
    
    def register_data_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function to be called when new data is acquired.
        
        Args:
            callback: Function to call with each new measurement
        """
        if callback not in self.data_callbacks:
            self.data_callbacks.append(callback)
    
    def unregister_data_callback(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Unregister a data callback.
        
        Args:
            callback: Callback to remove
            
        Returns:
            bool: True if callback was removed
        """
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
            return True
        return False
    
    def start(self) -> bool:
        """
        Start the acquisition session.
        
        Returns:
            bool: True if session was started successfully
        """
        if self.is_running:
            logger.warning(f"Acquisition session {self.session_id} already running")
            return False
        
        with self._lock:
            try:
                self.is_running = True
                
                # Get the acquisition mode and parameters
                mode = self._current_mode or "single"
                config = self.metadata["config"]
                
                # Start the appropriate acquisition mode
                if mode == "single":
                    # Single measurement mode
                    return self._start_single_acquisition()
                elif mode == "continuous":
                    # Continuous measurement mode
                    interval = config.get("interval", 1.0)
                    return self._start_continuous_acquisition(interval)
                elif mode == "kinetic":
                    # Kinetic measurement mode
                    duration = config.get("duration", 10.0)
                    samples_per_sec = config.get("samples_per_second", 2.0)
                    return self._start_kinetic_acquisition(duration, samples_per_sec)
                else:
                    logger.error(f"Unknown acquisition mode: {mode}")
                    self.is_running = False
                    return False
                    
            except Exception as e:
                logger.error(f"Error starting acquisition session: {str(e)}", exc_info=True)
                self.is_running = False
                return False
    
    def stop(self) -> bool:
        """
        Stop the acquisition session.
        
        Returns:
            bool: True if session was stopped successfully
        """
        if not self.is_running:
            return True
        
        with self._lock:
            try:
                # Stop the nephelometer
                result = self.nephelometer.stop_measurement()
                
                # Update session state
                self.is_running = False
                self.end_time = time.time()
                self.metadata["end_time"] = self.end_time
                self.metadata["duration"] = self.end_time - self.start_time
                
                logger.info(f"Stopped acquisition session {self.session_id}")
                
                return result
                
            except Exception as e:
                logger.error(f"Error stopping acquisition session: {str(e)}", exc_info=True)
                return False
    
    def get_data(self) -> List[Dict[str, Any]]:
        """
        Get all measurements in this session.
        
        Returns:
            List[Dict[str, Any]]: List of measurements
        """
        with self._lock:
            return self.measurements.copy()
    
    def get_latest_data(self, count: int = 1) -> List[Dict[str, Any]]:
        """
        Get the most recent measurements.
        
        Args:
            count: Number of measurements to return
            
        Returns:
            List[Dict[str, Any]]: List of recent measurements
        """
        with self._lock:
            return self.measurements[-count:] if count > 0 else []
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get session metadata.
        
        Returns:
            Dict[str, Any]: Session metadata
        """
        with self._lock:
            return self.metadata.copy()
    
    def _start_single_acquisition(self) -> bool:
        """
        Start a single acquisition.
        
        Returns:
            bool: True if acquisition was started successfully
        """
        try:
            # Take a single measurement
            measurement = self.nephelometer.take_single_measurement(
                subtract_background=self.metadata["config"].get("subtract_background", True)
            )
            
            # Add measurement to the session
            self._add_measurement(measurement)
            
            # End the session
            self.is_running = False
            self.end_time = time.time()
            self.metadata["end_time"] = self.end_time
            self.metadata["duration"] = self.end_time - self.start_time
            
            logger.info(f"Completed single acquisition for session {self.session_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in single acquisition: {str(e)}", exc_info=True)
            self.is_running = False
            return False
    
    def _start_continuous_acquisition(self, interval: float) -> bool:
        """
        Start continuous acquisition.
        
        Args:
            interval: Interval between measurements in seconds
            
        Returns:
            bool: True if acquisition was started successfully
        """
        try:
            # Start continuous measurement
            result = self.nephelometer.start_continuous_measurement(
                interval_seconds=interval,
                callback=self._measurement_callback,
                subtract_background=self.metadata["config"].get("subtract_background", True)
            )
            
            if result:
                logger.info(f"Started continuous acquisition for session {self.session_id} with {interval}s interval")
            else:
                logger.error(f"Failed to start continuous acquisition for session {self.session_id}")
                self.is_running = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error starting continuous acquisition: {str(e)}", exc_info=True)
            self.is_running = False
            return False
    
    def _start_kinetic_acquisition(self, duration: float, samples_per_sec: float) -> bool:
        """
        Start kinetic acquisition.
        
        Args:
            duration: Duration of acquisition in seconds
            samples_per_sec: Samples per second
            
        Returns:
            bool: True if acquisition was started successfully
        """
        try:
            # Start kinetic measurement
            result = self.nephelometer.start_kinetic_measurement(
                duration_seconds=duration,
                samples_per_second=samples_per_sec,
                callback=self._measurement_callback,
                subtract_background=self.metadata["config"].get("subtract_background", True)
            )
            
            if result:
                logger.info(f"Started kinetic acquisition for session {self.session_id} "
                           f"with {duration}s duration at {samples_per_sec}Hz")
            else:
                logger.error(f"Failed to start kinetic acquisition for session {self.session_id}")
                self.is_running = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error starting kinetic acquisition: {str(e)}", exc_info=True)
            self.is_running = False
            return False
    
    def _measurement_callback(self, measurement: Dict[str, Any]) -> None:
        """
        Callback for new measurements.
        
        Args:
            measurement: Measurement data
        """
        # Add measurement to the session
        self._add_measurement(measurement)
    
    def _add_measurement(self, measurement: Dict[str, Any]) -> None:
        """
        Add a measurement to the session with improved debugging.
        
        Args:
            measurement: Measurement data
        """
        print("DEBUG: Entering _add_measurement method")
        try:
            # Use a non-blocking approach to acquire the lock
            if not self._lock.acquire(timeout=2):
                print("DEBUG: Could not acquire lock after 2 seconds!")
                return
            
            try:
                print("DEBUG: Lock acquired, adding session metadata")
                # Add session metadata to the measurement
                measurement["session_id"] = self.session_id
                if "timestamp" not in measurement:
                    measurement["timestamp"] = time.time()
                
                print("DEBUG: Adding to measurements list")
                # Add to measurements list
                self.measurements.append(measurement)
                
                print("DEBUG: Adding to data buffer")
                # Add to buffer for real-time access
                self.data_buffer.append(measurement)
                
                print(f"DEBUG: Calling {len(self.data_callbacks)} registered callbacks")
                # Call registered callbacks
                for i, callback in enumerate(self.data_callbacks):
                    try:
                        # print(f"DEBUG: Calling callback {i+1}")
                        callback(measurement)
                        print(f"DEBUG: Callback {i+1} completed")
                    except Exception as e:
                        print(f"DEBUG: Error in callback {i+1}: {str(e)}")
                        logger.error(f"Error in measurement callback: {str(e)}", exc_info=True)
                        
                print("DEBUG: All callbacks complete")
            finally:
                print("DEBUG: Releasing lock")
                self._lock.release()
                
            print("DEBUG: _add_measurement completed successfully")
        except Exception as e:
            print(f"DEBUG: Exception in _add_measurement: {str(e)}")
            logger.error(f"Error adding measurement: {str(e)}", exc_info=True)

class DataAcquisitionManager:
    """
    Manages data acquisition across multiple sessions.
    
    This class provides a centralized interface for data acquisition,
    including session management, configuration, and data access.
    """
    
    def __init__(self, device_manager: Optional[DeviceManager] = None):
        """
        Initialize the data acquisition manager.
        
        Args:
            device_manager: Optional device manager, will be created if not provided
        """
        self.device_manager = device_manager
        self.nephelometer = None
        self.sessions = {}  # Maps session IDs to AcquisitionSession objects
        self.active_session_id = None
        self._lock = threading.Lock()
        
        logger.info("Initialized data acquisition manager")
    
    def connect(self, device_manager: Optional[DeviceManager] = None, port: Optional[str] = None) -> bool:
        """
        Connect to the device.
        
        Args:
            device_manager: Optional device manager to use
            port: Optional port to connect to
            
        Returns:
            bool: True if connection was successful
        """
        with self._lock:
            try:
                # Use provided device manager or create a new one
                if device_manager:
                    self.device_manager = device_manager
                elif not self.device_manager:
                    self.device_manager = DeviceManager()
                
                # Connect to the device
                if not self.device_manager.is_connected():
                    if not self.device_manager.connect(port=port):
                        logger.error("Failed to connect to device")
                        return False
                
                # Create nephelometer instance
                self.nephelometer = Nephelometer(self.device_manager)
                
                # Initialize nephelometer
                if not self.nephelometer.initialize():
                    logger.error("Failed to initialize nephelometer")
                    return False
                
                logger.info("Connected to device and initialized nephelometer")
                return True
                
            except Exception as e:
                logger.error(f"Error connecting to device: {str(e)}", exc_info=True)
                return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the device.
        
        Returns:
            bool: True if disconnection was successful
        """
        with self._lock:
            try:
                # Stop all active sessions
                for session_id, session in self.sessions.items():
                    if session.is_active():
                        session.stop()
                
                # Disconnect from the device
                if self.device_manager and self.device_manager.is_connected():
                    self.device_manager.disconnect()
                
                logger.info("Disconnected from device")
                return True
                
            except Exception as e:
                logger.error(f"Error disconnecting from device: {str(e)}", exc_info=True)
                return False
    
    def create_session(self, config: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None) -> str:
        """
        Create a new acquisition session.
        
        Args:
            config: Optional configuration for the session
            session_id: Optional session ID, will be generated if not provided
            
        Returns:
            str: Session ID
        """
        with self._lock:
            # Check if nephelometer is initialized
            if not self.nephelometer:
                if not self.connect():
                    raise RuntimeError("Unable to connect to device")
            
            # Create new session
            session = AcquisitionSession(self.nephelometer, session_id)
            
            # Configure the session if config is provided
            if config:
                session.configure(config)
            
            # Store the session
            self.sessions[session.session_id] = session
            
            # Set as active session
            self.active_session_id = session.session_id
            
            logger.info(f"Created new acquisition session: {session.session_id}")
            
            return session.session_id
    
    def start_session(self, session_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Start an acquisition session.
        
        Args:
            session_id: Session ID to start, or None for the active session
            config: Optional configuration to apply before starting
            
        Returns:
            bool: True if session was started successfully
        """
        with self._lock:
            # Determine which session to use
            session_id = session_id or self.active_session_id
            
            if not session_id:
                # No active session, create a new one
                session_id = self.create_session(config)
            elif session_id not in self.sessions:
                logger.error(f"Session {session_id} not found")
                return False
            
            session = self.sessions[session_id]
            
            # Apply configuration if provided
            if config and not session.configure(config):
                logger.error(f"Failed to configure session {session_id}")
                return False
            
            # Start the session
            if not session.start():
                logger.error(f"Failed to start session {session_id}")
                return False
            
            # Set as active session
            self.active_session_id = session_id
            
            logger.info(f"Started acquisition session: {session_id}")
            
            return True
    
    def stop_session(self, session_id: Optional[str] = None) -> bool:
        """
        Stop an acquisition session.
        
        Args:
            session_id: Session ID to stop, or None for the active session
            
        Returns:
            bool: True if session was stopped successfully
        """
        with self._lock:
            # Determine which session to use
            session_id = session_id or self.active_session_id
            
            if not session_id or session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found")
                return False
            
            session = self.sessions[session_id]
            
            # Stop the session
            result = session.stop()
            
            logger.info(f"Stopped acquisition session: {session_id}")
            
            return result
    
    def get_session(self, session_id: Optional[str] = None) -> Optional[AcquisitionSession]:
        """
        Get an acquisition session.
        
        Args:
            session_id: Session ID to get, or None for the active session
            
        Returns:
            Optional[AcquisitionSession]: The acquisition session, or None if not found
        """
        session_id = session_id or self.active_session_id
        
        if not session_id or session_id not in self.sessions:
            return None
        
        return self.sessions[session_id]
    
    def get_session_data(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get data from an acquisition session.
        
        Args:
            session_id: Session ID to get data from, or None for the active session
            
        Returns:
            List[Dict[str, Any]]: List of measurements from the session
        """
        session = self.get_session(session_id)
        
        if not session:
            return []
        
        return session.get_data()
    
    def get_latest_data(self, session_id: Optional[str] = None, count: int = 1) -> List[Dict[str, Any]]:
        """
        Get the latest data from an acquisition session.
        
        Args:
            session_id: Session ID to get data from, or None for the active session
            count: Number of measurements to return
            
        Returns:
            List[Dict[str, Any]]: List of latest measurements from the session
        """
        session = self.get_session(session_id)
        
        if not session:
            return []
        
        return session.get_latest_data(count)
    
    def get_session_metadata(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata from an acquisition session.
        
        Args:
            session_id: Session ID to get metadata from, or None for the active session
            
        Returns:
            Dict[str, Any]: Session metadata
        """
        session = self.get_session(session_id)
        
        if not session:
            return {}
        
        return session.get_metadata()
    
    def list_sessions(self) -> List[str]:
        """
        List all acquisition sessions.
        
        Returns:
            List[str]: List of session IDs
        """
        return list(self.sessions.keys())
    
    def register_data_callback(self, callback: Callable[[Dict[str, Any]], None], 
                              session_id: Optional[str] = None) -> bool:
        """
        Register a callback for new data in a session.
        
        Args:
            callback: Function to call with each new measurement
            session_id: Session ID to register callback for, or None for the active session
            
        Returns:
            bool: True if callback was registered successfully
        """
        session = self.get_session(session_id)
        
        if not session:
            return False
        
        session.register_data_callback(callback)
        return True
    
    def unregister_data_callback(self, callback: Callable[[Dict[str, Any]], None],
                                session_id: Optional[str] = None) -> bool:
        """
        Unregister a data callback from a session.
        
        Args:
            callback: Callback to remove
            session_id: Session ID to unregister callback from, or None for the active session
            
        Returns:
            bool: True if callback was unregistered successfully
        """
        session = self.get_session(session_id)
        
        if not session:
            return False
        
        return session.unregister_data_callback(callback)
    
    def configure_acquisition(self, config: Dict[str, Any], session_id: Optional[str] = None) -> bool:
        """
        Configure an acquisition session.
        
        Args:
            config: Configuration parameters
            session_id: Session ID to configure, or None for the active session
            
        Returns:
            bool: True if configuration was successful
        """
        session = self.get_session(session_id)
        
        if not session:
            # Create a new session with this configuration
            self.create_session(config)
            return True
        
        return session.configure(config)
    
    def take_single_measurement(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Take a single measurement.
        
        This is a convenience method that creates a session, takes a measurement,
        and returns the result.
        
        Args:
            config: Optional configuration for the measurement
            
        Returns:
            Dict[str, Any]: Measurement data
        """
        # Create a new session
        session_id = self.create_session(config)
        
        # Configure for single measurement mode
        self.configure_acquisition({"mode": "single"}, session_id)
        
        # Start the session
        if not self.start_session(session_id):
            logger.error("Failed to start single measurement session")
            return {}
        
        # Get the measurement data
        data = self.get_session_data(session_id)
        
        return data[0] if data else {}

# Create a global instance for easy access
acquisition_manager = DataAcquisitionManager()