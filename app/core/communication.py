"""
Handles low-level serial communication with the ESP32 device.
"""
import serial
import serial.tools.list_ports
import time
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
import re

from config import settings
from utils.logging import get_logger
from core.exceptions import (
    CommunicationError, CommandTimeoutError, InvalidResponseError, 
    DeviceDisconnectedError, ProtocolError
)
from core.protocol import Protocol

# Initialize module logger
logger = get_logger(__name__)

class SerialCommunication:
    """
    Handles low-level serial communication with the ESP32 device.
    
    This class manages the serial connection and raw data transmission/reception.
    """
    
    def __init__(self):
        """Initialize the serial communication handler."""
        self._port = None
        self._serial = None
        self._connected = False
        self._reading_thread = None
        self._stop_event = threading.Event()
        self._data_buffer = ""
        self._data_callback = None
        self._lock = threading.Lock()
    
    def list_available_ports(self) -> List[str]:
        """
        List all available serial ports.
        
        Returns:
            List[str]: List of available serial port names
        """
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append(port.device)
        return ports
    
    def detect_device_port(self) -> Optional[str]:
        """
        Try to automatically detect the ESP32 device port.
        
        Returns:
            Optional[str]: Detected port name or None if not found
        """
        # Common vendor/product IDs for ESP32 (may need expansion)
        ESP32_VIDS_PIDS = [
            (0x10C4, 0xEA60),  # Silicon Labs CP210x
            (0x1A86, 0x7523),  # QinHeng CH340
            (0x0403, 0x6001),  # FTDI FT232
        ]
        
        for port in serial.tools.list_ports.comports():
            # Check if the device has a matching VID/PID
            if hasattr(port, 'vid') and hasattr(port, 'pid'):
                if (port.vid, port.pid) in ESP32_VIDS_PIDS:
                    return port.device
            
            # Check description for common ESP32 indicators
            if hasattr(port, 'description'):
                if any(x in port.description.lower() for x in ['esp32', 'cp210x', 'ch340', 'ft232']):
                    return port.device
        
        return None
    
    def connect(self, port: Optional[str] = None, baudrate: Optional[int] = None) -> bool:
        """
        Connect to the ESP32 device.
        
        Args:
            port: Serial port to connect to, or None to auto-detect
            baudrate: Baudrate for the connection
            
        Returns:
            bool: True if connection successful
            
        Raises:
            CommunicationError: If connection fails
        """
        if self._connected:
            self.disconnect()
        
        # Use provided values or settings defaults
        if port is None:
            port = self.detect_device_port()
            if port is None:
                port = settings.get("DEFAULT_PORT")
                if port is None:
                    raise CommunicationError("No port specified and auto-detection failed")
        
        if baudrate is None:
            baudrate = settings.DEFAULT_BAUDRATE
        
        self._port = port
        timeout = settings.get("SERIAL_TIMEOUT", 1.0)
        
        logger.debug("Connecting to %s at %d baud", port, baudrate)
        
        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Start the reading thread
            self._stop_event.clear()
            self._reading_thread = threading.Thread(
                target=self._read_thread,
                name="SerialReadThread",
                daemon=True
            )
            self._reading_thread.start()
            
            # Connection established
            self._connected = True
            logger.info("Connected to device on %s at %d baud", port, baudrate)
            return True
                
        except serial.SerialException as e:
            logger.error("Failed to connect to %s: %s", port, str(e))
            raise CommunicationError(f"Connection failed: {str(e)}") from e
    
    def disconnect(self) -> None:
        """
        Disconnect from the device.
        """
        logger.debug("Disconnecting from device")
        
        # Stop the reading thread
        if self._reading_thread and self._reading_thread.is_alive():
            self._stop_event.set()
            self._reading_thread.join(timeout=1.0)
        
        # Close the serial connection
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception as e:
                logger.warning("Error closing serial port: %s", str(e))
        
        # Reset state
        self._serial = None
        self._connected = False
        self._port = None
        self._data_buffer = ""
        
        logger.info("Disconnected from device")
    
    def is_connected(self) -> bool:
        """
        Check if the device is currently connected.
        
        Returns:
            bool: True if connected
        """
        return (self._connected and 
                self._serial is not None and 
                self._serial.is_open and
                self._reading_thread is not None and 
                self._reading_thread.is_alive())
    
    def send_data(self, data: str) -> None:
        """
        Send raw data to the device.
        
        Args:
            data: Data to send
            
        Raises:
            DeviceDisconnectedError: If device is not connected
        """
        if not self.is_connected():
            raise DeviceDisconnectedError("Device is not connected")
        
        with self._lock:
            try:
                self._serial.write(data.encode('utf-8'))
                self._serial.flush()
                logger.debug("Sent data: %s", data.strip())
            except Exception as e:
                logger.error("Error sending data: %s", str(e))
                raise DeviceDisconnectedError(f"Error sending data: {str(e)}") from e
    
    def send_binary(self, data: bytes) -> None:
        """
        Send binary data to the device.
        
        Args:
            data: Binary data to send
            
        Raises:
            DeviceDisconnectedError: If device is not connected
        """
        if not self.is_connected():
            raise DeviceDisconnectedError("Device is not connected")
        
        with self._lock:
            try:
                self._serial.write(data)
                self._serial.flush()
                logger.debug("Sent %d bytes of binary data", len(data))
            except Exception as e:
                logger.error("Error sending binary data: %s", str(e))
                raise DeviceDisconnectedError(f"Error sending binary data: {str(e)}") from e
    
    def register_data_callback(self, callback: Callable[[str], None]) -> None:
        """
        Register a callback for incoming data.
        
        Args:
            callback: Function to call with each line of data
        """
        self._data_callback = callback
    
    def unregister_data_callback(self) -> None:
        """Unregister the data callback."""
        self._data_callback = None
    
    def _read_thread(self) -> None:
        """
        Background thread that reads data from the serial port.
        """
        logger.debug("Serial read thread started")
        
        while not self._stop_event.is_set():
            if not self._serial or not self._serial.is_open:
                # Serial port closed or not available
                logger.warning("Serial port closed unexpectedly")
                break
            
            try:
                # Read available data
                if self._serial.in_waiting > 0:
                    data = self._serial.read(self._serial.in_waiting).decode('utf-8', errors='replace')
                    self._process_incoming_data(data)
                else:
                    # No data available, sleep briefly to avoid busy waiting
                    time.sleep(0.01)
            except serial.SerialException as e:
                logger.error("Serial read error: %s", str(e))
                break
            except Exception as e:
                logger.error("Error in read thread: %s", str(e), exc_info=True)
                # Continue trying to read unless stop requested
                time.sleep(0.1)
        
        logger.debug("Serial read thread stopped")
    
    def _process_incoming_data(self, data: str) -> None:
        """
        Process incoming data from the serial port.
        
        This method handles buffering partial messages and extracting complete lines.
        
        Args:
            data: Data received from the serial port
        """
        # Add data to buffer
        self._data_buffer += data
        
        # Process complete lines
        while '\n' in self._data_buffer:
            # Split at the first newline
            line, self._data_buffer = self._data_buffer.split('\n', 1)
            
            # Skip empty lines
            if not line.strip():
                continue
            
            # Call the callback if registered
            if self._data_callback:
                try:
                    self._data_callback(line)
                except Exception as e:
                    logger.error("Error in data callback: %s", str(e), exc_info=True)
            
            # Log the received data
            logger.debug("Received data: %s", line)