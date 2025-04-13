"""
Manages device connection, command execution, and data handling.

This module builds on top of the low-level communication module to provide
a higher-level interface for device interaction.
"""
import json
import threading
import queue
import time
from typing import Dict, List, Any, Optional, Tuple, Callable, Union

from config import settings
from utils.logging import get_logger
from core.exceptions import (
    CommunicationError, CommandTimeoutError, InvalidResponseError, 
    DeviceDisconnectedError, ProtocolError
)
from core.communication import SerialCommunication
from core.protocol import Protocol, MessageType, ResponseType, StatusCode

# Initialize module logger
logger = get_logger(__name__)

class DeviceManager:
    """
    Manages connection to and communication with the ESP32 device.
    
    This class builds on the low-level SerialCommunication to implement
    the command/response protocol and handle data streaming.
    """
    
    def __init__(self):
        """Initialize the device manager."""
        self._comm = SerialCommunication()
        self._comm.register_data_callback(self._on_data_received)
        
        self._command_queue = queue.Queue()
        self._response_queues = {}
        self._data_callbacks = {}
        self._event_callbacks = {}
        
        self._next_command_id = 1
        self._command_lock = threading.Lock()
        self._callback_lock = threading.Lock()
        
        # Device information
        self._device_info = {}
    
    def connect(self, port: Optional[str] = None, baudrate: Optional[int] = None) -> bool:
        """
        Connect to the device.
        
        Args:
            port: Serial port to connect to, or None to auto-detect
            baudrate: Baudrate for the connection
            
        Returns:
            bool: True if connection successful
        """
        try:
            # Connect to the device
            if not self._comm.connect(port, baudrate):
                return False
            
            # Query device information
            return self._query_device_info()
                
        except CommunicationError as e:
            logger.error("Connection error: %s", str(e))
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the device."""
        # Clear all pending commands and responses
        while not self._command_queue.empty():
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                break
        
        # Clear response queues
        self._response_queues.clear()
        
        # Disconnect the communication link
        self._comm.disconnect()
    
    def is_connected(self) -> bool:
        """
        Check if the device is currently connected.
        
        Returns:
            bool: True if connected
        """
        return self._comm.is_connected()
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the connected device.
        
        Returns:
            Dict: Device information
        """
        return self._device_info.copy()
    
    def send_command(self, command: str, params: Dict[str, Any] = None, 
                    timeout: float = 5.0) -> Dict[str, Any]:
        """
        Send a command to the device and wait for a response.
        
        Args:
            command: Command name
            params: Command parameters
            timeout: Timeout in seconds
            
        Returns:
            Dict: Response data
            
        Raises:
            CommandTimeoutError: If command times out
            InvalidResponseError: If invalid response received
            DeviceDisconnectedError: If device disconnects during command
        """
        if not self.is_connected():
            raise DeviceDisconnectedError("Device is not connected")
        
        params = params or {}
        
        # Generate a unique command ID
        with self._command_lock:
            cmd_id = self._next_command_id
            self._next_command_id = (self._next_command_id + 1) % 65536  # Wrap at 16 bits
        
        # Create response queue for this command
        response_queue = queue.Queue()
        self._response_queues[cmd_id] = response_queue
        
        # Create command message
        cmd_msg = Protocol.create_command(command, params, cmd_id)
        
        # Encode command to JSON
        cmd_json = Protocol.encode_message(cmd_msg)
        
        # Send the command
        try:
            self._comm.send_data(cmd_json)
            logger.debug("Sent command: %s (id=%d)", command, cmd_id)
        except Exception as e:
            # Remove the response queue
            self._response_queues.pop(cmd_id, None)
            logger.error("Error sending command: %s", str(e))
            raise DeviceDisconnectedError(f"Error sending command: {str(e)}") from e
        
        # Wait for response
        try:
            response = response_queue.get(timeout=timeout)
            logger.debug("Received response for command %s (id=%d): %s", 
                        command, cmd_id, response)
            
            # Check response status
            if response.get("status", -1) != StatusCode.SUCCESS:
                error_data = response.get("data", "Unknown error")
                logger.warning("Command %s failed: %s", command, error_data)
            
            return response
        except queue.Empty:
            logger.warning("Command timeout: %s (id=%d)", command, cmd_id)
            self._response_queues.pop(cmd_id, None)
            raise CommandTimeoutError(f"Command {command} timed out after {timeout}s")
        finally:
            # Clean up the response queue
            self._response_queues.pop(cmd_id, None)
    
    def register_data_callback(self, data_type: str, 
                              callback: Callable[[Dict[str, Any]], None]) -> str:
        """
        Register a callback for streaming data of a specific type.
        
        Args:
            data_type: Type of data to receive
            callback: Function to call when data is received
            
        Returns:
            str: Callback ID for later removal
        """
        with self._callback_lock:
            # Generate a unique callback ID
            callback_id = f"{data_type}_{id(callback)}"
            
            # Store the callback
            if data_type not in self._data_callbacks:
                self._data_callbacks[data_type] = {}
            
            self._data_callbacks[data_type][callback_id] = callback
            
            logger.debug("Registered callback %s for data type %s", callback_id, data_type)
            return callback_id
    
    def unregister_data_callback(self, callback_id: str) -> bool:
        """
        Unregister a data callback.
        
        Args:
            callback_id: Callback ID to remove
            
        Returns:
            bool: True if callback was removed
        """
        with self._callback_lock:
            # Parse the callback ID to get the data type
            try:
                data_type, _ = callback_id.split('_', 1)
            except ValueError:
                logger.warning("Invalid callback ID: %s", callback_id)
                return False
            
            # Remove the callback
            if data_type in self._data_callbacks:
                if callback_id in self._data_callbacks[data_type]:
                    self._data_callbacks[data_type].pop(callback_id)
                    logger.debug("Unregistered callback %s", callback_id)
                    
                    # Clean up empty dictionaries
                    if not self._data_callbacks[data_type]:
                        self._data_callbacks.pop(data_type)
                    
                    return True
            
            logger.warning("Callback %s not found", callback_id)
            return False
    
    def register_event_callback(self, event_type: str, 
                               callback: Callable[[Dict[str, Any]], None]) -> str:
        """
        Register a callback for device events.
        
        Args:
            event_type: Type of event to receive
            callback: Function to call when event is received
            
        Returns:
            str: Callback ID for later removal
        """
        with self._callback_lock:
            # Generate a unique callback ID
            callback_id = f"{event_type}_{id(callback)}"
            
            # Store the callback
            if event_type not in self._event_callbacks:
                self._event_callbacks[event_type] = {}
            
            self._event_callbacks[event_type][callback_id] = callback
            
            logger.debug("Registered callback %s for event type %s", callback_id, event_type)
            return callback_id
    
    def unregister_event_callback(self, callback_id: str) -> bool:
        """
        Unregister an event callback.
        
        Args:
            callback_id: Callback ID to remove
            
        Returns:
            bool: True if callback was removed
        """
        with self._callback_lock:
            # Parse the callback ID to get the event type
            try:
                event_type, _ = callback_id.split('_', 1)
            except ValueError:
                logger.warning("Invalid callback ID: %s", callback_id)
                return False
            
            # Remove the callback
            if event_type in self._event_callbacks:
                if callback_id in self._event_callbacks[event_type]:
                    self._event_callbacks[event_type].pop(callback_id)
                    logger.debug("Unregistered callback %s", callback_id)
                    
                    # Clean up empty dictionaries
                    if not self._event_callbacks[event_type]:
                        self._event_callbacks.pop(event_type)
                    
                    return True
            
            logger.warning("Callback %s not found", callback_id)
            return False
    
    def start_data_stream(self, data_type: str, params: Dict[str, Any] = None) -> bool:
        """
        Start a data stream from the device.
        
        Args:
            data_type: Type of data to stream
            params: Parameters for the stream
            
        Returns:
            bool: True if stream started successfully
        """
        params = params or {}
        
        # Send command to start streaming
        try:
            response = self.send_command("stream_start", {
                "type": data_type,
                **params
            })
            
            return response.get("status") == StatusCode.SUCCESS
        except Exception as e:
            logger.error("Error starting data stream: %s", str(e))
            return False
    
    def stop_data_stream(self, data_type: str) -> bool:
        """
        Stop a data stream.
        
        Args:
            data_type: Type of data stream to stop
            
        Returns:
            bool: True if stream stopped successfully
        """
        try:
            response = self.send_command("stream_stop", {
                "type": data_type
            })
            
            return response.get("status") == StatusCode.SUCCESS
        except Exception as e:
            logger.error("Error stopping data stream: %s", str(e))
            return False
    
    def _query_device_info(self) -> bool:
        """
        Query device information.
        
        Returns:
            bool: True if successful
        """
        try:
            # Try to get device information
            response = self.send_command("get_info", {}, timeout=2.0)
            
            if response.get("status") == StatusCode.SUCCESS:
                self._device_info = response.get("data", {})
                logger.info("Connected to device: %s", self._device_info.get("name", "Unknown"))
                return True
            else:
                logger.error("Failed to get device information: %s", response.get("data", "Unknown error"))
                return False
                
        except CommandTimeoutError:
            logger.error("Timeout querying device information")
            return False
        except Exception as e:
            logger.error("Error querying device information: %s", str(e))
            return False

    def _on_data_received(self, data: str) -> None:
        """
        Handle data received from the device.
        
        This callback is called by the SerialCommunication class when
        a complete line is received.
        
        Args:
            data: Raw data line received
        """
        # Skip empty lines
        if not data or not data.strip():
            return
            
        # Try to parse the data as JSON
        try:
            # Check if this is likely a JSON message
            if data.strip().startswith('{'):
                message = Protocol.decode_message(data)
                self._handle_message(message)
            else:
                # This is probably debug output from the device
                if "error" in data.lower() or "warning" in data.lower():
                    logger.warning("Device debug output: %s", data.strip())
                else:
                    logger.debug("Device output: %s", data.strip())
        except ProtocolError as e:  # <-- Added 'as e' here
            # This is expected sometimes and doesn't need to be logged every time
            if "Empty message" not in str(e) and "Non-JSON data" not in str(e):
                logger.debug("Protocol error: %s", str(e))
        except Exception as e:
            logger.error("Error handling message: %s", str(e), exc_info=True)
                            
    def _handle_message(self, message: Dict[str, Any]) -> None:
        """
        Handle a parsed message from the device.
        
        Args:
            message: Parsed JSON message
        """
        try:
            # Determine message type
            message_type = Protocol.get_message_type(message)
            
            # Handle based on message type
            if message_type == MessageType.RESPONSE:
                self._handle_response(message)
            elif message_type == MessageType.DATA:
                self._handle_data_message(message)
            elif message_type == MessageType.EVENT:
                self._handle_event_message(message)
            else:
                logger.warning("Received unexpected message type: %s", message_type)
        except ProtocolError as e:
            logger.warning("Protocol error: %s", str(e))
        except Exception as e:
            logger.error("Error in message handling: %s", str(e), exc_info=True)
    
    def _handle_response(self, response: Dict[str, Any]) -> None:
        """
        Handle a response message from the device.
        
        Args:
            response: Response message
        """
        # Validate the response
        try:
            Protocol.validate_response(response)
        except ProtocolError as e:
            logger.warning("Invalid response: %s", str(e))
            return
        
        # Check if this is a response to a known command
        cmd_id = response.get("id")
        if cmd_id in self._response_queues:
            # Put the response in the queue
            self._response_queues[cmd_id].put(response)
        else:
            logger.warning("Received response for unknown command ID: %s", cmd_id)
    
    def _handle_data_message(self, message: Dict[str, Any]) -> None:
        """
        Handle a data message from the device.
        
        Args:
            message: Data message
        """
        # Validate the data message
        try:
            Protocol.validate_data_message(message)
        except ProtocolError as e:
            logger.warning("Invalid data message: %s", str(e))
            return
        
        # Extract data type and payload
        data_type = message.get("type")
        if not data_type:
            logger.warning("Received data message without type: %s", message)
            return
        
        # Call registered callbacks for this data type
        with self._callback_lock:
            callbacks = self._data_callbacks.get(data_type, {}).copy()
        
        if not callbacks:
            # Log only occasionally to avoid flooding
            if hash(str(message)) % 100 == 0:
                logger.debug("Received data of type %s but no callbacks registered", data_type)
            return
        
        # Call each callback
        for callback_id, callback in callbacks.items():
            try:
                callback(message)
            except Exception as e:
                logger.error("Error in data callback %s: %s", callback_id, str(e), exc_info=True)
    
    def _handle_event_message(self, message: Dict[str, Any]) -> None:
        """
        Handle an event message from the device.
        
        Args:
            message: Event message
        """
        # Validate the event message
        try:
            Protocol.validate_event_message(message)
        except ProtocolError as e:
            logger.warning("Invalid event message: %s", str(e))
            return
        
        # Extract event type and payload
        event_type = message.get("type")
        if not event_type:
            logger.warning("Received event message without type: %s", message)
            return
        
        # Log the event
        logger.info("Received event: %s", event_type)
        
        # Call registered callbacks for this event type
        with self._callback_lock:
            callbacks = self._event_callbacks.get(event_type, {}).copy()
            
            # Also call callbacks for "all" events
            all_callbacks = self._event_callbacks.get("all", {}).copy()
            callbacks.update(all_callbacks)
        
        if not callbacks:
            logger.debug("Received event of type %s but no callbacks registered", event_type)
            return
        
        # Call each callback
        for callback_id, callback in callbacks.items():
            try:
                callback(message)
            except Exception as e:
                logger.error("Error in event callback %s: %s", callback_id, str(e), exc_info=True)