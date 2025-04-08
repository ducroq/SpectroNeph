"""
Implementation of the communication protocol between the application and the ESP32.
"""
import json
from typing import Dict, Any, Optional, Union, List, Tuple

from utils.logging import get_logger
from core.exceptions import ProtocolError, InvalidResponseError

# Initialize module logger
logger = get_logger(__name__)

class MessageType:
    """Message types for protocol messages."""
    COMMAND = "cmd"      # Command from app to device
    RESPONSE = "resp"    # Response from device to app
    DATA = "data"        # Data message from device
    EVENT = "event"      # Event message from device

class ResponseType:
    """Response types for protocol responses."""
    ACK = "ack"          # Command acknowledged
    DATA = "data"        # Data response
    ERROR = "error"      # Error response

class StatusCode:
    """Status codes for protocol responses."""
    SUCCESS = 0          # Command executed successfully
    INVALID_COMMAND = 1  # Invalid command
    INVALID_PARAMS = 2   # Invalid parameters
    EXECUTION_ERROR = 3  # Error executing command
    TIMEOUT = 4          # Command timed out
    BUSY = 5             # Device is busy
    NOT_IMPLEMENTED = 6  # Command not implemented

class Protocol:
    """
    Implements the communication protocol between the application and the ESP32.
    
    This class handles message formatting, validation, and parsing.
    """
    
    @staticmethod
    def create_command(command: str, params: Dict[str, Any] = None, command_id: int = 0) -> Dict[str, Any]:
        """
        Create a command message.
        
        Args:
            command: Command name
            params: Command parameters
            command_id: Command ID for matching responses
            
        Returns:
            Dict: Command message
        """
        return {
            "cmd": command,
            "id": command_id,
            "params": params or {}
        }
    
    @staticmethod
    def create_response(response_type: str, command_id: int, data: Any = None, 
                        status: int = StatusCode.SUCCESS) -> Dict[str, Any]:
        """
        Create a response message.
        
        Args:
            response_type: Response type (ack, data, error)
            command_id: Command ID matching the command
            data: Response data
            status: Status code
            
        Returns:
            Dict: Response message
        """
        return {
            "resp": response_type,
            "id": command_id,
            "data": data,
            "status": status
        }
    
    @staticmethod
    def create_data_message(data_type: str, data: Any) -> Dict[str, Any]:
        """
        Create a data message.
        
        Args:
            data_type: Type of data
            data: Data payload
            
        Returns:
            Dict: Data message
        """
        return {
            "data": True,
            "type": data_type,
            "data": data,
            "timestamp": 0  # Set by device
        }
    
    @staticmethod
    def create_event_message(event_type: str, data: Any = None) -> Dict[str, Any]:
        """
        Create an event message.
        
        Args:
            event_type: Type of event
            data: Event data
            
        Returns:
            Dict: Event message
        """
        return {
            "event": True,
            "type": event_type,
            "data": data,
            "timestamp": 0  # Set by device
        }
    
    @staticmethod
    def encode_message(message: Dict[str, Any]) -> str:
        """
        Encode a message to JSON string.
        
        Args:
            message: Message to encode
            
        Returns:
            str: JSON string
        """
        return json.dumps(message) + "\n"
    
    @staticmethod
    def decode_message(message_str: str) -> Dict[str, Any]:
        """
        Decode a JSON string to a message.
        
        Args:
            message_str: JSON string to decode
            
        Returns:
            Dict: Decoded message
            
        Raises:
            ProtocolError: If message cannot be decoded
        """
        try:
            return json.loads(message_str)
        except json.JSONDecodeError as e:
            raise ProtocolError(f"Invalid JSON: {e}")
    
    @staticmethod
    def validate_command(command: Dict[str, Any]) -> bool:
        """
        Validate a command message.
        
        Args:
            command: Command message to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ProtocolError: If command is invalid
        """
        if not isinstance(command, dict):
            raise ProtocolError("Command must be a dictionary")
            
        if "cmd" not in command:
            raise ProtocolError("Command missing 'cmd' field")
            
        if not isinstance(command["cmd"], str):
            raise ProtocolError("Command 'cmd' field must be a string")
            
        if "id" not in command:
            raise ProtocolError("Command missing 'id' field")
            
        if not isinstance(command["id"], int):
            raise ProtocolError("Command 'id' field must be an integer")
            
        if "params" in command and not isinstance(command["params"], dict):
            raise ProtocolError("Command 'params' field must be a dictionary")
            
        return True
    
    @staticmethod
    def validate_response(response: Dict[str, Any]) -> bool:
        """
        Validate a response message.
        
        Args:
            response: Response message to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ProtocolError: If response is invalid
        """
        if not isinstance(response, dict):
            raise ProtocolError("Response must be a dictionary")
            
        if "resp" not in response:
            raise ProtocolError("Response missing 'resp' field")
            
        if not isinstance(response["resp"], str):
            raise ProtocolError("Response 'resp' field must be a string")
            
        if response["resp"] not in [ResponseType.ACK, ResponseType.DATA, ResponseType.ERROR]:
            raise ProtocolError(f"Invalid response type: {response['resp']}")
            
        if "id" not in response:
            raise ProtocolError("Response missing 'id' field")
            
        if not isinstance(response["id"], int):
            raise ProtocolError("Response 'id' field must be an integer")
            
        if "status" not in response:
            raise ProtocolError("Response missing 'status' field")
            
        if not isinstance(response["status"], int):
            raise ProtocolError("Response 'status' field must be an integer")
            
        return True
    
    @staticmethod
    def validate_data_message(message: Dict[str, Any]) -> bool:
        """
        Validate a data message.
        
        Args:
            message: Data message to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ProtocolError: If message is invalid
        """
        if not isinstance(message, dict):
            raise ProtocolError("Data message must be a dictionary")
            
        if "data" not in message or message["data"] is not True:
            raise ProtocolError("Data message missing 'data' field or 'data' is not True")
            
        if "type" not in message:
            raise ProtocolError("Data message missing 'type' field")
            
        if not isinstance(message["type"], str):
            raise ProtocolError("Data message 'type' field must be a string")
            
        return True
    
    @staticmethod
    def validate_event_message(message: Dict[str, Any]) -> bool:
        """
        Validate an event message.
        
        Args:
            message: Event message to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ProtocolError: If message is invalid
        """
        if not isinstance(message, dict):
            raise ProtocolError("Event message must be a dictionary")
            
        if "event" not in message or message["event"] is not True:
            raise ProtocolError("Event message missing 'event' field or 'event' is not True")
            
        if "type" not in message:
            raise ProtocolError("Event message missing 'type' field")
            
        if not isinstance(message["type"], str):
            raise ProtocolError("Event message 'type' field must be a string")
            
        return True
    
    @staticmethod
    def get_message_type(message: Dict[str, Any]) -> str:
        """
        Determine the type of a message.
        
        Args:
            message: Message to check
            
        Returns:
            str: Message type (cmd, resp, data, event)
            
        Raises:
            ProtocolError: If message type cannot be determined
        """
        if "cmd" in message:
            return MessageType.COMMAND
        elif "resp" in message:
            return MessageType.RESPONSE
        elif "data" in message and message["data"] is True:
            return MessageType.DATA
        elif "event" in message and message["event"] is True:
            return MessageType.EVENT
        else:
            raise ProtocolError("Unknown message type")
    
    @staticmethod
    def check_response_status(response: Dict[str, Any]) -> None:
        """
        Check the status of a response and raise an exception if it indicates an error.
        
        Args:
            response: Response message to check
            
        Raises:
            InvalidResponseError: If response status indicates an error
        """
        if response.get("status", 0) != StatusCode.SUCCESS:
            status = response.get("status", -1)
            data = response.get("data", "Unknown error")
            raise InvalidResponseError(f"Command failed with status {status}: {data}")