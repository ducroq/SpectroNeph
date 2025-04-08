# core/__init__.py
"""
Core package for the nephelometer application.

This package provides the core functionality for communicating with the
ESP32 device and managing the device connection.
"""

from core.exceptions import (
    NephelometerError, CommunicationError, CommandTimeoutError, 
    InvalidResponseError, DeviceDisconnectedError, ProtocolError,
    ConfigurationError, HardwareError, SensorError, ExperimentError, DataError
)

from core.protocol import Protocol, MessageType, ResponseType, StatusCode
from core.communication import SerialCommunication
from core.device import DeviceManager

__all__ = [
    # Exceptions
    'NephelometerError', 'CommunicationError', 'CommandTimeoutError',
    'InvalidResponseError', 'DeviceDisconnectedError', 'ProtocolError',
    'ConfigurationError', 'HardwareError', 'SensorError', 'ExperimentError', 'DataError',
    
    # Protocol classes
    'Protocol', 'MessageType', 'ResponseType', 'StatusCode',
    
    # Core classes
    'SerialCommunication', 'DeviceManager'
]