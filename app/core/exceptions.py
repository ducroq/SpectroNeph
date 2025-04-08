"""
Custom exceptions for the nephelometer application.
"""

class NephelometerError(Exception):
    """Base exception for all nephelometer-related errors."""
    pass

class CommunicationError(NephelometerError):
    """Exception raised for errors in the device communication."""
    pass

class CommandTimeoutError(CommunicationError):
    """Exception raised when a command times out."""
    pass

class InvalidResponseError(CommunicationError):
    """Exception raised when an invalid response is received."""
    pass

class DeviceDisconnectedError(CommunicationError):
    """Exception raised when the device is disconnected."""
    pass

class ProtocolError(CommunicationError):
    """Exception raised for protocol-related errors."""
    pass

class ConfigurationError(NephelometerError):
    """Exception raised for configuration errors."""
    pass

class HardwareError(NephelometerError):
    """Exception raised for hardware-related errors."""
    pass

class SensorError(HardwareError):
    """Exception raised for sensor-related errors."""
    pass

class ExperimentError(NephelometerError):
    """Exception raised for experiment-related errors."""
    pass

class DataError(NephelometerError):
    """Exception raised for data-related errors."""
    pass