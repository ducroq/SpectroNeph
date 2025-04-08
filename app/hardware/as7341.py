from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import time
from enum import Enum, auto

from config import settings
from utils.logging import get_logger
from core.communication import SerialCommunication, CommandTimeoutError, DeviceDisconnectedError

# Initialize module logger
logger = get_logger(__name__)

class AS7341Gain(Enum):
    """Gain settings for the AS7341 sensor."""
    GAIN_0_5X = 0    # 0.5x
    GAIN_1X = 1      # 1x
    GAIN_2X = 2      # 2x
    GAIN_4X = 3      # 4x
    GAIN_8X = 4      # 8x
    GAIN_16X = 5     # 16x
    GAIN_32X = 6     # 32x
    GAIN_64X = 7     # 64x
    GAIN_128X = 8    # 128x
    GAIN_256X = 9    # 256x
    GAIN_512X = 10   # 512x

class AS7341Channel(Enum):
    """Channel definitions for the AS7341 sensor."""
    F1 = "F1"      # 415nm
    F2 = "F2"      # 445nm
    F3 = "F3"      # 480nm
    F4 = "F4"      # 515nm
    F5 = "F5"      # 555nm
    F6 = "F6"      # 590nm
    F7 = "F7"      # 630nm
    F8 = "F8"      # 680nm
    CLEAR = "Clear"  # Clear channel
    NIR = "NIR"      # Near IR channel

class AS7341Error(Exception):
    """Exception raised for errors in the AS7341 sensor."""
    pass

class AS7341:
    """
    Interface for the AS7341 spectral sensor.
    
    This class handles communication with the AS7341 spectral sensor via the ESP32.
    It provides methods for configuring the sensor, reading spectral data, and
    controlling the LED.
    """
    
    def __init__(self, comm: SerialCommunication):
        """
        Initialize the AS7341 interface.
        
        Args:
            comm: SerialCommunication instance for device communication
        """
        self._comm = comm
        self._data_callback_id = None
        self._streaming = False
        self._last_reading = {}
        
        # Default configuration
        self._config = {
            "integration_time": settings.DEFAULT_INTEGRATION_TIME,
            "gain": settings.DEFAULT_GAIN,
            "led_current": settings.DEFAULT_LED_CURRENT
        }
    
    def initialize(self) -> bool:
        """
        Initialize the AS7341 sensor.
        
        Returns:
            bool: True if initialization successful
        """
        logger.info("Initializing AS7341 sensor")
        
        try:
            # Check if sensor is connected
            response = self._comm.send_command("as7341_init", {})
            
            if response.get("status") != 0:
                logger.error("AS7341 initialization failed: %s", response.get("data"))
                return False
            
            # Apply default configuration
            self.set_config(self._config)
            
            logger.info("AS7341 initialized successfully")
            return True
            
        except CommandTimeoutError:
            logger.error("AS7341 initialization timed out")
            return False
        except DeviceDisconnectedError:
            logger.error("Device disconnected during AS7341 initialization")
            return False
        except Exception as e:
            logger.error("Error initializing AS7341: %s", str(e), exc_info=True)
            return False
    
    def set_config(self, config: Dict[str, Any]) -> bool:
        """
        Configure the AS7341 sensor.
        
        Args:
            config: Configuration parameters
                - integration_time: Integration time in ms (1-1000)
                - gain: Gain setting (0-10, see AS7341Gain enum)
                - led_current: LED current in mA (0-20)
                
        Returns:
            bool: True if configuration successful
        """
        # Update local config with provided values
        self._config.update(config)
        
        # Validate configuration
        if "integration_time" in config and not (1 <= config["integration_time"] <= 1000):
            logger.warning("Invalid integration time: %s (must be 1-1000ms)", config["integration_time"])
            return False
            
        if "gain" in config and not (0 <= config["gain"] <= 10):
            logger.warning("Invalid gain setting: %s (must be 0-10)", config["gain"])
            return False
            
        if "led_current" in config and not (0 <= config["led_current"] <= 20):
            logger.warning("Invalid LED current: %s (must be 0-20mA)", config["led_current"])
            return False
        
        # Send configuration to device
        try:
            response = self._comm.send_command("as7341_config", self._config)
            
            if response.get("status") != 0:
                logger.error("Failed to configure AS7341: %s", response.get("data"))
                return False
                
            logger.debug("AS7341 configured: %s", self._config)
            return True
            
        except Exception as e:
            logger.error("Error configuring AS7341: %s", str(e))
            return False
    
    def read_spectral_data(self) -> Dict[str, int]:
        """
        Read spectral data from all channels.
        
        Returns:
            Dict[str, int]: Channel values keyed by channel name
        
        Raises:
            AS7341Error: If reading fails
        """
        try:
            response = self._comm.send_command("as7341_read", {})
            
            if response.get("status") != 0:
                error_msg = f"Failed to read AS7341: {response.get('data')}"
                logger.error(error_msg)
                raise AS7341Error(error_msg)
                
            # Extract and store channel data
            channel_data = response.get("data", {})
            self._last_reading = channel_data.copy()
            
            logger.debug("Read spectral data: %s", channel_data)
            return channel_data
            
        except CommandTimeoutError as e:
            raise AS7341Error(f"Timeout reading AS7341: {str(e)}")
        except DeviceDisconnectedError as e:
            raise AS7341Error(f"Device disconnected while reading AS7341: {str(e)}")
        except Exception as e:
            raise AS7341Error(f"Error reading AS7341: {str(e)}")
    
    def start_streaming(self, callback: Callable[[Dict[str, int]], None], 
                       interval_ms: int = 100) -> bool:
        """
        Start streaming spectral data.
        
        Args:
            callback: Function to call with each data reading
            interval_ms: Sampling interval in milliseconds
            
        Returns:
            bool: True if streaming started successfully
        """
        if self._streaming:
            self.stop_streaming()
        
        # Register data callback
        def data_handler(data_message: Dict[str, Any]) -> None:
            channel_data = data_message.get("data", {})
            self._last_reading = channel_data.copy()
            callback(channel_data)
        
        self._data_callback_id = self._comm.register_data_callback("as7341", data_handler)
        
        # Start streaming
        try:
            success = self._comm.start_data_stream("as7341", {
                "interval_ms": interval_ms
            })
            
            if success:
                self._streaming = True
                logger.info("Started AS7341 data streaming at %d ms interval", interval_ms)
                return True
            else:
                # Clean up on failure
                self._comm.unregister_data_callback(self._data_callback_id)
                self._data_callback_id = None
                logger.error("Failed to start AS7341 data streaming")
                return False
                
        except Exception as e:
            # Clean up on error
            if self._data_callback_id:
                self._comm.unregister_data_callback(self._data_callback_id)
                self._data_callback_id = None
            logger.error("Error starting AS7341 data streaming: %s", str(e))
            return False
    
    def stop_streaming(self) -> bool:
        """
        Stop streaming spectral data.
        
        Returns:
            bool: True if streaming stopped successfully
        """
        if not self._streaming:
            return True
        
        # Stop the data stream
        success = self._comm.stop_data_stream("as7341")
        
        # Unregister callback
        if self._data_callback_id:
            self._comm.unregister_data_callback(self._data_callback_id)
            self._data_callback_id = None
        
        self._streaming = False
        logger.info("Stopped AS7341 data streaming")
        return success
    
    def set_led(self, enabled: bool, current_ma: Optional[int] = None) -> bool:
        """
        Control the LED.
        
        Args:
            enabled: Whether to enable the LED
            current_ma: LED current in mA (0-20), or None to use config value
            
        Returns:
            bool: True if successful
        """
        # Use provided current or current from config
        if current_ma is None:
            current_ma = self._config["led_current"]
        
        # Validate current
        if not (0 <= current_ma <= 20):
            logger.warning("Invalid LED current: %s (must be 0-20mA)", current_ma)
            return False
        
        # Update config
        self._config["led_current"] = current_ma
        
        # Send command
        try:
            response = self._comm.send_command("as7341_led", {
                "enabled": enabled,
                "current": current_ma
            })
            
            if response.get("status") != 0:
                logger.error("Failed to control LED: %s", response.get("data"))
                return False
                
            logger.debug("LED %s at %d mA", "enabled" if enabled else "disabled", current_ma)
            return True
            
        except Exception as e:
            logger.error("Error controlling LED: %s", str(e))
            return False
    
    def get_last_reading(self) -> Dict[str, int]:
        """
        Get the most recent spectral reading.
        
        Returns:
            Dict[str, int]: Channel values keyed by channel name
        """
        return self._last_reading.copy()
    
    def calculate_channel_ratios(self, reading: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        """
        Calculate ratios between different channels.
        
        Args:
            reading: Reading to analyze, or None to use last reading
            
        Returns:
            Dict[str, float]: Channel ratios
        """
        data = reading or self._last_reading
        
        if not data:
            return {}
        
        # Ensure we have all necessary channels
        required_channels = ["F1", "F4", "F8", "Clear"]
        if not all(ch in data for ch in required_channels):
            missing = [ch for ch in required_channels if ch not in data]
            logger.warning("Missing channels for ratio calculation: %s", missing)
            return {}
        
        # Calculate ratios
        ratios = {
            "violet_green": data["F1"] / max(data["F4"], 1),  # Violet to Green
            "violet_red": data["F1"] / max(data["F8"], 1),    # Violet to Red
            "green_red": data["F4"] / max(data["F8"], 1),     # Green to Red
            "normalized_violet": data["F1"] / max(data["Clear"], 1)  # Violet normalized to Clear
        }
        
        return ratios