"""
Enhanced Nephelometer class with auto-adjustment capabilities.

This module extends the existing Nephelometer class with features for automatic
adjustment of sensor settings based on signal levels and measurement conditions.
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import time
import threading
from enum import Enum
from collections import deque
import numpy as np

from hardware.nephelometer import Nephelometer, AgglutinationState, MeasurementMode
from hardware.as7341 import AS7341, AS7341Error, AS7341Gain
from core.communication import SerialCommunication
from utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

class EnhancedNephelometer(Nephelometer):
    """
    Enhanced Nephelometer with auto-adjustment capabilities.
    
    This class extends the base Nephelometer with features for automatically
    adjusting sensor settings based on signal levels and measurement conditions.
    """
    
    def __init__(self, comm: SerialCommunication):
        """
        Initialize the enhanced nephelometer.
        
        Args:
            comm: SerialCommunication instance for device communication
        """
        # Initialize base class
        super().__init__(comm)
        
        # Auto-adjustment settings
        self._auto_gain_enabled = False
        self._auto_integration_time_enabled = False
        self._auto_led_current_enabled = False
        
        # Target signal levels (in raw counts)
        self._target_signal_min = 1000
        self._target_signal_max = 50000
        self._target_signal_optimal = 25000
        
        # Adjustment thresholds
        self._gain_adjustment_threshold = 0.5  # Adjust if signal is 50% off target
        self._integration_time_adjustment_threshold = 0.3  # Adjust if signal is 30% off target
        self._led_adjustment_threshold = 0.4  # Adjust if signal is 40% off target
        
        # Current configuration
        self._current_config = {
            "gain": 5,  # Default to 16x gain
            "integration_time": 100,  # Default to 100ms
            "led_current": 10,  # Default to 10mA
        }
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the nephelometer with enhanced settings.
        
        Args:
            config: Configuration parameters
                - gain: Gain setting (0-10)
                - integration_time: Integration time in ms (1-1000)
                - led_current: LED current in mA (0-20)
                - enable_auto_gain: Whether to enable automatic gain adjustment
                - enable_auto_integration_time: Whether to enable automatic integration time adjustment
                - enable_auto_led_current: Whether to enable automatic LED current adjustment
                - target_signal_min: Minimum acceptable signal level
                - target_signal_max: Maximum acceptable signal level
                - target_signal_optimal: Optimal signal level
                
        Returns:
            bool: True if configuration successful
        """
        # Extract auto-adjustment settings
        self._auto_gain_enabled = config.pop('enable_auto_gain', False)
        self._auto_integration_time_enabled = config.pop('enable_auto_integration_time', False)
        self._auto_led_current_enabled = config.pop('enable_auto_led_current', False)
        
        # Extract target signal levels if provided
        if 'target_signal_min' in config:
            self._target_signal_min = config.pop('target_signal_min')
        if 'target_signal_max' in config:
            self._target_signal_max = config.pop('target_signal_max')
        if 'target_signal_optimal' in config:
            self._target_signal_optimal = config.pop('target_signal_optimal')
        
        # Update current configuration
        for key in ['gain', 'integration_time', 'led_current']:
            if key in config:
                self._current_config[key] = config[key]
        
        # Call base class configure method
        return super().configure(config)
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Dict: Current configuration parameters
        """
        return self._current_config.copy()
    
    def take_single_measurement(self, subtract_background: bool = True, 
                               auto_adjust: bool = True) -> Dict[str, Any]:
        """
        Take a single measurement with optional auto-adjustment.
        
        Args:
            subtract_background: Whether to subtract the background reading
            auto_adjust: Whether to perform auto-adjustment of settings
            
        Returns:
            Dict: Measurement data with processed values
            
        Raises:
            NephelometerError: If measurement fails
        """
        # Take initial measurement
        measurement = super().take_single_measurement(subtract_background)
        
        # Perform auto-adjustment if enabled
        if auto_adjust:
            adjusted, adjustment_info = self._auto_adjust_settings(measurement)
            
            # If settings were adjusted, take a new measurement
            if adjusted:
                logger.info("Auto-adjusted settings: %s", adjustment_info)
                measurement = super().take_single_measurement(subtract_background)
                # Add adjustment info to measurement
                measurement["adjustment_info"] = adjustment_info
        
        return measurement
    
    def start_continuous_measurement(self, interval_seconds: float, 
                                    callback: Callable[[Dict[str, Any]], None],
                                    subtract_background: bool = True,
                                    auto_adjust: bool = True,
                                    adjust_interval: int = 5) -> bool:
        """
        Start continuous measurements with auto-adjustment.
        
        Args:
            interval_seconds: Time between measurements in seconds
            callback: Function to call with each measurement
            subtract_background: Whether to subtract background
            auto_adjust: Whether to enable auto-adjustment
            adjust_interval: How often to perform adjustment (every N samples)
            
        Returns:
            bool: True if measurement started successfully
        """
        # Store auto-adjustment settings
        self._continuous_auto_adjust = auto_adjust
        self._continuous_adjust_interval = adjust_interval
        self._continuous_sample_count = 0
        
        # Create wrapper callback for auto-adjustment
        original_callback = callback
        
        def auto_adjust_callback(data: Dict[str, Any]):
            if auto_adjust:
                self._continuous_sample_count += 1
                
                # Perform auto-adjustment every N samples
                if self._continuous_sample_count >= adjust_interval:
                    self._continuous_sample_count = 0
                    adjusted, adjustment_info = self._auto_adjust_settings(data)
                    
                    if adjusted:
                        # Add adjustment info to data
                        data["auto_adjusted"] = True
                        data["adjustment_info"] = adjustment_info
            
            # Call original callback
            original_callback(data)
        
        # Start continuous measurement with our wrapper callback
        return super().start_continuous_measurement(
            interval_seconds=interval_seconds,
            callback=auto_adjust_callback,
            subtract_background=subtract_background
        )
    
    def start_kinetic_measurement(self, duration_seconds: float, samples_per_second: float,
                                 callback: Callable[[Dict[str, Any]], None],
                                 subtract_background: bool = True,
                                 initial_auto_adjust: bool = True) -> bool:
        """
        Start kinetic measurement with initial auto-adjustment.
        
        For kinetic measurements, we only adjust at the beginning to ensure
        consistent settings throughout the measurement.
        
        Args:
            duration_seconds: Total duration of the measurement
            samples_per_second: Sampling rate in Hz
            callback: Function to call with each measurement
            subtract_background: Whether to subtract background readings
            initial_auto_adjust: Whether to perform initial auto-adjustment
            
        Returns:
            bool: True if measurement started successfully
        """
        if initial_auto_adjust:
            # Take a test measurement for auto-adjustment
            test_measurement = self.take_single_measurement(
                subtract_background=subtract_background,
                auto_adjust=True
            )
            
            # Add adjustment info to first measurement
            original_callback = callback
            
            def wrapper_callback(data: Dict[str, Any]):
                # Add adjustment info to first measurement only
                if not hasattr(wrapper_callback, 'first_called'):
                    wrapper_callback.first_called = True
                    if "adjustment_info" in test_measurement:
                        data["initial_adjustment"] = test_measurement["adjustment_info"]
                
                # Call original callback
                original_callback(data)
            
            # Use wrapper callback
            callback = wrapper_callback
        
        # Start kinetic measurement with possibly adjusted settings
        return super().start_kinetic_measurement(
            duration_seconds=duration_seconds,
            samples_per_second=samples_per_second,
            callback=callback,
            subtract_background=subtract_background
        )
    
    def _auto_adjust_settings(self, measurement: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Automatically adjust settings based on measurement data.
        
        Args:
            measurement: Measurement data
            
        Returns:
            Tuple[bool, Dict]: Whether settings were adjusted and adjustment info
        """
        # Extract raw data
        raw_data = measurement.get("raw", {})
        if not raw_data:
            return False, {}
        
        # Get relevant channel values
        # We'll use Clear channel, F4 (green), and max channel for adjustment
        clear_value = raw_data.get("Clear", 0)
        f4_value = raw_data.get("F4", 0)  # Green channel
        max_channel_value = max([v for k, v in raw_data.items() 
                               if k in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]], 
                              default=0)
        
        # Determine if signal is too low or too high
        signal_too_low = max_channel_value < self._target_signal_min
        signal_too_high = max_channel_value > self._target_signal_max
        
        # Track adjustments
        adjustments = {}
        adjusted = False
        
        # Adjust settings in priority order: gain, integration time, LED current
        
        # 1. Adjust gain if enabled and needed
        if self._auto_gain_enabled and (signal_too_low or signal_too_high):
            current_gain = self._current_config["gain"]
            
            if signal_too_low and current_gain < 10:  # Can increase gain
                # Calculate how much to increase by
                ratio = self._target_signal_optimal / max(1, max_channel_value)
                
                # Only adjust if significantly off target
                if ratio > (1 + self._gain_adjustment_threshold):
                    # Determine steps to increase gain (each step doubles the gain)
                    steps = min(10 - current_gain, int(np.log2(ratio)))
                    
                    if steps > 0:
                        new_gain = current_gain + steps
                        
                        # Apply new gain
                        super().configure({"gain": new_gain})
                        self._current_config["gain"] = new_gain
                        
                        adjustments["gain"] = {
                            "before": current_gain,
                            "after": new_gain,
                            "reason": "Signal too low"
                        }
                        adjusted = True
            
            elif signal_too_high and current_gain > 0:  # Can decrease gain
                # Calculate how much to decrease by
                ratio = max_channel_value / self._target_signal_optimal
                
                # Only adjust if significantly off target
                if ratio > (1 + self._gain_adjustment_threshold):
                    # Determine steps to decrease gain (each step halves the gain)
                    steps = min(current_gain, int(np.log2(ratio)))
                    
                    if steps > 0:
                        new_gain = current_gain - steps
                        
                        # Apply new gain
                        super().configure({"gain": new_gain})
                        self._current_config["gain"] = new_gain
                        
                        adjustments["gain"] = {
                            "before": current_gain,
                            "after": new_gain,
                            "reason": "Signal too high"
                        }
                        adjusted = True
        
        # 2. Adjust integration time if enabled and needed
        if (self._auto_integration_time_enabled and 
            (signal_too_low or signal_too_high) and 
            not adjustments):  # Only if no gain adjustment was made
            
            current_time = self._current_config["integration_time"]
            
            if signal_too_low and current_time < 500:  # Can increase time
                # Calculate how much to increase by
                ratio = self._target_signal_optimal / max(1, max_channel_value)
                
                # Only adjust if significantly off target
                if ratio > (1 + self._integration_time_adjustment_threshold):
                    # Increase by ratio, but cap at 500ms
                    new_time = min(500, int(current_time * min(ratio, 3)))
                    
                    # Ensure we make a meaningful change
                    if new_time > current_time * 1.2:  # At least 20% increase
                        # Apply new integration time
                        super().configure({"integration_time": new_time})
                        self._current_config["integration_time"] = new_time
                        
                        adjustments["integration_time"] = {
                            "before": current_time,
                            "after": new_time,
                            "reason": "Signal too low"
                        }
                        adjusted = True
            
            elif signal_too_high and current_time > 20:  # Can decrease time
                # Calculate how much to decrease by
                ratio = max_channel_value / self._target_signal_optimal
                
                # Only adjust if significantly off target
                if ratio > (1 + self._integration_time_adjustment_threshold):
                    # Decrease by ratio, but ensure at least 20ms
                    new_time = max(20, int(current_time / min(ratio, 3)))
                    
                    # Ensure we make a meaningful change
                    if new_time < current_time * 0.8:  # At least 20% decrease
                        # Apply new integration time
                        super().configure({"integration_time": new_time})
                        self._current_config["integration_time"] = new_time
                        
                        adjustments["integration_time"] = {
                            "before": current_time,
                            "after": new_time,
                            "reason": "Signal too high"
                        }
                        adjusted = True
        
        # 3. Adjust LED current if enabled and needed
        if (self._auto_led_current_enabled and 
            signal_too_low and 
            not adjustments):  # Only if no other adjustments were made
            
            current_led = self._current_config["led_current"]
            
            if current_led < 20:  # Can increase LED current
                # Calculate how much to increase by
                ratio = self._target_signal_optimal / max(1, max_channel_value)
                
                # Only adjust if significantly off target
                if ratio > (1 + self._led_adjustment_threshold):
                    # Increase by ratio, but cap at 20mA
                    new_led = min(20, int(current_led * min(ratio, 2)))
                    
                    # Ensure we make a meaningful change
                    if new_led > current_led:
                        # Apply new LED current
                        self.set_led(True, new_led)
                        self._current_config["led_current"] = new_led
                        
                        adjustments["led_current"] = {
                            "before": current_led,
                            "after": new_led,
                            "reason": "Signal too low"
                        }
                        adjusted = True
        
        return adjusted, adjustments