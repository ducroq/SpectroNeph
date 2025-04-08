from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import time
import threading
from enum import Enum
from collections import deque
import numpy as np

from config import settings
from utils.logging import get_logger
from core.communication import SerialCommunication
from hardware.as7341 import AS7341, AS7341Error

# Initialize module logger
logger = get_logger(__name__)

class AgglutinationState(Enum):
    """Classification of agglutination states."""
    NONE = 0        # No detectable agglutination
    MINIMAL = 1     # Minimal/threshold agglutination
    MODERATE = 2    # Moderate agglutination
    STRONG = 3      # Strong agglutination
    COMPLETE = 4    # Complete/maximum agglutination

class MeasurementMode(Enum):
    """Measurement modes for the nephelometer."""
    SINGLE = "single"            # Single measurement
    CONTINUOUS = "continuous"    # Continuous measurements at fixed intervals
    KINETIC = "kinetic"          # High-frequency measurements for kinetic studies

class NephelometerError(Exception):
    """Exception raised for errors in the nephelometer."""
    pass

class Nephelometer:
    """
    High-level nephelometer control class.
    
    This class provides a high-level interface for controlling the nephelometer,
    including experiment setup, measurements, and data analysis.
    """
    
    def __init__(self, comm: SerialCommunication):
        """
        Initialize the nephelometer.
        
        Args:
            comm: SerialCommunication instance for device communication
        """
        self._comm = comm
        self._sensor = AS7341(comm)
        self._measurement_lock = threading.Lock()
        self._is_measuring = False
        self._stop_measurement = threading.Event()
        self._measurement_thread = None
        self._data_callbacks = []
        
        # Data buffers
        self._measurement_buffer = deque(maxlen=settings.get("BUFFER_SIZE", 1000))
        self._background_reading = {}
        
        # Experiment state
        self._experiment_running = False
        self._experiment_start_time = 0
    
    def initialize(self) -> bool:
        """
        Initialize the nephelometer hardware.
        
        Returns:
            bool: True if initialization successful
        """
        logger.info("Initializing nephelometer")
        
        # Initialize sensor
        if not self._sensor.initialize():
            logger.error("Failed to initialize AS7341 sensor")
            return False
        
        logger.info("Nephelometer initialized successfully")
        return True
    
    def set_led(self, enabled: bool, current_ma: Optional[int] = None) -> bool:
        """
        Control the LED.
        
        Args:
            enabled: Whether to enable the LED
            current_ma: LED current in mA (0-20), or None to use default
            
        Returns:
            bool: True if successful
        """
        return self._sensor.set_led(enabled, current_ma)
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the nephelometer.
        
        Args:
            config: Configuration parameters
                - integration_time: Integration time in ms (1-1000)
                - gain: Gain setting (0-10)
                - led_current: LED current in mA (0-20)
                
        Returns:
            bool: True if configuration successful
        """
        return self._sensor.set_config(config)
    
    def take_background_reading(self) -> Dict[str, int]:
        """
        Take a background reading for baseline subtraction.
        
        Returns:
            Dict[str, int]: Background reading values
            
        Raises:
            NephelometerError: If reading fails
        """
        logger.info("Taking background reading")
        
        with self._measurement_lock:
            try:
                # Ensure LED is on
                self._sensor.set_led(True)
                
                # Wait for light to stabilize
                time.sleep(0.1)
                
                # Take multiple readings and average them
                num_readings = 5
                readings = []
                
                for _ in range(num_readings):
                    reading = self._sensor.read_spectral_data()
                    readings.append(reading)
                    time.sleep(0.05)
                
                # Calculate average reading
                background = {}
                for channel in readings[0].keys():
                    values = [r[channel] for r in readings]
                    background[channel] = int(sum(values) / len(values))
                
                self._background_reading = background
                logger.info("Background reading completed")
                
                return background
                
            except AS7341Error as e:
                raise NephelometerError(f"Error taking background reading: {str(e)}")
    
    def take_single_measurement(self, subtract_background: bool = True) -> Dict[str, Any]:
        """
        Take a single measurement.
        
        Args:
            subtract_background: Whether to subtract the background reading
            
        Returns:
            Dict: Measurement data with processed values
            
        Raises:
            NephelometerError: If measurement fails
        """
        logger.debug("Taking single measurement")
        
        with self._measurement_lock:
            try:
                # Ensure LED is on
                self._sensor.set_led(True)
                
                # Take the measurement
                raw_data = self._sensor.read_spectral_data()
                
                # Process the data
                processed_data = self._process_measurement(raw_data, subtract_background)
                
                return processed_data
                
            except AS7341Error as e:
                raise NephelometerError(f"Error taking measurement: {str(e)}")
    
    def start_continuous_measurement(self, interval_seconds: float, 
                                    callback: Callable[[Dict[str, Any]], None],
                                    subtract_background: bool = True) -> bool:
        """
        Start continuous measurements at a specified interval.
        
        Args:
            interval_seconds: Time between measurements in seconds
            callback: Function to call with each measurement
            subtract_background: Whether to subtract background readings
            
        Returns:
            bool: True if measurement started successfully
        """
        if self._is_measuring:
            logger.warning("Measurement already in progress")
            return False
        
        # Set up measurement state
        self._stop_measurement.clear()
        self._is_measuring = True
        self._data_callbacks = [callback]
        
        # Start measurement thread
        self._measurement_thread = threading.Thread(
            target=self._continuous_measurement_thread,
            args=(interval_seconds, subtract_background),
            name="NephelometerMeasurement",
            daemon=True
        )
        self._measurement_thread.start()
        
        logger.info("Started continuous measurement with %.2f second interval", interval_seconds)
        return True
    
    def start_kinetic_measurement(self, duration_seconds: float, samples_per_second: float,
                                 callback: Callable[[Dict[str, Any]], None],
                                 subtract_background: bool = True) -> bool:
        """
        Start high-frequency kinetic measurements.
        
        Args:
            duration_seconds: Total duration of the measurement
            samples_per_second: Sampling rate in Hz
            callback: Function to call with each measurement
            subtract_background: Whether to subtract background readings
            
        Returns:
            bool: True if measurement started successfully
        """
        if self._is_measuring:
            logger.warning("Measurement already in progress")
            return False
        
        # Validate parameters
        max_rate = settings.get("MAX_SAMPLING_RATE", 10.0)
        if samples_per_second > max_rate:
            logger.warning("Requested sampling rate %.1f Hz exceeds maximum %.1f Hz",
                          samples_per_second, max_rate)
            samples_per_second = max_rate
        
        interval_seconds = 1.0 / samples_per_second
        
        # Set up measurement state
        self._stop_measurement.clear()
        self._is_measuring = True
        self._data_callbacks = [callback]
        
        # Start measurement thread
        self._measurement_thread = threading.Thread(
            target=self._kinetic_measurement_thread,
            args=(duration_seconds, interval_seconds, subtract_background),
            name="NephelometerKinetic",
            daemon=True
        )
        self._measurement_thread.start()
        
        logger.info("Started kinetic measurement for %.1f seconds at %.1f Hz",
                   duration_seconds, samples_per_second)
        return True
    
    def stop_measurement(self) -> bool:
        """
        Stop any ongoing measurements.
        
        Returns:
            bool: True if successful
        """
        if not self._is_measuring:
            return True
        
        # Signal thread to stop
        self._stop_measurement.set()
        
        # Wait for thread to finish
        if self._measurement_thread and self._measurement_thread.is_alive():
            self._measurement_thread.join(timeout=2.0)
            
        # Clean up
        self._is_measuring = False
        self._data_callbacks = []
        
        logger.info("Stopped measurement")
        return True
    
    def register_data_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register an additional callback for measurement data.
        
        Args:
            callback: Function to call with each measurement
        """
        if callback not in self._data_callbacks:
            self._data_callbacks.append(callback)
    
    def unregister_data_callback(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Unregister a data callback.
        
        Args:
            callback: Callback to remove
            
        Returns:
            bool: True if callback was removed
        """
        if callback in self._data_callbacks:
            self._data_callbacks.remove(callback)
            return True
        return False
    
    def start_experiment(self, experiment_id: str) -> bool:
        """
        Start an experiment session.
        
        Args:
            experiment_id: Unique identifier for the experiment
            
        Returns:
            bool: True if experiment started successfully
        """
        if self._experiment_running:
            logger.warning("Experiment already in progress")
            return False
        
        self._experiment_running = True
        self._experiment_start_time = time.time()
        
        # Log experiment start
        exp_logger = get_logger("experiment", experiment_id=experiment_id)
        exp_logger.info("Starting experiment: %s", experiment_id)
        
        return True
    
    def stop_experiment(self) -> bool:
        """
        Stop the current experiment session.
        
        Returns:
            bool: True if successful
        """
        if not self._experiment_running:
            return True
        
        # Stop any ongoing measurements
        if self._is_measuring:
            self.stop_measurement()
        
        self._experiment_running = False
        
        # Log experiment end
        duration = time.time() - self._experiment_start_time
        logger.info("Experiment completed. Duration: %.1f seconds", duration)
        
        return True
    
    def analyze_agglutination(self, measurement: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a measurement for agglutination characteristics.
        
        Args:
            measurement: Measurement data to analyze
            
        Returns:
            Dict: Analysis results including agglutination metrics
        """
        # Extract raw and processed data
        raw_data = measurement.get("raw", {})
        processed = measurement.get("processed", {})
        
        # Get channel ratios
        ratios = measurement.get("ratios", {})
        
        # Calculate agglutination metrics
        # This is a simplified implementation that should be refined based on experimental data
        result = {
            "agglutination_score": 0,
            "agglutination_state": AgglutinationState.NONE,
            "particle_size_estimate": 0,
            "confidence": 0
        }
        
        # Example logic for determining agglutination from spectral ratios
        # Actual implementation should be calibrated based on experimental data
        if "violet_red" in ratios:
            violet_red_ratio = ratios["violet_red"]
            
            # Simplified classification based on violet/red ratio
            # These thresholds should be determined experimentally
            if violet_red_ratio > 2.0:
                state = AgglutinationState.COMPLETE
                score = 4.0
                confidence = 0.9
            elif violet_red_ratio > 1.5:
                state = AgglutinationState.STRONG
                score = 3.0
                confidence = 0.8
            elif violet_red_ratio > 1.2:
                state = AgglutinationState.MODERATE
                score = 2.0
                confidence = 0.7
            elif violet_red_ratio > 1.0:
                state = AgglutinationState.MINIMAL
                score = 1.0
                confidence = 0.6
            else:
                state = AgglutinationState.NONE
                score = 0.0
                confidence = 0.8
                
            result["agglutination_score"] = score
            result["agglutination_state"] = state
            result["confidence"] = confidence
            
            # Simplified particle size estimation based on violet/green ratio
            # This should be calibrated with known particle sizes
            if "violet_green" in ratios:
                violet_green_ratio = ratios["violet_green"]
                # Particle size in μm (hypothetical formula, requires calibration)
                result["particle_size_estimate"] = 0.1 + 0.9 * (1.0 - min(1.0, violet_green_ratio / 3.0))
        
        return result
    
    def _continuous_measurement_thread(self, interval_seconds: float, 
                                      subtract_background: bool) -> None:
        """
        Background thread for continuous measurements.
        
        Args:
            interval_seconds: Time between measurements
            subtract_background: Whether to subtract background readings
        """
        logger.debug("Continuous measurement thread started")
        
        next_time = time.time()
        
        try:
            while not self._stop_measurement.is_set():
                current_time = time.time()
                
                # Check if it's time for the next measurement
                if current_time >= next_time:
                    # Take a measurement
                    try:
                        measurement = self.take_single_measurement(subtract_background)
                        
                        # Add timestamp
                        measurement["timestamp"] = current_time
                        
                        # Store in buffer
                        self._measurement_buffer.append(measurement)
                        
                        # Call callbacks
                        for callback in self._data_callbacks:
                            try:
                                callback(measurement)
                            except Exception as e:
                                logger.error("Error in measurement callback: %s", str(e), exc_info=True)
                        
                        # Calculate next measurement time
                        next_time = current_time + interval_seconds
                        
                    except Exception as e:
                        logger.error("Error taking measurement: %s", str(e), exc_info=True)
                        next_time = current_time + interval_seconds
                
                # Sleep briefly to avoid busy waiting
                # But not longer than the time to the next measurement
                sleep_time = min(0.01, next_time - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except Exception as e:
            logger.error("Error in continuous measurement thread: %s", str(e), exc_info=True)
        
        # Ensure LED is off
        try:
            self._sensor.set_led(False)
        except Exception:
            pass
            
        self._is_measuring = False
        logger.debug("Continuous measurement thread stopped")
    
    def _kinetic_measurement_thread(self, duration_seconds: float, 
                                   interval_seconds: float,
                                   subtract_background: bool) -> None:
        """
        Background thread for kinetic measurements.
        
        Args:
            duration_seconds: Total duration of the measurement
            interval_seconds: Time between measurements
            subtract_background: Whether to subtract background readings
        """
        logger.debug("Kinetic measurement thread started")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        next_time = start_time
        
        try:
            # Pre-allocate measurement buffer
            num_samples = int(duration_seconds / interval_seconds) + 1
            measurements = []
            
            while time.time() < end_time and not self._stop_measurement.is_set():
                current_time = time.time()
                
                # Check if it's time for the next measurement
                if current_time >= next_time:
                    # Take a measurement
                    try:
                        measurement = self.take_single_measurement(subtract_background)
                        
                        # Add timestamp and elapsed time
                        measurement["timestamp"] = current_time
                        measurement["elapsed_seconds"] = current_time - start_time
                        
                        # Store in buffer
                        measurements.append(measurement)
                        
                        # Call callbacks
                        for callback in self._data_callbacks:
                            try:
                                callback(measurement)
                            except Exception as e:
                                logger.error("Error in measurement callback: %s", str(e), exc_info=True)
                        
                        # Calculate next measurement time
                        next_time += interval_seconds
                        
                    except Exception as e:
                        logger.error("Error taking measurement: %s", str(e), exc_info=True)
                        next_time = current_time + interval_seconds
                
                # Sleep briefly to avoid busy waiting
                # But not longer than the time to the next measurement
                sleep_time = min(0.001, next_time - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            # Process the full kinetic data set
            self._process_kinetic_data(measurements)
            
        except Exception as e:
            logger.error("Error in kinetic measurement thread: %s", str(e), exc_info=True)
        
        # Ensure LED is off
        try:
            self._sensor.set_led(False)
        except Exception:
            pass
            
        self._is_measuring = False
        logger.debug("Kinetic measurement thread stopped")
    
    def _process_measurement(self, raw_data: Dict[str, int], 
                            subtract_background: bool) -> Dict[str, Any]:
        """
        Process a raw measurement.
        
        Args:
            raw_data: Raw spectral data from sensor
            subtract_background: Whether to subtract background readings
            
        Returns:
            Dict: Processed measurement data
        """
        # Start with a copy of the raw data
        processed = {
            "raw": raw_data.copy(),
            "processed": {},
            "ratios": {}
        }
        
        # Apply background subtraction if requested and available
        if subtract_background and self._background_reading:
            for channel, value in raw_data.items():
                if channel in self._background_reading:
                    # Subtract with floor at zero
                    processed["processed"][channel] = max(0, value - self._background_reading[channel])
                else:
                    processed["processed"][channel] = value
        else:
            processed["processed"] = raw_data.copy()
        
        # Calculate channel ratios
        ratios = self._sensor.calculate_channel_ratios(processed["processed"])
        processed["ratios"] = ratios
        
        return processed
    
    def _process_kinetic_data(self, measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a complete kinetic measurement dataset.
        
        Args:
            measurements: List of measurements from a kinetic run
            
        Returns:
            Dict: Kinetic analysis results
        """
        if not measurements:
            return {}
        
        try:
            # Extract timestamps and key channels
            timestamps = [m["timestamp"] for m in measurements]
            elapsed = [m["elapsed_seconds"] for m in measurements]
            
            # Calculate rate of change for key ratios
            if all("ratios" in m for m in measurements):
                # Extract violet/red ratio over time
                vr_ratios = [m["ratios"].get("violet_red", 0) for m in measurements]
                
                # Calculate rate of change (if we have enough points)
                if len(vr_ratios) > 5:
                    # Simple linear regression to get slope
                    x = np.array(elapsed)
                    y = np.array(vr_ratios)
                    
                    # Calculate slope (rate)
                    if np.std(x) > 0:
                        slope = np.cov(x, y)[0, 1] / np.var(x)
                    else:
                        slope = 0
                        
                    kinetic_results = {
                        "reaction_rate": slope,
                        "initial_value": vr_ratios[0] if vr_ratios else 0,
                        "final_value": vr_ratios[-1] if vr_ratios else 0,
                        "change": vr_ratios[-1] - vr_ratios[0] if vr_ratios else 0
                    }
                    
                    # Call callbacks with kinetic results
                    for callback in self._data_callbacks:
                        try:
                            callback({"type": "kinetic_results", "data": kinetic_results})
                        except Exception as e:
                            logger.error("Error in kinetic callback: %s", str(e))
                    
                    return kinetic_results
            
            return {}
                
        except Exception as e:
            logger.error("Error processing kinetic data: %s", str(e), exc_info=True)
            return {}