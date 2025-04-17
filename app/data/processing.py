"""
Signal processing and algorithms for the SpectroNeph system.

This module handles all data processing functionality, including:
- Signal filtering and smoothing
- Baseline correction
- Ratio calculation
- Feature extraction
- Agglutination detection algorithms
"""

import numpy as np
import scipy.signal as signal
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import json

from config import settings
from utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

class SignalProcessor:
    """
    Signal processing class for SpectroNeph data.
    
    This class provides methods for processing spectral data,
    including filtering, normalization, and feature extraction.
    """
    
    def __init__(self):
        """Initialize the signal processor."""
        pass
    
    def filter_measurement(self, measurement: Dict[str, Any], 
                          method: str = "moving_average", 
                          **kwargs) -> Dict[str, Any]:
        """
        Apply a filter to a measurement.
        
        Args:
            measurement: Measurement data to filter
            method: Filtering method ('moving_average', 'savgol', 'median', etc.)
            **kwargs: Additional arguments specific to the filtering method
            
        Returns:
            Dict[str, Any]: Filtered measurement
        """
        # Create a copy of the measurement
        filtered = measurement.copy()
        
        # Get the raw data
        raw_data = measurement.get("raw", {})
        if not raw_data:
            logger.warning("No raw data found in measurement")
            return filtered
        
        # Filter based on the selected method
        if method == "moving_average":
            filtered_data = self._moving_average_filter(raw_data, **kwargs)
        elif method == "savgol":
            filtered_data = self._savgol_filter(raw_data, **kwargs)
        elif method == "median":
            filtered_data = self._median_filter(raw_data, **kwargs)
        else:
            logger.warning(f"Unknown filtering method: {method}")
            return filtered
        
        # Store the filtered data
        if "processed" not in filtered:
            filtered["processed"] = {}
        filtered["processed"]["filtered"] = filtered_data
        
        return filtered
    
    def normalize_measurement(self, measurement: Dict[str, Any],
                            reference_channel: str = "Clear",
                            **kwargs) -> Dict[str, Any]:
        """
        Normalize a measurement by a reference channel.
        
        Args:
            measurement: Measurement data to normalize
            reference_channel: Channel to use as reference
            **kwargs: Additional arguments for normalization
            
        Returns:
            Dict[str, Any]: Normalized measurement
        """
        # Create a copy of the measurement
        normalized = measurement.copy()
        
        # Get the data to normalize (raw or processed)
        use_raw = kwargs.get("use_raw", False)
        if use_raw:
            data = measurement.get("raw", {})
        else:
            data = measurement.get("processed", {})
            if not data:
                # Fall back to raw if processed not available
                data = measurement.get("raw", {})
        
        if not data:
            logger.warning("No data found for normalization")
            return normalized
        
        # Get the reference value
        if reference_channel not in data:
            logger.warning(f"Reference channel {reference_channel} not found in data")
            return normalized
        
        reference_value = data[reference_channel]
        if reference_value <= 0:
            logger.warning(f"Reference value is <= 0, cannot normalize")
            return normalized
        
        # Normalize each channel
        normalized_data = {}
        for channel, value in data.items():
            # Make sure we're normalizing a number, not a dict or other type
            if isinstance(value, (int, float)) and channel != reference_channel:
                normalized_data[channel] = value / reference_value
            else:
                # Keep original value for non-numeric or reference channel
                normalized_data[channel] = value
        
        # Also store the reference value
        normalized_data[f"{reference_channel}_reference"] = reference_value
        
        # Store the normalized data
        if "processed" not in normalized:
            normalized["processed"] = {}
        normalized["processed"]["normalized"] = normalized_data
        
        return normalized
    
    def calculate_ratios(self, measurement: Dict[str, Any], 
                        ratio_definitions: Optional[Dict[str, Tuple[str, str]]] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Calculate spectral ratios from a measurement.
        
        Args:
            measurement: Measurement data
            ratio_definitions: Dictionary mapping ratio names to (numerator, denominator) channel pairs
            **kwargs: Additional arguments for ratio calculation
            
        Returns:
            Dict[str, Any]: Measurement with calculated ratios
        """
        # Create a copy of the measurement
        result = measurement.copy()
        
        # Determine which data to use (raw, processed, or filtered)
        data_source = kwargs.get("data_source", "raw")
        if data_source == "raw":
            data = measurement.get("raw", {})
        elif data_source == "processed":
            data = measurement.get("processed", {})
        elif data_source == "filtered" and "processed" in measurement and "filtered" in measurement["processed"]:
            data = measurement["processed"]["filtered"]
        elif data_source == "normalized" and "processed" in measurement and "normalized" in measurement["processed"]:
            data = measurement["processed"]["normalized"]
        else:
            # Fall back to raw data
            data = measurement.get("raw", {})
        
        if not data:
            logger.warning("No data found for ratio calculation")
            return result
        
        # Use default ratio definitions if none provided
        if not ratio_definitions:
            ratio_definitions = {
                "violet_red": ("F1", "F8"),
                "violet_green": ("F1", "F4"),
                "green_red": ("F4", "F8"),
                "blue_green": ("F2", "F4"),
                "green_yellow": ("F4", "F6"),
                "normalized_violet": ("F1", "Clear")
            }
        
        # Calculate ratios
        ratios = {}
        for ratio_name, (numerator, denominator) in ratio_definitions.items():
            if numerator in data and denominator in data:
                # Avoid division by zero
                denom_value = max(1, data[denominator])
                ratios[ratio_name] = data[numerator] / denom_value
            else:
                logger.warning(f"Could not calculate {ratio_name} ratio: channels not found in data")
        
        # Store the calculated ratios
        result["ratios"] = ratios
        
        return result
    
    def subtract_background(self, measurement: Dict[str, Any], 
                           background: Dict[str, Any],
                           **kwargs) -> Dict[str, Any]:
        """
        Subtract background from a measurement.
        
        Args:
            measurement: Measurement data
            background: Background measurement to subtract
            **kwargs: Additional arguments for background subtraction
            
        Returns:
            Dict[str, Any]: Measurement with background subtracted
        """
        # Create a copy of the measurement
        result = measurement.copy()
        
        # Get raw data from both measurements
        measurement_data = measurement.get("raw", {})
        background_data = background.get("raw", {})
        
        if not measurement_data or not background_data:
            logger.warning("Missing data for background subtraction")
            return result
        
        # Subtract background from each channel, ensuring non-negative values
        subtracted_data = {}
        for channel, value in measurement_data.items():
            if channel in background_data:
                # Subtract with floor at zero
                subtracted_data[channel] = max(0, value - background_data[channel])
            else:
                # If channel not in background, use original value
                subtracted_data[channel] = value
        
        # Store the background-subtracted data
        if "processed" not in result:
            result["processed"] = {}
        result["processed"]["background_subtracted"] = subtracted_data
        
        return result
    
    def detect_outliers(self, measurements: List[Dict[str, Any]],
                       method: str = "zscore",
                       threshold: float = 3.0,
                       **kwargs) -> List[bool]:
        """
        Detect outliers in a series of measurements.
        
        Args:
            measurements: List of measurement data
            method: Outlier detection method ('zscore', 'iqr', etc.)
            threshold: Threshold for outlier detection
            **kwargs: Additional arguments for outlier detection
            
        Returns:
            List[bool]: Boolean mask indicating outliers (True = outlier)
        """
        if not measurements:
            return []
        
        # Extract the data to analyze
        channel = kwargs.get("channel", "F4")  # Default to green channel
        data_source = kwargs.get("data_source", "raw")
        
        # Extract values for the selected channel
        values = []
        for m in measurements:
            if data_source == "raw" and "raw" in m and channel in m["raw"]:
                values.append(m["raw"][channel])
            elif data_source == "processed" and "processed" in m and channel in m["processed"]:
                values.append(m["processed"][channel])
            elif data_source == "ratios" and "ratios" in m and channel in m["ratios"]:
                values.append(m["ratios"][channel])
            else:
                # Use NaN for missing values
                values.append(float('nan'))
        
        values = np.array(values)
        mask = np.isnan(values)
        valid_values = values[~mask]
        
        if len(valid_values) < 2:
            # Not enough data for outlier detection
            return [False] * len(measurements)
        
        # Detect outliers based on the selected method
        outliers = np.zeros(len(values), dtype=bool)
        
        if method == "zscore":
            # Z-score method
            mean = np.mean(valid_values)
            std = np.std(valid_values)
            if std > 0:
                zscores = np.abs((values - mean) / std)
                outliers[~mask] = zscores[~mask] > threshold
        elif method == "iqr":
            # Interquartile range method
            q1 = np.percentile(valid_values, 25)
            q3 = np.percentile(valid_values, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers[~mask] = (valid_values < lower_bound) | (valid_values > upper_bound)
        else:
            logger.warning(f"Unknown outlier detection method: {method}")
            return [False] * len(measurements)
        
        return outliers.tolist()
    
    def extract_features(self, measurement: Dict[str, Any], 
                        features: Optional[List[str]] = None, 
                        **kwargs) -> Dict[str, Any]:
        """
        Extract features from a measurement.
        
        Args:
            measurement: Measurement data
            features: List of features to extract
            **kwargs: Additional arguments for feature extraction
            
        Returns:
            Dict[str, Any]: Extracted features
        """
        if not features:
            features = [
                "max_channel", "min_channel", "total_intensity", 
                "dominant_wavelength", "spectral_centroid", "peak_ratio"
            ]
        
        # Get the data to analyze
        data_source = kwargs.get("data_source", "raw")
        
        if data_source == "raw":
            data = measurement.get("raw", {})
        elif data_source == "processed":
            data = measurement.get("processed", {})
        elif data_source == "filtered" and "processed" in measurement and "filtered" in measurement["processed"]:
            data = measurement["processed"]["filtered"]
        else:
            # Fall back to raw data
            data = measurement.get("raw", {})
        
        if not data:
            logger.warning("No data found for feature extraction")
            return {}
        
        # Define wavelengths for each channel (in nm)
        wavelengths = {
            "F1": 415,
            "F2": 445,
            "F3": 480,
            "F4": 515,
            "F5": 555,
            "F6": 590,
            "F7": 630,
            "F8": 680
        }
        
        # Filter to only spectral channels
        spectral_data = {k: v for k, v in data.items() if k in wavelengths}
        
        if not spectral_data:
            logger.warning("No spectral data found for feature extraction")
            return {}
        
        result = {}
        
        # Extract requested features
        for feature in features:
            if feature == "max_channel":
                max_channel = max(spectral_data.items(), key=lambda x: x[1])
                result["max_channel"] = max_channel[0]
                result["max_value"] = max_channel[1]
                
            elif feature == "min_channel":
                min_channel = min(spectral_data.items(), key=lambda x: x[1])
                result["min_channel"] = min_channel[0]
                result["min_value"] = min_channel[1]
                
            elif feature == "total_intensity":
                result["total_intensity"] = sum(spectral_data.values())
                
            elif feature == "dominant_wavelength":
                # Wavelength of the channel with maximum value
                max_channel = max(spectral_data.items(), key=lambda x: x[1])[0]
                result["dominant_wavelength"] = wavelengths[max_channel]
                
            elif feature == "spectral_centroid":
                # Weighted average of wavelengths
                total = sum(spectral_data.values())
                if total > 0:
                    centroid = sum(wavelengths[ch] * val for ch, val in spectral_data.items()) / total
                    result["spectral_centroid"] = centroid
                else:
                    result["spectral_centroid"] = 0
                    
            elif feature == "peak_ratio":
                # Ratio of the maximum to the mean
                mean_value = sum(spectral_data.values()) / len(spectral_data)
                if mean_value > 0:
                    max_value = max(spectral_data.values())
                    result["peak_ratio"] = max_value / mean_value
                else:
                    result["peak_ratio"] = 0
        
        return result
    
    def analyze_kinetics(self, measurements: List[Dict[str, Any]], 
                        parameter: str = "violet_red",
                        **kwargs) -> Dict[str, Any]:
        """
        Analyze kinetics from a series of measurements.
        
        Args:
            measurements: List of measurement data
            parameter: Parameter to analyze ('violet_red', 'F1', etc.)
            **kwargs: Additional arguments for kinetic analysis
            
        Returns:
            Dict[str, Any]: Kinetic analysis results
        """
        if not measurements:
            return {}
        
        # Extract timestamps and values
        timestamps = []
        values = []
        
        for m in measurements:
            # Get timestamp
            if "timestamp" in m:
                timestamps.append(m["timestamp"])
            elif "elapsed_seconds" in m:
                timestamps.append(m["elapsed_seconds"])
            else:
                # If no timestamp, use the index
                timestamps.append(len(timestamps))
            
            # Get the parameter value
            if parameter in m.get("ratios", {}):
                values.append(m["ratios"][parameter])
            elif parameter in m.get("raw", {}):
                values.append(m["raw"][parameter])
            elif "processed" in m and parameter in m["processed"]:
                values.append(m["processed"][parameter])
            else:
                # Use NaN for missing values
                values.append(float('nan'))
        
        # Convert to numpy arrays
        timestamps = np.array(timestamps)
        values = np.array(values)
        
        # Remove NaN values
        mask = ~np.isnan(values)
        timestamps = timestamps[mask]
        values = values[mask]
        
        if len(timestamps) < 2:
            logger.warning("Not enough data for kinetic analysis")
            return {}
        
        # Normalize timestamps to start from 0
        if timestamps[0] != 0:
            timestamps = timestamps - timestamps[0]
        
        # Calculate basic kinetic parameters
        result = {
            "parameter": parameter,
            "initial_value": values[0],
            "final_value": values[-1],
            "change": values[-1] - values[0],
            "duration": timestamps[-1] - timestamps[0]
        }
        
        # Calculate rate using linear regression
        if len(timestamps) > 2:
            # Simple linear regression
            slope, intercept = np.polyfit(timestamps, values, 1)
            result["rate"] = slope
            result["intercept"] = intercept
            
            # Calculate R-squared
            y_pred = slope * timestamps + intercept
            ss_total = np.sum((values - np.mean(values)) ** 2)
            ss_residual = np.sum((values - y_pred) ** 2)
            if ss_total > 0:
                result["r_squared"] = 1 - (ss_residual / ss_total)
            else:
                result["r_squared"] = 0
        
        # Analyze different phases if enough data points
        if len(timestamps) >= 10:
            # Try to identify lag, log, and plateau phases
            # Simple approach: divide the time series into three equal parts
            n = len(timestamps)
            lag_end = n // 3
            log_end = 2 * n // 3
            
            # Lag phase rate
            if lag_end > 1:
                lag_slope, _ = np.polyfit(timestamps[:lag_end], values[:lag_end], 1)
                result["lag_rate"] = lag_slope
            
            # Log (exponential) phase rate
            if log_end - lag_end > 1:
                log_slope, _ = np.polyfit(timestamps[lag_end:log_end], values[lag_end:log_end], 1)
                result["log_rate"] = log_slope
            
            # Plateau phase rate
            if n - log_end > 1:
                plateau_slope, _ = np.polyfit(timestamps[log_end:], values[log_end:], 1)
                result["plateau_rate"] = plateau_slope
        
        return result
    
    def classify_agglutination(self, measurement: Dict[str, Any], 
                             threshold_config: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Classify agglutination state based on measurement data.
        
        Args:
            measurement: Measurement data
            threshold_config: Thresholds for classification
            
        Returns:
            Dict[str, Any]: Classification results
        """
        # Default thresholds if not provided
        if not threshold_config:
            threshold_config = {
                "none_threshold": 1.0,
                "minimal_threshold": 1.2,
                "moderate_threshold": 1.5,
                "strong_threshold": 2.0
            }
        
        # Get ratios
        ratios = measurement.get("ratios", {})
        if not ratios:
            logger.warning("No ratios found for agglutination classification")
            return {
                "state": "unknown",
                "score": 0.0,
                "confidence": 0.0
            }
        
        # Get violet/red ratio (primary indicator of agglutination)
        violet_red = ratios.get("violet_red", 0.0)
        
        # Classify based on thresholds
        if violet_red >= threshold_config["strong_threshold"]:
            state = "complete"
            score = 4.0
            confidence = 0.9
        elif violet_red >= threshold_config["moderate_threshold"]:
            state = "strong"
            score = 3.0
            confidence = 0.8
        elif violet_red >= threshold_config["minimal_threshold"]:
            state = "moderate"
            score = 2.0
            confidence = 0.7
        elif violet_red >= threshold_config["none_threshold"]:
            state = "minimal"
            score = 1.0
            confidence = 0.6
        else:
            state = "none"
            score = 0.0
            confidence = 0.8
        
        # Adjust confidence based on secondary indicators
        # Examples: consistency between different ratios, signal strength, etc.
        violet_green = ratios.get("violet_green", 0.0)
        green_red = ratios.get("green_red", 0.0)
        
        # Consistency check between ratios
        # If violet/green and green/red ratios are consistent with violet/red,
        # confidence increases. If not, confidence decreases.
        expected_product = violet_green * green_red
        if expected_product > 0:
            ratio_consistency = abs(violet_red - expected_product) / max(violet_red, expected_product)
            if ratio_consistency < 0.2:
                confidence = min(1.0, confidence * 1.1)  # Increase confidence
            elif ratio_consistency > 0.5:
                confidence = confidence * 0.9  # Decrease confidence
        
        return {
            "state": state,
            "score": score,
            "confidence": confidence,
            "primary_ratio": violet_red
        }
    
    def _moving_average_filter(self, data: Dict[str, Any], window_size: int = 3) -> Dict[str, Any]:
        """
        Apply a moving average filter to spectral data.
        
        Args:
            data: Spectral data to filter
            window_size: Window size for the moving average
            
        Returns:
            Dict[str, Any]: Filtered data
        """
        result = {}
        
        if window_size < 2:
            return data.copy()
        
        # Get spectral channels in order
        channels = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
        
        # Extract values for spectral channels
        spectral_values = []
        for ch in channels:
            if ch in data:
                spectral_values.append(data[ch])
        
        # Apply moving average to spectral channels
        if spectral_values:
            # Pad the values to handle edge effects
            padded = np.pad(spectral_values, (window_size//2, window_size//2), mode='edge')
            
            # Apply moving average
            kernel = np.ones(window_size) / window_size
            filtered = np.convolve(padded, kernel, mode='valid')
            
            # Copy filtered values back to result
            for i, ch in enumerate(channels):
                if ch in data:
                    result[ch] = filtered[i]
        
        # Copy non-spectral channels directly
        for ch, val in data.items():
            if ch not in channels:
                result[ch] = val
        
        return result
    
    def _savgol_filter(self, data: Dict[str, Any], window_size: int = 5, poly_order: int = 2) -> Dict[str, Any]:
        """
        Apply a Savitzky-Golay filter to spectral data.
        
        Args:
            data: Spectral data to filter
            window_size: Window size for the filter
            poly_order: Polynomial order for the filter
            
        Returns:
            Dict[str, Any]: Filtered data
        """
        result = {}
        
        if window_size < poly_order + 2:
            # Window size must be larger than polynomial order
            window_size = poly_order + 2
        
        if window_size % 2 == 0:
            # Window size must be odd
            window_size += 1
        
        # Get spectral channels in order
        channels = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
        
        # Extract values for spectral channels
        spectral_values = []
        for ch in channels:
            if ch in data:
                spectral_values.append(data[ch])
        
        # Apply Savitzky-Golay filter to spectral channels
        if len(spectral_values) >= window_size:
            # Apply filter
            try:
                filtered = signal.savgol_filter(spectral_values, window_size, poly_order)
                
                # Copy filtered values back to result
                for i, ch in enumerate(channels):
                    if ch in data:
                        result[ch] = filtered[i]
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.warning(f"Failed to apply Savitzky-Golay filter: {str(e)}")
                # Fall back to original data
                for ch in channels:
                    if ch in data:
                        result[ch] = data[ch]
        else:
            # Not enough data points for filtering
            for ch in channels:
                if ch in data:
                    result[ch] = data[ch]
        
        # Copy non-spectral channels directly
        for ch, val in data.items():
            if ch not in channels:
                result[ch] = val
        
        return result
    
    def _median_filter(self, data: Dict[str, Any], window_size: int = 3) -> Dict[str, Any]:
        """
        Apply a median filter to spectral data.
        
        Args:
            data: Spectral data to filter
            window_size: Window size for the filter
            
        Returns:
            Dict[str, Any]: Filtered data
        """
        result = {}
        
        if window_size < 3:
            window_size = 3
        
        if window_size % 2 == 0:
            # Window size must be odd
            window_size += 1
        
        # Get spectral channels in order
        channels = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
        
        # Extract values for spectral channels
        spectral_values = []
        for ch in channels:
            if ch in data:
                spectral_values.append(data[ch])
        
        # Apply median filter to spectral channels
        if len(spectral_values) >= window_size:
            # Pad the values to handle edge effects
            padded = np.pad(spectral_values, (window_size//2, window_size//2), mode='edge')
            
            # Apply median filter
            filtered = signal.medfilt(padded, window_size)
            
            # Copy filtered values back to result
            for i, ch in enumerate(channels):
                if ch in data:
                    result[ch] = filtered[i + window_size//2]
        else:
            # Not enough data points for filtering
            for ch in channels:
                if ch in data:
                    result[ch] = data[ch]
        
        # Copy non-spectral channels directly
        for ch, val in data.items():
            if ch not in channels:
                result[ch] = val
        
        return result


# Create a global instance for convenience
signal_processor = SignalProcessor()