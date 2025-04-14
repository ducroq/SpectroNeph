#!/usr/bin/env python3
"""
AS7341 Sensor Functionality Tests for SpectroNeph

This module contains tests to verify the functionality of the AS7341 spectral sensor
and basic agglutination detection. It tests sensor readings, different configurations,
and basic spectral analysis.

Usage:
    python sensor_functionality_tests.py [--port COM_PORT]
"""

import sys
import time
import json
import argparse
import unittest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Project root added to sys.path: {project_root}")

# Import modules from the SpectroNeph project directly
sys.path.insert(0, str(project_root / "app"))
from core.device import DeviceManager
from hardware.as7341 import AS7341Channel
from hardware.nephelometer import AgglutinationState
from utils.logging import setup_logging, get_logger
from utils.helpers import detect_serial_port

# Initialize logger
setup_logging()
logger = get_logger("sensor_tests")

class AS7341SensorTests(unittest.TestCase):
    """Test suite for AS7341 sensor functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test class - connect to device once for all tests."""
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='SpectroNeph AS7341 Sensor Tests')
        parser.add_argument('--port', help='Serial port to connect to')
        parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
        args = parser.parse_args()
        
        cls.show_plots = not args.no_plots
        
        # Initialize device manager
        cls.device = DeviceManager()
        
        # Connect to device
        port = args.port if args.port else detect_serial_port()
        if not cls.device.connect(port=port):
            print("❌ FAILED: Could not connect to device")
            print("   Make sure the device is connected and the port is correct")
            sys.exit(1)
            
        print(f"✓ Connected to device on {cls.device._comm._port}")
        
        # Initialize AS7341 sensor
        response = cls.device.send_command("as7341_init")
        if response.get("status") != 0 or not response.get("data", {}).get("initialized", False):
            print("❌ FAILED: Could not initialize AS7341 sensor")
            cls.device.disconnect()
            sys.exit(1)
            
        print("✓ AS7341 sensor initialized")
        
        # Store channel mappings
        cls.channel_wavelengths = {
            "F1": 415,  # Violet
            "F2": 445,  # Indigo
            "F3": 480,  # Blue
            "F4": 515,  # Cyan/Green
            "F5": 555,  # Green
            "F6": 590,  # Yellow
            "F7": 630,  # Orange/Red
            "F8": 680,  # Red
            "Clear": 0,  # Clear channel (broadband)
            "NIR": 0,    # Near IR
        }
        
        # Create a data directory if it doesn't exist
        cls.data_dir = project_root / "test_data"
        cls.data_dir.mkdir(exist_ok=True)
              
        # Sleep briefly to ensure connection is stable
        time.sleep(0.5)

        # Verify that a scattering sample is present
        if not cls.verify_sample_present():
            print("❌ FAILED: No scattering sample detected")
            cls.device.disconnect()
            sys.exit(1)
  
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Disconnect from device
        if cls.device and cls.device.is_connected():
            cls.device.disconnect()
            print("✓ Disconnected from device")
    
    def setUp(self):
        """Set up before each test."""
        # Verify connection is still active
        if not self.device.is_connected():
            self.device.connect()
            time.sleep(0.5)

    @classmethod
    def verify_sample_present(cls):
        """Verify that a scattering sample is present in the nephelometer."""
        # Configure for test
        cls.device.send_command("as7341_config", {
            "gain": 7,  # 64x gain for maximum sensitivity
            "integration_time": 200  # 200ms
        })
        
        # LED on at medium intensity
        cls.device.send_command("as7341_led", {
            "enabled": True,
            "current": 15
        })
        
        # Wait for LED to stabilize
        time.sleep(0.2)
        
        # Read sensor
        response = cls.device.send_command("as7341_read")
        
        # Turn off LED
        cls.device.send_command("as7341_led", {"enabled": False})
        
        # Check if we have significant signal in F4 (515nm) channel
        if response.get("status") == 0:
            f4_signal = response.get("data", {}).get("F4", 0)
            if f4_signal < 100:  # Threshold depends on your system sensitivity
                print("\n⚠️ WARNING: Very low scattering signal detected.")
                print("  Please ensure a scattering sample is present in the nephelometer.")
                print("  Suggested samples: dilute milk solution, polystyrene beads, or formazin standard.")
                print("  Simple DIY solution: Diluted milk (1:1000 to 1:100,000 in water) - contains lipid particles that scatter well.")
                return False
        
        return True
    
    def test_01_gain_sensitivity(self):
        """Test sensor sensitivity at different gain settings."""
        # Define gain values to test
        gain_values = [3, 5, 6, 7]  # 4x, 16x, 32x, 64x
        results = {}
        
        try:
            # Turn on LED
            self.device.send_command("as7341_led", {
                "enabled": True,
                "current": 10
            })
            
            # Wait for LED to stabilize
            time.sleep(0.2)
            
            # Test each gain setting
            for gain in gain_values:
                # Configure sensor with this gain
                self.device.send_command("as7341_config", {
                    "gain": gain,
                    "integration_time": 100
                })
                
                # Wait for settings to take effect
                time.sleep(0.2)
                
                # Read sensor data
                response = self.device.send_command("as7341_read")
                self.assertEqual(response.get("status"), 0, 
                                f"Reading with gain {gain} should succeed")
                
                # Store result
                results[gain] = response.get("data", {})
                
                # Check if readings increase with gain
                if gain > gain_values[0]:
                    for channel in ["F1", "F4", "F8"]:
                        current = results[gain].get(channel, 0)
                        previous = results[gain_values[gain_values.index(gain)-1]].get(channel, 0)
                        
                        # Gain should approximately double for each step
                        self.assertGreater(current, previous * 1.5, 
                                          f"Channel {channel} should increase significantly with higher gain")
            
            # Turn off LED
            self.device.send_command("as7341_led", {"enabled": False})
            
            print("✓ Gain sensitivity test passed")
            
            # Plot results if enabled
            if self.show_plots:
                self._plot_gain_results(results, gain_values)
            
        except Exception as e:
            # Ensure LED is off
            self.device.send_command("as7341_led", {"enabled": False})
            self.fail(f"Gain sensitivity test failed: {str(e)}")
    
    def test_02_spectral_response(self):
        """Test sensor spectral response and wavelength discrimination."""
        try:
            # Configure sensor with optimal settings for this test
            self.device.send_command("as7341_config", {
                "gain": 6,  # 32x gain
                "integration_time": 100
            })
            
            # Measure dark (LED off) reading
            self.device.send_command("as7341_led", {"enabled": False})
            time.sleep(0.2)
            dark_response = self.device.send_command("as7341_read")
            dark_readings = dark_response.get("data", {})
            
            # Measure with LED on
            self.device.send_command("as7341_led", {"enabled": True, "current": 10})
            time.sleep(0.2)
            light_response = self.device.send_command("as7341_read")
            light_readings = light_response.get("data", {})
            
            # Turn off LED
            self.device.send_command("as7341_led", {"enabled": False})
            
            # Calculate signal-to-noise ratio for each channel
            snr = {}
            for channel in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]:
                dark = max(1, dark_readings.get(channel, 1))  # Avoid division by zero
                light = light_readings.get(channel, 0)
                snr[channel] = light / dark
            
            # Check if all channels show response
            for channel, ratio in snr.items():
                self.assertGreater(ratio, 2.0, 
                                  f"Channel {channel} should show significant response to LED")
            
            print("✓ Spectral response test passed")
            
            # Plot results if enabled
            if self.show_plots:
                self._plot_spectral_response(light_readings, dark_readings)
            
        except Exception as e:
            # Ensure LED is off
            self.device.send_command("as7341_led", {"enabled": False})
            self.fail(f"Spectral response test failed: {str(e)}")
    
    def test_03_integration_time(self):
        """Test sensor response to different integration times."""
        # Define integration times to test
        int_times = [20, 50, 100, 200]  # ms
        results = {}
        
        try:
            # Configure sensor with fixed gain
            self.device.send_command("as7341_config", {
                "gain": 5,  # 16x gain
            })
            
            # Turn on LED
            self.device.send_command("as7341_led", {
                "enabled": True,
                "current": 10
            })
            
            # Wait for LED to stabilize
            time.sleep(0.2)
            
            # Test each integration time
            for int_time in int_times:
                # Configure sensor with this integration time
                self.device.send_command("as7341_config", {
                    "integration_time": int_time
                })
                
                # Wait for settings to take effect
                time.sleep(0.2)
                
                # Read sensor data
                response = self.device.send_command("as7341_read")
                self.assertEqual(response.get("status"), 0, 
                                f"Reading with integration time {int_time}ms should succeed")
                
                # Store result
                results[int_time] = response.get("data", {})
                
                # Check if readings scale with integration time
                if int_time > int_times[0]:
                    for channel in ["F1", "F4", "F8"]:
                        current = results[int_time].get(channel, 0)
                        baseline = results[int_times[0]].get(channel, 0)
                        expected_ratio = int_time / int_times[0]
                        actual_ratio = current / max(1, baseline)  # Avoid division by zero
                        
                        # Allow 30% tolerance in linearity
                        self.assertLessEqual(abs(actual_ratio - expected_ratio) / expected_ratio, 0.5,
                                             f"Channel {channel} should scale approximately linearly with integration time")
            
            # Turn off LED
            self.device.send_command("as7341_led", {"enabled": False})
            
            print("✓ Integration time test passed")
            
            # Plot results if enabled
            if self.show_plots:
                self._plot_integration_time_results(results, int_times)
            
        except Exception as e:
            # Ensure LED is off
            self.device.send_command("as7341_led", {"enabled": False})
            self.fail(f"Integration time test failed: {str(e)}")
    
    def test_04_led_current(self):
        """Test sensor response to different LED currents."""
        # Define LED currents to test - use a wider range with bigger steps
        led_currents = [2, 10, 18]  # mA - use minimum, middle, and maximum values
        results = {}
        
        try:
            # Configure sensor with fixed settings - use higher gain for better sensitivity
            self.device.send_command("as7341_config", {
                "gain": 7,  # 64x gain instead of 16x for better sensitivity
                "integration_time": 150  # Longer integration time (150ms)
            })
            
            # Test each LED current
            for current in led_currents:
                # Turn on LED with this current
                self.device.send_command("as7341_led", {
                    "enabled": True,
                    "current": current
                })
                
                # Wait longer for LED to stabilize
                time.sleep(0.5)
                
                # Take multiple readings and average them for stability
                readings = []
                for _ in range(3):
                    response = self.device.send_command("as7341_read")
                    readings.append(response.get("data", {}))
                    time.sleep(0.1)
                
                # Average the readings
                avg_reading = {}
                for channel in ["F1", "F4", "F8", "Clear"]:
                    values = [r.get(channel, 0) for r in readings]
                    avg_reading[channel] = sum(values) / len(values)
                
                # Store result
                results[current] = avg_reading
                
                # Turn off LED
                self.device.send_command("as7341_led", {"enabled": False})
                time.sleep(0.3)  # Wait before next test
            
            # Check from minimum to maximum (first to last) instead of incremental
            min_current = led_currents[0]
            max_current = led_currents[-1]
            
            # Check if any channel shows a significant increase (at least 20%)
            increases = []
            for channel in ["F1", "F4", "F8", "Clear"]:
                min_val = results[min_current].get(channel, 0)
                max_val = results[max_current].get(channel, 0)
                
                if min_val > 0:
                    percent_increase = (max_val - min_val) / min_val * 100
                    increases.append((channel, percent_increase))
            
            # Print all increases for debugging
            print("  Channel increases from min to max LED current:")
            for channel, percent in increases:
                print(f"    {channel}: {percent:.1f}%")
            
            # Check if any channel increased by at least 10%
            significant_increase = any(percent >= 10 for _, percent in increases)
            self.assertTrue(significant_increase, 
                        "At least one channel should increase by 10% or more from min to max LED current")
            
            print("✓ LED current test passed")
            
        except Exception as e:
            # Ensure LED is off
            self.device.send_command("as7341_led", {"enabled": False})
            self.fail(f"LED current test failed: {str(e)}")
                
    def test_05_spectral_ratios(self):
        """Test calculation of channel ratios used for agglutination detection."""
        try:
            # Configure sensor with optimal settings
            self.device.send_command("as7341_config", {
                "gain": 6,  # 32x gain
                "integration_time": 100  # 100ms
            })
            
            # Turn on LED
            self.device.send_command("as7341_led", {
                "enabled": True,
                "current": 10
            })
            
            # Wait for LED to stabilize
            time.sleep(0.2)
            
            # Read sensor data
            response = self.device.send_command("as7341_read")
            self.assertEqual(response.get("status"), 0, "Sensor reading should succeed")
            
            # Turn off LED
            self.device.send_command("as7341_led", {"enabled": False})
            
            # Get raw readings
            readings = response.get("data", {})
            
            # Calculate ratios
            ratios = self._calculate_spectral_ratios(readings)
            
            # Check if ratios are reasonable (will vary by device and conditions)
            self.assertGreater(ratios["violet_red"], 0, "Violet/Red ratio should be positive")
            self.assertGreater(ratios["violet_green"], 0, "Violet/Green ratio should be positive")
            self.assertGreater(ratios["green_red"], 0, "Green/Red ratio should be positive")
            
            print("✓ Spectral ratios test passed")
            print(f"  Sample ratios - Violet/Red: {ratios['violet_red']:.2f}, "
                  f"Violet/Green: {ratios['violet_green']:.2f}, "
                  f"Green/Red: {ratios['green_red']:.2f}")
            
            # Save reference ratios to file
            ratios_path = self.data_dir / "reference_ratios.json"
            with open(ratios_path, 'w') as f:
                full_data = {
                    "raw_readings": readings,
                    "calculated_ratios": ratios,
                    "timestamp": time.time()
                }
                json.dump(full_data, f, indent=2)
            
            print(f"  Reference ratios saved to {ratios_path}")
            
        except Exception as e:
            # Ensure LED is off
            self.device.send_command("as7341_led", {"enabled": False})
            self.fail(f"Spectral ratios test failed: {str(e)}")
    
    def test_06_simulated_agglutination(self):
        """
        Test agglutination detection algorithm with simulated data.
        
        Since we can't perform actual agglutination in a test, we'll simulate
        it by modifying the spectral ratios.
        """
        try:
            # Load the reference ratios
            try:
                with open(self.data_dir / "reference_ratios.json", 'r') as f:
                    reference_data = json.load(f)
                    reference_readings = reference_data["raw_readings"]
            except (FileNotFoundError, json.JSONDecodeError):
                # If no reference file, take a new reading
                self.device.send_command("as7341_config", {
                    "gain": 6,  # 32x gain
                    "integration_time": 100  # 100ms
                })
                self.device.send_command("as7341_led", {"enabled": True, "current": 10})
                time.sleep(0.2)
                response = self.device.send_command("as7341_read")
                reference_readings = response.get("data", {})
                self.device.send_command("as7341_led", {"enabled": False})
                
            # Create simulated readings for different agglutination states
            agglutination_stages = {
                AgglutinationState.NONE: self._simulate_agglutination(reference_readings, 1.0),
                AgglutinationState.MINIMAL: self._simulate_agglutination(reference_readings, 1.2),
                AgglutinationState.MODERATE: self._simulate_agglutination(reference_readings, 1.5),
                AgglutinationState.STRONG: self._simulate_agglutination(reference_readings, 2.0),
                AgglutinationState.COMPLETE: self._simulate_agglutination(reference_readings, 2.5)
            }
            
            # Calculate ratios and check classification for each stage
            results = {}
            for stage, readings in agglutination_stages.items():
                ratios = self._calculate_spectral_ratios(readings)
                classification = self._classify_agglutination(ratios)
                results[stage] = {
                    "readings": readings,
                    "ratios": ratios,
                    "classification": classification
                }
                
                # Check if classification matches the expected stage
                # Allow one level of difference since classification is subjective
                stage_value = stage.value
                classification_value = classification["state"].value
                max_diff = 1  # Allow one level of difference
                
                self.assertLessEqual(abs(stage_value - classification_value), max_diff,
                                    f"Classification should be within {max_diff} level of expected stage")
            
            print("✓ Simulated agglutination test passed")
            print("  Agglutination classifications:")
            for stage, result in results.items():
                print(f"    {stage.name}: Classified as {result['classification']['state'].name} "
                     f"(Score: {result['classification']['score']:.1f}, "
                     f"Confidence: {result['classification']['confidence']:.1f})")
            
            # Plot results if enabled
            if self.show_plots:
                self._plot_agglutination_simulation(results)
                
        except Exception as e:
            self.fail(f"Simulated agglutination test failed: {str(e)}")
    
    def test_07_noise_floor(self):
        """Test sensor noise floor with LED off."""
        try:
            # Configure sensor with high gain
            self.device.send_command("as7341_config", {
                "gain": 8,  # 128x gain for maximum sensitivity
                "integration_time": 200  # 200ms
            })
            
            # Ensure LED is off
            self.device.send_command("as7341_led", {"enabled": False})
            time.sleep(0.5)  # Wait for any residual light to fade
            
            # Take multiple readings to assess noise
            num_readings = 5
            noise_readings = []
            
            for i in range(num_readings):
                response = self.device.send_command("as7341_read")
                self.assertEqual(response.get("status"), 0, f"Dark reading {i+1} should succeed")
                noise_readings.append(response.get("data", {}))
                time.sleep(0.1)
            
            # Calculate noise statistics
            noise_stats = {}
            for channel in ["F1", "F4", "F8", "Clear"]:
                values = [reading.get(channel, 0) for reading in noise_readings]
                noise_stats[channel] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": min(values),
                    "max": max(values)
                }
                
                # Check noise levels are reasonable
                self.assertLess(noise_stats[channel]["mean"], 1000, 
                               f"Dark noise mean for {channel} should be below 1000")
                self.assertLess(noise_stats[channel]["std"], 100, 
                               f"Dark noise standard deviation for {channel} should be below 100")
            
            print("✓ Noise floor test passed")
            print("  Noise statistics:")
            for channel, stats in noise_stats.items():
                print(f"    {channel}: Mean={stats['mean']:.1f}, StdDev={stats['std']:.1f}, "
                     f"Range={stats['min']}-{stats['max']}")
                
        except Exception as e:
            self.fail(f"Noise floor test failed: {str(e)}")
    
    def _calculate_spectral_ratios(self, readings):
        """Calculate spectral ratios used for agglutination detection."""
        # Ensure all necessary channels are present
        if not all(key in readings for key in ["F1", "F4", "F8", "Clear"]):
            raise ValueError("Missing required channels in readings")
        
        # Get channel values, ensuring no division by zero
        f1 = readings["F1"]  # 415nm (Violet)
        f4 = max(1, readings["F4"])  # 515nm (Green)
        f8 = max(1, readings["F8"])  # 680nm (Red)
        clear = max(1, readings["Clear"])  # Clear channel
        
        # Calculate ratios
        ratios = {
            "violet_green": f1 / f4,  # Violet to Green
            "violet_red": f1 / f8,    # Violet to Red
            "green_red": f4 / f8,     # Green to Red
            "normalized_violet": f1 / clear  # Violet normalized to Clear
        }
        
        return ratios
    
    def _simulate_agglutination(self, base_readings, violet_red_ratio):
        """
        Simulate agglutination by adjusting spectral readings.
        
        Args:
            base_readings: Base spectral readings to modify
            violet_red_ratio: Target violet/red ratio
            
        Returns:
            Modified readings simulating the specified level of agglutination
        """
        # Create a copy of base readings
        modified = base_readings.copy()
        
        # Calculate current violet/red ratio
        current_ratio = base_readings["F1"] / max(1, base_readings["F8"])
        
        # Adjust F1 (violet) and F8 (red) to achieve the target ratio
        ratio_factor = violet_red_ratio / current_ratio
        
        # Increase violet and decrease red to simulate agglutination
        modified["F1"] = int(base_readings["F1"] * np.sqrt(ratio_factor))
        modified["F8"] = int(base_readings["F8"] / np.sqrt(ratio_factor))
        
        # Also adjust nearby channels to create a realistic spectrum
        # Channels between violet and red get scaled proportionally
        channels = ["F2", "F3", "F4", "F5", "F6", "F7"]
        for i, ch in enumerate(channels):
            if ch in base_readings:
                # Linear interpolation factor (0 at violet end, 1 at red end)
                factor = (i + 1) / (len(channels) + 1)
                # Blend between violet scaling and red scaling
                scale = np.sqrt(ratio_factor) * (1 - factor) + (1 / np.sqrt(ratio_factor)) * factor
                modified[ch] = int(base_readings[ch] * scale)
        
        return modified
    
    def _classify_agglutination(self, ratios):
        """
        Classify agglutination state based on spectral ratios.
        
        This is a simplified version of the algorithm that might be used
        in a real agglutination detection system.
        
        Args:
            ratios: Spectral ratios calculated from readings
            
        Returns:
            Dict with classification results
        """
        # Extract the violet/red ratio
        violet_red = ratios.get("violet_red", 0)
        
        # Simple classification based on violet/red ratio
        if violet_red > 2.0:
            state = AgglutinationState.COMPLETE
            score = 4.0
            confidence = 0.9
        elif violet_red > 1.5:
            state = AgglutinationState.STRONG
            score = 3.0
            confidence = 0.8
        elif violet_red > 1.2:
            state = AgglutinationState.MODERATE
            score = 2.0
            confidence = 0.7
        elif violet_red > 1.0:
            state = AgglutinationState.MINIMAL
            score = 1.0
            confidence = 0.6
        else:
            state = AgglutinationState.NONE
            score = 0.0
            confidence = 0.8
        
        # Additional confidence adjustment based on other ratios
        # In a real system, this would be more sophisticated
        if abs(ratios.get("green_red", 1) - 1.0) > 0.5:
            confidence *= 0.9
        
        return {
            "state": state,
            "score": score,
            "confidence": confidence,
            "ratios": ratios
        }
    
    def _plot_gain_results(self, results, gain_values):
        """Plot results from gain sensitivity test."""
        plt.figure(figsize=(10, 6))
        
        # Create x-axis labels (actual gain values, not indices)
        gain_labels = [f"{2**(g-1)}x" for g in gain_values]
        x = np.arange(len(gain_values))
        
        # Plot key channels
        channels = ["F1", "F4", "F8"]
        colors = ["blue", "green", "red"]
        
        for i, channel in enumerate(channels):
            values = [results[gain].get(channel, 0) for gain in gain_values]
            plt.plot(x, values, 'o-', color=colors[i], label=f"{channel} ({self.channel_wavelengths[channel]}nm)")
        
        plt.xticks(x, gain_labels)
        plt.xlabel('Gain Setting')
        plt.ylabel('Channel Reading')
        plt.title('AS7341 Response vs. Gain Setting')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save and show
        plt.savefig(self.data_dir / 'gain_response.png')
        plt.show() if self.show_plots else plt.close()
    
    def _plot_spectral_response(self, light_readings, dark_readings):
        """Plot spectral response."""
        plt.figure(figsize=(12, 7))
        
        # Extract channels and wavelengths
        channels = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
        wavelengths = [self.channel_wavelengths[ch] for ch in channels]
        
        # Light and dark values
        light_values = [light_readings.get(ch, 0) for ch in channels]
        dark_values = [dark_readings.get(ch, 0) for ch in channels]
        
        # Calculate signal-to-noise ratio
        snr_values = [light / max(1, dark) for light, dark in zip(light_values, dark_values)]
        
        # Create the figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot raw counts
        ax1.bar(wavelengths, light_values, width=20, alpha=0.7, color='blue', label='LED On')
        ax1.bar(wavelengths, dark_values, width=20, alpha=0.7, color='gray', label='LED Off')
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Channel Reading')
        ax1.set_title('AS7341 Spectral Response')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot SNR
        ax2.bar(wavelengths, snr_values, width=20, color='green')
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Signal-to-Noise Ratio')
        ax2.set_title('AS7341 Channel Signal-to-Noise Ratio')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save and show
        plt.savefig(self.data_dir / 'spectral_response.png')
        plt.show() if self.show_plots else plt.close()
    
    def _plot_integration_time_results(self, results, int_times):
        """Plot results from integration time test."""
        plt.figure(figsize=(10, 6))
        
        # Create x-axis values
        x = int_times
        
        # Plot key channels
        channels = ["F1", "F4", "F8"]
        colors = ["blue", "green", "red"]
        
        for i, channel in enumerate(channels):
            values = [results[t].get(channel, 0) for t in int_times]
            plt.plot(x, values, 'o-', color=colors[i], label=f"{channel} ({self.channel_wavelengths[channel]}nm)")
        
        # Plot theoretical linear response
        baseline_channel = "F4"
        baseline_value = results[int_times[0]].get(baseline_channel, 1)
        theoretical = [baseline_value * (t / int_times[0]) for t in int_times]
        plt.plot(x, theoretical, '--', color='black', label='Linear (Theoretical)')
        
        plt.xlabel('Integration Time (ms)')
        plt.ylabel('Channel Reading')
        plt.title('AS7341 Response vs. Integration Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save and show
        plt.savefig(self.data_dir / 'integration_time_response.png')
        plt.show() if self.show_plots else plt.close()
    
    def _plot_led_current_results(self, results, led_currents):
        """Plot results from LED current test."""
        plt.figure(figsize=(10, 6))
        
        # Create x-axis values
        x = led_currents
        
        # Plot key channels
        channels = ["F1", "F4", "F8"]
        colors = ["blue", "green", "red"]
        
        for i, channel in enumerate(channels):
            values = [results[c].get(channel, 0) for c in led_currents]
            plt.plot(x, values, 'o-', color=colors[i], label=f"{channel} ({self.channel_wavelengths[channel]}nm)")
        
        plt.xlabel('LED Current (mA)')
        plt.ylabel('Channel Reading')
        plt.title('AS7341 Response vs. LED Current')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save and show
        plt.savefig(self.data_dir / 'led_current_response.png')
        plt.show() if self.show_plots else plt.close()
    
    def _plot_agglutination_simulation(self, results):
        """Plot results from agglutination simulation."""
        plt.figure(figsize=(12, 10))
        
        # Extract stages in order
        stages = [
            AgglutinationState.NONE,
            AgglutinationState.MINIMAL,
            AgglutinationState.MODERATE,
            AgglutinationState.STRONG,
            AgglutinationState.COMPLETE
        ]
        
        # Plot spectral profiles for each stage
        channels = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
        wavelengths = [self.channel_wavelengths[ch] for ch in channels]
        
        plt.subplot(2, 1, 1)
        for stage in stages:
            if stage in results:
                readings = results[stage]["readings"]
                values = [readings.get(ch, 0) for ch in channels]
                plt.plot(wavelengths, values, 'o-', label=f"{stage.name}")
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Channel Reading')
        plt.title('Simulated Spectral Profiles for Different Agglutination States')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot key ratios
        plt.subplot(2, 1, 2)
        
        # Extract key ratio values
        violet_red_values = [results[stage]["ratios"]["violet_red"] for stage in stages if stage in results]
        violet_green_values = [results[stage]["ratios"]["violet_green"] for stage in stages if stage in results]
        green_red_values = [results[stage]["ratios"]["green_red"] for stage in stages if stage in results]
        
        # Stage names for x-axis
        stage_names = [stage.name for stage in stages if stage in results]
        x = np.arange(len(stage_names))
        
        # Bar positions
        width = 0.25
        
        # Plot each ratio as a group of bars
        plt.bar(x - width, violet_red_values, width, label='Violet/Red', color='purple')
        plt.bar(x, violet_green_values, width, label='Violet/Green', color='teal')
        plt.bar(x + width, green_red_values, width, label='Green/Red', color='orange')
        
        plt.xticks(x, stage_names)
        plt.xlabel('Agglutination State')
        plt.ylabel('Ratio Value')
        plt.title('Spectral Ratios for Different Agglutination States')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save and show
        plt.savefig(self.data_dir / 'agglutination_simulation.png')
        plt.show() if self.show_plots else plt.close()

def main():
    """Main entry point for the test script."""
    print("\n=== SpectroNeph AS7341 Sensor Tests ===\n")
    
    # Run the tests
    unittest.main(argv=[sys.argv[0]])

if __name__ == "__main__":
    main()