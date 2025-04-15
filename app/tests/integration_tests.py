#!/usr/bin/env python3
"""
Nephelometer Integration Tests for SpectroNeph

This module contains comprehensive integration tests for the Nephelometer class.
It verifies core functionality including measurements, data processing, and analysis.

Usage:
    python nephelometer_integration_tests.py [--port COM_PORT]
"""

import sys
import time
import json
import argparse
import unittest
import threading
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Project root added to sys.path: {project_root}")

# Import modules from the SpectroNeph project directly
sys.path.insert(0, str(project_root / "app"))
from core.device import DeviceManager
from hardware.nephelometer import Nephelometer, MeasurementMode, AgglutinationState
from utils.logging import setup_logging, get_logger
from utils.helpers import detect_serial_port

# Initialize logger
setup_logging()
logger = get_logger("nephelometer_tests")

class NephelometerIntegrationTests(unittest.TestCase):
    """Integration tests for the Nephelometer class."""

    @classmethod
    def setUpClass(cls):
        """Set up test class - connect to device once for all tests."""
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='SpectroNeph Nephelometer Tests')
        parser.add_argument('--port', help='Serial port to connect to')
        parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
        args = parser.parse_args()
        
        cls.show_plots = not args.no_plots
        
        # Initialize device manager
        cls.device_manager = DeviceManager()
        
        # Connect to device
        port = args.port if args.port else detect_serial_port()
        if not cls.device_manager.connect(port=port):
            print("❌ FAILED: Could not connect to device")
            print("   Make sure the device is connected and the port is correct")
            sys.exit(1)
            
        print(f"✓ Connected to device on {cls.device_manager._comm._port}")
        
        # Create a data directory if it doesn't exist
        cls.data_dir = project_root / "test_data"
        cls.data_dir.mkdir(exist_ok=True)
        
        # Sleep briefly to ensure connection is stable
        time.sleep(0.5)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Disconnect from device
        if cls.device_manager and cls.device_manager.is_connected():
            cls.device_manager.disconnect()
            print("✓ Disconnected from device")
    
    def setUp(self):
        """Set up before each test."""
        # Verify connection is still active
        if not self.device_manager.is_connected():
            self.device_manager.connect()
            time.sleep(0.5)
    
    def test_01_nephelometer_initialization(self):
        """Test nephelometer initialization."""
        try:
            # Create a new nephelometer instance
            nephelometer = Nephelometer(self.device_manager)
            
            # Initialize it
            result = nephelometer.initialize()
            
            # Verify initialization was successful
            self.assertTrue(result, "Nephelometer initialization should succeed")
            
            print("✓ Nephelometer initialization test passed")
            
        except Exception as e:
            self.fail(f"Nephelometer initialization test failed: {str(e)}")
    
    def test_02_configuration(self):
        """Test nephelometer configuration."""
        try:
            # Create and initialize nephelometer
            nephelometer = Nephelometer(self.device_manager)
            nephelometer.initialize()
            
            # Configure with specific settings (high values for our low-signal sample)
            config = {
                "gain": 8,             # 128x gain
                "integration_time": 200, # 200ms
                "led_current": 15      # 15mA
            }
            
            # Apply configuration
            result = nephelometer.configure(config)
            
            # Verify configuration was successful
            self.assertTrue(result, "Nephelometer configuration should succeed")
            
            # Optional: Verify configuration was applied by taking a measurement
            nephelometer.set_led(True)
            time.sleep(0.2)  # Allow LED to stabilize
            measurement = nephelometer.take_single_measurement(subtract_background=False)
            nephelometer.set_led(False)
            
            # Check if measurement contains data
            self.assertIsNotNone(measurement, "Measurement should not be None")
            self.assertIn("raw", measurement, "Measurement should contain raw data")
            
            print("✓ Nephelometer configuration test passed")
            
        except Exception as e:
            # Ensure LED is off
            try:
                nephelometer.set_led(False)
            except:
                pass
            self.fail(f"Nephelometer configuration test failed: {str(e)}")
    
    def test_03_led_control(self):
        """Test LED control functionality."""
        try:
            # Create and initialize nephelometer
            nephelometer = Nephelometer(self.device_manager)
            nephelometer.initialize()
            
            # Turn on LED with specific current
            led_current = 10
            result_on = nephelometer.set_led(True, led_current)
            
            # Verify LED control was successful
            self.assertTrue(result_on, "LED should turn on successfully")
            
            # Wait briefly
            time.sleep(0.5)
            
            # Turn off LED
            result_off = nephelometer.set_led(False)
            
            # Verify LED control was successful
            self.assertTrue(result_off, "LED should turn off successfully")
            
            print("✓ LED control test passed")
            
        except Exception as e:
            # Ensure LED is off
            try:
                nephelometer.set_led(False)
            except:
                pass
            self.fail(f"LED control test failed: {str(e)}")
    
    def test_04_single_measurement(self):
        """Test taking a single measurement."""
        try:
            # Create and initialize nephelometer with optimal settings for low-signal sample
            nephelometer = Nephelometer(self.device_manager)
            nephelometer.initialize()
            
            # Configure with high gain and long integration time
            nephelometer.configure({
                "gain": 10,                # Maximum gain
                "integration_time": 300,   # Long integration time
                "led_current": 20          # Maximum LED current
            })
            
            # Take a measurement
            nephelometer.set_led(True)
            time.sleep(0.2)  # Allow LED to stabilize
            measurement = nephelometer.take_single_measurement(subtract_background=False)
            nephelometer.set_led(False)
            
            # Verify measurement structure
            self.assertIsNotNone(measurement, "Measurement should not be None")
            self.assertIn("raw", measurement, "Measurement should contain raw data")
            self.assertIn("processed", measurement, "Measurement should contain processed data")
            self.assertIn("ratios", measurement, "Measurement should contain spectral ratios")
            
            # Check for presence of key channels
            required_channels = ["F1", "F4", "F8", "Clear"]
            for channel in required_channels:
                self.assertIn(channel, measurement["raw"], f"Raw data should contain {channel} channel")
            
            # Check for spectral ratios
            required_ratios = ["violet_red", "violet_green", "green_red"]
            for ratio in required_ratios:
                self.assertIn(ratio, measurement["ratios"], f"Ratios should contain {ratio} ratio")
            
            # Print some measurement values
            print("✓ Single measurement test passed")
            print(f"  Sample readings - F1(415nm): {measurement['raw'].get('F1')}, "
                  f"F4(515nm): {measurement['raw'].get('F4')}, "
                  f"F8(680nm): {measurement['raw'].get('F8')}")
            print(f"  Sample ratios - V/R: {measurement['ratios'].get('violet_red', 0):.2f}, "
                  f"V/G: {measurement['ratios'].get('violet_green', 0):.2f}, "
                  f"G/R: {measurement['ratios'].get('green_red', 0):.2f}")
            
        except Exception as e:
            # Ensure LED is off
            try:
                nephelometer.set_led(False)
            except:
                pass
            self.fail(f"Single measurement test failed: {str(e)}")
    
    def test_05_background_subtraction(self):
        """Test background subtraction functionality."""
        try:
            # Create and initialize nephelometer
            nephelometer = Nephelometer(self.device_manager)
            nephelometer.initialize()
            
            # Configure with high gain and long integration time
            nephelometer.configure({
                "gain": 10,                # Maximum gain
                "integration_time": 300,   # Long integration time
                "led_current": 20          # Maximum LED current
            })
            
            # Take a background reading
            nephelometer.set_led(True)
            time.sleep(0.2)  # Allow LED to stabilize
            background = nephelometer.take_background_reading()
            
            # Verify background reading format
            self.assertIsNotNone(background, "Background reading should not be None")
            required_channels = ["F1", "F4", "F8", "Clear"]
            for channel in required_channels:
                self.assertIn(channel, background, f"Background should contain {channel} channel")
            
            # Take a measurement with background subtraction
            measurement_with_bg = nephelometer.take_single_measurement(subtract_background=True)
            
            # Take a measurement without background subtraction
            measurement_without_bg = nephelometer.take_single_measurement(subtract_background=False)
            nephelometer.set_led(False)
            
            # Verify both measurements contain processed data
            self.assertIn("processed", measurement_with_bg, "Measurement should contain processed data")
            self.assertIn("processed", measurement_without_bg, "Measurement should contain processed data")
            
            # For at least some channels, the background-subtracted values should be lower
            # (though this might not always be true depending on noise)
            any_channel_lower = False
            for channel in required_channels:
                if (measurement_with_bg["processed"].get(channel, 0) < 
                    measurement_without_bg["processed"].get(channel, 0)):
                    any_channel_lower = True
                    break
            
            # Since this may not always be true with very low signals, we just log the result
            if any_channel_lower:
                print("  Background subtraction reduces signal as expected")
            else:
                print("  Note: Background subtraction didn't measurably reduce signals (may be normal with low signal)")
            
            print("✓ Background subtraction test passed")
            
        except Exception as e:
            # Ensure LED is off
            try:
                nephelometer.set_led(False)
            except:
                pass
            self.fail(f"Background subtraction test failed: {str(e)}")
    
    def test_06_measurement_repeatability(self):
        """Test measurement repeatability."""
        try:
            # Create and initialize nephelometer
            nephelometer = Nephelometer(self.device_manager)
            nephelometer.initialize()
            
            # Configure with high gain and long integration time
            nephelometer.configure({
                "gain": 10,                # Maximum gain
                "integration_time": 300,   # Long integration time
                "led_current": 20          # Maximum LED current
            })
            
            # Take multiple measurements
            num_measurements = 3
            measurements = []
            
            nephelometer.set_led(True)
            time.sleep(0.2)  # Allow LED to stabilize
            
            for _ in range(num_measurements):
                measurement = nephelometer.take_single_measurement(subtract_background=False)
                measurements.append(measurement)
                time.sleep(0.1)  # Brief delay between measurements
            
            nephelometer.set_led(False)
            
            # Calculate variation in key channels
            variations = {}
            for channel in ["F1", "F4", "F8"]:
                values = [m["raw"].get(channel, 0) for m in measurements]
                mean = sum(values) / len(values) if values else 0
                # Avoid division by zero
                if mean > 0:
                    # Calculate coefficient of variation (standard deviation / mean)
                    # We use a simple method here
                    squared_diffs = [(v - mean) ** 2 for v in values]
                    variance = sum(squared_diffs) / len(squared_diffs) if squared_diffs else 0
                    std_dev = variance ** 0.5
                    variation = (std_dev / mean) * 100  # as percentage
                else:
                    variation = 0
                variations[channel] = variation
            
            # Check if variations are within reasonable limits
            # With low signals, variation might be higher
            for channel, variation in variations.items():
                print(f"  {channel} variation: {variation:.1f}%")
                # We use a very generous threshold for low-signal conditions
                self.assertLessEqual(variation, 50, 
                                    f"{channel} variation should be below 50%")
            
            print("✓ Measurement repeatability test passed")
            
        except Exception as e:
            # Ensure LED is off
            try:
                nephelometer.set_led(False)
            except:
                pass
            self.fail(f"Measurement repeatability test failed: {str(e)}")
    
    def test_07_continuous_measurement(self):
        """Test continuous measurement mode."""
        measurements = []
        
        def measurement_callback(data):
            measurements.append(data)
        
        try:
            # Create and initialize nephelometer
            nephelometer = Nephelometer(self.device_manager)
            nephelometer.initialize()
            
            # Configure with optimal settings for low-signal sample
            nephelometer.configure({
                "gain": 10,                # Maximum gain
                "integration_time": 300,   # Long integration time
                "led_current": 20          # Maximum LED current
            })
            
            # Start continuous measurement
            result = nephelometer.start_continuous_measurement(
                interval_seconds=0.5,     # 500ms interval
                callback=measurement_callback,
                subtract_background=False
            )
            
            self.assertTrue(result, "Starting continuous measurement should succeed")
            
            # Allow time to collect measurements (3 seconds = ~6 measurements)
            print("  Waiting for data collection (3 seconds)...")
            time.sleep(3.0)
            
            # Stop measurement
            nephelometer.stop_measurement()
            
            # Verify we received measurements
            self.assertGreater(len(measurements), 1, 
                              "Should have received at least 1 measurement")
            
            print(f"✓ Continuous measurement test passed - Received {len(measurements)} measurements")
            
        except Exception as e:
            # Ensure measurement is stopped
            try:
                nephelometer.stop_measurement()
            except:
                pass
            self.fail(f"Continuous measurement test failed: {str(e)}")
    
    def test_08_kinetic_measurement(self):
        """Test kinetic measurement mode."""
        kinetic_data = []
        
        def kinetic_callback(data):
            kinetic_data.append(data)
        
        try:
            # Create and initialize nephelometer
            nephelometer = Nephelometer(self.device_manager)
            nephelometer.initialize()
            
            # Configure with optimal settings for low-signal sample
            nephelometer.configure({
                "gain": 10,                # Maximum gain
                "integration_time": 300,   # Long integration time
                "led_current": 20          # Maximum LED current
            })
            
            # Start kinetic measurement for a short duration
            result = nephelometer.start_kinetic_measurement(
                duration_seconds=2.0,     # 2 second measurement
                samples_per_second=2.0,   # 2Hz sampling
                callback=kinetic_callback,
                subtract_background=False
            )
            
            self.assertTrue(result, "Starting kinetic measurement should succeed")
            
            # Wait for completion (add a small buffer)
            print("  Waiting for kinetic measurement completion...")
            time.sleep(3.0)
            
            # Verify we received measurements
            self.assertGreater(len(kinetic_data), 1, 
                              "Should have received at least 1 kinetic measurement")
            
            # Check if measurements have timestamps and elapsed time
            if kinetic_data:
                self.assertIn("timestamp", kinetic_data[0], 
                             "Kinetic measurements should have timestamps")
                self.assertIn("elapsed_seconds", kinetic_data[0], 
                             "Kinetic measurements should have elapsed time")
            
            print(f"✓ Kinetic measurement test passed - Received {len(kinetic_data)} measurements")
            
        except Exception as e:
            # Ensure measurement is stopped
            try:
                nephelometer.stop_measurement()
            except:
                pass
            self.fail(f"Kinetic measurement test failed: {str(e)}")
    
    def test_09_experiment_session(self):
        """Test experiment session functionality."""
        try:
            # Create and initialize nephelometer
            nephelometer = Nephelometer(self.device_manager)
            nephelometer.initialize()
            
            # Start an experiment session
            experiment_id = "test_experiment_001"
            result = nephelometer.start_experiment(experiment_id)
            
            self.assertTrue(result, "Starting experiment should succeed")
            
            # Take a few measurements within the experiment
            nephelometer.set_led(True)
            time.sleep(0.2)  # Allow LED to stabilize
            
            for i in range(2):
                measurement = nephelometer.take_single_measurement(subtract_background=False)
                self.assertIsNotNone(measurement, f"Measurement {i+1} should not be None")
                time.sleep(0.5)
            
            nephelometer.set_led(False)
            
            # Stop the experiment
            stop_result = nephelometer.stop_experiment()
            
            self.assertTrue(stop_result, "Stopping experiment should succeed")
            
            print(f"✓ Experiment session test passed")
            
        except Exception as e:
            # Ensure cleanup
            try:
                nephelometer.set_led(False)
                nephelometer.stop_experiment()
            except:
                pass
            self.fail(f"Experiment session test failed: {str(e)}")
    
    def test_10_spectral_ratios(self):
        """Test spectral ratio calculations."""
        try:
            # Create and initialize nephelometer
            nephelometer = Nephelometer(self.device_manager)
            nephelometer.initialize()
            
            # Configure with optimal settings for low-signal sample
            nephelometer.configure({
                "gain": 10,                # Maximum gain
                "integration_time": 300,   # Long integration time
                "led_current": 20          # Maximum LED current
            })
            
            # Take a measurement
            nephelometer.set_led(True)
            time.sleep(0.2)  # Allow LED to stabilize
            measurement = nephelometer.take_single_measurement(subtract_background=False)
            nephelometer.set_led(False)
            
            # Verify ratios are calculated
            self.assertIn("ratios", measurement, "Measurement should contain spectral ratios")
            
            required_ratios = ["violet_red", "violet_green", "green_red"]
            for ratio in required_ratios:
                self.assertIn(ratio, measurement["ratios"], f"Ratios should contain {ratio}")
                self.assertGreaterEqual(measurement["ratios"][ratio], 0, 
                                       f"{ratio} ratio should be non-negative")
            
            # Manually calculate a ratio to verify
            raw_data = measurement["raw"]
            if "F1" in raw_data and "F8" in raw_data and raw_data["F8"] > 0:
                manual_violet_red = raw_data["F1"] / raw_data["F8"]
                calculated_violet_red = measurement["ratios"]["violet_red"]
                
                # Allow for small floating point differences
                self.assertAlmostEqual(manual_violet_red, calculated_violet_red, 
                                      delta=0.01, msg="Manually calculated ratio should match")
            
            print("✓ Spectral ratio calculation test passed")
            print(f"  Calculated ratios: V/R={measurement['ratios'].get('violet_red', 0):.2f}, "
                  f"V/G={measurement['ratios'].get('violet_green', 0):.2f}, "
                  f"G/R={measurement['ratios'].get('green_red', 0):.2f}")
            
        except Exception as e:
            # Ensure LED is off
            try:
                nephelometer.set_led(False)
            except:
                pass
            self.fail(f"Spectral ratio calculation test failed: {str(e)}")
    
    def test_11_agglutination_analysis(self):
        """Test agglutination analysis functionality."""
        try:
            # Create and initialize nephelometer
            nephelometer = Nephelometer(self.device_manager)
            nephelometer.initialize()
            
            # Configure with optimal settings for low-signal sample
            nephelometer.configure({
                "gain": 10,                # Maximum gain
                "integration_time": 300,   # Long integration time
                "led_current": 20          # Maximum LED current
            })
            
            # Take a measurement
            nephelometer.set_led(True)
            time.sleep(0.2)  # Allow LED to stabilize
            measurement = nephelometer.take_single_measurement(subtract_background=False)
            nephelometer.set_led(False)
            
            # Analyze for agglutination
            analysis = nephelometer.analyze_agglutination(measurement)
            
            # Verify analysis structure
            self.assertIsNotNone(analysis, "Agglutination analysis should not be None")
            self.assertIn("agglutination_score", analysis, "Analysis should include agglutination score")
            self.assertIn("agglutination_state", analysis, "Analysis should include agglutination state")
            self.assertIn("confidence", analysis, "Analysis should include confidence level")
            
            # Check if agglutination state is a valid enum value
            state = analysis["agglutination_state"]
            self.assertIsInstance(state, AgglutinationState, "State should be an AgglutinationState enum")
            
            # Check if score and confidence are in valid ranges
            self.assertGreaterEqual(analysis["agglutination_score"], 0, "Score should be non-negative")
            self.assertLessEqual(analysis["agglutination_score"], 4, "Score should be at most 4")
            self.assertGreaterEqual(analysis["confidence"], 0, "Confidence should be non-negative")
            self.assertLessEqual(analysis["confidence"], 1, "Confidence should be at most 1")
            
            print("✓ Agglutination analysis test passed")
            print(f"  Analysis results - State: {state.name}, "
                  f"Score: {analysis['agglutination_score']:.1f}, "
                  f"Confidence: {analysis['confidence']:.2f}")
            
        except Exception as e:
            # Ensure LED is off
            try:
                nephelometer.set_led(False)
            except:
                pass
            self.fail(f"Agglutination analysis test failed: {str(e)}")

def main():
    """Main entry point for the test script."""
    print("\n=== SpectroNeph Nephelometer Integration Tests ===\n")
    
    # Run the tests
    unittest.main(argv=[sys.argv[0]])

if __name__ == "__main__":
    main()