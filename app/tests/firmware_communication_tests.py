#!/usr/bin/env python3
"""
Firmware Communication Tests for SpectroNeph

This module contains tests to verify communication between the Python application
and the ESP32 firmware. It tests basic connectivity, command/response protocol,
sensor functionality, and data streaming.

Usage:
    python firmware_communication_tests.py [--port COM_PORT]
"""

import sys
import time
import json
import argparse
import unittest
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Project root added to sys.path: {project_root}")

# Import modules from the SpectroNeph project - use direct imports
sys.path.insert(0, str(project_root / "app"))
from core.device import DeviceManager
from core.communication import SerialCommunication
from core.exceptions import (
    CommunicationError,
    CommandTimeoutError,
    InvalidResponseError,
    DeviceDisconnectedError,
)
from config.settings import Settings
from utils.logging import setup_logging, get_logger
from utils.helpers import detect_serial_port

# Initialize logger
setup_logging()
logger = get_logger("tests")

class FirmwareCommunicationTests(unittest.TestCase):
    """Test suite for firmware communication."""

    @classmethod
    def setUpClass(cls):
        """Set up test class - connect to device once for all tests."""
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description="SpectroNeph Firmware Communication Tests"
        )
        parser.add_argument("--port", help="Serial port to connect to")
        args = parser.parse_args()

        # Initialize device manager
        cls.device = DeviceManager()

        # Connect to device - use provided port or auto-detect
        port = args.port if args.port else detect_serial_port()
        if not cls.device.connect(port=port):
            print("❌ FAILED: Could not connect to device")
            print("   Make sure the device is connected and the port is correct")
            sys.exit(1)

        print(f"✓ Connected to device on {cls.device._comm._port}")

        # Collect device info for reference
        cls.device_info = cls.device.get_device_info()
        print(
            f"✓ Device info: {cls.device_info.get('name', 'Unknown')}, "
            f"Version: {cls.device_info.get('version', 'Unknown')}"
        )

        # Sleep briefly to ensure connection is stable
        time.sleep(0.5)

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

    def test_01_ping_command(self):
        """Test basic ping command."""
        try:
            response = self.device.send_command("ping")

            self.assertEqual(response.get("status"), 0, "Ping command should succeed")
            self.assertTrue(
                response.get("data", {}).get("pong", False),
                "Ping response should contain 'pong': true",
            )

            print("✓ Ping command test passed")
        except Exception as e:
            self.fail(f"Ping command test failed: {str(e)}")

    def test_02_get_info(self):
        """Test get_info command."""
        try:
            response = self.device.send_command("get_info")

            self.assertEqual(
                response.get("status"), 0, "get_info command should succeed"
            )

            # Check if response contains expected fields
            data = response.get("data", {})
            self.assertIn("name", data, "Device info should include 'name'")
            self.assertIn("version", data, "Device info should include 'version'")
            self.assertIn("uptime", data, "Device info should include 'uptime'")

            print(
                f"✓ get_info command test passed - Device: {data.get('name')}, "
                f"Version: {data.get('version')}"
            )
        except Exception as e:
            self.fail(f"get_info command test failed: {str(e)}")

    def test_03_as7341_init(self):
        """Test AS7341 initialization."""
        try:
            response = self.device.send_command("as7341_init")

            self.assertEqual(
                response.get("status"), 0, "as7341_init command should succeed"
            )
            self.assertTrue(
                response.get("data", {}).get("initialized", False),
                "AS7341 should be successfully initialized",
            )

            print("✓ AS7341 initialization test passed")
        except Exception as e:
            self.fail(f"AS7341 initialization test failed: {str(e)}")

    def test_04_as7341_config(self):
        """Test AS7341 configuration."""
        try:
            # Test with default values
            response = self.device.send_command(
                "as7341_config",
                {
                    "gain": 5,  # 16x gain
                    "integration_time": 100,  # 100ms
                    "led_current": 10,  # 10mA
                },
            )

            self.assertEqual(
                response.get("status"), 0, "as7341_config command should succeed"
            )

            # Check returned configuration
            data = response.get("data", {})
            self.assertEqual(data.get("gain"), 5, "Gain should be set to 5 (16x)")
            self.assertEqual(
                data.get("integration_time"), 100, "Integration time should be 100ms"
            )
            self.assertEqual(data.get("led_current"), 10, "LED current should be 10mA")

            print("✓ AS7341 configuration test passed")
        except Exception as e:
            self.fail(f"AS7341 configuration test failed: {str(e)}")

    def test_05_as7341_led(self):
        """Test AS7341 LED control."""
        try:
            # Turn on LED
            response_on = self.device.send_command(
                "as7341_led", {"enabled": True, "current": 5}  # 5mA for testing
            )

            self.assertEqual(
                response_on.get("status"), 0, "LED enable command should succeed"
            )
            self.assertTrue(
                response_on.get("data", {}).get("enabled", False),
                "LED should be enabled",
            )

            # Wait briefly
            time.sleep(0.5)

            # Turn off LED
            response_off = self.device.send_command("as7341_led", {"enabled": False})

            self.assertEqual(
                response_off.get("status"), 0, "LED disable command should succeed"
            )
            self.assertFalse(
                response_off.get("data", {}).get("enabled", True),
                "LED should be disabled",
            )

            print("✓ AS7341 LED control test passed")
        except Exception as e:
            self.fail(f"AS7341 LED control test failed: {str(e)}")

    def test_06_as7341_read(self):
        """Test reading sensor data from AS7341."""
        try:
            # Configure sensor with reasonable settings
            self.device.send_command(
                "as7341_config",
                {
                    "gain": 6,  # 32x gain for better signal
                    "integration_time": 100,  # 100ms
                },
            )

            # Turn on LED for measurement
            self.device.send_command(
                "as7341_led", {"enabled": True, "current": 10}  # 10mA
            )

            # Wait for LED to stabilize
            time.sleep(0.2)

            # Read sensor data
            response = self.device.send_command("as7341_read")

            # Turn off LED
            self.device.send_command("as7341_led", {"enabled": False})

            self.assertEqual(
                response.get("status"), 0, "as7341_read command should succeed"
            )

            # Check if response contains expected channels
            data = response.get("data", {})
            expected_channels = [
                "F1",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F7",
                "F8",
                "Clear",
                "NIR",
            ]

            for channel in expected_channels:
                self.assertIn(
                    channel, data, f"Response should include {channel} channel"
                )
                # Check if readings are in a reasonable range
                self.assertGreaterEqual(
                    data.get(channel, 0), 0, f"{channel} reading should be >= 0"
                )
                self.assertLessEqual(
                    data.get(channel, 0), 65535, f"{channel} reading should be <= 65535"
                )

            print("✓ AS7341 sensor reading test passed")
            print(
                f"  Sample readings - F1(415nm): {data.get('F1')}, F4(515nm): {data.get('F4')}, "
                f"F8(680nm): {data.get('F8')}"
            )
        except Exception as e:
            self.fail(f"AS7341 sensor reading test failed: {str(e)}")

    def test_07_stream_control(self):
        """Test streaming functionality."""
        received_data = []

        # Callback function for data
        def data_callback(data_message):
            received_data.append(data_message)
            print(f"  Received data packet: type={data_message.get('type')}")

        try:
            # Register data callback
            callback_id = self.device.register_data_callback("as7341", data_callback)

            # Start data stream
            response = self.device.start_data_stream(
                "as7341", {"interval_ms": 500}  # Sample every 500ms
            )

            self.assertTrue(response, "Stream start should return True")

            # Wait for some data (3 seconds = ~6 samples)
            print("  Waiting for data stream (3 seconds)...")
            time.sleep(3)

            # Stop the stream
            stop_result = self.device.stop_data_stream("as7341")
            self.assertTrue(stop_result, "Stream stop should return True")

            # Unregister callback
            self.device.unregister_data_callback(callback_id)

            # Check if we received data
            self.assertGreater(
                len(received_data), 0, "Should have received at least one data packet"
            )

            # Check content of at least one packet
            if received_data:
                packet = received_data[0]
                self.assertEqual(
                    packet.get("type"), "as7341", "Data packet should have correct type"
                )
                self.assertIn("data", packet, "Data packet should contain data object")

                # Check expected channels in first packet
                data = packet.get("data", {})
                for channel in [
                    "F1",
                    "F2",
                    "F3",
                    "F4",
                    "F5",
                    "F6",
                    "F7",
                    "F8",
                    "Clear",
                    "NIR",
                ]:
                    self.assertIn(
                        channel, data, f"Data should include {channel} channel"
                    )

            print(
                f"✓ Streaming test passed - Received {len(received_data)} data packets"
            )
        except Exception as e:
            self.fail(f"Streaming test failed: {str(e)}")
        finally:
            # Ensure stream is stopped
            try:
                self.device.stop_data_stream("as7341")
            except:
                pass

    def test_08_command_timeout(self):
        """Test command timeout handling."""
        try:
            # Create a non-existent command that will cause a timeout
            with self.assertRaises((CommandTimeoutError, InvalidResponseError)):
                self.device.send_command("nonexistent_command", {}, timeout=1.0)

            print("✓ Command timeout test passed")
        except Exception as e:
            self.fail(f"Command timeout test failed: {str(e)}")

    def test_09_advanced_commands(self):
        """Test advanced device commands."""
        try:
            # Get active streams (should be none at this point)
            response = self.device.send_command("get_streams")

            self.assertEqual(
                response.get("status"), 0, "get_streams command should succeed"
            )
            self.assertEqual(
                response.get("data", {}).get("count", -1),
                0,
                "No streams should be active",
            )

            # Try getting device diagnostics
            diag_response = self.device.send_command("diagnostics")
            print(f"Diagnostics response: {diag_response}")

            self.assertEqual(
                diag_response.get("status"), 0, "diagnostics command should succeed"
            )

            # Check diagnostics data
            diag_data = diag_response.get("data", {})
            self.assertIn(
                "system", diag_data, "Diagnostics should include system information"
            )
            self.assertIn(
                "sensor", diag_data, "Diagnostics should include sensor information"
            )
            self.assertIn(
                "result", diag_data, "Diagnostics should include overall result"
            )

            print("✓ Advanced commands test passed")
        except Exception as e:
            self.fail(f"Advanced commands test failed: {str(e)}")

    def test_10_as7341_read(self):
        """Test reading diferential sensor data from AS7341."""
        try:
            # Configure sensor with reasonable settings
            self.device.send_command(
                "as7341_config",
                {
                    "gain": 6,  # 32x gain for better signal
                    "integration_time": 100,  # 100ms
                },
            )

            # Set LED current for measurement
            self.device.send_command(
                "as7341_led", {"enabled": False, "current": 10}  # 10mA
            )

            # Read sensor data
            response = self.device.send_command("as7341_differential_read")

            self.assertEqual(
                response.get("status"), 0, "as7341_differential_read command should succeed"
            )

            # Check if response contains expected channels
            data = response.get("data", {})
            expected_channels = [
                "F1",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F7",
                "F8",
                "Clear",
                "NIR",
            ]

            for channel in expected_channels:
                self.assertIn(
                    channel, data, f"Response should include {channel} channel"
                )
                # Check if readings are in a reasonable range
                self.assertGreaterEqual(
                    data.get(channel, 0), 0, f"{channel} reading should be >= 0"
                )
                self.assertLessEqual(
                    data.get(channel, 0), 65535, f"{channel} reading should be <= 65535"
                )

            print("✓ AS7341 differential sensor reading test passed")
            print(
                f"  Sample readings - F1(415nm): {data.get('F1')}, F4(515nm): {data.get('F4')}, "
                f"F8(680nm): {data.get('F8')}"
            )
        except Exception as e:
            self.fail(f"AS7341 sensor reading test failed: {str(e)}")



def main():
    """Main entry point for the test script."""
    print("\n=== SpectroNeph Firmware Communication Tests ===\n")

    # Run the tests
    unittest.main(argv=[sys.argv[0]])


if __name__ == "__main__":
    main()
