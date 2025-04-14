#!/usr/bin/env python3
"""
Simplified Firmware Test for SpectroNeph

This script provides a simplified test for the firmware communication,
with improved port detection and error handling.

Usage:
    python simplified_firmware_test.py [--port COM_PORT]
"""

import sys
import time
import json
import argparse
from pathlib import Path

# Add the necessary paths to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
app_dir = project_root / "app"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(app_dir))

# Import core functionality - use direct imports
from core.device import DeviceManager
from core.exceptions import DeviceDisconnectedError

def detect_serial_port():
    """Try to detect the device port automatically."""
    try:
        from serial.tools import list_ports
        
        # Common vendor/product IDs for ESP32
        ESP32_VIDS_PIDS = [
            (0x10C4, 0xEA60),  # Silicon Labs CP210x
            (0x1A86, 0x7523),  # QinHeng CH340
            (0x0403, 0x6001),  # FTDI FT232
        ]
        
        print("Scanning for serial ports...")
        ports = list(list_ports.comports())
        if not ports:
            print("No serial ports detected.")
            return None
            
        print(f"Found {len(ports)} ports:")
        for port in ports:
            print(f"  {port.device} - {port.description}")
            # Check if the device has ESP32 VID/PID
            if hasattr(port, 'vid') and hasattr(port, 'pid') and port.vid is not None:
                if (port.vid, port.pid) in ESP32_VIDS_PIDS:
                    print(f"    ✓ Detected ESP32 device: {port.device}")
                    return port.device
            
            # Check description for common ESP32 indicators
            if hasattr(port, 'description'):
                if any(x in port.description.lower() for x in ['esp32', 'cp210x', 'ch340', 'ft232']):
                    print(f"    ✓ Detected ESP32 device: {port.device}")
                    return port.device
        
        # If no definite match, suggest using the first port
        if ports:
            print(f"  ⚠️ No ESP32 device detected, suggesting first port: {ports[0].device}")
            return ports[0].device
            
        return None
    except Exception as e:
        print(f"Error detecting serial ports: {str(e)}")
        return None

def run_firmware_tests(port=None):
    """Run a series of firmware communication tests."""
    print("\n" + "="*60)
    print("SpectroNeph Firmware Communication Tests")
    print("="*60)
    
    # Detect port if not provided
    if port is None:
        port = detect_serial_port()
        if port is None:
            print("❌ No serial port detected. Please specify a port with --port COMx")
            return False
    
    print(f"Using port: {port}")
    
    # Initialize device manager
    device = DeviceManager()
    
    # Connect to the device
    print("\nTest 1: Device Connection")
    print("-" * 30)
    try:
        if device.connect(port=port):
            print("✓ Connected to device")
        else:
            print("❌ Failed to connect to device")
            return False
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        return False
    
    try:
        # Test 2: Get device info
        print("\nTest 2: Device Information")
        print("-" * 30)
        info = device.send_command("get_info")
        if info.get("status") == 0:
            print("✓ Device info received:")
            data = info.get("data", {})
            print(f"  Device: {data.get('name', 'Unknown')}")
            print(f"  Firmware: {data.get('version', 'Unknown')}")
            print(f"  Uptime: {data.get('uptime', 0)}ms")
        else:
            print("❌ Failed to get device info")
        
        # Test 3: Ping command
        print("\nTest 3: Ping Command")
        print("-" * 30)
        ping_response = device.send_command("ping")
        if ping_response.get("status") == 0 and ping_response.get("data", {}).get("pong", False):
            print("✓ Ping command successful")
        else:
            print("❌ Ping command failed")
        
        # Test 4: AS7341 sensor initialization
        print("\nTest 4: AS7341 Sensor Initialization")
        print("-" * 30)
        init_result = device.send_command("as7341_init")
        if init_result.get("status") == 0 and init_result.get("data", {}).get("initialized", False):
            print("✓ AS7341 sensor initialized successfully")
            sensor_available = True
        else:
            print("⚠️ AS7341 initialization failed or sensor not available")
            sensor_available = False
        
        # Only run sensor tests if sensor is available
        if sensor_available:
            # Test 5: AS7341 configuration
            print("\nTest 5: AS7341 Configuration")
            print("-" * 30)
            config_result = device.send_command("as7341_config", {
                "gain": 5,  # 16x gain
                "integration_time": 100,  # 100ms
                "led_current": 10  # 10mA
            })
            if config_result.get("status") == 0:
                print("✓ AS7341 configuration successful")
            else:
                print("❌ AS7341 configuration failed")
            
            # Test 6: LED control
            print("\nTest 6: LED Control")
            print("-" * 30)
            led_result = device.send_command("as7341_led", {
                "enabled": True,
                "current": 5  # 5mA for testing
            })
            if led_result.get("status") == 0:
                print("✓ LED turned on")
                # Wait for LED to stabilize
                time.sleep(0.5)
                
                # Test 7: Sensor reading
                print("\nTest 7: Sensor Reading")
                print("-" * 30)
                read_result = device.send_command("as7341_read")
                if read_result.get("status") == 0:
                    print("✓ Sensor reading successful")
                    data = read_result.get("data", {})
                    
                    # Print a few key channel values
                    print("  Channel readings:")
                    channels = ["F1", "F4", "F8", "Clear"]
                    for channel in channels:
                        if channel in data:
                            print(f"    {channel}: {data[channel]}")
                else:
                    print("❌ Sensor reading failed")
                
                # Turn off LED
                device.send_command("as7341_led", {"enabled": False})
                print("✓ LED turned off")
            else:
                print("❌ LED control failed")
                
            # Test 8: Data streaming (brief test)
            print("\nTest 8: Data Streaming")
            print("-" * 30)
            
            # Set up data callback
            received_data = []
            
            def data_callback(data_message):
                received_data.append(data_message)
                count = len(received_data)
                if count <= 3:  # Limit output to first 3 packets
                    print(f"  Received data packet {count}")
            
            # Register callback and start stream
            callback_id = device.register_data_callback("as7341", data_callback)
            
            stream_start = device.start_data_stream("as7341", {
                "interval_ms": 500  # 500ms interval
            })
            
            if stream_start:
                print("✓ Data stream started")
                
                # Wait for a few data packets
                print("  Waiting for data (3 seconds)...")
                time.sleep(3)
                
                # Stop stream
                stream_stop = device.stop_data_stream("as7341")
                if stream_stop:
                    print("✓ Data stream stopped")
                else:
                    print("❌ Failed to stop data stream")
                
                # Check results
                if received_data:
                    print(f"✓ Received {len(received_data)} data packets")
                else:
                    print("❌ No data packets received")
                
                # Unregister callback
                device.unregister_data_callback(callback_id)
            else:
                print("❌ Failed to start data stream")
        
        # Always disconnect
        device.disconnect()
        print("\n✅ Tests completed. Device disconnected.")
        return True
        
    except DeviceDisconnectedError:
        print("❌ Device disconnected during tests")
        return False
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        try:
            device.disconnect()
        except:
            pass
        return False

def main():
    """Main entry point."""
    # Parse command line args
    parser = argparse.ArgumentParser(description='SpectroNeph Firmware Communication Tests')
    parser.add_argument('--port', help='Serial port to connect to')
    args = parser.parse_args()
    
    # Run tests
    success = run_firmware_tests(args.port)
    
    # Return appropriate exit code
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())