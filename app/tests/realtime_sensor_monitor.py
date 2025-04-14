#!/usr/bin/env python3
"""
Real-Time AS7341 Sensor Monitor

This script provides continuous readings from the AS7341 sensor to help with
optical alignment and focus adjustment. It displays live spectral readings and
key metrics that can be used to optimize the optical setup.

To use:
1. Configure the parameters below
2. Run the script in VS Code
3. Watch the terminal for real-time readings
4. Press Ctrl+C to stop
"""

import sys
import time
import threading
import os
from pathlib import Path

# ============ CONFIGURATION PARAMETERS (MODIFY THESE) ============
# Serial port (set to None for auto-detection)
PORT = None  # Example: "COM5" on Windows or "/dev/ttyUSB0" on Linux

# Sensor settings
GAIN = 6  # 0=0.5x, 1=1x, 2=2x, 3=4x, 4=8x, 5=16x, 6=32x, 7=64x, 8=128x, 9=256x, 10=512x
INTEGRATION_TIME = 100  # milliseconds (1-1000)
LED_CURRENT = 10  # milliamps (0-20)
LED_ENABLED = True  # True to turn on LED, False to keep it off

# Update interval
UPDATE_INTERVAL = 0.2  # seconds between readings

# Display options
SHOW_BARGRAPH = True  # Set to False for simpler display
# ================================================================

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Project root added to sys.path: {project_root}")

# Import modules from the SpectroNeph project directly
sys.path.insert(0, str(project_root / "app"))
from core.device import DeviceManager
from utils.logging import setup_logging, get_logger
from utils.helpers import detect_serial_port

# Initialize logger
setup_logging()
logger = get_logger("realtime_monitor")

class RealTimeSensorMonitor:
    """Real-time monitor for AS7341 sensor readings."""
    
    def __init__(self, port=None, gain=6, integration_time=100, led_current=10, led_enabled=True, interval=0.2):
        """Initialize the monitor."""
        self.interval = interval
        self.running = False
        self.thread = None
        self.led_on = led_enabled
        self.device = DeviceManager()

        port = port if port else detect_serial_port()
        
        # Connect to the device
        if not self.device.connect(port):
            print("❌ FAILED: Could not connect to device")
            print("   Make sure the device is connected and the port is correct")
            sys.exit(1)
            
        print(f"✓ Connected to device on {self.device._comm._port}")
        
        # Initialize sensor
        response = self.device.send_command("as7341_init")
        if response.get("status") != 0 or not response.get("data", {}).get("initialized", False):
            print("❌ FAILED: Could not initialize AS7341 sensor")
            self.device.disconnect()
            sys.exit(1)
            
        print("✓ AS7341 sensor initialized")
        
        # Configure sensor with specified settings
        self.configure_sensor(gain, integration_time, led_current)
        
        # Turn LED on/off based on setting
        self.device.send_command("as7341_led", {
            "enabled": self.led_on,
            "current": self.current_led_current
        })
        print(f"✓ LED {'ON' if self.led_on else 'OFF'}")
        
    def configure_sensor(self, gain, integration_time, led_current):
        """Configure the sensor."""
        self.device.send_command("as7341_config", {
            "gain": gain,
            "integration_time": integration_time,
            "led_current": led_current
        })
        
        # Store current settings
        self.current_gain = gain
        self.current_integration_time = integration_time
        self.current_led_current = led_current
        
        print(f"✓ Sensor configured: Gain={2**(gain-1)}x, Integration={integration_time}ms, LED={led_current}mA")
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop)
        self.thread.daemon = True
        self.thread.start()
        
        print("Starting real-time monitoring. Press Ctrl+C to stop.")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        # Turn off LED
        if self.led_on:
            self.device.send_command("as7341_led", {"enabled": False})
            self.led_on = False
    
    def _monitoring_loop(self):
        """Monitoring thread function."""
        last_readings = None
        
        while self.running:
            try:
                # Read sensor data
                response = self.device.send_command("as7341_read")
                
                if response.get("status") == 0:
                    readings = response.get("data", {})
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(readings)
                    
                    # Calculate changes if we have previous readings
                    changes = None
                    if last_readings:
                        changes = self._calculate_changes(readings, last_readings)
                    
                    # Display readings
                    self._display_readings(readings, metrics, changes)
                    
                    # Store for next comparison
                    last_readings = readings
                
                # Wait for the next reading
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {str(e)}")
                time.sleep(1.0)
    
    def _calculate_metrics(self, readings):
        """Calculate useful metrics from the readings."""
        # Get key channel values
        f1 = readings.get("F1", 0)  # 415nm (Violet)
        f4 = max(1, readings.get("F4", 1))  # 515nm (Green)
        f8 = max(1, readings.get("F8", 1))  # 680nm (Red)
        clear = max(1, readings.get("Clear", 1))  # Clear channel
        
        # Calculate ratios
        violet_red_ratio = f1 / f8
        violet_green_ratio = f1 / f4
        green_red_ratio = f4 / f8
        
        # Calculate intensity metrics
        total_intensity = sum(readings.get(ch, 0) for ch in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"])
        max_channel = max(readings.items(), key=lambda x: x[1] if x[0] in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"] else 0)
        
        return {
            "violet_red_ratio": violet_red_ratio,
            "violet_green_ratio": violet_green_ratio,
            "green_red_ratio": green_red_ratio,
            "total_intensity": total_intensity,
            "max_channel": max_channel
        }
    
    def _calculate_changes(self, current, previous):
        """Calculate changes between readings."""
        # Track changes in key metrics
        changes = {}
        
        # Calculate total intensity change
        total_current = sum(current.get(ch, 0) for ch in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"])
        total_previous = sum(previous.get(ch, 0) for ch in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"])
        
        if total_previous > 0:
            percent_change = (total_current - total_previous) / total_previous * 100
            changes["total_intensity"] = percent_change
        else:
            changes["total_intensity"] = 0
            
        return changes
    
    def _display_readings(self, readings, metrics, changes):
        """Display readings and metrics in the console."""
        global SHOW_BARGRAPH
        
        # Clear the screen (cross-platform)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Display current settings
        print(f"Settings: Gain={2**(self.current_gain-1)}x, Integration={self.current_integration_time}ms, LED={'ON' if self.led_on else 'OFF'} ({self.current_led_current}mA)")
        print("-" * 80)
        
        # Display key metrics for optical alignment
        print(f"ALIGNMENT METRICS:")
        print(f"Total Signal: {metrics['total_intensity']:5d} counts", end="")
        
        # Show change indicator if available
        if changes and "total_intensity" in changes:
            change = changes["total_intensity"]
            if abs(change) > 0.5:  # Only show if change is significant
                if change > 0:
                    print(f" ↑ {change:+.1f}%", end="")
                else:
                    print(f" ↓ {change:+.1f}%", end="")
        print()
        
        print(f"Strongest Channel: {metrics['max_channel'][0]} ({metrics['max_channel'][1]} counts)")
        print(f"Signal Ratios: V/R={metrics['violet_red_ratio']:.2f}, V/G={metrics['violet_green_ratio']:.2f}, G/R={metrics['green_red_ratio']:.2f}")
        print("-" * 80)
        
        # Display channel readings 
        channels = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "Clear", "NIR"]
        wavelengths = ["415nm", "445nm", "480nm", "515nm", "555nm", "590nm", "630nm", "680nm", "Clear", "NIR"]
        
        print(f"CHANNEL READINGS:")
        
        if SHOW_BARGRAPH:
            max_val = max(readings.get(ch, 0) for ch in channels)
            scale_factor = 50.0 / max(1, max_val)  # Scale to max 50 characters
            
            for i, ch in enumerate(channels):
                value = readings.get(ch, 0)
                bar_len = int(value * scale_factor)
                bar = "█" * bar_len
                print(f"{ch:5s} ({wavelengths[i]:5s}): {value:5d} {bar}")
        else:
            # Simple table format
            headers = ["Channel", "Value", "Channel", "Value", "Channel", "Value"]
            print(f"{headers[0]:7s} {headers[1]:7s}   {headers[2]:7s} {headers[3]:7s}   {headers[4]:7s} {headers[5]:7s}")
            print("-" * 65)
            
            # Display in 3 columns
            for i in range(0, len(channels), 3):
                if i+2 < len(channels):
                    ch1, ch2, ch3 = channels[i], channels[i+1], channels[i+2]
                    val1 = readings.get(ch1, 0)
                    val2 = readings.get(ch2, 0)
                    val3 = readings.get(ch3, 0)
                    print(f"{ch1:7s} {val1:7d}   {ch2:7s} {val2:7d}   {ch3:7s} {val3:7d}")
                elif i+1 < len(channels):
                    ch1, ch2 = channels[i], channels[i+1]
                    val1 = readings.get(ch1, 0)
                    val2 = readings.get(ch2, 0)
                    print(f"{ch1:7s} {val1:7d}   {ch2:7s} {val2:7d}")
                else:
                    ch1 = channels[i]
                    val1 = readings.get(ch1, 0)
                    print(f"{ch1:7s} {val1:7d}")
        
        print("-" * 80)
        print("Press Ctrl+C to stop monitoring")

def main():
    """Main entry point for the script."""
    global PORT, GAIN, INTEGRATION_TIME, LED_CURRENT, LED_ENABLED, UPDATE_INTERVAL, SHOW_BARGRAPH
    
    try:
        # Create and start the monitor with the configured parameters
        monitor = RealTimeSensorMonitor(
            port=PORT,
            gain=GAIN,
            integration_time=INTEGRATION_TIME,
            led_current=LED_CURRENT,
            led_enabled=LED_ENABLED,
            interval=UPDATE_INTERVAL
        )
        
        monitor.start_monitoring()
        
        # Keep the main thread alive
        while True:
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
        if 'monitor' in locals():
            monitor.stop_monitoring()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        
    finally:
        # Clean up
        if 'monitor' in locals():
            monitor.stop_monitoring()
            monitor.device.disconnect()
            print("Disconnected from device.")

if __name__ == "__main__":
    main()
