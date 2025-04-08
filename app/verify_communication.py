from core.device import DeviceManager
import time
import json

# Initialize device manager
device = DeviceManager()

# Connect to the device
if device.connect():
    print("Connected to device")
    
    # Get device info
    info = device.send_command("get_info")
    print(f"Device info: {json.dumps(info, indent=2)}")
    
    # Initialize AS7341
    init_result = device.send_command("as7341_init")
    print(f"AS7341 initialization: {init_result}")
    
    # Configure sensor
    config_result = device.send_command("as7341_config", {
        "gain": 5,  # 16x gain
        "integration_time": 100,  # 100ms
        "led_current": 10  # 10mA
    })
    print(f"AS7341 configuration: {json.dumps(config_result, indent=2)}")
    
    # Turn on LED
    led_result = device.send_command("as7341_led", {
        "enabled": True,
        "current": 10
    })
    print(f"LED control: {json.dumps(led_result, indent=2)}")
    
    # Read sensor data
    data_result = device.send_command("as7341_read")
    print(f"Sensor data: {json.dumps(data_result, indent=2)}")
    
    # Turn off LED
    device.send_command("as7341_led", {"enabled": False})
    
    # Disconnect
    device.disconnect()
    print("Disconnected from device")
else:
    print("Failed to connect to device")