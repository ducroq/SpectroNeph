def detect_serial_port():
    """Try to detect the device port automatically."""
    try:
        from serial.tools import list_ports

        # Common vendor/product IDs for ESP32
        ESP32_VIDS_PIDS = [
            (0x10C4, 0xEA60),  # Silicon Labs CP210x
            (0x1A86, 0x7523),  # QinHeng CH340
            (0x1A86, 0x55D3),  # QinHeng CH343
            (0x1A86, 0x55D4),  # QinHeng CH343
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
            if hasattr(port, "vid") and hasattr(port, "pid") and port.vid is not None:
                if (port.vid, port.pid) in ESP32_VIDS_PIDS:
                    print(f"    ✓ Detected ESP32 device: {port.device}")
                    return port.device

            # Check description for common ESP32 indicators
            if hasattr(port, "description"):
                desc = port.description.lower()
                if any(
                    x in desc
                    for x in ["esp32", "cp210x", "ch340", "ft232", "usb serial"]
                ):
                    print(f"    ✓ Detected ESP32 device: {port.device}")
                    return port.device

        # If no definite match, suggest using the first port
        if ports:
            print(
                f"  ⚠️ No ESP32 device detected, suggesting first port: {ports[0].device}"
            )
            return ports[0].device

        return None
    except Exception as e:
        print(f"Error detecting serial ports: {str(e)}")
        return None