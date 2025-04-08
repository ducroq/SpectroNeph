# AS7341 Nephelometer Firmware

This is the ESP32 firmware for the AS7341-based nephelometer. It implements the device side of the communication protocol and interfaces with the AS7341 spectral sensor.

## Prerequisites

- [PlatformIO](https://platformio.org/) (recommended) or Arduino IDE
- Required libraries:
  - ArduinoJson (v6.x)
  - Adafruit AS7341 Library
  - Adafruit BusIO

## Hardware Setup

### Required Components

- ESP32 development board (ESP32-DevKit or ESP32-S3)
- AS7341 spectral sensor breakout board
- Optional: External LED for illumination

### Wiring

Connect the AS7341 to the ESP32 using I2C:

| AS7341 Pin | ESP32 Pin | Description |
|------------|-----------|-------------|
| VIN        | 3.3V      | Power supply |
| GND        | GND       | Ground |
| SDA        | GPIO21    | I2C Data (configurable in config.h) |
| SCL        | GPIO22    | I2C Clock (configurable in config.h) |
| LED        | -         | Internal LED controlled by AS7341 |

If using an external LED:

| External LED | ESP32 Pin | Description |
|--------------|-----------|-------------|
| Anode (+)    | GPIO13    | Through appropriate resistor (configurable in config.h) |
| Cathode (-)  | GND       | Ground |

## Building and Flashing

### Using PlatformIO

1. Open the project in PlatformIO
2. Select the appropriate environment (esp32dev or esp32s3)
3. Build and upload the firmware:

```bash
# Build the firmware
pio run

# Upload the firmware
pio run --target upload

# Monitor serial output
pio device monitor
```

### Using Arduino IDE

1. Install required libraries through the Library Manager
2. Open main.cpp in Arduino IDE
3. Select the appropriate board in the Board Manager
4. Compile and upload the firmware

## Configuration

The firmware can be configured by modifying the `config.h` file. Key configuration options include:

- I2C pins for the AS7341 sensor
- External LED pin
- Serial communication parameters
- Default sensor settings
- Debug message levels

## Communication Protocol

The firmware implements a JSON-based command/response protocol over the serial port. Commands are sent as JSON objects with the following format:

```json
{
  "cmd": "command_name",
  "id": 123,
  "params": {
    "param1": "value1",
    "param2": "value2"
  }
}
```

Responses have the following format:

```json
{
  "resp": "data",
  "id": 123,
  "status": 0,
  "data": {
    "result1": "value1",
    "result2": "value2"
  }
}
```

## Available Commands

- `ping`: Simple ping command to check connectivity
- `get_info`: Get device information
- `as7341_init`: Initialize the AS7341 sensor
- `as7341_config`: Configure the AS7341 sensor
- `as7341_read`: Read spectral data from the AS7341
- `as7341_led`: Control the AS7341 LED
- `stream_start`: Start a data stream
- `stream_stop`: Stop a data stream
- `get_streams`: Get a list of active streams
- `reset`: Reset the device

## Data Streaming

The firmware supports continuous data streaming for real-time monitoring. Data messages have the following format:

```json
{
  "data": true,
  "type": "as7341",
  "timestamp": 12345,
  "data": {
    "F1": 1024,
    "F2": 2048,
    "F3": 3072,
    "F4": 4096,
    "F5": 5120,
    "F6": 6144,
    "F7": 7168,
    "F8": 8192,
    "Clear": 9216,
    "NIR": 10240
  }
}
```

## Troubleshooting

- If the AS7341 is not detected, check the I2C connections and addresses
- If commands are not being recognized, ensure the JSON format is correct
- Enable debug messages by setting `LOG_LEVEL` to a higher value in `config.h`

## License

This project is licensed under the Apache License - see the LICENSE file for details.