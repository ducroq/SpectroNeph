/**
 * @file config.h
 * @brief Configuration settings for the firmware
 */

#ifndef CONFIG_H
#define CONFIG_H

// Hardware configuration
#define I2C_SDA_PIN 4  // I2C SDA pin for AS7341 (I2C0: 8,9 and I2C1: 4,5)
#define I2C_SCL_PIN 5  // I2C SCL pin for AS7341
#define LED_PIN -1     // Optional external LED pin (set to -1 if not used)

// Serial communication
#define SERIAL_BAUD_RATE 115200 // Serial baud rate
#define JSON_BUFFER_SIZE 2048   // Size of JSON buffer for commands/responses
#define SERIAL_RX_SIZE 2048     // Size of serial receive buffer
#define SERIAL_TX_SIZE 2048     // Size of serial transmit buffer

// AS7341 sensor settings
#define DEFAULT_GAIN AS7341_GAIN_16X // Default gain setting
#define DEFAULT_ATIME 100            // Default integration time (ms)
#define DEFAULT_LED_CURRENT 10       // Default LED current (mA, 0-20)
#define MAX_LED_CURRENT 20           // Maximum LED current (mA)

// Protocol settings
#define CMD_BUFFER_SIZE 5                 // Number of commands to buffer
#define MAX_DATA_STREAMS 3                // Maximum number of concurrent data streams
#define DEVICE_NAME "AS7341 Nephelometer" // Device name
#define FIRMWARE_VERSION "0.1.0"          // Firmware version

// Streaming settings
#define MIN_STREAM_INTERVAL_MS 10      // Minimum interval between stream updates (ms)
#define MAX_STREAM_INTERVAL_MS 60000   // Maximum interval between stream updates (ms)
#define DEFAULT_STREAM_INTERVAL_MS 100 // Default interval between stream updates (ms)

// Debug settings
#define ENABLE_DEBUG_MESSAGES 1 // Enable debug messages (1=true, 0=false)
#define LOG_LEVEL 5             // 0=None, 1=Error, 2=Warn, 3=Info, 4=Debug, 5=Verbose

// Power management settings
#define ENABLE_POWER_SAVING 0          // Enable power saving features (1=true, 0=false)
#define SLEEP_AFTER_IDLE_MS 60000      // Enter light sleep after 60 seconds of inactivity

// WiFi settings (if applicable)
// #define ENABLE_WIFI 1               // Enable WiFi functionality (uncomment if needed)

#endif // CONFIG_H