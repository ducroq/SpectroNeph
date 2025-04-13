/**
 * @file main.cpp
 * @brief Main entry point for the firmware
 */

#include <Arduino.h>
#include "protocol.h"
#include "commands.h"
#include "as7341.h"
#include "streaming.h"
#include "config.h"
#include "esp_task_wdt.h"
#include "esp_wifi.h"
#include "esp_sleep.h"
#include "power_management.h"

// Forward declarations
void setupHardware();

unsigned long lastActivityTime = 0;

void setup()
{
    // Initialize hardware
    setupHardware();

    // Configure watchdog timer (timeout in seconds)
    const int WDT_TIMEOUT = 30;           // 30 second timeout
    esp_task_wdt_init(WDT_TIMEOUT, true); // Enable panic so ESP32 restarts
    esp_task_wdt_add(NULL);               // Add current thread to WDT watch

    // Initialize protocol handler
    protocol.begin();

    // Initialize power management
    powerManagement.begin();

    // Register command handlers
    registerCommands();

    // Initialize AS7341 sensor
    if (!as7341.begin())
    {
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 1
        Serial.println("Failed to initialize AS7341");
#endif
    }

    // Initialize streaming manager
    streaming.begin();

    // Send startup event
    StaticJsonDocument<JSON_BUFFER_SIZE> doc;
    JsonObject eventData = doc.to<JsonObject>();
    eventData["uptime"] = millis();
    eventData["sensor_connected"] = as7341.isConnected();
    protocol.sendEvent("device_ready", eventData);

#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 3
    Serial.println("Device ready");
#endif
}

void loop()
{
    // Reset watchdog timer to prevent timeout
    esp_task_wdt_reset();

    // Process incoming commands
    protocol.update();

    // Update active data streams
    streaming.update();

    // Check if we should enter sleep mode
    powerManagement.checkSleepConditions();

    // Small delay to prevent tight loops
    delay(1);
}

void setupHardware()
{
    // Initialize serial with expanded buffer size
    Serial.setRxBufferSize(SERIAL_RX_SIZE);
    Serial.setTxBufferSize(SERIAL_TX_SIZE);
    Serial.begin(SERIAL_BAUD_RATE);
    delay(100); // Short delay to allow serial to initialize

    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);

#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 3
    Serial.println("\n\nAS7341 Nephelometer");
    Serial.print("Firmware version: ");
    Serial.println(FIRMWARE_VERSION);
#endif

// Configure pins
#if defined(LED_PIN) && LED_PIN >= 0
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
#endif
}