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

// Forward declarations
void setupHardware();

void setup()
{
    // Initialize hardware
    setupHardware();

    // Initialize protocol handler
    protocol.begin();

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
    // Process incoming commands
    protocol.update();

    // Update active data streams
    streaming.update();

    // Small delay to prevent tight loops
    delay(1);
}

void setupHardware()
{
    // Initialize serial first for debug output
    Serial.begin(SERIAL_BAUD_RATE);
    delay(100); // Short delay to allow serial to initialize

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