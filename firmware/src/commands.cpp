/**
 * @file commands.cpp
 * @brief Implementation of command handlers
 */

#include "commands.h"
#include "protocol.h"
#include "as7341.h"
#include "streaming.h"
#include "config.h"

void registerCommands()
{
    // Register command handlers with the protocol
    protocol.registerCommand("ping", handlePing);
    protocol.registerCommand("get_info", handleGetInfo);
    protocol.registerCommand("as7341_init", handleAs7341Init);
    protocol.registerCommand("as7341_config", handleAs7341Config);
    protocol.registerCommand("as7341_read", handleAs7341Read);
    protocol.registerCommand("as7341_led", handleAs7341Led);
    protocol.registerCommand("stream_start", handleStreamStart);
    protocol.registerCommand("stream_stop", handleStreamStop);
    protocol.registerCommand("get_streams", handleGetStreams);
    protocol.registerCommand("reset", handleResetDevice);
    protocol.registerCommand("diagnostics", handleDiagnostics);

#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 3
    Serial.println("Command handlers registered");
#endif
}

// Updated signature for all command handlers
void handlePing(const JsonObject &params, JsonObject &response, const JsonObject &command)
{
    // Simple ping command that returns pong
    response["pong"] = true;
    response["time"] = millis();
}

void handleGetInfo(const JsonObject &params, JsonObject &response, const JsonObject &command)
{
    // Return device information
    response["name"] = DEVICE_NAME;
    response["version"] = FIRMWARE_VERSION;
    response["uptime"] = millis();

    // Hardware info
    JsonObject hardware = response.createNestedObject("hardware");
    hardware["chip"] = "ESP32";
    hardware["sdk"] = ESP.getSdkVersion();
    hardware["cpu_freq"] = ESP.getCpuFreqMHz();
    hardware["flash_size"] = ESP.getFlashChipSize() / 1024;
    hardware["free_heap"] = ESP.getFreeHeap();

    // Sensor info
    JsonObject sensor = response.createNestedObject("sensor");
    sensor["type"] = "AS7341";
    sensor["connected"] = as7341.isConnected();

    if (as7341.isConnected())
    {
        // Get current configuration
        JsonObject config = sensor.createNestedObject("config");
        as7341.getConfiguration(config);
    }
}

void handleAs7341Init(const JsonObject &params, JsonObject &response, const JsonObject &command)
{
    // Initialize the AS7341 sensor
    bool success = as7341.begin();

    if (!success)
    {
        response["error"] = "Failed to initialize AS7341";
    }

    response["initialized"] = success;
}

void handleAs7341Config(const JsonObject &params, JsonObject &response, const JsonObject &command)
{
    // Configure the AS7341 sensor
    uint8_t gain = params.containsKey("gain") ? params["gain"].as<uint8_t>() : DEFAULT_GAIN;
    uint16_t integrationTime = params.containsKey("integration_time") ? params["integration_time"].as<uint16_t>() : DEFAULT_ATIME;
    uint8_t ledCurrent = params.containsKey("led_current") ? params["led_current"].as<uint8_t>() : DEFAULT_LED_CURRENT;

    bool success = as7341.configure(gain, integrationTime, ledCurrent);

    if (!success)
    {
        response["warning"] = "Some configuration parameters were invalid";
    }

    // Return current configuration
    as7341.getConfiguration(response);
}

void handleAs7341Read(const JsonObject &params, JsonObject &response, const JsonObject &command)
{
    // Read spectral data from the AS7341
    bool success = as7341.readSpectralData(response);

    if (!success)
    {
        response["error"] = "Failed to read spectral data";
    }
}

void handleAs7341Led(const JsonObject &params, JsonObject &response, const JsonObject &command)
{
    // Control the AS7341 LED
    bool enable = params.containsKey("enabled") ? params["enabled"].as<bool>() : false;
    uint8_t current = params.containsKey("current") ? params["current"].as<uint8_t>() : DEFAULT_LED_CURRENT;

    // Check if external LED should be used
    bool useExternal = params.containsKey("external") ? params["external"].as<bool>() : false;

    bool success;
    if (useExternal)
    {
        success = as7341.setExternalLed(enable);
        response["type"] = "external";
    }
    else
    {
        success = as7341.setLed(enable, current);
        response["type"] = "onboard";
    }

    if (!success)
    {
        response["error"] = "Failed to control LED";
    }

    response["enabled"] = enable;
    if (!useExternal)
    {
        response["current"] = current;
    }
}

void handleStreamStart(const JsonObject &params, JsonObject &response, const JsonObject &command)
{
    // Start a data stream
    if (!params.containsKey("type"))
    {
        response["error"] = "Missing stream type";
        return;
    }

    String type = params["type"].as<String>();
    uint32_t intervalMs = params.containsKey("interval_ms") ? params["interval_ms"].as<uint32_t>() : DEFAULT_STREAM_INTERVAL_MS;

    // Extract stream parameters
    StaticJsonDocument<JSON_BUFFER_SIZE> doc;
    JsonObject streamParams = doc.to<JsonObject>();

    if (params.containsKey("params"))
    {
        JsonObject sourceParams = params["params"].as<JsonObject>();
        for (JsonPair kv : sourceParams)
        {
            streamParams[kv.key()] = kv.value();
        }
    }

    // Start the stream
    bool success = streaming.startStream(type, streamParams, intervalMs);

    if (!success)
    {
        response["error"] = "Failed to start stream";
    }

    response["type"] = type;
    response["interval_ms"] = intervalMs;
    response["active"] = success;
}

void handleStreamStop(const JsonObject &params, JsonObject &response, const JsonObject &command)
{
    // Stop a data stream
    if (!params.containsKey("type"))
    {
        response["error"] = "Missing stream type";
        return;
    }

    String type = params["type"].as<String>();

    // Check if stream is active
    bool wasActive = streaming.isStreamActive(type);

    // Stop the stream
    bool success = streaming.stopStream(type);

    if (!success && wasActive)
    {
        response["error"] = "Failed to stop stream";
    }

    response["type"] = type;
    response["was_active"] = wasActive;
}

void handleGetStreams(const JsonObject &params, JsonObject &response, const JsonObject &command)
{
    // Get a list of active streams
    JsonArray streams = response.createNestedArray("streams");
    streaming.getActiveStreams(streams);

    response["count"] = streams.size();
}

void handleResetDevice(const JsonObject &params, JsonObject &response, const JsonObject &command)
{
    // Stop all streams
    streaming.stopAllStreams();

    // Turn off LEDs
    as7341.setLed(false);
    as7341.setExternalLed(false);

    // Indicate success and schedule reset
    response["reset"] = true;
    response["message"] = "Device will reset in 1 second";

    // Schedule reset after sending response
    delay(100); // Give time for response to be sent
    ESP.restart();
}

void handleDiagnostics(const JsonObject &params, JsonObject &response, const JsonObject &command)
{
    // Simple diagnostics that won't cause a panic
    response["status"] = "running";
    response["timestamp"] = millis();
    
    // System information
    JsonObject systemTest = response.createNestedObject("system");
    systemTest["free_heap"] = ESP.getFreeHeap();
    systemTest["CPU_freq"] = ESP.getCpuFreqMHz();
    systemTest["flash_size"] = ESP.getFlashChipSize() / 1024; // in KB
    systemTest["uptime_ms"] = millis();
    systemTest["status"] = "pass";
    
    // Try/catch for sensor operations to avoid crashes
    JsonObject sensorTest = response.createNestedObject("sensor");
    bool sensorConnected = false;
    
    try {
        // Safely check sensor connection without accessing registers
        sensorConnected = as7341.isConnected();
        sensorTest["connected"] = sensorConnected;
        
        // Only try further tests if connected
        if (sensorConnected) {
            sensorTest["status"] = "pass";
        } else {
            sensorTest["status"] = "fail";
        }
    } catch (...) {
        sensorTest["status"] = "error";
        sensorTest["error"] = "Exception during sensor test";
    }
    
    // Communication test
    JsonObject commTest = response.createNestedObject("communication");
    commTest["serial"] = "pass";  // If we got here, serial is working
    commTest["status"] = "pass";
    
    // Overall result
    response["result"] = (
        systemTest["status"] == "pass" && 
        (sensorTest["status"] == "pass" || !sensorConnected) && 
        commTest["status"] == "pass"
    ) ? "pass" : "fail";
}