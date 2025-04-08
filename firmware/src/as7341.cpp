/**
 * @file as7341.cpp
 * @brief Implementation of AS7341 sensor driver
 */

#include "as7341.h"

// Create global instance
AS7341Driver as7341;

bool AS7341Driver::begin()
{
// Initialize I2C if pins are specified
#if defined(I2C_SDA_PIN) && defined(I2C_SCL_PIN) && I2C_SDA_PIN >= 0 && I2C_SCL_PIN >= 0
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
#endif

    // Initialize the sensor
    if (!as7341.begin())
    {
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 1
        Serial.println("Could not find AS7341");
#endif
        initialized = false;
        return false;
    }

// Initialize external LED pin if configured
#if defined(LED_PIN) && LED_PIN >= 0
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
#endif

    // Apply default configuration
    configure(DEFAULT_GAIN, DEFAULT_ATIME, DEFAULT_LED_CURRENT);

    initialized = true;

#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 3
    Serial.println("AS7341 initialized");
#endif

    return true;
}

bool AS7341Driver::configure(uint8_t gain, uint16_t integrationTime, uint8_t ledCurrent)
{
    if (!initialized && !begin())
    {
        return false;
    }

    bool success = true;

    // Set gain
    as7341_gain_t gainEnum;
    switch (gain)
    {
    case 0:
        gainEnum = AS7341_GAIN_0_5X;
        break;
    case 1:
        gainEnum = AS7341_GAIN_1X;
        break;
    case 2:
        gainEnum = AS7341_GAIN_2X;
        break;
    case 3:
        gainEnum = AS7341_GAIN_4X;
        break;
    case 4:
        gainEnum = AS7341_GAIN_8X;
        break;
    case 5:
        gainEnum = AS7341_GAIN_16X;
        break;
    case 6:
        gainEnum = AS7341_GAIN_32X;
        break;
    case 7:
        gainEnum = AS7341_GAIN_64X;
        break;
    case 8:
        gainEnum = AS7341_GAIN_128X;
        break;
    case 9:
        gainEnum = AS7341_GAIN_256X;
        break;
    case 10:
        gainEnum = AS7341_GAIN_512X;
        break;
    default:
        gainEnum = AS7341_GAIN_16X;
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 2
        Serial.print("Invalid gain value: ");
        Serial.println(gain);
#endif
        success = false;
        break;
    }

    as7341.setGain(gainEnum);
    currentGain = gain;

    // Set integration time (ATIME)
    uint16_t atime = integrationTimeToAtime(integrationTime);
    as7341.setATIME(atime);
    currentIntegrationTime = integrationTime;

    // Set LED current (if LED is enabled)
    if (ledEnabled)
    {
        uint8_t actualCurrent = ledCurrent;
        if (actualCurrent > MAX_LED_CURRENT)
        {
            actualCurrent = MAX_LED_CURRENT;
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 2
            Serial.print("LED current limited to ");
            Serial.println(MAX_LED_CURRENT);
#endif
            success = false;
        }

        as7341.setLEDCurrent(actualCurrent);
        currentLedCurrent = actualCurrent;
    }

#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 3
    Serial.print("AS7341 configured: gain=");
    Serial.print(gain);
    Serial.print(", integrationTime=");
    Serial.print(integrationTime);
    Serial.print(", ledCurrent=");
    Serial.println(ledCurrent);
#endif

    return success;
}

bool AS7341Driver::readSpectralData(JsonObject &readings)
{
    if (!initialized && !begin())
    {
        return false;
    }

    // Measure all channels
    if (!as7341.readAllChannels())
    {
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 1
        Serial.println("Failed to read channels");
#endif
        return false;
    }

    // Read each channel
    readings["F1"] = as7341.getChannel(AS7341_CHANNEL_415nm_F1); // 415nm
    readings["F2"] = as7341.getChannel(AS7341_CHANNEL_445nm_F2); // 445nm
    readings["F3"] = as7341.getChannel(AS7341_CHANNEL_480nm_F3); // 480nm
    readings["F4"] = as7341.getChannel(AS7341_CHANNEL_515nm_F4); // 515nm
    readings["F5"] = as7341.getChannel(AS7341_CHANNEL_555nm_F5); // 555nm
    readings["F6"] = as7341.getChannel(AS7341_CHANNEL_590nm_F6); // 590nm
    readings["F7"] = as7341.getChannel(AS7341_CHANNEL_630nm_F7); // 630nm
    readings["F8"] = as7341.getChannel(AS7341_CHANNEL_680nm_F8); // 680nm
    readings["Clear"] = as7341.getChannel(AS7341_CHANNEL_CLEAR); // Clear
    readings["NIR"] = as7341.getChannel(AS7341_CHANNEL_NIR);     // Near IR

    return true;
}

bool AS7341Driver::setLed(bool enable, uint8_t current)
{
    if (!initialized && !begin())
    {
        return false;
    }

    // Validate current
    uint8_t actualCurrent = current;
    if (actualCurrent > MAX_LED_CURRENT)
    {
        actualCurrent = MAX_LED_CURRENT;
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 2
        Serial.print("LED current limited to ");
        Serial.println(MAX_LED_CURRENT);
#endif
    }

    // Set current and enable LED
    as7341.setLEDCurrent(actualCurrent);
    currentLedCurrent = actualCurrent;

    // Enable/disable LED
    as7341.enableLED(enable);
    ledEnabled = enable;

#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 3
    Serial.print("AS7341 LED ");
    Serial.print(enable ? "enabled" : "disabled");
    Serial.print(" with current ");
    Serial.println(actualCurrent);
#endif

    return true;
}

bool AS7341Driver::setExternalLed(bool enable)
{
#if defined(LED_PIN) && LED_PIN >= 0
    digitalWrite(LED_PIN, enable ? HIGH : LOW);
    externalLedEnabled = enable;

#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 3
    Serial.print("External LED ");
    Serial.println(enable ? "enabled" : "disabled");
#endif

    return true;
#else
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 2
    Serial.println("External LED not configured");
#endif

    return false;
#endif
}

void AS7341Driver::getConfiguration(JsonObject &config)
{
    config["gain"] = currentGain;
    config["integration_time"] = currentIntegrationTime;
    config["led_current"] = currentLedCurrent;
    config["led_enabled"] = ledEnabled;
    config["external_led_enabled"] = externalLedEnabled;
}

bool AS7341Driver::isConnected()
{
    if (initialized)
    {
        return true;
    }

    // Try to initialize if not already initialized
    return begin();
}

uint16_t AS7341Driver::integrationTimeToAtime(uint16_t ms)
{
    // ATIME formula: Integration time = (ATIME + 1) * (ASTEP + 1) * 2.78µs
    // We use a default ASTEP of 999 for a time step of about 2.78ms
    const uint16_t ASTEP = 999;
    const float TIME_STEP_MS = 2.78; // ms

    // Calculate ATIME based on desired integration time
    float timeStepMs = (ASTEP + 1) * 0.00278;
    uint16_t atime = round(ms / timeStepMs) - 1;

    // Ensure ATIME is in valid range (0-65535)
    if (atime > 65535)
    {
        atime = 65535;
    }

    return atime;
}

uint16_t AS7341Driver::atimeToIntegrationTime(uint16_t atime)
{
    // Convert ATIME back to integration time in ms
    const uint16_t ASTEP = 999;
    const float TIME_STEP_MS = 2.78; // ms

    float timeStepMs = (ASTEP + 1) * 0.00278;
    uint16_t ms = (atime + 1) * timeStepMs;

    return ms;
}