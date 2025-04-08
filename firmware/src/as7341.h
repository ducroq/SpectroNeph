/**
 * @file as7341.h
 * @brief AS7341 sensor driver
 */

#ifndef AS7341_DRIVER_H
#define AS7341_DRIVER_H

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_AS7341.h>
#include <ArduinoJson.h>

#include "config.h"

/**
 * @brief AS7341 sensor controller
 *
 * This class provides an interface to the AS7341 spectral sensor.
 */
class AS7341Driver
{
public:
    /**
     * @brief Initialize the AS7341 sensor
     *
     * @return true if initialization was successful
     */
    bool begin();

    /**
     * @brief Configure the AS7341 sensor
     *
     * @param gain Gain setting
     * @param integrationTime Integration time in milliseconds
     * @param ledCurrent LED current in mA (0-20)
     * @return true if configuration was successful
     */
    bool configure(uint8_t gain = DEFAULT_GAIN,
                   uint16_t integrationTime = DEFAULT_ATIME,
                   uint8_t ledCurrent = DEFAULT_LED_CURRENT);

    /**
     * @brief Read spectral data from the sensor
     *
     * @param readings JSON object to store readings
     * @return true if read was successful
     */
    bool readSpectralData(JsonObject &readings);

    /**
     * @brief Control the AS7341 LED
     *
     * @param enable Whether to enable the LED
     * @param current LED current in mA (0-20)
     * @return true if successful
     */
    bool setLed(bool enable, uint8_t current = DEFAULT_LED_CURRENT);

    /**
     * @brief Control an external LED
     *
     * @param enable Whether to enable the LED
     * @return true if successful
     */
    bool setExternalLed(bool enable);

    /**
     * @brief Get the current configuration
     *
     * @param config JSON object to store configuration
     */
    void getConfiguration(JsonObject &config);

    /**
     * @brief Check if sensor is connected
     *
     * @return true if sensor is connected
     */
    bool isConnected();

private:
    Adafruit_AS7341 as7341;
    bool initialized = false;

    // Current configuration
    uint8_t currentGain = DEFAULT_GAIN;
    uint16_t currentIntegrationTime = DEFAULT_ATIME;
    uint8_t currentLedCurrent = DEFAULT_LED_CURRENT;
    bool ledEnabled = false;
    bool externalLedEnabled = false;

    /**
     * @brief Convert integration time to ATIME register value
     *
     * @param ms Integration time in milliseconds
     * @return uint16_t ATIME register value
     */
    uint16_t integrationTimeToAtime(uint16_t ms);

    /**
     * @brief Convert ATIME register value to integration time
     *
     * @param atime ATIME register value
     * @return uint16_t Integration time in milliseconds
     */
    uint16_t atimeToIntegrationTime(uint16_t atime);
};

extern AS7341Driver as7341;

#endif // AS7341_DRIVER_H