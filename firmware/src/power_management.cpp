/**
 * @file power_management.cpp
 * @brief Implementation of power management functionality
 */

#include "power_management.h"
#include "as7341.h"
#include "esp_sleep.h"
#include "esp_wifi.h"

// Conditionally include WiFi if needed
#if defined(ESP_PLATFORM) && defined(ENABLE_WIFI)
#include <WiFi.h>
#endif

// Create global instance
PowerManagement powerManagement;

bool PowerManagement::begin()
{
    // Initialize timestamp
    lastActivityTime = millis();

#if ENABLE_POWER_SAVING
    // Configure any power-saving settings
#if defined(ESP_PLATFORM)
// Set WiFi power save mode if applicable
#if defined(ENABLE_WIFI) && defined(ESP_ARDUINO_VERSION_MAJOR)
    WiFi.setSleep(true);
#endif
#endif
#endif

    return true;
}

void PowerManagement::updateActivityTimestamp()
{
    lastActivityTime = millis();
}

void PowerManagement::checkSleepConditions()
{
#if ENABLE_POWER_SAVING
    // If it's been too long since last activity, enter light sleep
    if (millis() - lastActivityTime > SLEEP_AFTER_IDLE_MS)
    {
        enterLightSleep(5000); // Wake every 5 seconds to check
    }
#endif
}

void PowerManagement::enterLightSleep(uint32_t sleepTimeMs)
{
#if ENABLE_POWER_SAVING && defined(ESP_PLATFORM)
    // Prepare for sleep
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 2
    Serial.println("Entering light sleep mode due to inactivity");
    Serial.flush();
#endif

    // Turn off LED and sensor
    as7341.setLed(false);
    as7341.enableExternalLed(false);

    // Enter light sleep (can be woken by timer or external interrupt)
    if (sleepTimeMs > 0)
    {
        esp_sleep_enable_timer_wakeup(sleepTimeMs * 1000); // Convert to microseconds
    }
    esp_light_sleep_start();

    // When we wake up, reset the activity timestamp
    updateActivityTimestamp();
#endif
}