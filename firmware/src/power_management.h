/**
 * @file power_management.h
 * @brief Power management functionality for battery optimization
 */

 #ifndef POWER_MANAGEMENT_H
 #define POWER_MANAGEMENT_H
 
 #include <Arduino.h>
 #include "config.h"
 
 /**
  * @brief Power management class
  * 
  * This class manages device power states and sleep modes to optimize battery life.
  */
 class PowerManagement {
 public:
     /**
      * @brief Initialize power management
      * 
      * @return true if initialization was successful
      */
     bool begin();
     
     /**
      * @brief Update activity timestamp
      * 
      * Call this whenever there is user activity to prevent sleep
      */
     void updateActivityTimestamp();
     
     /**
      * @brief Check sleep conditions and enter sleep if appropriate
      * 
      * Call this regularly from the main loop
      */
     void checkSleepConditions();
     
     /**
      * @brief Force the device to enter light sleep
      * 
      * @param sleepTimeMs Time to sleep in milliseconds
      */
     void enterLightSleep(uint32_t sleepTimeMs = 0);
     
 private:
     unsigned long lastActivityTime = 0;
 };
 
 // Global instance
 extern PowerManagement powerManagement;
 
 #endif // POWER_MANAGEMENT_H