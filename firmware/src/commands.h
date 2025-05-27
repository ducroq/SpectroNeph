/**
 * @file commands.h
 * @brief Command handlers for the device
 */

 #ifndef COMMANDS_H
 #define COMMANDS_H
 
 #include <Arduino.h>
 #include <ArduinoJson.h>
 
 /**
  * @brief Register all command handlers
  */
 void registerCommands();
 
 // Command handler functions
 void handlePing(const JsonObject &params, JsonObject &response, const JsonObject &command);
 void handleGetInfo(const JsonObject &params, JsonObject &response, const JsonObject &command);
 void handleAs7341Init(const JsonObject &params, JsonObject &response, const JsonObject &command);
 void handleAs7341Config(const JsonObject &params, JsonObject &response, const JsonObject &command);
 void handleAs7341Read(const JsonObject &params, JsonObject &response, const JsonObject &command);
 void handleAs7341DifferentialRead(const JsonObject &params, JsonObject &response, const JsonObject &command);
 void handleAs7341Led(const JsonObject &params, JsonObject &response, const JsonObject &command);
 void handleStreamStart(const JsonObject &params, JsonObject &response, const JsonObject &command);
 void handleStreamStop(const JsonObject &params, JsonObject &response, const JsonObject &command);
 void handleGetStreams(const JsonObject &params, JsonObject &response, const JsonObject &command);
 void handleResetDevice(const JsonObject &params, JsonObject &response, const JsonObject &command);
 void handleDiagnostics(const JsonObject &params, JsonObject &response, const JsonObject &command);

 #endif // COMMANDS_H