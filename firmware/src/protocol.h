/**
 * @file protocol.h
 * @brief Communication protocol handler for nephelometer
 */

#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <Arduino.h>
#include <ArduinoJson.h>
#include <functional>
#include <map>
#include <vector>

#include "config.h"

/**
 * @brief Status codes for command responses
 */
enum class StatusCode
{
    SUCCESS = 0,         // Command executed successfully
    INVALID_COMMAND = 1, // Invalid command
    INVALID_PARAMS = 2,  // Invalid parameters
    EXECUTION_ERROR = 3, // Error executing command
    TIMEOUT = 4,         // Command timed out
    BUSY = 5,            // Device is busy
    NOT_IMPLEMENTED = 6  // Command not implemented
};

/**
 * @brief Response types for command responses
 */
enum class ResponseType
{
    ACK,  // Acknowledge (success)
    DATA, // Data response
    ERROR // Error response
};

/**
 * @brief Function type for command handlers
 */
typedef std::function<void(const JsonObject &, JsonObject &, JsonObject &)> CommandHandler;

/**
 * @brief Protocol class for handling communication
 *
 * This class manages the communication protocol between the device and the host.
 * It handles command parsing, response formatting, and data streaming.
 */
class Protocol
{
public:
    /**
     * @brief Initialize the protocol handler
     *
     * @return true if initialization was successful
     */
    bool begin();

    /**
     * @brief Process incoming data
     *
     * This method should be called regularly to process incoming commands.
     */
    void update();

    /**
     * @brief Register a command handler
     *
     * @param command Command name
     * @param handler Function to handle the command
     */
    void registerCommand(const String &command, CommandHandler handler);

    /**
     * @brief Send a data message to the host
     *
     * @param type Data type
     * @param data Data to send
     * @return true if the message was sent successfully
     */
    bool sendData(const String &type, const JsonObject &data);

    /**
     * @brief Send an event message to the host
     *
     * @param type Event type
     * @param data Event data
     * @return true if the message was sent successfully
     */
    bool sendEvent(const String &type, const JsonObject &data);

    /**
     * @brief Check if the given command exists
     *
     * @param command Command to check
     * @return true if the command exists
     */
    bool commandExists(const String &command) const;

private:
    /**
     * @brief Process a command from the host
     *
     * @param command Command JSON object
     */
    void processCommand(const JsonObject &command);

    /**
     * @brief Send a response to the host
     *
     * @param type Response type
     * @param commandId Command ID to respond to
     * @param data Response data
     * @param status Status code
     * @return true if the response was sent successfully
     */
    bool sendResponse(ResponseType type, int commandId, const JsonVariant &data = JsonVariant(),
                      StatusCode status = StatusCode::SUCCESS);

    /**
     * @brief Send an error response with a string message
     * 
     * @param type Response type
     * @param commandId Command ID to respond to
     * @param message Error message
     * @param status Status code
     * @return true if the response was sent successfully
     */
    bool sendErrorMessage(ResponseType type, int commandId, const String &message,
        StatusCode status = StatusCode::EXECUTION_ERROR) {
        StaticJsonDocument<JSON_BUFFER_SIZE> doc;
        JsonVariant variant = doc.to<JsonVariant>();
        variant.set(message);
        return sendResponse(type, commandId, variant, status);
    }                      

    /**
     * @brief Send a JSON document to the host
     *
     * @param doc JSON document to send
     * @return true if the document was sent successfully
     */
    bool sendJsonDocument(const JsonDocument &doc);

    // Map of registered command handlers
    std::map<String, CommandHandler> commandHandlers;

    // Buffer for command processing
    char commandBuffer[JSON_BUFFER_SIZE];
    size_t bufferPos = 0;

    // Queue of received commands
    std::vector<String> commandQueue;
};

extern Protocol protocol;

#endif // PROTOCOL_H