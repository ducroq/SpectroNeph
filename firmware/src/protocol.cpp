/**
 * @file protocol.cpp
 * @brief Implementation of the communication protocol
 */

#include "protocol.h"

// Create global protocol instance
Protocol protocol;

bool Protocol::begin()
{
    // Initialize serial with expanded buffer size
    Serial.begin(SERIAL_BAUD_RATE);
    Serial.setRxBufferSize(SERIAL_RX_SIZE);
    Serial.setTxBufferSize(SERIAL_TX_SIZE);

    // Clear the command buffer
    memset(commandBuffer, 0, JSON_BUFFER_SIZE);
    bufferPos = 0;

    // Clear the command queue
    commandQueue.clear();

    // Wait for serial to be ready
    delay(100);

#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 3
    Serial.println("Protocol initialized");
#endif

    return true;
}

void Protocol::update()
{
    // Process incoming serial data
    while (Serial.available() > 0)
    {
        char c = Serial.read();

        // Add character to buffer if it's not a newline
        if (c != '\n' && c != '\r')
        {
            if (bufferPos < JSON_BUFFER_SIZE - 1)
            {
                commandBuffer[bufferPos++] = c;
            }
        }
        else if (c == '\n')
        {
            // Null-terminate the buffer
            commandBuffer[bufferPos] = '\0';

            // Process the command if buffer has content
            if (bufferPos > 0)
            {
                // Add to command queue
                commandQueue.push_back(String(commandBuffer));
            }

            // Reset buffer
            memset(commandBuffer, 0, JSON_BUFFER_SIZE);
            bufferPos = 0;
        }
    }

    // Process commands in the queue
    while (!commandQueue.empty())
    {
        String cmdStr = commandQueue.front();
        commandQueue.erase(commandQueue.begin());

        // Parse and process the command
        StaticJsonDocument<JSON_BUFFER_SIZE> doc;
        DeserializationError error = deserializeJson(doc, cmdStr);

        if (error)
        {
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 1
            Serial.print("JSON parse error: ");
            Serial.println(error.c_str());
#endif
            continue;
        }

        // Check if it's a command
        JsonObject cmd = doc.as<JsonObject>();
        if (cmd.containsKey("cmd"))
        {
            processCommand(cmd);
        }
    }
}

void Protocol::registerCommand(const String &command, CommandHandler handler)
{
    commandHandlers[command] = handler;

#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 4
    Serial.print("Registered command handler: ");
    Serial.println(command);
#endif
}

bool Protocol::sendData(const String &type, const JsonObject &data)
{
    // Use DynamicJsonDocument for potentially larger data
    DynamicJsonDocument doc(JSON_BUFFER_SIZE * 2); // Double the size for safety

    // Create data message
    doc["data"] = true;
    doc["type"] = type;
    doc["timestamp"] = millis();

    // Copy data to message
    JsonObject dataObj = doc.createNestedObject("data");
    for (JsonPair kv : data)
    {
        dataObj[kv.key()] = kv.value();
    }

    // Send the message
    return sendJsonDocument(doc);
}

bool Protocol::sendEvent(const String &type, const JsonObject &data)
{
    // Use DynamicJsonDocument for potentially larger events
    DynamicJsonDocument doc(JSON_BUFFER_SIZE * 2); // Double the size for safety

    // Create event message
    doc["event"] = true;
    doc["type"] = type;
    doc["timestamp"] = millis();

    // Copy data to message
    JsonObject dataObj = doc.createNestedObject("data");
    for (JsonPair kv : data)
    {
        dataObj[kv.key()] = kv.value();
    }

    // Send the message
    return sendJsonDocument(doc);
}

bool Protocol::commandExists(const String &command) const
{
    return commandHandlers.find(command) != commandHandlers.end();
}

void Protocol::processCommand(const JsonObject &command)
{
    // Extract command fields
    String cmdName = command["cmd"].as<String>();
    int cmdId = command["id"].as<int>();

#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 3
    Serial.print("Received command: ");
    Serial.print(cmdName);
    Serial.print(" (id=");
    Serial.print(cmdId);
    Serial.println(")");
#endif

    // Check if command exists
    if (!commandExists(cmdName))
    {
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 2
        Serial.print("Unknown command: ");
        Serial.println(cmdName);
#endif

        sendErrorMessage(ResponseType::ERROR, cmdId,
                         String("Unknown command: ") + cmdName,
                         StatusCode::INVALID_COMMAND);
        return;
    }

    // Extract parameters
    JsonObject params = command["params"].as<JsonObject>();

    // Prepare response
    StaticJsonDocument<JSON_BUFFER_SIZE> responseDoc;
    JsonObject responseData = responseDoc.to<JsonObject>();

    // Call the command handler
    try
    {
        commandHandlers[cmdName](params, responseData, command);

        // Send success response
        sendResponse(ResponseType::DATA, cmdId, responseData);
    }
    catch (const std::exception &e)
    {
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 1
        Serial.print("Command execution error: ");
        Serial.println(e.what());
#endif

        sendErrorMessage(ResponseType::ERROR, cmdId,
                         String("Execution error: ") + e.what(),
                         StatusCode::EXECUTION_ERROR);
    }
    catch (...)
    {
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 1
        Serial.println("Unknown command execution error");
#endif

        sendErrorMessage(ResponseType::ERROR, cmdId,
                         "Unknown execution error",
                         StatusCode::EXECUTION_ERROR);
    }
}

bool Protocol::sendResponse(ResponseType type, int commandId,
                            const JsonVariant &data,
                            StatusCode status)
{
    StaticJsonDocument<JSON_BUFFER_SIZE> doc;

    // Set response type
    switch (type)
    {
    case ResponseType::ACK:
        doc["resp"] = "ack";
        break;
    case ResponseType::DATA:
        doc["resp"] = "data";
        break;
    case ResponseType::ERROR:
        doc["resp"] = "error";
        break;
    }

    // Set command ID and status
    doc["id"] = commandId;
    doc["status"] = static_cast<int>(status);

    // Add data if provided
    doc["data"] = data;

    // Send the response
    return sendJsonDocument(doc);
}

bool Protocol::sendJsonDocument(const JsonDocument &doc)
{
    // Serialize and send
    String output;
    serializeJson(doc, output);
    output += "\n";

    size_t bytesWritten = Serial.print(output);

#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 5
    Serial.print("Sent JSON: ");
    Serial.println(output);
#endif

    return bytesWritten == output.length();
}
