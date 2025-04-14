/**
 * @file streaming.cpp
 * @brief Implementation of data streaming functionality
 */

#include "streaming.h"
#include "as7341.h"

// Create global streaming instance
StreamingManager streaming;

bool StreamingManager::begin()
{
    // Clear active streams
    activeStreams.clear();

#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 3
    Serial.println("Streaming manager initialized");
#endif

    return true;
}

void StreamingManager::update()
{
    uint32_t currentTime = millis();

    // Update each active stream
    for (auto &pair : activeStreams)
    {
        DataStream &stream = pair.second;

        // Skip inactive streams
        if (!stream.active)
        {
            continue;
        }

        // Check if it's time to update
        if (currentTime - stream.lastUpdateMs >= stream.intervalMs)
        {
            // Update the stream
            if (updateStream(stream))
            {
                // Update last update time
                stream.lastUpdateMs = currentTime;
            }
        }
    }
}

bool StreamingManager::startStream(const String &type, const JsonObject &params, uint32_t intervalMs)
{
    // Validate interval
    if (intervalMs < MIN_STREAM_INTERVAL_MS)
    {
        intervalMs = MIN_STREAM_INTERVAL_MS;

#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 2
        Serial.print("Stream interval limited to minimum: ");
        Serial.println(MIN_STREAM_INTERVAL_MS);
#endif
    }

    if (intervalMs > MAX_STREAM_INTERVAL_MS)
    {
        intervalMs = MAX_STREAM_INTERVAL_MS;

#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 2
        Serial.print("Stream interval limited to maximum: ");
        Serial.println(MAX_STREAM_INTERVAL_MS);
#endif
    }

    // Check if we've reached the maximum number of streams
    if (activeStreams.size() >= MAX_DATA_STREAMS && !activeStreams.count(type))
    {
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 1
        Serial.println("Maximum number of streams reached");
#endif
        return false;
    }

    // Create or update the stream
    DataStream &stream = activeStreams[type];
    stream.type = type;
    stream.intervalMs = intervalMs;
    stream.lastUpdateMs = 0; // Force immediate update
    stream.active = true;

 #if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 3
    Serial.print("Started stream: ");
    Serial.print(type);
    Serial.print(" with interval ");
    Serial.print(intervalMs);
    Serial.println("ms");
#endif

    return true;
}

bool StreamingManager::stopStream(const String &type)
{
    // Check if stream exists
    if (!activeStreams.count(type))
    {
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 2
        Serial.print("Stream not found: ");
        Serial.println(type);
#endif
        return false;
    }

    // Deactivate the stream
    activeStreams[type].active = false;

    // Remove the stream
    activeStreams.erase(type);

#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 3
    Serial.print("Stopped stream: ");
    Serial.println(type);
#endif

    return true;
}

bool StreamingManager::isStreamActive(const String &type) const
{
    return activeStreams.count(type) > 0 && activeStreams.at(type).active;
}

void StreamingManager::getActiveStreams(JsonArray &streams) const
{
    for (const auto &pair : activeStreams)
    {
        const DataStream &stream = pair.second;

        if (stream.active)
        {
            JsonObject streamInfo = streams.createNestedObject();
            streamInfo["type"] = stream.type;
            streamInfo["interval_ms"] = stream.intervalMs;
        }
    }
}

void StreamingManager::stopAllStreams()
{
    // Create a copy of stream keys to avoid iterator invalidation
    std::vector<String> streamKeys;
    for (const auto &pair : activeStreams)
    {
        streamKeys.push_back(pair.first);
    }

    // Stop each stream
    for (const String &type : streamKeys)
    {
        stopStream(type);
    }

#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 3
    Serial.println("Stopped all streams");
#endif
}

bool StreamingManager::updateStream(DataStream &stream)
{
    // Update based on stream type
    if (stream.type == "as7341")
    {
        return updateAs7341Stream(stream);
    }

// Unknown stream type
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 2
    Serial.print("Unknown stream type: ");
    Serial.println(stream.type);
#endif

    return false;
}

bool StreamingManager::updateAs7341Stream(const DataStream &stream)
{
    // Check if AS7341 is connected
    if (!as7341.isConnected())
    {
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 1
        Serial.println("AS7341 not connected");
#endif
        return false;
    }

    // Create data document - use larger buffer
    DynamicJsonDocument doc(JSON_BUFFER_SIZE * 2);
    JsonObject data = doc.to<JsonObject>();    

    // Read spectral data
    if (!as7341.readSpectralData(data))
    {
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 1
        Serial.println("Failed to read AS7341 data");
#endif
        return false;
    }

    // Send data message
    if (!protocol.sendData("as7341", data))
    {
#if ENABLE_DEBUG_MESSAGES && LOG_LEVEL >= 1
        Serial.println("Failed to send AS7341 data");
#endif
        return false;
    }

    return true;
}