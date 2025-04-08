/**
 * @file streaming.h
 * @brief Data streaming management
 */

#ifndef STREAMING_H
#define STREAMING_H

#include <Arduino.h>
#include <ArduinoJson.h>
#include <map>
#include <vector>

#include "config.h"
#include "protocol.h"

/**
 * @brief Data stream structure
 */
struct DataStream
{
    String type;           // Type of stream (e.g., "as7341")
    uint32_t intervalMs;   // Interval between updates in milliseconds
    uint32_t lastUpdateMs; // Time of last update
    bool active;           // Whether the stream is active
    JsonObject params;     // Stream parameters

    DataStream() : type(""), intervalMs(DEFAULT_STREAM_INTERVAL_MS),
                   lastUpdateMs(0), active(false) {}
};

/**
 * @brief Streaming manager class
 *
 * This class manages data streams from various sources.
 */
class StreamingManager
{
public:
    /**
     * @brief Initialize the streaming manager
     *
     * @return true if initialization was successful
     */
    bool begin();

    /**
     * @brief Update all active streams
     *
     * This method should be called regularly to update active streams.
     */
    void update();

    /**
     * @brief Start a data stream
     *
     * @param type Type of stream to start
     * @param params Stream parameters
     * @param intervalMs Interval between updates in milliseconds
     * @return true if stream was started successfully
     */
    bool startStream(const String &type, const JsonObject &params, uint32_t intervalMs = DEFAULT_STREAM_INTERVAL_MS);

    /**
     * @brief Stop a data stream
     *
     * @param type Type of stream to stop
     * @return true if stream was stopped successfully
     */
    bool stopStream(const String &type);

    /**
     * @brief Check if a stream is active
     *
     * @param type Type of stream to check
     * @return true if stream is active
     */
    bool isStreamActive(const String &type) const;

    /**
     * @brief Get a list of active streams
     *
     * @param streams Array to store stream information
     */
    void getActiveStreams(JsonArray &streams) const;

    /**
     * @brief Stop all active streams
     */
    void stopAllStreams();

private:
    // Map of active streams
    std::map<String, DataStream> activeStreams;

    /**
     * @brief Update a specific stream
     *
     * @param stream Stream to update
     * @return true if update was successful
     */
    bool updateStream(DataStream &stream);

    /**
     * @brief Update AS7341 stream
     *
     * @param stream Stream data
     * @return true if update was successful
     */
    bool updateAs7341Stream(const DataStream &stream);
};

extern StreamingManager streaming;

#endif // STREAMING_H