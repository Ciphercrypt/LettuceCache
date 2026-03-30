#pragma once
#include <string>
#include <optional>
#include <vector>
#include <utility>
#include <mutex>
#include <hiredis/hiredis.h>

namespace lettucecache::cache {

class RedisCacheAdapter {
public:
    RedisCacheAdapter(const std::string& host, int port);
    ~RedisCacheAdapter();

    RedisCacheAdapter(const RedisCacheAdapter&) = delete;
    RedisCacheAdapter& operator=(const RedisCacheAdapter&) = delete;

    std::optional<std::string> get(const std::string& key);
    bool set(const std::string& key, const std::string& value, int ttl_seconds = 3600);
    struct KVEntry {
        std::string key;
        std::string value;
        int ttl_seconds;
    };
    // Atomically set multiple key→value pairs (each with its own TTL) via MULTI/EXEC.
    bool multiSet(const std::vector<KVEntry>& entries);
    bool del(const std::string& key);
    bool exists(const std::string& key);

    // Domain index — tracks entry IDs per domain for bulk invalidation
    bool sadd(const std::string& set_key, const std::string& member);
    std::vector<std::string> smembers(const std::string& set_key);
    bool srem(const std::string& set_key, const std::string& member);

    // Tombstone — written before eviction to prevent stale L2 hits.
    // TTL defaults to 24 h so concurrent reads within the eviction window are covered.
    bool setTombstone(const std::string& entry_id, int ttl_seconds = 86400);
    bool isTombstoned(const std::string& entry_id);

    // Redis Streams — async write/read path for cache build requests
    bool xadd(const std::string& stream,
               const std::string& field,
               const std::string& value);
    std::vector<std::pair<std::string, std::string>> xread(
        const std::string& stream,
        const std::string& last_id,
        int count = 10);

    bool ping();

private:
    redisContext* ctx_{nullptr};
    std::string host_;
    int port_;
    mutable std::mutex redis_mutex_;

    bool reconnect();
    static void freeReply(redisReply* reply);
};

} // namespace lettucecache::cache
