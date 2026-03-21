#pragma once
#include <string>
#include <optional>
#include <vector>
#include <utility>
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
    bool del(const std::string& key);
    bool exists(const std::string& key);

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

    bool reconnect();
    static void freeReply(redisReply* reply);
};

} // namespace lettucecache::cache
