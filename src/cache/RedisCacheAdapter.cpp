#include "RedisCacheAdapter.h"
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace lettucecache::cache {

RedisCacheAdapter::RedisCacheAdapter(const std::string& host, int port)
    : host_(host), port_(port)
{
    ctx_ = redisConnect(host.c_str(), port);
    if (!ctx_ || ctx_->err) {
        std::string err = ctx_ ? ctx_->errstr : "allocation failure";
        spdlog::error("Redis connection failed: {}", err);
        if (ctx_) redisFree(ctx_);
        ctx_ = nullptr;
        throw std::runtime_error("Redis connection failed: " + err);
    }
    spdlog::info("Redis connected to {}:{}", host, port);
}

RedisCacheAdapter::~RedisCacheAdapter() {
    if (ctx_) redisFree(ctx_);
}

bool RedisCacheAdapter::reconnect() {
    if (ctx_) { redisFree(ctx_); ctx_ = nullptr; }
    ctx_ = redisConnect(host_.c_str(), port_);
    if (!ctx_ || ctx_->err) {
        spdlog::warn("Redis reconnect failed");
        if (ctx_) { redisFree(ctx_); ctx_ = nullptr; }
        return false;
    }
    spdlog::info("Redis reconnected to {}:{}", host_, port_);
    return true;
}

void RedisCacheAdapter::freeReply(redisReply* reply) {
    if (reply) freeReplyObject(reply);
}

std::optional<std::string> RedisCacheAdapter::get(const std::string& key) {
    std::lock_guard<std::mutex> lock(redis_mutex_);
    if (!ctx_ && !reconnect()) return std::nullopt;

    auto* reply = static_cast<redisReply*>(
        redisCommand(ctx_, "GET %s", key.c_str()));
    if (!reply) { reconnect(); return std::nullopt; }

    std::optional<std::string> result;
    if (reply->type == REDIS_REPLY_STRING) {
        result = std::string(reply->str, reply->len);
    }
    freeReply(reply);
    return result;
}

bool RedisCacheAdapter::set(const std::string& key,
                              const std::string& value,
                              int ttl_seconds)
{
    std::lock_guard<std::mutex> lock(redis_mutex_);
    if (!ctx_ && !reconnect()) return false;

    auto* reply = static_cast<redisReply*>(
        redisCommand(ctx_, "SETEX %s %d %b",
                     key.c_str(), ttl_seconds,
                     value.data(), value.size()));
    bool ok = (reply != nullptr) && (reply->type == REDIS_REPLY_STATUS);
    freeReply(reply);
    return ok;
}

bool RedisCacheAdapter::del(const std::string& key) {
    std::lock_guard<std::mutex> lock(redis_mutex_);
    if (!ctx_ && !reconnect()) return false;

    auto* reply = static_cast<redisReply*>(
        redisCommand(ctx_, "DEL %s", key.c_str()));
    bool ok = (reply != nullptr)
           && (reply->type == REDIS_REPLY_INTEGER)
           && (reply->integer > 0);
    freeReply(reply);
    return ok;
}

bool RedisCacheAdapter::exists(const std::string& key) {
    std::lock_guard<std::mutex> lock(redis_mutex_);
    if (!ctx_ && !reconnect()) return false;

    auto* reply = static_cast<redisReply*>(
        redisCommand(ctx_, "EXISTS %s", key.c_str()));
    bool ok = (reply != nullptr)
           && (reply->type == REDIS_REPLY_INTEGER)
           && (reply->integer > 0);
    freeReply(reply);
    return ok;
}

bool RedisCacheAdapter::xadd(const std::string& stream,
                               const std::string& field,
                               const std::string& value)
{
    std::lock_guard<std::mutex> lock(redis_mutex_);
    if (!ctx_ && !reconnect()) return false;

    auto* reply = static_cast<redisReply*>(
        redisCommand(ctx_, "XADD %s * %s %b",
                     stream.c_str(), field.c_str(),
                     value.data(), value.size()));
    bool ok = (reply != nullptr) && (reply->type == REDIS_REPLY_STRING);
    freeReply(reply);
    return ok;
}

std::vector<std::pair<std::string, std::string>> RedisCacheAdapter::xread(
    const std::string& stream, const std::string& last_id, int count)
{
    std::lock_guard<std::mutex> lock(redis_mutex_);
    std::vector<std::pair<std::string, std::string>> result;
    if (!ctx_ && !reconnect()) return result;

    auto* reply = static_cast<redisReply*>(
        redisCommand(ctx_, "XREAD COUNT %d STREAMS %s %s",
                     count, stream.c_str(), last_id.c_str()));

    if (!reply || reply->type != REDIS_REPLY_ARRAY || reply->elements == 0) {
        freeReply(reply);
        return result;
    }

    // XREAD: [[stream_name, [[id, [f, v, ...]], ...]]]
    auto* stream_entry = reply->element[0];
    if (stream_entry->elements < 2) { freeReply(reply); return result; }

    auto* messages = stream_entry->element[1];
    for (size_t i = 0; i < messages->elements; ++i) {
        auto* msg = messages->element[i];
        if (msg->elements < 2) continue;
        std::string msg_id(msg->element[0]->str, msg->element[0]->len);
        auto* fields = msg->element[1];
        if (fields->elements >= 2) {
            std::string val(fields->element[1]->str,
                            static_cast<size_t>(fields->element[1]->len));
            result.emplace_back(std::move(msg_id), std::move(val));
        }
    }
    freeReply(reply);
    return result;
}

bool RedisCacheAdapter::sadd(const std::string& set_key, const std::string& member) {
    std::lock_guard<std::mutex> lock(redis_mutex_);
    if (!ctx_ && !reconnect()) return false;
    auto* reply = static_cast<redisReply*>(
        redisCommand(ctx_, "SADD %s %s", set_key.c_str(), member.c_str()));
    bool ok = (reply != nullptr) && (reply->type == REDIS_REPLY_INTEGER);
    freeReply(reply);
    return ok;
}

std::vector<std::string> RedisCacheAdapter::smembers(const std::string& set_key) {
    std::lock_guard<std::mutex> lock(redis_mutex_);
    std::vector<std::string> result;
    if (!ctx_ && !reconnect()) return result;
    auto* reply = static_cast<redisReply*>(
        redisCommand(ctx_, "SMEMBERS %s", set_key.c_str()));
    if (reply && reply->type == REDIS_REPLY_ARRAY) {
        for (size_t i = 0; i < reply->elements; ++i) {
            result.emplace_back(reply->element[i]->str,
                                static_cast<size_t>(reply->element[i]->len));
        }
    }
    freeReply(reply);
    return result;
}

bool RedisCacheAdapter::srem(const std::string& set_key, const std::string& member) {
    std::lock_guard<std::mutex> lock(redis_mutex_);
    if (!ctx_ && !reconnect()) return false;
    auto* reply = static_cast<redisReply*>(
        redisCommand(ctx_, "SREM %s %s", set_key.c_str(), member.c_str()));
    bool ok = (reply != nullptr) && (reply->type == REDIS_REPLY_INTEGER);
    freeReply(reply);
    return ok;
}

bool RedisCacheAdapter::setTombstone(const std::string& entry_id, int ttl_seconds) {
    return set("lc:tomb:" + entry_id, "1", ttl_seconds);
}

bool RedisCacheAdapter::isTombstoned(const std::string& entry_id) {
    return exists("lc:tomb:" + entry_id);
}

bool RedisCacheAdapter::multiSet(const std::vector<KVEntry>& entries) {
    std::lock_guard<std::mutex> lock(redis_mutex_);
    if (!ctx_ && !reconnect()) return false;

    auto* multi = static_cast<redisReply*>(redisCommand(ctx_, "MULTI"));
    if (!multi) { reconnect(); return false; }
    bool ok = (multi->type == REDIS_REPLY_STATUS);
    freeReply(multi);
    if (!ok) return false;

    for (const auto& e : entries) {
        auto* r = static_cast<redisReply*>(
            redisCommand(ctx_, "SETEX %s %d %b",
                         e.key.c_str(), e.ttl_seconds,
                         e.value.data(), e.value.size()));
        // Inside MULTI each SETEX returns QUEUED — discard
        freeReply(r);
    }

    auto* exec = static_cast<redisReply*>(redisCommand(ctx_, "EXEC"));
    bool exec_ok = (exec != nullptr) && (exec->type == REDIS_REPLY_ARRAY);
    freeReply(exec);
    return exec_ok;
}

bool RedisCacheAdapter::ping() {
    std::lock_guard<std::mutex> lock(redis_mutex_);
    if (!ctx_ && !reconnect()) return false;
    auto* reply = static_cast<redisReply*>(redisCommand(ctx_, "PING"));
    bool ok = (reply != nullptr) && (reply->type == REDIS_REPLY_STATUS);
    freeReply(reply);
    return ok;
}

} // namespace lettucecache::cache
