#include "EmbeddingClient.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace lettucecache::embedding {

namespace {

size_t writeCallback(char* ptr, size_t size, size_t nmemb, std::string* out) {
    out->append(ptr, size * nmemb);
    return size * nmemb;
}

} // anonymous namespace

// ──────────────────────────────────────────────────────────────────────────────
// CURL lifecycle — persistent handle eliminates per-call TLS handshake overhead
// ──────────────────────────────────────────────────────────────────────────────
void EmbeddingClient::initCurl() {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_ = curl_easy_init();
    if (!curl_) throw std::runtime_error("EmbeddingClient: curl_easy_init failed");
    json_headers_ = curl_slist_append(nullptr, "Content-Type: application/json");
    curl_easy_setopt(curl_, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(curl_, CURLOPT_TCP_KEEPIDLE, 30L);
    curl_easy_setopt(curl_, CURLOPT_TCP_KEEPINTVL, 10L);
}

void EmbeddingClient::teardownCurl() {
    if (json_headers_) { curl_slist_free_all(json_headers_); json_headers_ = nullptr; }
    if (curl_)         { curl_easy_cleanup(curl_);           curl_ = nullptr; }
    curl_global_cleanup();
}

EmbeddingClient::EmbeddingClient(const std::string& base_url) : base_url_(base_url) {
    initCurl();
}

EmbeddingClient::~EmbeddingClient() {
    teardownCurl();
}

// ──────────────────────────────────────────────────────────────────────────────
// Circuit breaker helpers
// ──────────────────────────────────────────────────────────────────────────────
bool EmbeddingClient::allowRequest() {
    std::lock_guard<std::mutex> lock(cb_mutex_);
    CircuitState s = state_.load();
    if (s == CircuitState::CLOSED) return true;

    if (s == CircuitState::OPEN) {
        auto elapsed = std::chrono::steady_clock::now() - open_since_;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
                >= RESET_TIMEOUT_S)
        {
            state_.store(CircuitState::HALF_OPEN);
            spdlog::info("EmbeddingClient: circuit OPEN → HALF_OPEN (probing)");
            return true;
        }
        return false;
    }
    return true;  // HALF_OPEN: allow the probe
}

void EmbeddingClient::recordSuccess() {
    std::lock_guard<std::mutex> lock(cb_mutex_);
    failure_count_ = 0;
    if (state_.load() != CircuitState::CLOSED) {
        state_.store(CircuitState::CLOSED);
        spdlog::info("EmbeddingClient: circuit → CLOSED");
    }
}

void EmbeddingClient::recordFailure() {
    std::lock_guard<std::mutex> lock(cb_mutex_);
    ++failure_count_;
    if (state_.load() == CircuitState::HALF_OPEN ||
        failure_count_ >= FAILURE_THRESHOLD)
    {
        state_.store(CircuitState::OPEN);
        open_since_ = std::chrono::steady_clock::now();
        spdlog::warn("EmbeddingClient: circuit → OPEN (failures={})", failure_count_);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// doPost — persistent CURL handle + circuit breaker gate
// curl_easy_reset() clears all options but keeps the TCP connection alive,
// so the next call reuses the existing HTTP/1.1 keep-alive connection.
// ──────────────────────────────────────────────────────────────────────────────
std::optional<std::string> EmbeddingClient::doPost(const std::string& path,
                                                    const std::string& body)
{
    if (!allowRequest()) {
        spdlog::debug("EmbeddingClient::doPost: circuit OPEN, skipping {}", path);
        return std::nullopt;
    }

    std::lock_guard<std::mutex> lock(curl_mutex_);
    std::string response;
    std::string url = base_url_ + path;

    curl_easy_reset(curl_);
    curl_easy_setopt(curl_, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl_, CURLOPT_POST, 1L);
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, json_headers_);
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT_MS, static_cast<long>(TIMEOUT_MS));
    curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT_MS,
                     static_cast<long>(CONNECT_TIMEOUT_MS));

    CURLcode res = curl_easy_perform(curl_);
    if (res != CURLE_OK) {
        spdlog::warn("EmbeddingClient: curl error on {}: {}", path,
                     curl_easy_strerror(res));
        recordFailure();
        return std::nullopt;
    }

    long http_code = 0;
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &http_code);
    if (http_code >= 500) {
        spdlog::warn("EmbeddingClient: HTTP {} from sidecar on {}", http_code, path);
        recordFailure();
        return std::nullopt;
    }

    recordSuccess();
    return response;
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────
std::vector<float> EmbeddingClient::embed(const std::string& text) {
    try {
        nlohmann::json req;
        req["text"] = text;
        auto raw = doPost("/embed", req.dump());
        if (!raw) return {};
        auto resp = nlohmann::json::parse(*raw);
        if (resp.contains("embedding") && resp["embedding"].is_array())
            return resp["embedding"].get<std::vector<float>>();
        spdlog::warn("EmbeddingClient::embed: unexpected response");
    } catch (const std::exception& e) {
        spdlog::error("EmbeddingClient::embed: {}", e.what());
    }
    return {};
}

std::vector<std::vector<float>> EmbeddingClient::embedBatch(
    const std::vector<std::string>& texts)
{
    std::vector<std::vector<float>> results;
    try {
        nlohmann::json req;
        req["texts"] = texts;
        auto raw = doPost("/embed_batch", req.dump());
        if (!raw) return results;
        auto resp = nlohmann::json::parse(*raw);
        if (resp.contains("embeddings") && resp["embeddings"].is_array()) {
            for (const auto& arr : resp["embeddings"])
                results.push_back(arr.get<std::vector<float>>());
        }
    } catch (const std::exception& e) {
        spdlog::error("EmbeddingClient::embedBatch: {}", e.what());
    }
    return results;
}

bool EmbeddingClient::healthCheck() {
    if (!allowRequest()) return false;
    std::lock_guard<std::mutex> lock(curl_mutex_);

    std::string response;
    curl_easy_reset(curl_);
    curl_easy_setopt(curl_, CURLOPT_URL, (base_url_ + "/health").c_str());
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT_MS, 2000L);

    CURLcode res = curl_easy_perform(curl_);
    long http_code = 0;
    curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &http_code);
    bool ok = (res == CURLE_OK) && (http_code == 200);
    if (ok) recordSuccess(); else recordFailure();
    return ok;
}

} // namespace lettucecache::embedding
