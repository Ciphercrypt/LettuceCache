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

std::string postJson(const std::string& url,
                     const std::string& body,
                     int timeout_ms)
{
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init failed");

    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(timeout_ms));
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 2000L);

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        throw std::runtime_error(std::string("curl error: ") + curl_easy_strerror(res));
    }
    return response;
}

} // anonymous namespace

EmbeddingClient::EmbeddingClient(const std::string& base_url)
    : base_url_(base_url)
{
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

std::vector<float> EmbeddingClient::embed(const std::string& text) {
    try {
        nlohmann::json req;
        req["text"] = text;
        std::string body = req.dump();

        std::string raw = postJson(base_url_ + "/embed", body, TIMEOUT_MS);
        auto resp = nlohmann::json::parse(raw);

        std::vector<float> vec;
        if (resp.contains("embedding") && resp["embedding"].is_array()) {
            for (auto& v : resp["embedding"]) {
                vec.push_back(v.get<float>());
            }
        } else {
            spdlog::warn("EmbeddingClient: unexpected response format");
        }
        return vec;
    } catch (const std::exception& e) {
        spdlog::error("EmbeddingClient::embed failed: {}", e.what());
        return {};
    }
}

std::vector<std::vector<float>> EmbeddingClient::embedBatch(
    const std::vector<std::string>& texts)
{
    std::vector<std::vector<float>> results;
    results.reserve(texts.size());
    try {
        nlohmann::json req;
        req["texts"] = texts;
        std::string body = req.dump();

        std::string raw = postJson(base_url_ + "/embed_batch", body, TIMEOUT_MS);
        auto resp = nlohmann::json::parse(raw);

        if (resp.contains("embeddings") && resp["embeddings"].is_array()) {
            for (auto& emb_arr : resp["embeddings"]) {
                std::vector<float> vec;
                for (auto& v : emb_arr) vec.push_back(v.get<float>());
                results.push_back(std::move(vec));
            }
        }
    } catch (const std::exception& e) {
        spdlog::error("EmbeddingClient::embedBatch failed: {}", e.what());
    }
    return results;
}

bool EmbeddingClient::healthCheck() {
    try {
        CURL* curl = curl_easy_init();
        if (!curl) return false;

        std::string response;
        curl_easy_setopt(curl, CURLOPT_URL, (base_url_ + "/health").c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 2000L);

        CURLcode res = curl_easy_perform(curl);
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        curl_easy_cleanup(curl);

        return (res == CURLE_OK) && (http_code == 200);
    } catch (...) {
        return false;
    }
}

} // namespace lettucecache::embedding
