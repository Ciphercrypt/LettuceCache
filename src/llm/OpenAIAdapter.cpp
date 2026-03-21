#include "OpenAIAdapter.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <sstream>

namespace lettucecache::llm {

namespace {

size_t writeCallback(char* ptr, size_t size, size_t nmemb, std::string* out) {
    out->append(ptr, size * nmemb);
    return size * nmemb;
}

} // anonymous namespace

OpenAIAdapter::OpenAIAdapter(const std::string& api_key,
                               const std::string& model,
                               int max_tokens,
                               double temperature)
    : api_key_(api_key), model_(model),
      max_tokens_(max_tokens), temperature_(temperature) {}

std::string OpenAIAdapter::complete(
    const std::string& prompt,
    const std::vector<std::string>& context)
{
    if (api_key_.empty()) {
        spdlog::warn("OpenAI API key not set — returning stub response");
        return "[LLM not configured] " + prompt;
    }

    // Build messages array: system context + user prompt
    nlohmann::json messages = nlohmann::json::array();

    if (!context.empty()) {
        std::ostringstream sys;
        sys << "Context from conversation:\n";
        for (size_t i = 0; i < context.size(); ++i) {
            sys << "[" << (i + 1) << "] " << context[i] << "\n";
        }
        messages.push_back({{"role", "system"}, {"content", sys.str()}});
    } else {
        messages.push_back({{"role", "system"},
            {"content", "You are a helpful assistant."}});
    }
    messages.push_back({{"role", "user"}, {"content", prompt}});

    nlohmann::json payload;
    payload["model"]       = model_;
    payload["messages"]    = messages;
    payload["max_tokens"]  = max_tokens_;
    payload["temperature"] = temperature_;

    std::string body = payload.dump();

    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init failed");

    std::string response;
    std::string auth_header = "Authorization: Bearer " + api_key_;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, auth_header.c_str());

    curl_easy_setopt(curl, CURLOPT_URL, OPENAI_CHAT_URL);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(TIMEOUT_MS));
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 5000L);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        spdlog::error("OpenAI curl error: {}", curl_easy_strerror(res));
        return "";
    }

    try {
        auto json_resp = nlohmann::json::parse(response);
        if (json_resp.contains("choices") && !json_resp["choices"].empty()) {
            return json_resp["choices"][0]["message"]["content"].get<std::string>();
        }
        if (json_resp.contains("error")) {
            spdlog::error("OpenAI API error: {}",
                json_resp["error"].value("message", "unknown"));
        }
    } catch (const std::exception& e) {
        spdlog::error("OpenAI response parse error: {}", e.what());
    }
    return "";
}

bool OpenAIAdapter::isAvailable() {
    return !api_key_.empty();
}

} // namespace lettucecache::llm
