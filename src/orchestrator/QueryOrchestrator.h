#pragma once
#include <string>
#include <vector>
#include "../cache/RedisCacheAdapter.h"
#include "../cache/FaissVectorStore.h"
#include "../embedding/EmbeddingClient.h"
#include "../llm/LLMAdapter.h"
#include "../validation/ValidationService.h"
#include "../builder/CacheBuilderWorker.h"
#include "../builder/IntelligentAdmissionPolicy.h"
#include "ContextBuilder.h"

namespace lettucecache::orchestrator {

struct QueryRequest {
    // ── Routing / tracing ──────────────────────────────────────────────────
    std::string query;
    std::vector<std::string> context;
    std::string user_id;
    std::string session_id;
    std::string domain;
    std::string correlation_id;

    // ── LLM call parameters (all feed into the cache key) ─────────────────
    // Framing — change what kind of response is possible
    std::string system_prompt;        // static deployment instructions
    std::string response_format;      // "text" | "json_object" | "json_schema"
    std::string response_schema;      // JSON schema; only for "json_schema" format
    std::string tools;                // JSON array of tool definitions; empty = none
    std::string tool_choice;          // "auto" | "none" | "required" | function name

    // Distribution — shift sampling distribution; bucketed in cache key
    float       temperature{0.0f};    // high values bypass cache entirely
    float       top_p{1.0f};          // nucleus sampling; default = full distribution
    int         max_tokens{0};        // 0 = model decides
    int         seed{-1};             // -1 = not provided
    std::string model;
};

struct QueryResponse {
    std::string answer;
    bool cache_hit{false};
    double confidence{0.0};
    std::string cache_entry_id;
    long long latency_ms{0};
};

class QueryOrchestrator {
public:
    QueryOrchestrator(
        cache::RedisCacheAdapter& redis,
        cache::FaissVectorStore& faiss,
        embedding::EmbeddingClient& embed_client,
        llm::LLMAdapter& llm,
        validation::ValidationService& validator,
        builder::CacheBuilderWorker& builder,
        builder::IntelligentAdmissionPolicy& policy
    );

    QueryResponse process(const QueryRequest& req);

private:
    cache::RedisCacheAdapter&           redis_;
    cache::FaissVectorStore&            faiss_;
    embedding::EmbeddingClient&         embed_client_;
    llm::LLMAdapter&                    llm_;
    validation::ValidationService&      validator_;
    builder::CacheBuilderWorker&        builder_;
    builder::IntelligentAdmissionPolicy& policy_;

    static constexpr double VALIDATION_THRESHOLD = 0.85;
    static constexpr int    L1_TTL_SECONDS       = 3600;
    // Requests with temperature >= this value are non-deterministic; skip
    // both cache lookup and cache population to avoid polluting the cache.
    static constexpr float  HIGH_TEMP_THRESHOLD  = 0.7f;
};

} // namespace lettucecache::orchestrator
