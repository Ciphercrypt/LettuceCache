#pragma once
#include <string>
#include <vector>
#include "../cache/RedisCacheAdapter.h"
#include "../cache/FaissVectorStore.h"
#include "../embedding/EmbeddingClient.h"
#include "../llm/LLMAdapter.h"
#include "../validation/ValidationService.h"
#include "../builder/CacheBuilderWorker.h"
#include "ContextBuilder.h"

namespace lettucecache::orchestrator {

struct QueryRequest {
    std::string query;
    std::vector<std::string> context;
    std::string user_id;
    std::string session_id;
    std::string domain;
    std::string correlation_id;
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
        builder::CacheBuilderWorker& builder
    );

    QueryResponse process(const QueryRequest& req);

private:
    cache::RedisCacheAdapter& redis_;
    cache::FaissVectorStore& faiss_;
    embedding::EmbeddingClient& embed_client_;
    llm::LLMAdapter& llm_;
    validation::ValidationService& validator_;
    builder::CacheBuilderWorker& builder_;

    static constexpr double VALIDATION_THRESHOLD = 0.85;
    static constexpr int L1_TTL_SECONDS          = 3600;
};

} // namespace lettucecache::orchestrator
