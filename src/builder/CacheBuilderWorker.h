#pragma once
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include "../cache/RedisCacheAdapter.h"
#include "../cache/FaissVectorStore.h"
#include "AdmissionController.h"
#include "IntelligentAdmissionPolicy.h"
#include "ResponseQualityFilter.h"
#include "Templatizer.h"

namespace lettucecache::builder {

struct CacheEntryRequest {
    std::string query;
    std::vector<std::string> context;
    std::string domain;
    std::string user_id;
    std::string signature_hash;       // full hash (L1 key component)
    std::string context_fingerprint;  // context-only hash (stored in CacheEntry for L2)
    std::string llm_response;
    std::vector<float> embedding;
};

class CacheBuilderWorker {
public:
    CacheBuilderWorker(
        cache::RedisCacheAdapter& redis,
        cache::FaissVectorStore& faiss,
        AdmissionController& admission,
        IntelligentAdmissionPolicy& policy,
        ResponseQualityFilter& quality_filter,
        Templatizer& templatizer
    );
    ~CacheBuilderWorker();

    CacheBuilderWorker(const CacheBuilderWorker&) = delete;
    CacheBuilderWorker& operator=(const CacheBuilderWorker&) = delete;

    void enqueue(const CacheEntryRequest& req);
    void start();
    void stop();
    size_t queueDepth() const;

private:
    cache::RedisCacheAdapter&     redis_;
    cache::FaissVectorStore&     faiss_;
    AdmissionController&         admission_;       // frequency tracking only
    IntelligentAdmissionPolicy&  policy_;          // CVF decision
    ResponseQualityFilter&       quality_filter_;  // pre-filter (hard rejects)
    Templatizer&                 templatizer_;

    std::queue<CacheEntryRequest> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_thread_;
    std::atomic<bool> running_{false};

    void run();
    void processEntry(const CacheEntryRequest& req);

    static constexpr int L1_TTL_SECONDS   = 3600;
    // Slots outlive L1 so that L2 hits always have slot values available.
    static constexpr int SLOT_TTL_SECONDS = L1_TTL_SECONDS * 2;
    static constexpr const char* STREAM_KEY = "lc:build:stream";
};

} // namespace lettucecache::builder
