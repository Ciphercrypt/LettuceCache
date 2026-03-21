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
#include "Templatizer.h"

namespace lettucecache::builder {

struct CacheEntryRequest {
    std::string query;
    std::vector<std::string> context;
    std::string domain;
    std::string user_id;
    std::string signature_hash;
    std::string llm_response;
    std::vector<float> embedding;
};

class CacheBuilderWorker {
public:
    CacheBuilderWorker(
        cache::RedisCacheAdapter& redis,
        cache::FaissVectorStore& faiss,
        AdmissionController& admission,
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
    cache::RedisCacheAdapter& redis_;
    cache::FaissVectorStore& faiss_;
    AdmissionController& admission_;
    Templatizer& templatizer_;

    std::queue<CacheEntryRequest> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_thread_;
    std::atomic<bool> running_{false};

    void run();
    void processEntry(const CacheEntryRequest& req);

    static constexpr int L1_TTL_SECONDS = 3600;
    static constexpr const char* STREAM_KEY = "lc:build:stream";
};

} // namespace lettucecache::builder
