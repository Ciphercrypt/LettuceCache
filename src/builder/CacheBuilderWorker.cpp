#include "CacheBuilderWorker.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <openssl/sha.h>
#include <sstream>
#include <iomanip>

namespace lettucecache::builder {

namespace {

std::string makeEntryId(const std::string& sig_hash, const std::string& query) {
    // Stable ID: sha256(sig:query)[:16]
    std::string raw = sig_hash + ":" + query;
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(raw.data()), raw.size(), hash);
    std::ostringstream oss;
    for (int i = 0; i < 8; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    return oss.str();
}

} // anonymous namespace

CacheBuilderWorker::CacheBuilderWorker(
    cache::RedisCacheAdapter& redis,
    cache::FaissVectorStore& faiss,
    AdmissionController& admission,
    IntelligentAdmissionPolicy& policy,
    ResponseQualityFilter& quality_filter,
    Templatizer& templatizer)
    : redis_(redis), faiss_(faiss),
      admission_(admission), policy_(policy),
      quality_filter_(quality_filter), templatizer_(templatizer) {}

CacheBuilderWorker::~CacheBuilderWorker() {
    stop();
}

void CacheBuilderWorker::start() {
    running_.store(true);
    worker_thread_ = std::thread(&CacheBuilderWorker::run, this);
    spdlog::info("CacheBuilderWorker started");
}

void CacheBuilderWorker::stop() {
    if (running_.exchange(false)) {
        cv_.notify_all();
        if (worker_thread_.joinable()) worker_thread_.join();
        spdlog::info("CacheBuilderWorker stopped");
    }
}

void CacheBuilderWorker::enqueue(const CacheEntryRequest& req) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(req);
    }
    cv_.notify_one();
}

size_t CacheBuilderWorker::queueDepth() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

void CacheBuilderWorker::run() {
    while (running_.load()) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, std::chrono::milliseconds(200),
                     [this] { return !queue_.empty() || !running_.load(); });

        while (!queue_.empty()) {
            CacheEntryRequest req = queue_.front();
            queue_.pop();
            lock.unlock();
            try {
                processEntry(req);
            } catch (const std::exception& e) {
                spdlog::error("CacheBuilderWorker::processEntry exception: {}", e.what());
            }
            lock.lock();
        }
    }
}

void CacheBuilderWorker::processEntry(const CacheEntryRequest& req) {
    // ── Stage 1: record request for frequency & domain tracking ───────────
    admission_.recordQuery(req.signature_hash);
    policy_.recordRequest(req.signature_hash, req.domain);

    // ── Stage 2: hard size gate (fast, no computation) ────────────────────
    if (!admission_.shouldAdmit(req.signature_hash, req.llm_response)) {
        spdlog::debug("CacheBuilder: size/cold rejected sig={}", req.signature_hash);
        return;
    }

    // ── Stage 3: quality pre-filter (hard rejects — no need for CVF) ──────
    // Conversational, session-bound, refusals, and AI-identity responses are
    // rejected here before the more expensive CVF computation.
    auto quality = quality_filter_.evaluate(req.llm_response, req.query);
    if (!quality.should_cache) {
        spdlog::info("CacheBuilder: quality rejected sig={} score={:.2f} [{}]",
                     req.signature_hash, quality.score, quality.reason);
        return;
    }

    if (req.embedding.empty()) {
        spdlog::warn("CacheBuilder: empty embedding sig={}, skipping", req.signature_hash);
        return;
    }

    // ── Stage 4: Intelligent Admission — multi-signal Cache Value Function ─
    // CVF = 0.30×frequency + 0.25×cost + 0.25×quality + 0.20×MMR_novelty
    // Adaptive per-domain threshold based on observed hit rates.
    auto decision = policy_.evaluate(
        req.signature_hash, req.llm_response, req.domain,
        req.embedding, quality.score);

    if (!decision.should_admit) {
        spdlog::info("CacheBuilder: CVF rejected sig={} value={:.3f} [{}]",
                     req.signature_hash, decision.value, decision.reason);
        return;
    }

    // 4. Templatize the response
    auto tpl_result = templatizer_.templatize(req.llm_response);

    // 5. Build cache entry
    std::string entry_id = makeEntryId(req.signature_hash, req.query);

    cache::CacheEntry entry;
    entry.id                = entry_id;
    entry.embedding         = req.embedding;
    entry.context_signature = req.signature_hash;
    entry.template_str      = tpl_result.templ;
    entry.domain            = req.domain;

    // 5. Store serialized slot values in Redis alongside L1 key
    nlohmann::json slot_json = tpl_result.slot_values;
    std::string slot_key = "lc:slots:" + entry_id;
    redis_.set(slot_key, slot_json.dump(), L1_TTL_SECONDS);

    // 6. Write L1: exact response for this signature
    std::string l1_key = "lc:l1:" + req.signature_hash;
    redis_.set(l1_key, req.llm_response, L1_TTL_SECONDS);

    // 7. Add to FAISS L2
    faiss_.add(entry);

    spdlog::info("CacheBuilder: stored entry_id={} sig={} domain={}",
                 entry_id, req.signature_hash, req.domain);
}

} // namespace lettucecache::builder
