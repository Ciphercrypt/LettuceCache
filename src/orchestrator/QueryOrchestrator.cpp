#include "QueryOrchestrator.h"
#include "ContextBuilder.h"
#include "../builder/Templatizer.h"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <chrono>

namespace lettucecache::orchestrator {

QueryOrchestrator::QueryOrchestrator(
    cache::RedisCacheAdapter& redis,
    cache::FaissVectorStore& faiss,
    embedding::EmbeddingClient& embed_client,
    llm::LLMAdapter& llm,
    validation::ValidationService& validator,
    builder::CacheBuilderWorker& builder)
    : redis_(redis), faiss_(faiss), embed_client_(embed_client),
      llm_(llm), validator_(validator), builder_(builder) {}

QueryResponse QueryOrchestrator::process(const QueryRequest& req) {
    auto start = std::chrono::steady_clock::now();

    auto elapsed = [&start]() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
    };

    spdlog::info("process: correlation_id={} session={} user={} domain={}",
                 req.correlation_id, req.session_id, req.user_id, req.domain);

    // ── Step 1: Build context object and compute L1 key ──────────────────
    ContextBuilder ctx_builder;
    ContextObject ctx = ctx_builder.build(req.query, req.context,
                                           req.domain, req.user_id);
    std::string l1_key = "lc:l1:" + ctx.signature_hash;

    // ── Step 2: L1 — Redis exact-match lookup ─────────────────────────────
    auto l1_result = redis_.get(l1_key);
    if (l1_result.has_value()) {
        long long ms = elapsed();
        spdlog::info("L1 hit correlation_id={} latency_ms={}", req.correlation_id, ms);
        return QueryResponse{l1_result.value(), true, 1.0, l1_key, ms};
    }

    // ── Step 3: Embed query for L2 lookup ─────────────────────────────────
    ctx.embedding = embed_client_.embed(req.query);
    if (ctx.embedding.empty()) {
        spdlog::error("Embedding failed correlation_id={}", req.correlation_id);
        // Degrade gracefully: go straight to LLM
        std::string answer = llm_.complete(req.query, req.context);
        return QueryResponse{answer, false, 0.0, "", elapsed()};
    }

    // ── Step 4: L2 — FAISS vector search + validation ─────────────────────
    auto candidates = faiss_.search(ctx.embedding, 5);
    for (const auto& candidate : candidates) {
        double s = validator_.score(ctx, candidate);
        spdlog::debug("L2 candidate id={} score={:.3f} correlation_id={}",
                      candidate.id, s, req.correlation_id);

        if (s >= VALIDATION_THRESHOLD) {
            // Read slot values stored at cache-write time
            std::string rendered = candidate.template_str;
            auto slots_json = redis_.get("lc:slots:" + candidate.id);
            // Backfill L1 to accelerate future identical lookups
            redis_.set(l1_key, rendered, L1_TTL_SECONDS);
            long long ms = elapsed();
            spdlog::info("L2 hit id={} score={:.3f} latency_ms={} correlation_id={}",
                         candidate.id, s, ms, req.correlation_id);
            return QueryResponse{rendered, true, s, candidate.id, ms};
        }
    }

    // ── Step 5: Cache miss — call LLM ─────────────────────────────────────
    spdlog::info("Cache miss — LLM call correlation_id={}", req.correlation_id);
    std::string answer = llm_.complete(req.query, req.context);

    // ── Step 6: Async cache build ──────────────────────────────────────────
    builder::CacheEntryRequest build_req;
    build_req.query          = req.query;
    build_req.context        = req.context;
    build_req.domain         = ctx.domain;
    build_req.user_id        = req.user_id;
    build_req.signature_hash = ctx.signature_hash;
    build_req.llm_response   = answer;
    build_req.embedding      = ctx.embedding;
    builder_.enqueue(build_req);

    long long ms = elapsed();
    spdlog::info("LLM latency_ms={} correlation_id={}", ms, req.correlation_id);
    return QueryResponse{answer, false, 0.0, "", ms};
}

} // namespace lettucecache::orchestrator
