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
    builder::CacheBuilderWorker& builder,
    builder::IntelligentAdmissionPolicy& policy)
    : redis_(redis), faiss_(faiss), embed_client_(embed_client),
      llm_(llm), validator_(validator), builder_(builder), policy_(policy) {}

QueryResponse QueryOrchestrator::process(const QueryRequest& req) {
    auto start = std::chrono::steady_clock::now();

    auto elapsed = [&start]() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
    };

    spdlog::info("process: correlation_id={} session={} user={} domain={}",
                 req.correlation_id, req.session_id, req.user_id, req.domain);

    // ── Step 1: Build context object and compute L1 key ──────────────────
    CacheDimensions dims;
    dims.system_prompt    = req.system_prompt;
    dims.response_format  = req.response_format;
    dims.response_schema  = req.response_schema;
    dims.tools            = req.tools;
    dims.tool_choice      = req.tool_choice;
    dims.temperature      = req.temperature;
    dims.top_p            = req.top_p;
    dims.max_tokens       = req.max_tokens;
    dims.seed             = req.seed;
    dims.model            = req.model;

    ContextBuilder ctx_builder;
    ContextObject ctx = ctx_builder.build(req.query, req.context,
                                           req.domain, req.user_id, dims);
    std::string l1_key = "lc:l1:" + ctx.signature_hash;

    // ── Step 1b: Bypass cache for high-temperature requests ───────────────
    // High-temperature outputs are stochastic one-offs. Caching them would
    // pollute the index with responses that are unlikely to match future queries.
    if (req.temperature >= HIGH_TEMP_THRESHOLD) {
        spdlog::info("High-temp bypass temperature={:.1f} correlation_id={}",
                     req.temperature, req.correlation_id);
        std::string answer = llm_.complete(req.query, req.context);
        return QueryResponse{answer, false, 0.0, "", elapsed()};
    }

    // ── Step 2: L1 — Redis exact-match lookup ─────────────────────────────
    auto l1_result = redis_.get(l1_key);
    if (l1_result.has_value()) {
        long long ms = elapsed();
        policy_.recordCacheHit(ctx.domain);   // feed domain hit rate for adaptive threshold
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

        if (s >= validator_.thresholdForDomain(ctx.domain)) {
            // Skip entries that have been tombstoned by a concurrent DELETE.
            if (redis_.isTombstoned(candidate.id)) {
                spdlog::debug("L2 candidate id={} tombstoned, skipping", candidate.id);
                continue;
            }

            // Reconstruct the full response: read slot values from Redis and
            // call Templatizer::render() to fill {{SLOT_N}} placeholders.
            // Falls back to the raw template if slots are unavailable (e.g. expired).
            std::string rendered = candidate.template_str;
            auto slots_json = redis_.get("lc:slots:" + candidate.domain + ":" + candidate.id);
            if (slots_json.has_value()) {
                try {
                    auto slot_values =
                        nlohmann::json::parse(*slots_json).get<std::vector<std::string>>();
                    rendered = builder::Templatizer::render(candidate.template_str,
                                                            slot_values);
                } catch (const std::exception& e) {
                    spdlog::warn("L2 render failed for id={}: {}", candidate.id, e.what());
                }
            } else {
                spdlog::debug("L2 slot values expired for id={}, returning raw template",
                              candidate.id);
            }

            // Backfill L1 with the rendered (slot-filled) response
            redis_.set(l1_key, rendered, L1_TTL_SECONDS);
            policy_.recordCacheHit(ctx.domain);   // adaptive threshold feedback
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
    build_req.query               = req.query;
    build_req.context             = req.context;
    build_req.domain              = ctx.domain;
    build_req.user_id             = req.user_id;
    build_req.signature_hash      = ctx.signature_hash;
    build_req.context_fingerprint = ctx.context_fingerprint;
    build_req.llm_response        = answer;
    build_req.embedding           = ctx.embedding;
    builder_.enqueue(build_req);

    long long ms = elapsed();
    spdlog::info("LLM latency_ms={} correlation_id={}", ms, req.correlation_id);
    return QueryResponse{answer, false, 0.0, "", ms};
}

} // namespace lettucecache::orchestrator
