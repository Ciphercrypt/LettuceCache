#pragma once
#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "../cache/FaissVectorStore.h"

namespace lettucecache::builder {

// Config is a free struct at namespace scope to avoid clang's restriction on
// default member initializers in nested structs referenced in default arguments.
struct IntelligentAdmissionConfig {
    float w_frequency  = 0.30f;
    float w_cost       = 0.25f;
    float w_quality    = 0.25f;
    float w_novelty    = 0.20f;

    float base_threshold      = 0.42f;
    float threshold_range     = 0.08f;
    float freq_half_life_secs = 120.0f;
    int   freq_window_secs    = 300;

    float near_duplicate_threshold = 0.94f;
    float novelty_soft_start       = 0.82f;
    int   domain_min_observations  = 20;
};

// ─────────────────────────────────────────────────────────────────────────────
// IntelligentAdmissionPolicy — multi-signal Cache Value Function (CVF)
//
// Replaces the binary frequency gate of AdmissionController with a composite
// score drawn from four signals, synthesising the open gap identified in the
// literature survey (Intelligent Cache Admission for LLM Semantic Caching):
//
//   CVF(entry) = w_freq  × frequency_score      [0-1]
//              + w_cost  × generation_cost_score [0-1]
//              + w_qual  × quality_score         [0-1]  (caller-provided)
//              + w_novel × novelty_score         [0-1]  (MMR vs. existing cache)
//
// Signal design rationale (paper references):
//
//  frequency_score  — Exponential-decay recency weighting (LeCaR/CACHEUS).
//                     Recent requests carry more weight than old ones;
//                     normalised via 1 - exp(-sum) to [0,1].
//
//  generation_cost  — "Cost-aware admission" (CacheSack, LHD, Cost-Aware
//                     Caching). Expensive model responses (GPT-4-class) are
//                     more valuable to cache than cheap ones. Estimated from
//                     approximate token count × model-tier multiplier.
//
//  quality_score    — Passed from ResponseQualityFilter. Factual, structured
//                     responses score high; conversational/refusals already
//                     rejected before reaching this layer.
//
//  novelty_score    — MMR-inspired diversity (Carbonell & Goldstein SIGIR 1998).
//                     1 − max_cosine(embedding, existing_cache). Near-duplicate
//                     entries (sim > NEAR_DUPLICATE_THRESHOLD) are hard-rejected;
//                     partial overlaps receive a graduated penalty.
//
// Adaptive per-domain threshold (vCache / SCALM insight):
//   Domains with historically high cache hit rates (cache is working) receive
//   a slightly relaxed threshold so more entries flow in. Domains with low hit
//   rates tighten the threshold to avoid polluting the index.
//
// Entry point:
//   1. Call recordRequest() on every query (hit or miss) to update frequency.
//   2. Call recordCacheHit() on L1/L2 hits for adaptive threshold tracking.
//   3. Call evaluate() in the async write path (CacheBuilderWorker) to get the
//      admit/reject decision with a full breakdown.
// ─────────────────────────────────────────────────────────────────────────────

class IntelligentAdmissionPolicy {
public:
    struct Decision {
        bool   should_admit;
        float  value;              // composite CVF score in [0,1]
        float  freq_score;
        float  cost_score;
        float  novel_score;
        float  effective_threshold;
        std::string reason;
    };

    using Config = IntelligentAdmissionConfig;

    IntelligentAdmissionPolicy(cache::FaissVectorStore& faiss,
                                const Config& config = Config{});

    // Called on every incoming query (before L1/L2 lookup) to update frequency.
    void recordRequest(const std::string& sig_hash, const std::string& domain);

    // Called whenever a cache hit is served (L1 or L2) to update hit rates.
    void recordCacheHit(const std::string& domain);

    // Full admission decision combining all four CVF signals.
    // quality_score: value from ResponseQualityFilter::Result::score (caller provides).
    // embedding: the query embedding for novelty check (may be empty → skip novelty).
    Decision evaluate(const std::string& sig_hash,
                      const std::string& response,
                      const std::string& domain,
                      const std::vector<float>& embedding,
                      float quality_score) const;

private:
    cache::FaissVectorStore& faiss_;
    Config cfg_;

    // ── Frequency state ────────────────────────────────────────────────────
    struct FreqEntry {
        std::vector<std::chrono::steady_clock::time_point> timestamps;
    };
    mutable std::unordered_map<std::string, FreqEntry> freq_map_;

    // ── Domain adaptive threshold state ───────────────────────────────────
    struct DomainStats {
        int requests{0};
        int hits{0};
        float hit_rate() const {
            return requests > 0 ? static_cast<float>(hits) / requests : 0.3f;
        }
    };
    mutable std::unordered_map<std::string, DomainStats> domain_stats_;

    // Single shared_mutex guards both freq_map_ and domain_stats_.
    // Shared lock for read-only signals (frequencyScore, effectiveThreshold),
    // exclusive lock for mutations (recordRequest, recordCacheHit, evictExpired).
    // Eliminates dual-mutex deadlock risk from independent lock acquisition order.
    mutable std::shared_mutex policy_mutex_;

    // ── Signal computations ───────────────────────────────────────────────
    float frequencyScore(const std::string& sig_hash) const;
    float costScore(const std::string& response) const;
    float noveltyScore(const std::vector<float>& embedding) const;
    float effectiveThreshold(const std::string& domain) const;

    void evictExpiredFrequency() const;

    static float modelTierMultiplier();           // reads LLM_MODEL env var
    static float cosineSim(const std::vector<float>& a, const std::vector<float>& b);
    static float sigmoid(float x);
};

} // namespace lettucecache::builder
