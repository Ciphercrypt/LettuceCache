#include "IntelligentAdmissionPolicy.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <shared_mutex>
#include <spdlog/spdlog.h>

namespace lettucecache::builder {

// ─────────────────────────────────────────────────────────────────────────────
// Static helpers
// ─────────────────────────────────────────────────────────────────────────────

float IntelligentAdmissionPolicy::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float IntelligentAdmissionPolicy::cosineSim(const std::vector<float>& a,
                                              const std::vector<float>& b) {
    // Embeddings are L2-normalised by the sidecar → dot product = cosine.
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return dot;
}

// Read LLM_MODEL env var and return a cost-tier multiplier.
// Based on approximate relative API pricing (scaled, not absolute).
float IntelligentAdmissionPolicy::modelTierMultiplier() {
    const char* model = std::getenv("LLM_MODEL");
    if (!model) return 1.0f;  // default: mini class
    std::string m(model);
    // GPT-4 class (expensive)
    if (m.find("gpt-4o") != std::string::npos && m.find("mini") == std::string::npos)
        return 3.0f;
    if (m.find("gpt-4-turbo") != std::string::npos) return 3.0f;
    if (m.find("gpt-4") != std::string::npos)       return 2.5f;
    // Mid-tier
    if (m.find("gpt-4o-mini") != std::string::npos) return 1.0f;
    if (m.find("claude-3-5") != std::string::npos)  return 2.0f;
    if (m.find("claude-3") != std::string::npos)    return 1.5f;
    // Cheap
    if (m.find("gpt-3.5") != std::string::npos)     return 0.4f;
    return 1.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

IntelligentAdmissionPolicy::IntelligentAdmissionPolicy(
    cache::FaissVectorStore& faiss, const Config& config)
    : faiss_(faiss), cfg_(config) {}

// ─────────────────────────────────────────────────────────────────────────────
// Request / hit tracking
// ─────────────────────────────────────────────────────────────────────────────

void IntelligentAdmissionPolicy::evictExpiredFrequency() const {
    // Caller must hold exclusive lock on policy_mutex_
    const auto now = std::chrono::steady_clock::now();
    for (auto it = freq_map_.begin(); it != freq_map_.end(); ) {
        auto& ts = it->second.timestamps;
        ts.erase(std::remove_if(ts.begin(), ts.end(), [&](const auto& t) {
            return std::chrono::duration_cast<std::chrono::seconds>(now - t).count()
                   > cfg_.freq_window_secs;
        }), ts.end());
        if (ts.empty()) it = freq_map_.erase(it);
        else            ++it;
    }
}

void IntelligentAdmissionPolicy::recordRequest(const std::string& sig_hash,
                                                const std::string& domain) {
    std::unique_lock<std::shared_mutex> lock(policy_mutex_);
    evictExpiredFrequency();
    freq_map_[sig_hash].timestamps.push_back(std::chrono::steady_clock::now());
    domain_stats_[domain].requests++;
}

void IntelligentAdmissionPolicy::recordCacheHit(const std::string& domain) {
    std::unique_lock<std::shared_mutex> lock(policy_mutex_);
    auto& stats = domain_stats_[domain];
    stats.requests++;
    stats.hits++;
}

// ─────────────────────────────────────────────────────────────────────────────
// Signal: frequency score
//
// Exponential-decay weighting (LeCaR / CACHEUS insight): recent requests
// count more than old ones. Score = sum of exp(-λ × age_seconds) over all
// timestamps in the window, then normalised to [0,1] via 1 - exp(-sum).
//
// Half-life 120 s → a single request just now maps to ≈0.63; two fresh
// requests → ≈0.87; one request 5 minutes ago → ≈0.33.
// ─────────────────────────────────────────────────────────────────────────────
float IntelligentAdmissionPolicy::frequencyScore(const std::string& sig_hash) const {
    std::shared_lock<std::shared_mutex> lock(policy_mutex_);
    auto it = freq_map_.find(sig_hash);
    if (it == freq_map_.end()) return 0.0f;

    const float lambda = std::log(2.0f) / cfg_.freq_half_life_secs;
    const auto  now    = std::chrono::steady_clock::now();
    float decay_sum    = 0.0f;

    for (const auto& ts : it->second.timestamps) {
        const float age = static_cast<float>(
            std::chrono::duration_cast<std::chrono::seconds>(now - ts).count());
        decay_sum += std::exp(-lambda * age);
    }
    // 1 − exp(−x) maps [0,∞) → [0,1): asymptotes at 1.0
    return 1.0f - std::exp(-decay_sum);
}

// ─────────────────────────────────────────────────────────────────────────────
// Signal: generation cost score
//
// Based on CacheSack (cost-aware knapsack) and LHD (utility per space).
// More expensive responses are more valuable to cache: they save more
// money per cache hit. Approximated from character count (≈4 chars/token)
// and a model-tier multiplier.
//
// sigmoid((weighted_tokens − 150) / 350) maps:
//   ~50 tokens × tier 0.4  →  ≈0.37
//   ~150 tokens × tier 1.0 →  ≈0.50
//   ~300 tokens × tier 1.0 →  ≈0.65
//   ~200 tokens × tier 3.0 →  ≈0.80
// ─────────────────────────────────────────────────────────────────────────────
float IntelligentAdmissionPolicy::costScore(const std::string& response) const {
    const float approx_tokens = static_cast<float>(response.size()) / 4.0f;
    const float tier          = modelTierMultiplier();
    const float weighted      = approx_tokens * tier;
    return sigmoid((weighted - 150.0f) / 350.0f);
}

// ─────────────────────────────────────────────────────────────────────────────
// Signal: novelty score (MMR-inspired, Carbonell & Goldstein SIGIR 1998)
//
// Penalises entries whose semantic neighbourhood overlaps heavily with what
// is already cached — analogous to MMR penalising documents similar to
// already-selected documents. Prevents the cache from filling with
// near-duplicate entries covering the same semantic cluster.
//
// novelty = 1 − max_cosine_sim(new_embedding, existing_cache)
//
// Hard-reject zone: cosine > near_duplicate_threshold (e.g. 0.94)
//   → the existing entry already covers this query; adding another copy
//     consumes space without improving recall diversity.
//
// Soft ramp: cosine in [novelty_soft_start, near_duplicate_threshold]
//   → partial overlap; novelty score decreases linearly to 0.
//
// If FAISS is empty or untrained, returns 1.0 (fully novel by default).
// ─────────────────────────────────────────────────────────────────────────────
float IntelligentAdmissionPolicy::noveltyScore(const std::vector<float>& embedding) const {
    if (embedding.empty()) return 1.0f;

    auto candidates = faiss_.search(embedding, 1);
    if (candidates.empty()) return 1.0f;   // nothing in cache yet

    const float max_sim = cosineSim(embedding, candidates[0].embedding);

    // Hard-reject zone: existing entry is semantically identical
    if (max_sim >= cfg_.near_duplicate_threshold) return 0.0f;

    // Soft penalty ramp between soft_start and near_duplicate_threshold
    if (max_sim >= cfg_.novelty_soft_start) {
        const float range    = cfg_.near_duplicate_threshold - cfg_.novelty_soft_start;
        const float overlap  = max_sim - cfg_.novelty_soft_start;
        return std::max(0.0f, 1.0f - (overlap / range));
    }

    // Genuinely novel content
    return std::max(0.0f, 1.0f - max_sim);
}

// ─────────────────────────────────────────────────────────────────────────────
// Adaptive per-domain threshold (vCache / SCALM insight)
//
// Domains where the cache is already working well (high hit rate) receive a
// slightly relaxed threshold, encouraging more entries. Domains where the
// cache rarely hits tighten the threshold — the cache content there is not
// reusable, so we avoid polluting it with low-value entries.
// ─────────────────────────────────────────────────────────────────────────────
float IntelligentAdmissionPolicy::effectiveThreshold(const std::string& domain) const {
    std::shared_lock<std::shared_mutex> lock(policy_mutex_);
    auto it = domain_stats_.find(domain);
    if (it == domain_stats_.end() ||
        it->second.requests < cfg_.domain_min_observations)
        return cfg_.base_threshold;

    const float hr    = it->second.hit_rate();
    // hit_rate 0.30 → no adjustment (neutral)
    // hit_rate 0.60 → relax by full range (lower threshold)
    // hit_rate 0.05 → tighten by full range (raise threshold)
    const float delta = (hr - 0.30f) * (cfg_.threshold_range / 0.30f);
    return std::clamp(cfg_.base_threshold - delta,
                      cfg_.base_threshold - cfg_.threshold_range,
                      cfg_.base_threshold + cfg_.threshold_range);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main evaluation — composite Cache Value Function
// ─────────────────────────────────────────────────────────────────────────────
IntelligentAdmissionPolicy::Decision IntelligentAdmissionPolicy::evaluate(
    const std::string& sig_hash,
    const std::string& response,
    const std::string& domain,
    const std::vector<float>& embedding,
    float quality_score) const
{
    // ── Compute individual signals ─────────────────────────────────────────
    const float freq   = frequencyScore(sig_hash);
    const float cost   = costScore(response);
    const float novel  = noveltyScore(embedding);
    const float thresh = effectiveThreshold(domain);

    // ── Hard reject: near-duplicate already in cache ───────────────────────
    // (novel == 0 means cosine ≥ near_duplicate_threshold)
    if (novel == 0.0f) {
        return {false, 0.0f, freq, cost, novel, thresh,
                "near-duplicate: semantically identical entry already cached"};
    }

    // ── Composite CVF ─────────────────────────────────────────────────────
    const float value = cfg_.w_frequency * freq
                      + cfg_.w_cost      * cost
                      + cfg_.w_quality   * quality_score
                      + cfg_.w_novelty   * novel;

    const bool admit = value >= thresh;

    // ── Reason string for structured logs ─────────────────────────────────
    std::string reason;
    if (admit) {
        if (novel > 0.80f && value > 0.65f)
            reason = "high-value novel entry";
        else if (cost > 0.70f)
            reason = "expensive generation — high cache ROI";
        else
            reason = "passes CVF threshold";
    } else {
        // Identify the dominant bottleneck signal
        if (freq   < 0.40f) reason = "low frequency (not yet seen enough)";
        else if (novel < 0.30f) reason = "low novelty (semantically covered by existing entry)";
        else if (cost  < 0.30f) reason = "low generation cost (cheap to re-generate)";
        else                    reason = "composite CVF below domain threshold";
    }

    spdlog::info("IntelligentAdmission: sig={} value={:.3f}(f={:.2f} c={:.2f} "
                 "q={:.2f} n={:.2f}) thresh={:.2f} admit={} | {}",
                 sig_hash.substr(0, 8), value, freq, cost, quality_score, novel,
                 thresh, admit, reason);

    return {admit, value, freq, cost, novel, thresh, reason};
}

} // namespace lettucecache::builder
