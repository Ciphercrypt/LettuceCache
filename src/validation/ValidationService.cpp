#include "ValidationService.h"
#include "../quantization/TurboQuantizer.h"
#include <cmath>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace lettucecache::validation {

ValidationService::ValidationService(double threshold,
                                      quantization::TurboQuantizer* tq)
    : threshold_(threshold), tq_(tq)
{
    // Parse DOMAIN_THRESHOLDS env var: JSON object e.g. {"faq":0.75,"compliance":0.92}
    const char* env = std::getenv("DOMAIN_THRESHOLDS");
    if (env) {
        try {
            auto obj = nlohmann::json::parse(env);
            for (auto& [domain, val] : obj.items()) {
                domain_thresholds_[domain] = val.get<double>();
            }
            spdlog::info("ValidationService: loaded {} domain threshold overrides",
                         domain_thresholds_.size());
        } catch (const std::exception& e) {
            spdlog::warn("ValidationService: failed to parse DOMAIN_THRESHOLDS: {}", e.what());
        }
    }
}

double ValidationService::thresholdForDomain(const std::string& domain) const {
    auto it = domain_thresholds_.find(domain);
    return (it != domain_thresholds_.end()) ? it->second : threshold_;
}

// ──────────────────────────────────────────────────────────────────────────────
// cosineSimilarity
//
// Two paths:
//  1. TurboQuant path: unbiased inner-product via TQ_prod codes.
//     E[TQ_ip] = <query, stored>  (TurboQuant paper Theorem 2).
//     Eliminates the MSE bias that shifts the 0.85 threshold — at 1-bit MSE
//     every cosine score would be scaled by 2/π ≈ 0.64, making 0.85 unreachable.
//
//  2. Fast dot-product: the Python sidecar L2-normalises all embeddings
//     (normalize_embeddings=True), so cosine = dot product.
//     The original implementation computed sqrt(norm_a) * sqrt(norm_b) = 1.0
//     on every call — wasted cycles eliminated here.
// ──────────────────────────────────────────────────────────────────────────────
double ValidationService::cosineSimilarity(const std::vector<float>& query,
                                            const cache::CacheEntry& candidate) const
{
    if (tq_ && !candidate.tq_codes.empty()) {
        return static_cast<double>(tq_->inner_product(query, candidate.tq_codes));
    }

    if (query.empty() || candidate.embedding.empty() ||
        query.size() != candidate.embedding.size()) return 0.0;

    double dot = 0.0;
    for (size_t i = 0; i < query.size(); ++i) {
        dot += static_cast<double>(query[i]) *
               static_cast<double>(candidate.embedding[i]);
    }
    return dot;
}

double ValidationService::contextSignatureScore(
    const orchestrator::ContextObject& ctx,
    const cache::CacheEntry& candidate)
{
    // Compare context_fingerprint (deployment context only, no query intent).
    // Two differently-phrased queries in the same deployment context share the
    // same fingerprint — FAISS cosine handles query-level semantic matching.
    return (ctx.context_fingerprint == candidate.context_signature) ? 1.0 : 0.0;
}

double ValidationService::domainScore(
    const orchestrator::ContextObject& ctx,
    const cache::CacheEntry& candidate)
{
    return (ctx.domain == candidate.domain) ? 1.0 : 0.0;
}

double ValidationService::score(const orchestrator::ContextObject& query_ctx,
                                 const cache::CacheEntry& candidate) const
{
    double cos_sim   = cosineSimilarity(query_ctx.embedding, candidate);
    double ctx_score = contextSignatureScore(query_ctx, candidate);
    double dom_score = domainScore(query_ctx, candidate);

    double final_score = W_COSINE  * cos_sim
                       + W_CONTEXT * ctx_score
                       + W_DOMAIN  * dom_score;

    spdlog::debug("Validation: cosine={:.4f} ctx={:.1f} domain={:.1f} score={:.4f} tq={}",
                  cos_sim, ctx_score, dom_score, final_score,
                  (tq_ && !candidate.tq_codes.empty()) ? "yes" : "no");
    return final_score;
}

bool ValidationService::isHit(const orchestrator::ContextObject& query_ctx,
                               const cache::CacheEntry& candidate) const
{
    return score(query_ctx, candidate) >= threshold_;
}

} // namespace lettucecache::validation
