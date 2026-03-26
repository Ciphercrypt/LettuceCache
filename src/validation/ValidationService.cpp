#include "ValidationService.h"
#include <cmath>
#include <spdlog/spdlog.h>

namespace lettucecache::validation {

ValidationService::ValidationService(double threshold)
    : threshold_(threshold) {}

double ValidationService::cosineSimilarity(
    const std::vector<float>& a,
    const std::vector<float>& b)
{
    // Embeddings are L2-normalised by the Python sidecar - dot product == cosine.
    // Skips the two sqrt calls that were always computing sqrt(1.0) * sqrt(1.0).
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.0;
    double dot = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    return dot;
}

double ValidationService::contextSignatureScore(
    const orchestrator::ContextObject& ctx,
    const cache::CacheEntry& candidate)
{
    // Exact SHA-256 match on intent:domain:user_scope
    return (ctx.signature_hash == candidate.context_signature) ? 1.0 : 0.0;
}

double ValidationService::domainScore(
    const orchestrator::ContextObject& ctx,
    const cache::CacheEntry& candidate)
{
    return (ctx.domain == candidate.domain) ? 1.0 : 0.0;
}

double ValidationService::score(
    const orchestrator::ContextObject& query_ctx,
    const cache::CacheEntry& candidate) const
{
    double cos_sim    = cosineSimilarity(query_ctx.embedding, candidate.embedding);
    double ctx_score  = contextSignatureScore(query_ctx, candidate);
    double dom_score  = domainScore(query_ctx, candidate);

    double final_score = W_COSINE  * cos_sim
                       + W_CONTEXT * ctx_score
                       + W_DOMAIN  * dom_score;

    spdlog::debug("Validation: cosine={:.3f} ctx={:.1f} domain={:.1f} final={:.3f}",
                  cos_sim, ctx_score, dom_score, final_score);
    return final_score;
}

bool ValidationService::isHit(
    const orchestrator::ContextObject& query_ctx,
    const cache::CacheEntry& candidate) const
{
    return score(query_ctx, candidate) >= threshold_;
}

} // namespace lettucecache::validation
