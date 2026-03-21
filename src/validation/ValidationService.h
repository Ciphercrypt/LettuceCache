#pragma once
#include "../orchestrator/ContextBuilder.h"
#include "../cache/FaissVectorStore.h"

namespace lettucecache::validation {

class ValidationService {
public:
    explicit ValidationService(double threshold = 0.85);

    // Returns composite score in [0, 1].
    double score(
        const orchestrator::ContextObject& query_ctx,
        const cache::CacheEntry& candidate
    ) const;

    bool isHit(
        const orchestrator::ContextObject& query_ctx,
        const cache::CacheEntry& candidate
    ) const;

private:
    double threshold_;

    static double cosineSimilarity(
        const std::vector<float>& a,
        const std::vector<float>& b);

    static double contextSignatureScore(
        const orchestrator::ContextObject& ctx,
        const cache::CacheEntry& candidate);

    static double domainScore(
        const orchestrator::ContextObject& ctx,
        const cache::CacheEntry& candidate);

    // Composite weights
    static constexpr double W_COSINE  = 0.60;
    static constexpr double W_CONTEXT = 0.25;
    static constexpr double W_DOMAIN  = 0.15;
};

} // namespace lettucecache::validation
