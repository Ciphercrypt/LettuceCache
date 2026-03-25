#pragma once
#include "../orchestrator/ContextBuilder.h"
#include "../cache/FaissVectorStore.h"

namespace lettucecache::quantization { class TurboQuantizer; }

namespace lettucecache::validation {

class ValidationService {
public:
    // tq: optional TurboQuantizer for unbiased inner-product scoring.
    //     When a candidate has tq_codes populated, the cosine term uses
    //     TurboQuant_prod instead of full-precision dot product.
    explicit ValidationService(double threshold = 0.85,
                                quantization::TurboQuantizer* tq = nullptr);

    // Returns composite score in [0, 1].
    double score(const orchestrator::ContextObject& query_ctx,
                 const cache::CacheEntry& candidate) const;

    bool isHit(const orchestrator::ContextObject& query_ctx,
               const cache::CacheEntry& candidate) const;

private:
    double                        threshold_;
    quantization::TurboQuantizer* tq_;  // non-owning; nullptr = TQ disabled

    // Cosine similarity.
    // If candidate.tq_codes is non-empty and tq_ != nullptr, uses TurboQuant
    // unbiased inner-product estimate (eliminates MSE-bias at low bit-widths).
    // Otherwise: dot product of L2-normalised vectors (= cosine, no sqrt needed).
    double cosineSimilarity(const std::vector<float>& query,
                            const cache::CacheEntry& candidate) const;

    static double contextSignatureScore(const orchestrator::ContextObject& ctx,
                                        const cache::CacheEntry& candidate);

    static double domainScore(const orchestrator::ContextObject& ctx,
                               const cache::CacheEntry& candidate);

    static constexpr double W_COSINE  = 0.60;
    static constexpr double W_CONTEXT = 0.25;
    static constexpr double W_DOMAIN  = 0.15;
};

} // namespace lettucecache::validation
