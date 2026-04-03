#pragma once
#include <string>

namespace lettucecache::builder {

// ResponseQualityFilter decides whether an LLM response is worth caching.
//
// Signals evaluated (all run even if one would reject, for logging):
//
//  1. Length          — very short responses are almost always conversational
//  2. Conversational  — openers/closers that add no retrievable information
//  3. Refusal/error   — "I don't know", "I cannot", "I'm sorry" responses
//  4. Session-bound   — references prior turns ("as I mentioned earlier")
//  5. Dynamic content — time-sensitive markers ("today", "right now")
//  6. Personal density — fraction of personal pronouns (your/you/my/I)
//  7. Structural boost — lists, headers, multi-sentence factual content
//
// Composite score = clamp(
//     base_length_score
//   - conversational_penalty
//   - refusal_penalty
//   - session_penalty
//   - dynamic_penalty
//   - personal_penalty
//   + structural_boost,
//   0.0f, 1.0f)
//
// Responses scoring below `threshold` (default 0.40) are rejected.
//
// Note: purely conversational short responses are hard-rejected with
// score 0.0 regardless of threshold.

class ResponseQualityFilter {
public:
    struct Result {
        float  score;         // 0.0 = definitely skip, 1.0 = excellent to cache
        bool   should_cache;  // score >= threshold
        std::string reason;   // primary reason for decision (for structured logs)
    };

    explicit ResponseQualityFilter(float threshold = 0.40f);

    // Evaluate whether `response` is worth caching for the given `query`.
    // `domain` is used for domain-aware refusal whitelisting — patterns that
    // look like refusals in general text may be valid cacheable responses in
    // specific regulated domains (e.g. banking security responses).
    Result evaluate(const std::string& response,
                    const std::string& query,
                    const std::string& domain = {}) const;

    float threshold() const { return threshold_; }

private:
    float threshold_;

    // Individual signal helpers (all operate on lowercase text)
    float lengthScore(const std::string& response) const;
    float conversationalPenalty(const std::string& lower,
                                 size_t char_count) const;
    float refusalPenalty(const std::string& lower,
                         const std::string& domain) const;
    float sessionBoundPenalty(const std::string& lower) const;
    float dynamicContentPenalty(const std::string& lower) const;
    float personalDensityPenalty(const std::string& lower,
                                  size_t word_count) const;
    float structuralBoost(const std::string& response,
                           const std::string& lower,
                           size_t word_count) const;

    static std::string toLower(const std::string& s);
    static size_t countWords(const std::string& s);
    static bool startsWith(const std::string& s, const std::string& prefix);
    static size_t countOccurrences(const std::string& text,
                                    const std::string& needle);
};

} // namespace lettucecache::builder
