#include "ResponseQualityFilter.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <spdlog/spdlog.h>

namespace lettucecache::builder {

// ──────────────────────────────────────────────────────────────────────────────
// Pattern tables (all lowercase, matched against lowercased response)
// ──────────────────────────────────────────────────────────────────────────────

// Openers that identify a response as purely conversational/social.
// Hard-reject when the entire response is short and dominated by these.
static const char* const CONV_OPENERS[] = {
    "sure!", "sure,", "sure.", "of course", "certainly!", "certainly,",
    "absolutely!", "absolutely,", "got it!", "got it,",
    "no problem", "happy to help", "i'd be happy", "i'd be glad",
    "great question", "that's a great question", "that's an excellent",
    "i understand", "i see what you mean", "i hear you",
    "i'm sorry to hear", "i apologize for",
    "hello!", "hello,", "hi there", "hey there", "hi!", "hi,",
    "thank you for", "thanks for asking", "thanks for your question",
    "you're welcome", "you're very welcome", "no worries",
    "is there anything else", "how can i assist", "how can i help you",
    "let me know if you", "feel free to ask", "don't hesitate to",
    nullptr
};

// Conversational closers that indicate the response is a social wrap-up,
// not factual content. Penalise responses dominated by these.
static const char* const CONV_CLOSERS[] = {
    "hope that helps", "hope this helps", "hope this was helpful",
    "please let me know", "feel free to ask", "if you have more questions",
    "happy to answer any", "i'm here if you need",
    "take care", "have a great day", "have a wonderful",
    nullptr
};

// Hard indicators that the response refuses, hedges, or admits ignorance.
// These are almost never worth caching since another request will get the
// same empty/incorrect answer.
static const char* const REFUSAL_MARKERS[] = {
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "i cannot", "i can't", "i'm unable to", "i am unable to",
    "i don't have access", "i do not have access",
    "i don't have information", "i'm not aware",
    "as an ai", "as an artificial intelligence", "as a language model",
    "i'm just an ai", "i cannot provide", "it's not possible for me",
    nullptr
};

// References to earlier turns — binding the response to a single session.
// These responses can never be generalised to a new request.
static const char* const SESSION_MARKERS[] = {
    "as i mentioned", "as i said", "as we discussed",
    "as mentioned earlier", "as stated earlier", "as noted above",
    "earlier i said", "earlier i mentioned", "i previously said",
    "in my previous", "in my last", "continuing from where",
    "following up on", "building on what i said",
    nullptr
};

// Time-sensitive markers whose presence means the response will go stale.
static const char* const DYNAMIC_MARKERS[] = {
    " today ", " today.", " today,", " today\n",
    " currently ", "right now", "at the moment", "at this moment",
    "as of now", "as of today", "as of this",
    " yesterday ", " tomorrow ", "this week", "this month", "this year",
    "just released", "just announced", "breaking news",
    "your order", "your account", "your balance", "your subscription",
    "your recent", "your current", "your latest",
    nullptr
};

// Domain-aware refusal whitelist.
// Maps domain → patterns that should NOT be treated as refusals in that domain.
// E.g. "I cannot confirm your balance without authentication" is a valid
// security response in banking, not an LLM ignorance refusal.
struct DomainRefusalWhitelist {
    const char* domain;
    const char* pattern;
};

static const DomainRefusalWhitelist DOMAIN_REFUSAL_WHITELIST[] = {
    {"banking",    "i cannot confirm"},
    {"banking",    "i'm unable to verify"},
    {"banking",    "i am unable to verify"},
    {"banking",    "cannot be disclosed"},
    {"finance",    "i cannot confirm"},
    {"finance",    "cannot be disclosed"},
    {"compliance", "i cannot provide"},
    {"compliance", "i'm unable to provide"},
    {"compliance", "i am unable to provide"},
    {"compliance", "cannot be shared"},
    {"security",   "i cannot confirm"},
    {"security",   "i'm unable to verify"},
    {"security",   "cannot authenticate"},
    {nullptr,      nullptr}
};

// Personal pronouns indicating a highly individualised response.
// High density → response is tailored to a specific person/session.
static const char* const PERSONAL_WORDS[] = {
    "your ", "you've ", "you're ", "you'll ", "you'd ", "yours ",
    "you've", "you're",
    nullptr
};

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────
std::string ResponseQualityFilter::toLower(const std::string& s) {
    std::string out(s.size(), '\0');
    std::transform(s.begin(), s.end(), out.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return out;
}

size_t ResponseQualityFilter::countWords(const std::string& s) {
    std::istringstream iss(s);
    std::string w;
    size_t n = 0;
    while (iss >> w) ++n;
    return n;
}

bool ResponseQualityFilter::startsWith(const std::string& s,
                                        const std::string& prefix) {
    return s.size() >= prefix.size() &&
           s.compare(0, prefix.size(), prefix) == 0;
}

size_t ResponseQualityFilter::countOccurrences(const std::string& text,
                                                const std::string& needle) {
    size_t count = 0, pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

// ──────────────────────────────────────────────────────────────────────────────
// Signal: length score  (base signal — very short = conversational)
// ──────────────────────────────────────────────────────────────────────────────
float ResponseQualityFilter::lengthScore(const std::string& response) const {
    const size_t n = response.size();
    if (n < 30)  return 0.05f;
    if (n < 60)  return 0.20f;
    if (n < 120) return 0.35f;
    if (n < 250) return 0.50f;
    if (n < 500) return 0.65f;
    if (n < 1000) return 0.75f;
    return 0.80f;
}

// ──────────────────────────────────────────────────────────────────────────────
// Signal: conversational penalty
// If the response starts with a conversational opener AND is short (< 120 chars),
// it's almost certainly a pure social response → hard-reject score.
// Long responses with openers get a softer penalty (opener is just fluff before
// the actual content: "Sure! Here are the steps:...")
// ──────────────────────────────────────────────────────────────────────────────
float ResponseQualityFilter::conversationalPenalty(const std::string& lower,
                                                    size_t char_count) const {
    // Check openers
    for (const char* const* p = CONV_OPENERS; *p != nullptr; ++p) {
        if (startsWith(lower, *p)) {
            // Short response dominated by the opener → very high penalty
            return char_count < 120 ? 0.70f : 0.20f;
        }
    }

    // Count how many closer phrases appear
    int closer_hits = 0;
    for (const char* const* p = CONV_CLOSERS; *p != nullptr; ++p) {
        if (lower.find(*p) != std::string::npos) ++closer_hits;
    }
    // One closer in a long response is normal; many closers signal a
    // response that is mostly social wrap-up
    if (closer_hits >= 2) return 0.25f;
    if (closer_hits == 1 && char_count < 200) return 0.20f;

    return 0.0f;
}

// ──────────────────────────────────────────────────────────────────────────────
// Signal: refusal / ignorance penalty
// Domain-aware: patterns in DOMAIN_REFUSAL_WHITELIST are not treated as
// refusals for their configured domain (e.g. banking security responses).
// ──────────────────────────────────────────────────────────────────────────────
float ResponseQualityFilter::refusalPenalty(const std::string& lower,
                                             const std::string& domain) const {
    for (const char* const* p = REFUSAL_MARKERS; *p != nullptr; ++p) {
        if (lower.find(*p) == std::string::npos) continue;

        // Check domain whitelist before applying penalty
        if (!domain.empty()) {
            bool whitelisted = false;
            for (const auto* w = DOMAIN_REFUSAL_WHITELIST; w->domain != nullptr; ++w) {
                if (domain == w->domain && lower.find(w->pattern) != std::string::npos) {
                    whitelisted = true;
                    break;
                }
            }
            if (whitelisted) continue;
        }
        return 0.60f;
    }
    return 0.0f;
}

// ──────────────────────────────────────────────────────────────────────────────
// Signal: session-bound penalty
// ──────────────────────────────────────────────────────────────────────────────
float ResponseQualityFilter::sessionBoundPenalty(const std::string& lower) const {
    for (const char* const* p = SESSION_MARKERS; *p != nullptr; ++p) {
        if (lower.find(*p) != std::string::npos) return 0.80f;
    }
    return 0.0f;
}

// ──────────────────────────────────────────────────────────────────────────────
// Signal: dynamic / time-sensitive content penalty
// ──────────────────────────────────────────────────────────────────────────────
float ResponseQualityFilter::dynamicContentPenalty(const std::string& lower) const {
    int hits = 0;
    for (const char* const* p = DYNAMIC_MARKERS; *p != nullptr; ++p) {
        if (lower.find(*p) != std::string::npos) ++hits;
    }
    return std::min(static_cast<float>(hits) * 0.15f, 0.50f);
}

// ──────────────────────────────────────────────────────────────────────────────
// Signal: personal pronoun density penalty
// Responses tailored to a specific person should not be served to others.
// Penalty kicks in when personal pronouns exceed 10% of all words.
// ──────────────────────────────────────────────────────────────────────────────
float ResponseQualityFilter::personalDensityPenalty(const std::string& lower,
                                                     size_t word_count) const {
    if (word_count == 0) return 0.0f;
    size_t personal_count = 0;
    for (const char* const* p = PERSONAL_WORDS; *p != nullptr; ++p) {
        personal_count += countOccurrences(lower, *p);
    }
    const float density = static_cast<float>(personal_count) /
                          static_cast<float>(word_count);
    if (density <= 0.08f) return 0.0f;
    // Linear ramp: 0.08 → 0.20 density maps to 0 → 0.35 penalty
    return std::min((density - 0.08f) / 0.12f * 0.35f, 0.35f);
}

// ──────────────────────────────────────────────────────────────────────────────
// Signal: structural / informational boost
// Factual, structured responses are the highest-value cache entries.
// Indicators: numbered lists, bullet points, code blocks, headers,
//             multi-paragraph structure, high word count with low social density.
// ──────────────────────────────────────────────────────────────────────────────
float ResponseQualityFilter::structuralBoost(const std::string& response,
                                              const std::string& lower,
                                              size_t word_count) const {
    float boost = 0.0f;

    // Numbered list items ("1. ", "2. ", etc.)
    size_t numbered = 0;
    for (size_t i = 0; i + 2 < response.size(); ++i) {
        if ((response[i] == '\n' || i == 0) &&
            std::isdigit(static_cast<unsigned char>(response[i + (i==0?0:1)])) &&
            response[i + (i==0?1:2)] == '.')
        {
            ++numbered;
        }
    }
    if (numbered >= 3) boost += 0.20f;
    else if (numbered >= 1) boost += 0.10f;

    // Bullet / dash list items
    size_t bullets = countOccurrences(response, "\n- ") +
                     countOccurrences(response, "\n* ") +
                     countOccurrences(response, "\n• ");
    if (bullets >= 3) boost += 0.15f;
    else if (bullets >= 1) boost += 0.07f;

    // Code blocks (markdown or indentation)
    if (lower.find("```") != std::string::npos) boost += 0.15f;

    // Multiple paragraphs (two or more blank lines)
    size_t double_newlines = countOccurrences(response, "\n\n");
    if (double_newlines >= 2) boost += 0.10f;
    else if (double_newlines >= 1) boost += 0.05f;

    // High word count with no social penalty already boosts length score;
    // add a small bonus for genuinely long, substantive responses
    if (word_count > 80) boost += 0.10f;

    // Headers
    if (response.find("\n## ") != std::string::npos ||
        response.find("\n### ") != std::string::npos ||
        response.find("\n# ") != std::string::npos)
        boost += 0.10f;

    return std::min(boost, 0.40f);
}

// ──────────────────────────────────────────────────────────────────────────────
// Main evaluation
// ──────────────────────────────────────────────────────────────────────────────
ResponseQualityFilter::ResponseQualityFilter(float threshold)
    : threshold_(threshold) {}

ResponseQualityFilter::Result ResponseQualityFilter::evaluate(
    const std::string& response,
    const std::string& query,
    const std::string& domain) const
{
    const size_t chars = response.size();
    const std::string lower = toLower(response);
    const size_t words = countWords(response);

    // ── Hard reject: empty response ─────────────────────────────────────────
    if (chars == 0)
        return {0.0f, false, "empty response"};

    // ── Compute all signals ──────────────────────────────────────────────────
    const float len    = lengthScore(response);
    const float conv   = conversationalPenalty(lower, chars);
    const float refus  = refusalPenalty(lower, domain);
    const float sess   = sessionBoundPenalty(lower);
    const float dyn    = dynamicContentPenalty(lower);
    const float pers   = personalDensityPenalty(lower, words);
    const float boost  = structuralBoost(response, lower, words);

    // ── Hard reject: session-bound or refusal responses ──────────────────────
    // These can never produce a generalisable cache entry.
    if (sess >= 0.80f)
        return {0.0f, false,
                "session-bound: response references a specific prior turn"};
    if (refus >= 0.60f)
        return {0.0f, false,
                "refusal/ignorance: LLM admits it cannot answer"};

    // ── Composite score ──────────────────────────────────────────────────────
    float score = len - conv - dyn - pers + boost;
    score = std::max(0.0f, std::min(1.0f, score));

    // ── Decision ─────────────────────────────────────────────────────────────
    bool cacheable = score >= threshold_;

    std::string reason;
    if (!cacheable) {
        if (conv  > 0.40f) reason = "conversational/social response";
        else if (dyn  > 0.30f) reason = "time-sensitive/dynamic content";
        else if (pers > 0.20f) reason = "highly personalised response";
        else                   reason = "insufficient information density";
    } else {
        if      (boost > 0.25f) reason = "structured factual content";
        else if (words > 60)    reason = "substantive response";
        else                    reason = "passes quality threshold";
    }

    spdlog::debug("ResponseQualityFilter: score={:.2f} (len={:.2f} conv=-{:.2f} "
                  "dyn=-{:.2f} sess=-{:.2f} pers=-{:.2f} boost=+{:.2f}) "
                  "cacheable={} reason={}",
                  score, len, conv, dyn, sess, pers, boost,
                  cacheable, reason);

    return {score, cacheable, reason};
}

} // namespace lettucecache::builder
