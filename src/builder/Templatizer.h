#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace lettucecache::builder {

// Templatizer extracts a reusable template from an LLM response by
// replacing high-entropy tokens (numbers, proper-nouns heuristic,
// dates, UUIDs) with {{SLOT_N}} placeholders.
// At serve-time, the orchestrator fills slots from the current query.
class Templatizer {
public:
    struct TemplateResult {
        std::string templ;                         // response with {{SLOT_N}} markers
        std::vector<std::string> slot_values;      // original values in slot order
    };

    TemplateResult templatize(const std::string& response) const;

    // Render: fill {{SLOT_N}} markers back with provided values.
    static std::string render(const std::string& templ,
                               const std::vector<std::string>& slot_values);

private:
    static bool isHighEntropy(const std::string& token);
    static bool isNumeric(const std::string& token);
    static bool looksLikeDate(const std::string& token);
    static bool looksLikeUUID(const std::string& token);
    static bool looksLikeProperNoun(const std::string& token);
};

} // namespace lettucecache::builder
