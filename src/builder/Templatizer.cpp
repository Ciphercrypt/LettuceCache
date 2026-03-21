#include "Templatizer.h"
#include <sstream>
#include <regex>
#include <algorithm>
#include <spdlog/spdlog.h>

namespace lettucecache::builder {

namespace {

// Tokenize preserving whitespace and punctuation as separate tokens.
std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    for (unsigned char c : text) {
        if (std::isspace(c)) {
            if (!cur.empty()) { tokens.push_back(cur); cur.clear(); }
            tokens.push_back(std::string(1, static_cast<char>(c)));
        } else if (std::ispunct(c) && c != '-' && c != '_') {
            if (!cur.empty()) { tokens.push_back(cur); cur.clear(); }
            tokens.push_back(std::string(1, static_cast<char>(c)));
        } else {
            cur += static_cast<char>(c);
        }
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

} // anonymous namespace

bool Templatizer::isNumeric(const std::string& token) {
    if (token.empty()) return false;
    // Matches integers, decimals, percentages like 3.14 or 42 or 100%
    static const std::regex num_re(R"(^-?\d+(\.\d+)?%?$)");
    return std::regex_match(token, num_re);
}

bool Templatizer::looksLikeDate(const std::string& token) {
    // Simple: contains digit and separator patterns common in dates
    static const std::regex date_re(
        R"(\b(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})\b)");
    return std::regex_search(token, date_re);
}

bool Templatizer::looksLikeUUID(const std::string& token) {
    static const std::regex uuid_re(
        R"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})");
    return std::regex_search(token, uuid_re);
}

bool Templatizer::looksLikeProperNoun(const std::string& token) {
    if (token.size() < 2) return false;
    // Heuristic: starts with uppercase, rest lowercase, length > 3
    if (!std::isupper(static_cast<unsigned char>(token[0]))) return false;
    bool rest_lower = true;
    for (size_t i = 1; i < token.size(); ++i) {
        if (!std::islower(static_cast<unsigned char>(token[i])) &&
            !std::isdigit(static_cast<unsigned char>(token[i]))) {
            rest_lower = false;
            break;
        }
    }
    return rest_lower && token.size() > 3;
}

bool Templatizer::isHighEntropy(const std::string& token) {
    return isNumeric(token) || looksLikeDate(token) ||
           looksLikeUUID(token) || looksLikeProperNoun(token);
}

Templatizer::TemplateResult Templatizer::templatize(const std::string& response) const {
    auto tokens = tokenize(response);
    std::ostringstream out;
    std::vector<std::string> slot_values;
    int slot_idx = 0;

    for (const auto& tok : tokens) {
        if (isHighEntropy(tok)) {
            out << "{{SLOT_" << slot_idx++ << "}}";
            slot_values.push_back(tok);
        } else {
            out << tok;
        }
    }

    TemplateResult result;
    result.templ       = out.str();
    result.slot_values = std::move(slot_values);
    spdlog::debug("Templatizer: {} slots extracted", result.slot_values.size());
    return result;
}

std::string Templatizer::render(const std::string& templ,
                                 const std::vector<std::string>& slot_values)
{
    std::string output = templ;
    for (size_t i = 0; i < slot_values.size(); ++i) {
        std::string placeholder = "{{SLOT_" + std::to_string(i) + "}}";
        size_t pos = output.find(placeholder);
        if (pos != std::string::npos) {
            output.replace(pos, placeholder.size(), slot_values[i]);
        }
    }
    return output;
}

} // namespace lettucecache::builder
