#pragma once
#include <string>
#include <vector>

namespace lettucecache::embedding {

class EmbeddingClient {
public:
    explicit EmbeddingClient(const std::string& base_url);

    // Returns empty vector on failure.
    std::vector<float> embed(const std::string& text);

    // Batch embed — returns one embedding per input string.
    std::vector<std::vector<float>> embedBatch(const std::vector<std::string>& texts);

    bool healthCheck();

private:
    std::string base_url_;

    static constexpr int TIMEOUT_MS = 5000;
};

} // namespace lettucecache::embedding
