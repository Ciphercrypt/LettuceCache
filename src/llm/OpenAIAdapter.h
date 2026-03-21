#pragma once
#include "LLMAdapter.h"
#include <string>

namespace lettucecache::llm {

class OpenAIAdapter final : public LLMAdapter {
public:
    explicit OpenAIAdapter(const std::string& api_key,
                            const std::string& model = "gpt-3.5-turbo",
                            int max_tokens = 512,
                            double temperature = 0.2);

    std::string complete(
        const std::string& prompt,
        const std::vector<std::string>& context
    ) override;

    bool isAvailable() override;

private:
    std::string api_key_;
    std::string model_;
    int max_tokens_;
    double temperature_;

    static constexpr const char* OPENAI_CHAT_URL =
        "https://api.openai.com/v1/chat/completions";
    static constexpr int TIMEOUT_MS = 30000;
};

} // namespace lettucecache::llm
