#pragma once
#include <string>
#include <vector>

namespace lettucecache::llm {

class LLMAdapter {
public:
    virtual ~LLMAdapter() = default;

    virtual std::string complete(
        const std::string& prompt,
        const std::vector<std::string>& context
    ) = 0;

    virtual bool isAvailable() = 0;
};

} // namespace lettucecache::llm
