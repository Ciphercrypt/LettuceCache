#pragma once
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace lettucecache::cache {

struct CacheEntry;  // forward decl

// Abstract vector store interface.
// Concrete implementations: FaissVectorStore (in-process), MilvusVectorStore (future).
class IVectorStore {
public:
    virtual ~IVectorStore() = default;

    virtual void add(const CacheEntry& entry) = 0;
    virtual std::vector<CacheEntry> search(const std::vector<float>& query, int top_k = 5) = 0;
    // Returns the CacheEntry for the given entry_id, or nullopt if not found.
    // Used before remove() so callers can retrieve sig_hash/domain for key cleanup.
    virtual std::optional<CacheEntry> find(const std::string& entry_id) const = 0;
    virtual bool remove(const std::string& entry_id) = 0;
    virtual size_t size() const = 0;
    virtual void persist() = 0;
};

} // namespace lettucecache::cache
