#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

namespace lettucecache::cache {

struct CacheEntry {
    std::string id;
    std::vector<float> embedding;
    std::string context_signature;
    std::string template_str;
    std::string domain;
    int64_t faiss_id{-1};
    long long created_at{0};
};

class FaissVectorStore {
public:
    explicit FaissVectorStore(int dimension, const std::string& index_path);
    ~FaissVectorStore();

    FaissVectorStore(const FaissVectorStore&) = delete;
    FaissVectorStore& operator=(const FaissVectorStore&) = delete;

    void add(const CacheEntry& entry);
    std::vector<CacheEntry> search(const std::vector<float>& query_vec, int top_k = 5);
    bool remove(const std::string& entry_id);
    void persist();
    size_t size() const;

private:
    int dim_;
    std::string index_path_;
    std::unique_ptr<faiss::IndexFlatL2> quantizer_;
    std::unique_ptr<faiss::IndexIVFPQ> index_;
    std::unordered_map<int64_t, CacheEntry> id_to_entry_;
    std::unordered_map<std::string, int64_t> entry_id_to_faiss_id_;
    int64_t next_id_{0};
    mutable std::mutex mutex_;
    bool trained_{false};

    static constexpr int NLIST        = 100;
    static constexpr int M_PQ         = 8;
    static constexpr int NBITS        = 8;
    static constexpr int NPROBE       = 10;
    static constexpr int MIN_TRAIN_VEC = 256;

    void ensureTrained(const std::vector<float>& new_vec);
    void loadFromDisk();
};

} // namespace lettucecache::cache
