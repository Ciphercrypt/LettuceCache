#pragma once
#include "IVectorStore.h"
#include <optional>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <atomic>

namespace lettucecache::quantization { class TurboQuantizer; }

namespace lettucecache::cache {

// CacheEntry holds everything needed to score and return a candidate.
// tq_codes: TurboQuant_prod compressed embedding (empty = TQ disabled).
//   When populated, ValidationService uses unbiased inner-product estimation
//   instead of full-precision cosine (fixing the MSE-bias problem at low bits).
struct CacheEntry {
    std::string          id;
    std::vector<float>   embedding;         // full-precision, used for FAISS add
    std::vector<uint8_t> tq_codes;          // TurboQuant codes (optional)
    // context_signature: context_fingerprint (deployment context, no query intent).
    // Used by ValidationService for L2 context matching.
    std::string          context_signature;
    // signature_hash: full key (intent + all dims). Used by DELETE to reconstruct
    // the correct lc:l1:{sig_hash} Redis key for cleanup.
    std::string          signature_hash;
    std::string          template_str;
    std::string          domain;
    int64_t              faiss_id{-1};
    long long            created_at{0};
};

// FaissVectorStore — in-process FAISS-backed semantic vector store.
//
// Key improvements vs original:
//   • std::shared_mutex: concurrent readers (search) / exclusive writer (add/remove)
//   • Metadata persistence: id_to_entry_ serialised to <index_path>.meta.json
//     alongside the FAISS binary. Fixes the critical "L2 dead after restart" bug.
//   • Implements IVectorStore for future Milvus migration.
//   • Optional TurboQuantizer: if set, TQ codes computed on add() and stored
//     in CacheEntry.tq_codes so ValidationService can use unbiased inner products.
class FaissVectorStore : public IVectorStore {
public:
    explicit FaissVectorStore(int dimension, const std::string& index_path,
                               quantization::TurboQuantizer* tq = nullptr);
    ~FaissVectorStore() override;

    FaissVectorStore(const FaissVectorStore&)            = delete;
    FaissVectorStore& operator=(const FaissVectorStore&) = delete;

    // IVectorStore
    void add(const CacheEntry& entry) override;
    std::vector<CacheEntry> search(const std::vector<float>& query, int top_k = 5) override;
    std::optional<CacheEntry> find(const std::string& entry_id) const override;
    bool   remove(const std::string& entry_id) override;
    void   persist() override;
    size_t size() const override;

private:
    int         dim_;
    std::string index_path_;
    std::string meta_path_;   // index_path_ + ".meta.json"

    quantization::TurboQuantizer* tq_;  // non-owning; nullptr = TQ disabled

    // Phase 1: exact search until MIN_IVF_TRAIN_VEC real vectors are available.
    // IndexIDMap wraps IndexFlatIP to add add_with_ids / remove_ids support,
    // which IndexFlatIP alone does not provide.
    std::unique_ptr<faiss::IndexIDMap> flat_index_;

    // Phase 2: IVF+PQ ANN index — activated once training threshold is met
    std::unique_ptr<faiss::IndexFlatL2> ivf_quantizer_;
    std::unique_ptr<faiss::IndexIVFPQ>  ivf_index_;
    bool ivf_trained_{false};

    std::unordered_map<int64_t, CacheEntry>   id_to_entry_;
    std::unordered_map<std::string, int64_t>  entry_id_to_faiss_id_;
    int64_t next_id_{0};

    // shared_mutex: N concurrent search() readers; exclusive add/remove/persist
    mutable std::shared_mutex rw_mutex_;

    static constexpr int NLIST             = 100;
    static constexpr int M_PQ              = 8;
    static constexpr int NBITS             = 8;
    static constexpr int NPROBE            = 10;
    // 39 × NLIST — minimum real vectors for stable IVF cluster centroids
    static constexpr int MIN_IVF_TRAIN_VEC = 39 * NLIST;  // 3900

    void migrateToIVF();   // called under exclusive lock when threshold crossed
    void loadFromDisk();
    void saveMetadata() const;
    void loadMetadata();
};

} // namespace lettucecache::cache
