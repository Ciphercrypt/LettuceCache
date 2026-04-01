#include "FaissVectorStore.h"
#include "../quantization/TurboQuantizer.h"
#include <faiss/impl/FaissException.h>
#include <faiss/IndexFlat.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <chrono>
#include <fstream>
#include <stdexcept>

namespace lettucecache::cache {

// ──────────────────────────────────────────────────────────────────────────────
// Construction / Destruction
// ──────────────────────────────────────────────────────────────────────────────
FaissVectorStore::FaissVectorStore(int dimension,
                                   const std::string& index_path,
                                   quantization::TurboQuantizer* tq)
    : dim_(dimension), index_path_(index_path),
      meta_path_(index_path + ".meta.json"), tq_(tq)
{
    // Phase 1: always start with an exact flat index for perfect recall
    flat_index_    = std::make_unique<faiss::IndexFlatIP>(dim_);
    // Phase 2 components — constructed but unused until migration
    ivf_quantizer_ = std::make_unique<faiss::IndexFlatL2>(dim_);
    ivf_index_     = std::make_unique<faiss::IndexIVFPQ>(
        ivf_quantizer_.get(), dim_, NLIST, M_PQ, NBITS);
    ivf_index_->nprobe = NPROBE;

    std::ifstream test(index_path_);
    if (test.good()) {
        test.close();
        try {
            loadFromDisk();
            loadMetadata();
            spdlog::info("FaissVectorStore: loaded from disk, entries={} ivf={}",
                         id_to_entry_.size(), ivf_trained_ ? "yes" : "flat");
        } catch (const std::exception& e) {
            spdlog::warn("FaissVectorStore: could not load from disk ({}); starting fresh.",
                         e.what());
        }
    }

    if (tq_) {
        spdlog::info("FaissVectorStore: TurboQuant enabled, code_size={} bytes",
                     tq_->code_size());
    }
}

FaissVectorStore::~FaissVectorStore() {
    try { persist(); } catch (...) {}
}

// ──────────────────────────────────────────────────────────────────────────────
// migrateToIVF — called under exclusive lock when MIN_IVF_TRAIN_VEC crossed.
// Trains IVF+PQ on all real vectors already in flat_index_, then bulk-adds them.
// ──────────────────────────────────────────────────────────────────────────────
void FaissVectorStore::migrateToIVF() {
    const size_t n = id_to_entry_.size();
    spdlog::info("FaissVectorStore: migrating {} vectors from flat → IVF+PQ", n);

    std::vector<float> all_vecs;
    all_vecs.reserve(n * static_cast<size_t>(dim_));
    std::vector<faiss::idx_t> all_ids;
    all_ids.reserve(n);

    for (const auto& [fid, entry] : id_to_entry_) {
        all_vecs.insert(all_vecs.end(),
                        entry.embedding.begin(), entry.embedding.end());
        all_ids.push_back(fid);
    }

    // Recreate IVF+PQ with fresh state (previous object may have been reset)
    ivf_quantizer_ = std::make_unique<faiss::IndexFlatL2>(dim_);
    ivf_index_     = std::make_unique<faiss::IndexIVFPQ>(
        ivf_quantizer_.get(), dim_, NLIST, M_PQ, NBITS);
    ivf_index_->nprobe = NPROBE;

    ivf_index_->train(static_cast<int64_t>(n), all_vecs.data());
    ivf_index_->add_with_ids(static_cast<int64_t>(n), all_vecs.data(),
                              all_ids.data());
    ivf_trained_ = true;

    // Flat index no longer needed; reset to free memory
    flat_index_ = std::make_unique<faiss::IndexFlatIP>(dim_);

    spdlog::info("FaissVectorStore: IVF+PQ migration complete, trained on {} vectors", n);
}

// ──────────────────────────────────────────────────────────────────────────────
// add()  — exclusive write lock
// Uses flat index below MIN_IVF_TRAIN_VEC; migrates to IVF+PQ when threshold crossed.
// ──────────────────────────────────────────────────────────────────────────────
void FaissVectorStore::add(const CacheEntry& entry) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    if (static_cast<int>(entry.embedding.size()) != dim_) {
        spdlog::error("FaissVectorStore::add: dim mismatch expected={} got={}",
                      dim_, entry.embedding.size());
        return;
    }
    if (entry_id_to_faiss_id_.count(entry.id)) {
        spdlog::debug("FaissVectorStore::add: dedup id={}", entry.id);
        return;
    }

    int64_t fid = next_id_++;
    CacheEntry stored = entry;
    stored.faiss_id   = fid;
    stored.created_at = static_cast<long long>(
        std::chrono::system_clock::now().time_since_epoch().count());

    if (tq_ && stored.tq_codes.empty()) {
        stored.tq_codes = tq_->encode(entry.embedding);
    }

    if (ivf_trained_) {
        ivf_index_->add_with_ids(1, entry.embedding.data(), &fid);
    } else {
        flat_index_->add_with_ids(1, entry.embedding.data(), &fid);
    }

    id_to_entry_[fid]               = std::move(stored);
    entry_id_to_faiss_id_[entry.id] = fid;

    // Trigger migration when the training threshold is first crossed
    if (!ivf_trained_ &&
        static_cast<int>(id_to_entry_.size()) >= MIN_IVF_TRAIN_VEC)
    {
        migrateToIVF();
    }

    spdlog::debug("FaissVectorStore::add: id={} faiss_id={} ivf={} tq={}",
                  entry.id, fid,
                  ivf_trained_ ? "yes" : "flat",
                  tq_ ? "yes" : "no");
}

// ──────────────────────────────────────────────────────────────────────────────
// search()  — shared read lock (concurrent searches are allowed)
// Routes to flat index or IVF+PQ based on current phase.
// ──────────────────────────────────────────────────────────────────────────────
std::vector<CacheEntry> FaissVectorStore::search(const std::vector<float>& query,
                                                   int top_k)
{
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    std::vector<CacheEntry> results;

    if (id_to_entry_.empty()) {
        spdlog::debug("FaissVectorStore::search: index empty");
        return results;
    }
    if (static_cast<int>(query.size()) != dim_) {
        spdlog::error("FaissVectorStore::search: dim mismatch expected={} got={}",
                      dim_, query.size());
        return results;
    }

    int k = std::min(top_k, static_cast<int>(id_to_entry_.size()));
    std::vector<faiss::idx_t> labels(k, -1);
    std::vector<float> distances(k, 0.0f);

    try {
        if (ivf_trained_) {
            ivf_index_->search(1, query.data(), k, distances.data(), labels.data());
        } else {
            flat_index_->search(1, query.data(), k, distances.data(), labels.data());
        }
    } catch (const faiss::FaissException& e) {
        spdlog::error("FaissVectorStore::search FAISS exception: {}", e.what());
        return results;
    }

    for (int i = 0; i < k; ++i) {
        if (labels[i] < 0) continue;
        auto it = id_to_entry_.find(labels[i]);
        if (it != id_to_entry_.end()) {
            results.push_back(it->second);
        }
    }
    return results;
}

// ──────────────────────────────────────────────────────────────────────────────
// find()  — shared read lock
// Returns the CacheEntry for the given entry_id, or nullopt if not found.
// Called before remove() so the DELETE handler can retrieve the sig_hash and
// domain needed to clean up the corresponding Redis L1 and slot keys.
// ──────────────────────────────────────────────────────────────────────────────
std::optional<CacheEntry> FaissVectorStore::find(const std::string& entry_id) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    auto it = entry_id_to_faiss_id_.find(entry_id);
    if (it == entry_id_to_faiss_id_.end()) return std::nullopt;
    auto jt = id_to_entry_.find(it->second);
    if (jt == id_to_entry_.end()) return std::nullopt;
    return jt->second;
}

// ──────────────────────────────────────────────────────────────────────────────
// remove()  — exclusive write lock
// ──────────────────────────────────────────────────────────────────────────────
bool FaissVectorStore::remove(const std::string& entry_id) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    auto it = entry_id_to_faiss_id_.find(entry_id);
    if (it == entry_id_to_faiss_id_.end()) return false;

    int64_t fid = it->second;
    faiss::IDSelectorBatch selector(1, &fid);
    if (ivf_trained_) {
        ivf_index_->remove_ids(selector);
    } else {
        flat_index_->remove_ids(selector);
    }
    id_to_entry_.erase(fid);
    entry_id_to_faiss_id_.erase(it);

    spdlog::info("FaissVectorStore::remove: entry_id={}", entry_id);
    return true;
}

// ──────────────────────────────────────────────────────────────────────────────
// persist()  — exclusive write lock
// ──────────────────────────────────────────────────────────────────────────────
void FaissVectorStore::persist() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    if (id_to_entry_.empty()) return;

    faiss::Index* idx_to_write = ivf_trained_
        ? static_cast<faiss::Index*>(ivf_index_.get())
        : static_cast<faiss::Index*>(flat_index_.get());
    faiss::write_index(idx_to_write, index_path_.c_str());
    saveMetadata();
    spdlog::info("FaissVectorStore::persist: wrote {} and {} (ivf={})",
                 index_path_, meta_path_, ivf_trained_ ? "yes" : "flat");
}

size_t FaissVectorStore::size() const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    return id_to_entry_.size();
}

// ──────────────────────────────────────────────────────────────────────────────
// Metadata persistence
// ──────────────────────────────────────────────────────────────────────────────
void FaissVectorStore::saveMetadata() const {
    static constexpr char kHexDigits[] = "0123456789abcdef";

    nlohmann::json root = nlohmann::json::array();
    for (const auto& [fid, entry] : id_to_entry_) {
        nlohmann::json e;
        e["faiss_id"]          = fid;
        e["id"]                = entry.id;
        e["context_signature"] = entry.context_signature;
        e["signature_hash"]    = entry.signature_hash;
        e["template_str"]      = entry.template_str;
        e["domain"]            = entry.domain;
        e["created_at"]        = entry.created_at;
        e["embedding"]         = entry.embedding;
        e["ivf_trained"]       = ivf_trained_;

        if (!entry.tq_codes.empty()) {
            std::string hex;
            hex.reserve(entry.tq_codes.size() * 2);
            for (uint8_t b : entry.tq_codes) {
                hex += kHexDigits[b >> 4];
                hex += kHexDigits[b & 0xf];
            }
            e["tq_codes_hex"] = hex;
        }
        root.push_back(e);
    }

    std::ofstream ofs(meta_path_);
    if (!ofs) {
        spdlog::error("FaissVectorStore::saveMetadata: cannot open {}", meta_path_);
        return;
    }
    ofs << root.dump();
}

void FaissVectorStore::loadMetadata() {
    std::ifstream ifs(meta_path_);
    if (!ifs) {
        spdlog::debug("FaissVectorStore::loadMetadata: no sidecar at {}", meta_path_);
        return;
    }

    nlohmann::json root;
    try {
        ifs >> root;
    } catch (const std::exception& e) {
        spdlog::warn("FaissVectorStore::loadMetadata: parse error: {}", e.what());
        return;
    }

    for (const auto& e : root) {
        CacheEntry entry;
        int64_t fid            = e.at("faiss_id").get<int64_t>();
        entry.id               = e.at("id").get<std::string>();
        entry.context_signature = e.at("context_signature").get<std::string>();
        entry.signature_hash   = e.contains("signature_hash")
                                     ? e.at("signature_hash").get<std::string>()
                                     : "";  // empty for entries written before this field existed
        entry.template_str     = e.at("template_str").get<std::string>();
        entry.domain           = e.at("domain").get<std::string>();
        entry.created_at       = e.at("created_at").get<long long>();
        entry.faiss_id         = fid;
        entry.embedding        = e.at("embedding").get<std::vector<float>>();

        if (e.contains("tq_codes_hex")) {
            const std::string& hex = e.at("tq_codes_hex").get_ref<const std::string&>();
            std::vector<uint8_t> codes;
            codes.reserve(hex.size() / 2);
            auto nibble = [](char c) -> uint8_t {
                if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
                return static_cast<uint8_t>(c - 'a' + 10);
            };
            for (size_t i = 0; i + 1 < hex.size(); i += 2) {
                codes.push_back(static_cast<uint8_t>(
                    (nibble(hex[i]) << 4) | nibble(hex[i + 1])));
            }
            entry.tq_codes = std::move(codes);
        }

        // Read ivf_trained flag from first entry (all rows carry the same value)
        if (e.contains("ivf_trained")) {
            ivf_trained_ = e.at("ivf_trained").get<bool>();
        }

        entry_id_to_faiss_id_[entry.id] = fid;
        id_to_entry_[fid]               = std::move(entry);
        if (fid >= next_id_) next_id_   = fid + 1;
    }

    spdlog::info("FaissVectorStore::loadMetadata: restored {} entries (ivf={})",
                 id_to_entry_.size(), ivf_trained_ ? "yes" : "flat");
}

// ──────────────────────────────────────────────────────────────────────────────
// loadFromDisk — loads the persisted FAISS index (flat or IVF+PQ)
// ──────────────────────────────────────────────────────────────────────────────
void FaissVectorStore::loadFromDisk() {
    faiss::Index* raw = faiss::read_index(index_path_.c_str());
    if (!raw) throw std::runtime_error("faiss::read_index returned null");

    if (auto* flat = dynamic_cast<faiss::IndexFlatIP*>(raw)) {
        flat_index_.reset(flat);
        ivf_trained_ = false;
        spdlog::info("FaissVectorStore: loaded flat index from disk");
    } else if (auto* ivfpq = dynamic_cast<faiss::IndexIVFPQ*>(raw)) {
        ivf_index_.reset(ivfpq);
        ivf_index_->nprobe = NPROBE;
        ivf_trained_       = ivfpq->is_trained;
        spdlog::info("FaissVectorStore: loaded IVF+PQ index from disk (trained={})",
                     ivf_trained_);
    } else {
        delete raw;
        throw std::runtime_error("Loaded FAISS index is neither IndexFlatIP nor IndexIVFPQ");
    }
}

} // namespace lettucecache::cache
