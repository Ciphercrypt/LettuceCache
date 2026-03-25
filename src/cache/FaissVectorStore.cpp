#include "FaissVectorStore.h"
#include "../quantization/TurboQuantizer.h"
#include <faiss/impl/FaissException.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <chrono>
#include <cstdlib>
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
    quantizer_ = std::make_unique<faiss::IndexFlatL2>(dim_);
    index_     = std::make_unique<faiss::IndexIVFPQ>(
        quantizer_.get(), dim_, NLIST, M_PQ, NBITS);
    index_->nprobe = NPROBE;

    std::ifstream test(index_path_);
    if (test.good()) {
        test.close();
        try {
            loadFromDisk();   // loads FAISS binary
            loadMetadata();   // loads id_to_entry_ from sidecar JSON — fixes restart bug
            spdlog::info("FaissVectorStore: loaded from disk, entries={}",
                         id_to_entry_.size());
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
// IVF training (called under exclusive lock from add())
// ──────────────────────────────────────────────────────────────────────────────
void FaissVectorStore::ensureTrained(const std::vector<float>& new_vec) {
    if (trained_) return;

    std::vector<float> train_data;
    train_data.reserve((id_to_entry_.size() + 1) * static_cast<size_t>(dim_));
    for (const auto& [fid, entry] : id_to_entry_) {
        train_data.insert(train_data.end(),
                          entry.embedding.begin(), entry.embedding.end());
    }
    train_data.insert(train_data.end(), new_vec.begin(), new_vec.end());

    size_t n_vecs = train_data.size() / static_cast<size_t>(dim_);
    while (n_vecs < static_cast<size_t>(MIN_TRAIN_VEC)) {
        for (int d = 0; d < dim_; ++d)
            train_data.push_back(static_cast<float>(std::rand()) /
                                 static_cast<float>(RAND_MAX));
        ++n_vecs;
    }

    index_->train(static_cast<int64_t>(n_vecs), train_data.data());
    trained_ = true;
    spdlog::info("FaissVectorStore: IVF trained on {} vectors", n_vecs);
}

// ──────────────────────────────────────────────────────────────────────────────
// add()  — exclusive write lock
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

    ensureTrained(entry.embedding);

    int64_t fid = next_id_++;
    CacheEntry stored = entry;
    stored.faiss_id   = fid;
    stored.created_at = static_cast<long long>(
        std::chrono::system_clock::now().time_since_epoch().count());

    // Compute TurboQuant codes if TQ enabled and not already present
    if (tq_ && stored.tq_codes.empty()) {
        stored.tq_codes = tq_->encode(entry.embedding);
    }

    index_->add_with_ids(1, entry.embedding.data(), &fid);
    id_to_entry_[fid]               = std::move(stored);
    entry_id_to_faiss_id_[entry.id] = fid;

    spdlog::debug("FaissVectorStore::add: id={} faiss_id={} tq={}",
                  entry.id, fid, tq_ ? "yes" : "no");
}

// ──────────────────────────────────────────────────────────────────────────────
// search()  — shared read lock (concurrent searches are allowed)
// ──────────────────────────────────────────────────────────────────────────────
std::vector<CacheEntry> FaissVectorStore::search(const std::vector<float>& query,
                                                   int top_k)
{
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    std::vector<CacheEntry> results;

    if (!trained_ || id_to_entry_.empty()) {
        spdlog::debug("FaissVectorStore::search: index empty or untrained");
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
        index_->search(1, query.data(), k, distances.data(), labels.data());
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
// remove()  — exclusive write lock
// ──────────────────────────────────────────────────────────────────────────────
bool FaissVectorStore::remove(const std::string& entry_id) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    auto it = entry_id_to_faiss_id_.find(entry_id);
    if (it == entry_id_to_faiss_id_.end()) return false;

    int64_t fid = it->second;
    faiss::IDSelectorBatch selector(1, &fid);
    index_->remove_ids(selector);
    id_to_entry_.erase(fid);
    entry_id_to_faiss_id_.erase(it);

    spdlog::info("FaissVectorStore::remove: entry_id={}", entry_id);
    return true;
}

// ──────────────────────────────────────────────────────────────────────────────
// persist()  — exclusive write lock
// Writes both the FAISS binary and the metadata JSON sidecar.
// ──────────────────────────────────────────────────────────────────────────────
void FaissVectorStore::persist() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    if (!trained_) return;
    faiss::write_index(index_.get(), index_path_.c_str());
    saveMetadata();
    spdlog::info("FaissVectorStore::persist: wrote {} and {}",
                 index_path_, meta_path_);
}

size_t FaissVectorStore::size() const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    return id_to_entry_.size();
}

// ──────────────────────────────────────────────────────────────────────────────
// Metadata persistence — fixes the critical "L2 dead after restart" bug.
// The FAISS binary only stores the ANN index, not the entry metadata.
// We serialise id_to_entry_ to a JSON sidecar so lookups survive restarts.
// ──────────────────────────────────────────────────────────────────────────────
void FaissVectorStore::saveMetadata() const {
    static constexpr char kHexDigits[] = "0123456789abcdef";

    nlohmann::json root = nlohmann::json::array();
    for (const auto& [fid, entry] : id_to_entry_) {
        nlohmann::json e;
        e["faiss_id"]          = fid;
        e["id"]                = entry.id;
        e["context_signature"] = entry.context_signature;
        e["template_str"]      = entry.template_str;
        e["domain"]            = entry.domain;
        e["created_at"]        = entry.created_at;
        e["embedding"]         = entry.embedding;

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

        entry_id_to_faiss_id_[entry.id] = fid;
        id_to_entry_[fid]               = std::move(entry);
        if (fid >= next_id_) next_id_   = fid + 1;
    }

    spdlog::info("FaissVectorStore::loadMetadata: restored {} entries", id_to_entry_.size());
}

// ──────────────────────────────────────────────────────────────────────────────
// loadFromDisk — loads the FAISS binary index
// ──────────────────────────────────────────────────────────────────────────────
void FaissVectorStore::loadFromDisk() {
    faiss::Index* raw = faiss::read_index(index_path_.c_str());
    if (!raw) throw std::runtime_error("faiss::read_index returned null");
    auto* ivfpq = dynamic_cast<faiss::IndexIVFPQ*>(raw);
    if (!ivfpq) {
        delete raw;
        throw std::runtime_error("Loaded FAISS index is not IndexIVFPQ");
    }
    index_.reset(ivfpq);
    index_->nprobe = NPROBE;
    trained_       = index_->is_trained;
}

} // namespace lettucecache::cache
