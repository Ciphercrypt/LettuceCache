#include "TurboQuantizer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

namespace lettucecache::quantization {

// ── Precomputed Lloyd-Max codebooks for N(0,1) ───────────────────────────────
namespace {
constexpr float kC1[] = { -0.7979f, +0.7979f };
constexpr float kB1[] = { 0.0f };
constexpr float kC2[] = { -1.5104f, -0.4528f, +0.4528f, +1.5104f };
constexpr float kB2[] = { -0.9816f, 0.0f, +0.9816f };
constexpr float kC3[] = { -2.1520f,-1.3440f,-0.7560f,-0.2451f,+0.2451f,+0.7560f,+1.3440f,+2.1520f };
constexpr float kB3[] = { -1.7480f,-1.0500f,-0.5006f,0.0f,+0.5006f,+1.0500f,+1.7480f };
constexpr float kC4[] = {
    -2.7330f,-2.0690f,-1.6180f,-1.2560f,-0.9424f,-0.6568f,-0.3881f,-0.1284f,
    +0.1284f,+0.3881f,+0.6568f,+0.9424f,+1.2560f,+1.6180f,+2.0690f,+2.7330f };
constexpr float kB4[] = {
    -2.4010f,-1.8435f,-1.4370f,-1.0992f,-0.7996f,-0.5225f,-0.2583f,0.0f,
    +0.2583f,+0.5225f,+0.7996f,+1.0992f,+1.4370f,+1.8435f,+2.4010f };
} // anonymous namespace

TurboQuantizer::TurboQuantizer(size_t dim, uint64_t rotation_seed, uint64_t qjl_seed)
    : dim_(dim), padded_dim_(1)
{
    if (dim == 0) throw std::invalid_argument("TurboQuantizer: dim must be > 0");
    while (padded_dim_ < dim_) padded_dim_ <<= 1;

    const int mse_bits = TQ_BITS - 1;
    size_t mse_bytes   = (dim_ * mse_bits + 7) / 8;
    size_t qjl_bytes   = (dim_ + 7) / 8;

    mse_byte_offset_ = sizeof(float);
    qjl_byte_offset_ = mse_byte_offset_ + mse_bytes;
    code_size_bytes_ = qjl_byte_offset_ + qjl_bytes;

    initCodebooks();
    initRHT(rotation_seed);
    initQJL(qjl_seed);
}

void TurboQuantizer::initCodebooks() {
    codebooks_[0].centroids  = { std::begin(kC1), std::end(kC1) };
    codebooks_[0].boundaries = { std::begin(kB1), std::end(kB1) };
    codebooks_[1].centroids  = { std::begin(kC2), std::end(kC2) };
    codebooks_[1].boundaries = { std::begin(kB2), std::end(kB2) };
    codebooks_[2].centroids  = { std::begin(kC3), std::end(kC3) };
    codebooks_[2].boundaries = { std::begin(kB3), std::end(kB3) };
    codebooks_[3].centroids  = { std::begin(kC4), std::end(kC4) };
    codebooks_[3].boundaries = { std::begin(kB4), std::end(kB4) };
}

void TurboQuantizer::initRHT(uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution coin(0.5);
    rht_signs_.resize(dim_);
    for (auto& s : rht_signs_) s = coin(rng) ? 1.0f : -1.0f;
}

void TurboQuantizer::initQJL(uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    qjl_matrix_.resize(dim_ * dim_);
    for (auto& v : qjl_matrix_) v = nd(rng);
}

void TurboQuantizer::whtInplace(float* x, size_t n) {
    for (size_t len = 1; len < n; len <<= 1)
        for (size_t i = 0; i < n; i += len << 1)
            for (size_t j = 0; j < len; ++j) {
                float u = x[i+j], v = x[i+j+len];
                x[i+j] = u+v;  x[i+j+len] = u-v;
            }
}

void TurboQuantizer::rht(std::vector<float>& buf, bool forward) const {
    buf.resize(padded_dim_, 0.0f);
    const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(padded_dim_));
    if (forward) {
        for (size_t i = 0; i < dim_; ++i) buf[i] *= rht_signs_[i];
        whtInplace(buf.data(), padded_dim_);
        for (size_t i = 0; i < padded_dim_; ++i) buf[i] *= inv_sqrt_n;
    } else {
        whtInplace(buf.data(), padded_dim_);
        for (size_t i = 0; i < padded_dim_; ++i) buf[i] *= inv_sqrt_n;
        for (size_t i = 0; i < dim_; ++i) buf[i] *= rht_signs_[i];
    }
}

} // namespace lettucecache::quantization
