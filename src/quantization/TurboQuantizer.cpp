#include "TurboQuantizer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

namespace lettucecache::quantization {

// ──────────────────────────────────────────────────────────────────────────────
// Precomputed Lloyd-Max codebooks for N(0,1) distribution.
// Source: TurboQuant paper (arXiv:2504.19874), Table 1 and production hardening doc.
// All centroids are in ascending order. Boundaries are midpoints between centroids.
// ──────────────────────────────────────────────────────────────────────────────
namespace {

// 1-bit: 2 centroids
constexpr float kC1[] = { -0.7979f, +0.7979f };
constexpr float kB1[] = { 0.0f };

// 2-bit: 4 centroids
constexpr float kC2[] = { -1.5104f, -0.4528f, +0.4528f, +1.5104f };
constexpr float kB2[] = { -0.9816f, 0.0f, +0.9816f };

// 3-bit: 8 centroids
constexpr float kC3[] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f,
    +0.2451f, +0.7560f, +1.3440f, +2.1520f
};
constexpr float kB3[] = {
    -1.7480f, -1.0500f, -0.5006f, 0.0f,
    +0.5006f, +1.0500f, +1.7480f
};

// 4-bit: 16 centroids
constexpr float kC4[] = {
    -2.7330f, -2.0690f, -1.6180f, -1.2560f,
    -0.9424f, -0.6568f, -0.3881f, -0.1284f,
    +0.1284f, +0.3881f, +0.6568f, +0.9424f,
    +1.2560f, +1.6180f, +2.0690f, +2.7330f
};
constexpr float kB4[] = {
    -2.4010f, -1.8435f, -1.4370f, -1.0992f,
    -0.7996f, -0.5225f, -0.2583f, 0.0f,
    +0.2583f, +0.5225f, +0.7996f, +1.0992f,
    +1.4370f, +1.8435f, +2.4010f
};

} // anonymous namespace

// ──────────────────────────────────────────────────────────────────────────────
// Constructor
// ──────────────────────────────────────────────────────────────────────────────
TurboQuantizer::TurboQuantizer(size_t dim, uint64_t rotation_seed, uint64_t qjl_seed)
    : dim_(dim), padded_dim_(1)
{
    if (dim == 0) throw std::invalid_argument("TurboQuantizer: dim must be > 0");

    // Next power of 2 for Walsh-Hadamard Transform
    while (padded_dim_ < dim_) padded_dim_ <<= 1;

    // MSE quantization operates on ALL padded_dim_ coordinates after RHT.
    // The WHT distributes energy from dim_ inputs across ALL padded_dim_ outputs;
    // quantizing only dim_ of them discards the remaining signal and inflates MSE.
    // QJL correction still uses dim_ (residual is in the original input space).
    const int mse_bits  = TQ_BITS - 1;  // = 3
    size_t mse_bytes    = (padded_dim_ * mse_bits + 7) / 8;  // padded_dim_, not dim_
    size_t qjl_bytes    = (dim_ + 7) / 8;                    // dim_ (residual space)

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
    // S ∈ R^{d×d}, i.i.d. N(0,1). Stored row-major for cache-friendly row access.
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    qjl_matrix_.resize(dim_ * dim_);
    for (auto& v : qjl_matrix_) v = nd(rng);
}

// ──────────────────────────────────────────────────────────────────────────────
// Walsh-Hadamard Transform (unnormalized, in-place, n must be power of 2)
// ──────────────────────────────────────────────────────────────────────────────
void TurboQuantizer::whtInplace(float* x, size_t n) {
    for (size_t len = 1; len < n; len <<= 1) {
        for (size_t i = 0; i < n; i += len << 1) {
            for (size_t j = 0; j < len; ++j) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j]       = u + v;
                x[i + j + len] = u - v;
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Randomized Hadamard Transform
//
// forward: y = (1/√n) · WHT · diag(signs) · x_padded
// inverse: x = diag(signs) · (1/√n) · WHT · y
// Both use the same sign vector. Self-inverse up to the sign application order.
// ──────────────────────────────────────────────────────────────────────────────
void TurboQuantizer::rht(std::vector<float>& buf, bool forward) const {
    // buf must have padded_dim_ elements; zero-pad if shorter
    buf.resize(padded_dim_, 0.0f);

    if (forward) {
        // Step 1: sign flip on first dim_ elements
        for (size_t i = 0; i < dim_; ++i) buf[i] *= rht_signs_[i];
        // Step 2: WHT
        whtInplace(buf.data(), padded_dim_);
        // Step 3: normalize
        const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(padded_dim_));
        for (size_t i = 0; i < padded_dim_; ++i) buf[i] *= inv_sqrt_n;
    } else {
        // Inverse: WHT first, then normalize, then sign flip
        whtInplace(buf.data(), padded_dim_);
        const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(padded_dim_));
        for (size_t i = 0; i < padded_dim_; ++i) buf[i] *= inv_sqrt_n;
        for (size_t i = 0; i < dim_; ++i) buf[i] *= rht_signs_[i];
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Bit packing helpers
// ──────────────────────────────────────────────────────────────────────────────
void TurboQuantizer::packBits(std::vector<uint8_t>& buf, size_t offset,
                               size_t pos, int b, uint32_t idx)
{
    size_t bit_start = static_cast<size_t>(b) * pos;
    size_t byte_idx  = offset + bit_start / 8;
    size_t bit_off   = bit_start % 8;
    int    remaining = b;

    while (remaining > 0) {
        int   bits_avail  = static_cast<int>(8 - bit_off);
        int   bits_write  = std::min(remaining, bits_avail);
        uint8_t mask      = static_cast<uint8_t>((1u << bits_write) - 1u);
        buf[byte_idx] &= ~static_cast<uint8_t>(mask << bit_off);
        buf[byte_idx] |=  static_cast<uint8_t>((idx & mask) << bit_off);
        idx >>= bits_write;
        remaining -= bits_write;
        ++byte_idx;
        bit_off = 0;
    }
}

uint32_t TurboQuantizer::unpackBits(const std::vector<uint8_t>& buf, size_t offset,
                                     size_t pos, int b)
{
    size_t bit_start = static_cast<size_t>(b) * pos;
    size_t byte_idx  = offset + bit_start / 8;
    size_t bit_off   = bit_start % 8;
    int    remaining = b;
    uint32_t result  = 0;
    uint32_t shift   = 0;

    while (remaining > 0) {
        int   bits_avail = static_cast<int>(8 - bit_off);
        int   bits_read  = std::min(remaining, bits_avail);
        uint32_t mask    = (1u << bits_read) - 1u;
        result |= ((buf[byte_idx] >> bit_off) & mask) << shift;
        shift     += bits_read;
        remaining -= bits_read;
        ++byte_idx;
        bit_off = 0;
    }
    return result;
}

// ──────────────────────────────────────────────────────────────────────────────
// Codebook lookup: binary search on boundaries
// z_n01 is in N(0,1) scale (already multiplied by √d before calling)
// ──────────────────────────────────────────────────────────────────────────────
uint32_t TurboQuantizer::findBin(float z_n01, int bits) const {
    const auto& cb = codebooks_[bits - 1];
    auto it = std::lower_bound(cb.boundaries.begin(), cb.boundaries.end(), z_n01);
    return static_cast<uint32_t>(it - cb.boundaries.begin());
}

// ──────────────────────────────────────────────────────────────────────────────
// MSE encode: rotated_scaled is in N(0,1) scale (each coord * √d after RHT)
// ──────────────────────────────────────────────────────────────────────────────
size_t TurboQuantizer::encodeMSE(const std::vector<float>& rotated_scaled,
                                  int bits,
                                  std::vector<uint8_t>& out,
                                  size_t offset) const
{
    // Quantise ALL padded_dim_ coordinates — the WHT spreads signal across all of them.
    size_t packed_bytes = (padded_dim_ * bits + 7) / 8;
    std::fill(out.begin() + offset, out.begin() + offset + packed_bytes, uint8_t{0});
    for (size_t j = 0; j < padded_dim_; ++j) {
        uint32_t bin = findBin(rotated_scaled[j], bits);
        packBits(out, offset, j, bits, bin);
    }
    return packed_bytes;
}

// ──────────────────────────────────────────────────────────────────────────────
// MSE decode: returns padded_dim_-length buffer in rotated N(0,1/d) space
// ──────────────────────────────────────────────────────────────────────────────
std::vector<float> TurboQuantizer::decodeMSE(const std::vector<uint8_t>& codes,
                                              size_t offset,
                                              int bits) const
{
    // Decode all padded_dim_ coordinates. Scale uses padded_dim_ because after
    // RHT the rotated vector lives in padded_dim_-dimensional space with each
    // coordinate ~ N(0, 1/padded_dim_), so sqrt(padded_dim_) brings it to N(0,1).
    const auto& cb         = codebooks_[bits - 1];
    const float inv_sqrtn  = 1.0f / std::sqrt(static_cast<float>(padded_dim_));
    std::vector<float> buf(padded_dim_, 0.0f);
    for (size_t j = 0; j < padded_dim_; ++j) {
        uint32_t bin = unpackBits(codes, offset, j, bits);
        buf[j] = cb.centroids[bin] * inv_sqrtn;  // N(0,1) → N(0,1/padded_dim_)
    }
    return buf;
}

// ──────────────────────────────────────────────────────────────────────────────
// encode()
// TurboQuant_prod: MSE at (TQ_BITS-1) bits + QJL on residual
// ──────────────────────────────────────────────────────────────────────────────
std::vector<uint8_t> TurboQuantizer::encode(const float* x) const {
    const int mse_bits = TQ_BITS - 1;  // 3

    // 1. L2 norm
    float norm_sq = 0.0f;
    for (size_t i = 0; i < dim_; ++i) norm_sq += x[i] * x[i];
    const float norm = std::sqrt(norm_sq);

    std::vector<uint8_t> codes(code_size_bytes_, uint8_t{0});
    std::memcpy(codes.data(), &norm, sizeof(float));
    if (norm < 1e-10f) return codes;

    // 2. Normalize + forward RHT
    std::vector<float> buf(padded_dim_, 0.0f);
    const float inv_norm = 1.0f / norm;
    for (size_t i = 0; i < dim_; ++i) buf[i] = x[i] * inv_norm;
    rht(buf, true);

    // 3. Scale to N(0,1): multiply by √padded_dim (not √dim_, since RHT output
    //    has padded_dim_ coordinates each ~ N(0, 1/padded_dim_) for unit input).
    const float sqrt_n = std::sqrt(static_cast<float>(padded_dim_));
    std::vector<float> scaled(padded_dim_);
    for (size_t i = 0; i < padded_dim_; ++i) scaled[i] = buf[i] * sqrt_n;

    // 4. MSE encode at (TQ_BITS-1) bits
    encodeMSE(scaled, mse_bits, codes, mse_byte_offset_);

    // 5. Compute residual = x - x̂_mse (in original space)
    //    x̂_mse = norm * inv_rht(centroids / √d)
    std::vector<float> mse_buf = decodeMSE(codes, mse_byte_offset_, mse_bits);
    rht(mse_buf, false);                           // inverse RHT → unit-sphere space
    std::vector<float> residual(dim_);
    for (size_t i = 0; i < dim_; ++i) {
        float x_hat_i = mse_buf[i] * norm;         // scale back by norm
        residual[i]   = x[i] - x_hat_i;
    }

    // 6. QJL: sign_bits = sign(S · residual)
    //    S is dim_×dim_, row-major. sign bit row_j = 1 iff (S_row_j · residual) ≥ 0.
    for (size_t row = 0; row < dim_; ++row) {
        float dot = 0.0f;
        const float* s_row = qjl_matrix_.data() + row * dim_;
        for (size_t k = 0; k < dim_; ++k) dot += s_row[k] * residual[k];
        if (dot >= 0.0f) {
            codes[qjl_byte_offset_ + row / 8] |= static_cast<uint8_t>(1u << (row % 8));
        }
    }

    return codes;
}

std::vector<uint8_t> TurboQuantizer::encode(const std::vector<float>& x) const {
    if (x.size() != dim_)
        throw std::invalid_argument("TurboQuantizer::encode: dim mismatch");
    return encode(x.data());
}

// ──────────────────────────────────────────────────────────────────────────────
// decode()
// Uses only the MSE stage (not QJL) — QJL correction is for inner products only.
// ──────────────────────────────────────────────────────────────────────────────
std::vector<float> TurboQuantizer::decode(const uint8_t* codes) const {
    const int mse_bits = TQ_BITS - 1;

    float norm;
    std::memcpy(&norm, codes, sizeof(float));
    if (norm < 1e-10f) return std::vector<float>(dim_, 0.0f);

    std::vector<uint8_t> codes_vec(codes, codes + code_size_bytes_);
    std::vector<float> buf = decodeMSE(codes_vec, mse_byte_offset_, mse_bits);
    rht(buf, false);

    std::vector<float> result(dim_);
    for (size_t i = 0; i < dim_; ++i) result[i] = buf[i] * norm;
    return result;
}

std::vector<float> TurboQuantizer::decode(const std::vector<uint8_t>& codes) const {
    if (codes.size() < code_size_bytes_)
        throw std::invalid_argument("TurboQuantizer::decode: insufficient code size");
    return decode(codes.data());
}

// ──────────────────────────────────────────────────────────────────────────────
// inner_product()
// Unbiased estimate: <y, x̂_prod> = <y, x̂_mse> + (√(π/2)/d) · (S·y)ᵀ · sign_bits
//
// Proof sketch: QJL gives E[QJL_correction] = <y, residual> (Lemma 4 in paper).
// Adding to MSE term: E[total] = <y, x̂_mse> + <y, residual> = <y, x> (unbiased).
//
// Asymmetric computation: y stays full-precision, only x is compressed.
// ──────────────────────────────────────────────────────────────────────────────
float TurboQuantizer::inner_product(const float* y, const uint8_t* codes) const {
    const int mse_bits = TQ_BITS - 1;

    float norm;
    std::memcpy(&norm, codes, sizeof(float));
    if (norm < 1e-10f) return 0.0f;

    std::vector<uint8_t> codes_vec(codes, codes + code_size_bytes_);

    // ── MSE term: <y, x̂_mse> = norm · <RHT(y), centroids/√d> ──────────────
    // RHT is orthogonal: <y, Πᵀ·c> = <Π·y, c>
    std::vector<float> y_rot(padded_dim_, 0.0f);
    for (size_t i = 0; i < dim_; ++i) y_rot[i] = y[i];
    rht(y_rot, true);  // y_rot = RHT(y), padded_dim_ length

    const auto& cb         = codebooks_[mse_bits - 1];
    const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(padded_dim_));
    float mse_ip = 0.0f;
    for (size_t j = 0; j < padded_dim_; ++j) {
        uint32_t bin = unpackBits(codes_vec, mse_byte_offset_, j, mse_bits);
        mse_ip += cb.centroids[bin] * inv_sqrt_n * y_rot[j];
    }
    mse_ip *= norm;

    // ── QJL correction: (√(π/2)/d) · Σ_row sign_row · (S_row · y) ─────────
    // = (√(π/2)/d) · (S·y)ᵀ · sign_bits
    const float qjl_scale = std::sqrt(M_PI / 2.0f) / static_cast<float>(dim_);
    float qjl_corr = 0.0f;
    for (size_t row = 0; row < dim_; ++row) {
        // Read sign bit
        int sign = ((codes[qjl_byte_offset_ + row / 8] >> (row % 8)) & 1) ? 1 : -1;
        // Compute S_row · y
        float s_dot_y = 0.0f;
        const float* s_row = qjl_matrix_.data() + row * dim_;
        for (size_t k = 0; k < dim_; ++k) s_dot_y += s_row[k] * y[k];
        qjl_corr += static_cast<float>(sign) * s_dot_y;
    }
    qjl_corr *= qjl_scale;

    return mse_ip + qjl_corr;
}

float TurboQuantizer::inner_product(const std::vector<float>& y,
                                     const std::vector<uint8_t>& codes) const {
    if (y.size() != dim_)
        throw std::invalid_argument("TurboQuantizer::inner_product: dim mismatch");
    if (codes.size() < code_size_bytes_)
        throw std::invalid_argument("TurboQuantizer::inner_product: insufficient codes");
    return inner_product(y.data(), codes.data());
}

} // namespace lettucecache::quantization
