#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lettucecache::quantization {

// TurboQuant at 4 total bits: 3-bit MSE stage + 1-bit QJL residual stage.
// Achieves unbiased inner-product estimation — critical for the 0.85 composite
// validation threshold. Reference: arXiv:2504.19874 (TurboQuant, Zandieh et al.)
//
// Code layout for a single d-dimensional vector (padded to next power-of-2 n):
//   [float32 norm (4 B)] [packed (TQ_BITS-1)-bit MSE indices over n coords] [QJL bits over d coords]
//
// For TQ_BITS=4, d=384 (n=512):
//   norm: 4 bytes, MSE: ceil(512*3/8)=192 bytes, QJL: ceil(384/8)=48 bytes
//   Total: 244 bytes  vs  1536 bytes FP32  → 6.3× compression
//   (WHT distributes signal over all n padded coords; all n must be quantised)
static constexpr int TQ_BITS = 4;

class TurboQuantizer {
public:
    // rotation_seed  → Randomized Hadamard Transform sign vector
    // qjl_seed       → QJL Gaussian matrix S (d×d)
    explicit TurboQuantizer(size_t dim,
                             uint64_t rotation_seed = 42,
                             uint64_t qjl_seed      = 137);

    // Encode x using TurboQuant_prod (3-bit MSE + 1-bit QJL residual).
    std::vector<uint8_t> encode(const float* x) const;
    std::vector<uint8_t> encode(const std::vector<float>& x) const;

    // Decode codes back to full-precision. Useful for FAISS add after compression.
    std::vector<float> decode(const uint8_t* codes) const;
    std::vector<float> decode(const std::vector<uint8_t>& codes) const;

    // Unbiased inner-product estimate: E[<y, decode(encode(x))>] = <y, x>.
    // y must be full-precision (asymmetric computation, query stays uncompressed).
    float inner_product(const float* y, const uint8_t* codes) const;
    float inner_product(const std::vector<float>& y,
                        const std::vector<uint8_t>& codes) const;

    size_t code_size() const { return code_size_bytes_; }
    size_t dim()       const { return dim_; }

private:
    size_t   dim_;
    size_t   padded_dim_;       // next power-of-2 ≥ dim_ (for WHT)
    size_t   code_size_bytes_;  // norm(4) + mse_bytes + qjl_bytes

    size_t mse_byte_offset_;    // = 4
    size_t qjl_byte_offset_;    // = 4 + mse_bytes

    std::vector<float> rht_signs_;    // dim_ sign values ∈ {-1,+1}
    std::vector<float> qjl_matrix_;  // dim_ × dim_ Gaussian (row-major)

    struct Codebook {
        std::vector<float> centroids;   // 2^bits entries, ascending
        std::vector<float> boundaries;  // 2^bits - 1 entries
    };
    std::array<Codebook, 4> codebooks_;  // index = bits-1 (1…4 bits)

    void initCodebooks();
    void initRHT(uint64_t seed);
    void initQJL(uint64_t seed);

    // In-place Randomized Hadamard Transform on a padded_dim_-length buffer.
    // forward=true : sign-flip → WHT → scale 1/√n
    // forward=false: WHT → scale 1/√n → sign-flip  (self-inverse)
    void rht(std::vector<float>& buf, bool forward) const;

    // In-place unnormalized Walsh-Hadamard Transform. n must be power of 2.
    static void whtInplace(float* x, size_t n);

    // Scalar quantization helpers.
    uint32_t findBin(float z_n01, int bits) const;   // z in N(0,1) scale

    // Bit-pack/unpack: write/read `b`-bit index at position `pos` into buf[offset..].
    static void     packBits  (std::vector<uint8_t>& buf, size_t offset,
                                size_t pos, int b, uint32_t idx);
    static uint32_t unpackBits(const std::vector<uint8_t>& buf, size_t offset,
                                size_t pos, int b);

    // MSE encode into `out` starting at `offset`. Returns bytes written.
    size_t encodeMSE(const std::vector<float>& rotated_scaled, int bits,
                     std::vector<uint8_t>& out, size_t offset) const;

    // MSE decode: returns padded_dim_-length buffer in rotated space (N(0,1/d) scale).
    std::vector<float> decodeMSE(const std::vector<uint8_t>& codes,
                                  size_t offset, int bits) const;
};

} // namespace lettucecache::quantization
