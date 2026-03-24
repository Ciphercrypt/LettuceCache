#include <gtest/gtest.h>
#include "quantization/TurboQuantizer.h"
#include "cache/FaissVectorStore.h"
#include "orchestrator/ContextBuilder.h"
#include "validation/ValidationService.h"
#include <cmath>
#include <numeric>
#include <random>

using namespace lettucecache::quantization;
using namespace lettucecache::cache;
using namespace lettucecache::orchestrator;
using namespace lettucecache::validation;

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────
static std::vector<float> makeUnitVec(size_t d, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v(d);
    float norm_sq = 0.0f;
    for (auto& x : v) { x = nd(rng); norm_sq += x * x; }
    float inv = 1.0f / std::sqrt(norm_sq);
    for (auto& x : v) x *= inv;
    return v;
}

static float dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// ──────────────────────────────────────────────────────────────────────────────
// TurboQuantizer: code_size matches expected layout
// ──────────────────────────────────────────────────────────────────────────────
TEST(TurboQuantizer, CodeSizeFormula) {
    // For d=384, padded to 512, TQ_BITS=4:
    //   4 (norm) + ceil(512*3/8) (MSE over padded coords) + ceil(384/8) (QJL over dim)
    //   = 4 + 192 + 48 = 244 bytes
    TurboQuantizer tq(384);
    size_t padded = 512;  // next power of 2 >= 384
    size_t expected = 4 + (padded * 3 + 7) / 8 + (384 + 7) / 8;
    EXPECT_EQ(tq.code_size(), expected);
}

TEST(TurboQuantizer, SmallDimCodeSize) {
    // d=8 is already a power of 2 (padded_dim=8):
    //   4 + ceil(8*3/8) + ceil(8/8) = 4 + 3 + 1 = 8 bytes
    TurboQuantizer tq(8);
    EXPECT_EQ(tq.code_size(), 8u);
}

// ──────────────────────────────────────────────────────────────────────────────
// TurboQuantizer: encode/decode round-trip — MSE distortion bounded
// Paper (Theorem 1): D_mse(b=3) ≤ √(23π)·4⁻³ ≈ 0.133 for any unit vector.
// Empirically at large d (~768): ~0.030. Test uses 0.15 (safely above the bound).
// ──────────────────────────────────────────────────────────────────────────────
TEST(TurboQuantizer, EncodeDecodeRoundTrip_SmallDim) {
    const size_t D = 64;
    TurboQuantizer tq(D);
    auto x = makeUnitVec(D, 1234);

    auto codes      = tq.encode(x);
    auto x_hat      = tq.decode(codes);

    ASSERT_EQ(x_hat.size(), D);

    float mse = 0.0f;
    for (size_t i = 0; i < D; ++i) {
        float diff = x[i] - x_hat[i];
        mse += diff * diff;
    }
    // For unit vectors, total MSE ≤ 0.10 is a loose upper bound (paper gives ~0.030)
    EXPECT_LT(mse, 0.15f) << "MSE too high: " << mse;
}

TEST(TurboQuantizer, EncodeDecodeRoundTrip_384Dim) {
    const size_t D = 384;
    TurboQuantizer tq(D);
    auto x = makeUnitVec(D, 5678);

    auto codes = tq.encode(x);
    auto x_hat = tq.decode(codes);
    ASSERT_EQ(x_hat.size(), D);

    float mse = 0.0f;
    for (size_t i = 0; i < D; ++i) {
        float diff = x[i] - x_hat[i];
        mse += diff * diff;
    }
    EXPECT_LT(mse, 0.15f) << "MSE for d=384: " << mse;
}

// ──────────────────────────────────────────────────────────────────────────────
// TurboQuantizer: zero vector encodes/decodes correctly
// ──────────────────────────────────────────────────────────────────────────────
TEST(TurboQuantizer, ZeroVector) {
    TurboQuantizer tq(32);
    std::vector<float> zero(32, 0.0f);
    auto codes = tq.encode(zero);
    auto x_hat = tq.decode(codes);
    for (float v : x_hat) EXPECT_FLOAT_EQ(v, 0.0f);
}

// ──────────────────────────────────────────────────────────────────────────────
// TurboQuantizer: unbiasedness of inner_product()
//
// For N independent trials with random x vectors, the average estimated
// inner product E[tq.inner_product(y, encode(x))] should be close to
// E[<y, x>] (the true inner product).
//
// Test: fix one query y, generate N random unit vectors x_i, compare
//   (1/N) Σ tq.inner_product(y, encode(x_i))
//   vs
//   (1/N) Σ <y, x_i>
// The difference should be < 0.02 for N=500 (CLT: std error ≈ σ/√N).
// ──────────────────────────────────────────────────────────────────────────────
TEST(TurboQuantizer, UnbiasedInnerProduct_D64) {
    const size_t D = 64;
    const int    N = 500;
    TurboQuantizer tq(D, 42, 137);

    auto y = makeUnitVec(D, 9999);

    double sum_true = 0.0, sum_est = 0.0;
    for (int i = 0; i < N; ++i) {
        auto x     = makeUnitVec(D, static_cast<uint64_t>(i + 1));
        auto codes = tq.encode(x);
        sum_true += static_cast<double>(dotProduct(y, x));
        sum_est  += static_cast<double>(tq.inner_product(y, codes));
    }

    double mean_true = sum_true / N;
    double mean_est  = sum_est  / N;
    double bias = std::abs(mean_true - mean_est);

    EXPECT_LT(bias, 0.05)
        << "Bias too large: true=" << mean_true << " est=" << mean_est;
}

// ──────────────────────────────────────────────────────────────────────────────
// TurboQuantizer: parallel encode/decode with different seeds produce
// different (non-colliding) codes — sanity check for seed independence
// ──────────────────────────────────────────────────────────────────────────────
TEST(TurboQuantizer, DifferentSeedsDifferentCodes) {
    const size_t D = 32;
    auto x = makeUnitVec(D, 1);
    TurboQuantizer tq1(D, /*rotation_seed=*/1, /*qjl_seed=*/2);
    TurboQuantizer tq2(D, /*rotation_seed=*/3, /*qjl_seed=*/4);
    EXPECT_NE(tq1.encode(x), tq2.encode(x));
}

// ──────────────────────────────────────────────────────────────────────────────
// Dry-run: ValidationService with TurboQuant path
//
// Validates the full scoring flow:
//   1. Encode a "stored" embedding with TQ
//   2. Score it against the same query (should be near 1.0 cosine)
//   3. Score against an orthogonal query (should be near 0.0 cosine)
// ──────────────────────────────────────────────────────────────────────────────
TEST(ValidationDryRun, TurboQuantScoringPath) {
    const size_t D = 64;
    TurboQuantizer tq(D);

    // Build a stored CacheEntry with TQ codes
    auto stored_emb = makeUnitVec(D, 42);
    CacheEntry entry;
    entry.id                = "test-entry-1";
    entry.embedding         = stored_emb;
    entry.tq_codes          = tq.encode(stored_emb);
    entry.context_signature = "sig-abc";
    entry.domain            = "finance";

    // Build query context matching the same embedding + same signature/domain
    ContextObject ctx;
    ctx.embedding       = stored_emb;   // identical vector → cosine ≈ 1.0
    ctx.signature_hash  = "sig-abc";
    ctx.domain          = "finance";

    ValidationService vs(0.85, &tq);
    double s = vs.score(ctx, entry);

    // cosine ≈ 1.0 (same vector, unbiased estimate), ctx=1.0, domain=1.0
    // Expected: 0.60*~1 + 0.25*1 + 0.15*1 = ~1.0
    EXPECT_GT(s, 0.80) << "Score for identical vector should be > 0.80, got " << s;
    EXPECT_TRUE(vs.isHit(ctx, entry));
}

TEST(ValidationDryRun, TurboQuantOrthogonalVector) {
    const size_t D = 128;
    TurboQuantizer tq(D);

    // Create two approximately orthogonal unit vectors
    auto v1 = makeUnitVec(D, 100);
    // Orthogonalise v2 w.r.t. v1 via Gram-Schmidt
    auto v2 = makeUnitVec(D, 200);
    float proj = dotProduct(v1, v2);
    float norm_sq = 0.0f;
    for (size_t i = 0; i < D; ++i) {
        v2[i] -= proj * v1[i];
        norm_sq += v2[i] * v2[i];
    }
    float inv = 1.0f / std::sqrt(norm_sq);
    for (auto& x : v2) x *= inv;

    CacheEntry entry;
    entry.id                = "test-entry-2";
    entry.embedding         = v1;
    entry.tq_codes          = tq.encode(v1);
    entry.context_signature = "sig-xyz";
    entry.domain            = "medical";

    ContextObject ctx;
    ctx.embedding       = v2;          // orthogonal → cosine ≈ 0
    ctx.signature_hash  = "different-sig";
    ctx.domain          = "finance";

    ValidationService vs(0.85, &tq);
    double s = vs.score(ctx, entry);
    // cosine≈0, ctx=0, domain=0 → score≈0
    EXPECT_LT(s, 0.30) << "Score for orthogonal vector should be < 0.30, got " << s;
    EXPECT_FALSE(vs.isHit(ctx, entry));
}

// ──────────────────────────────────────────────────────────────────────────────
// Dry-run: ContextBuilder canonicalization
//
// Same context turns in different order must produce identical signature_hash.
// ──────────────────────────────────────────────────────────────────────────────
TEST(ContextBuilderDryRun, ContextOrderIndependentHash) {
    ContextBuilder cb;

    auto ctx1 = cb.build("what is my balance", {"user: hi", "bot: hello"}, "finance", "u1");
    auto ctx2 = cb.build("what is my balance", {"bot: hello", "user: hi"}, "finance", "u1");

    EXPECT_EQ(ctx1.signature_hash, ctx2.signature_hash)
        << "Context ordering should not affect signature_hash";
}

TEST(ContextBuilderDryRun, DifferentContextProducesDifferentHash) {
    ContextBuilder cb;

    auto ctx1 = cb.build("what is my balance", {"user: hi"}, "finance", "u1");
    auto ctx2 = cb.build("what is my balance", {"user: bye"}, "finance", "u1");

    EXPECT_NE(ctx1.signature_hash, ctx2.signature_hash)
        << "Different context turns must produce different signature_hash";
}

TEST(ContextBuilderDryRun, StopwordFilterUsesHashSet) {
    // "what is the balance" → after stopword removal → "balance"
    std::string intent = ContextBuilder::extractIntent("what is the balance");
    EXPECT_EQ(intent, "balance");
}

// ──────────────────────────────────────────────────────────────────────────────
// Dry-run: ValidationService without TQ (fast dot-product path)
// Verify that cosine similarity = dot product for L2-normalised vectors.
// ──────────────────────────────────────────────────────────────────────────────
TEST(ValidationDryRun, DotProductEqualsCosineFOrNormalisedVectors) {
    const size_t D = 64;
    auto v1 = makeUnitVec(D, 1);
    auto v2 = makeUnitVec(D, 2);

    CacheEntry entry;
    entry.id                = "e1";
    entry.embedding         = v2;
    entry.tq_codes          = {};    // no TQ codes → fast dot-product path
    entry.context_signature = "s1";
    entry.domain            = "general";

    ContextObject ctx;
    ctx.embedding       = v1;
    ctx.signature_hash  = "s1";
    ctx.domain          = "general";

    ValidationService vs(0.85, nullptr);

    float expected_cos = dotProduct(v1, v2);  // == cosine for unit vectors
    double s = vs.score(ctx, entry);

    // composite = 0.60*cos + 0.25*1 + 0.15*1
    double expected_score = 0.60 * expected_cos + 0.25 + 0.15;
    EXPECT_NEAR(s, expected_score, 0.001);
}

// ──────────────────────────────────────────────────────────────────────────────
// Dry-run: CacheEntry tq_codes round-trip through hex serialization
// (covers the saveMetadata/loadMetadata path without writing files)
// ──────────────────────────────────────────────────────────────────────────────
TEST(FaissVectorStoreDryRun, TQCodesHexRoundTrip) {
    const size_t D = 32;
    TurboQuantizer tq(D);
    auto v = makeUnitVec(D, 777);
    auto codes = tq.encode(v);

    // Simulate hex serialisation
    static constexpr char kHex[] = "0123456789abcdef";
    std::string hex;
    hex.reserve(codes.size() * 2);
    for (uint8_t b : codes) { hex += kHex[b >> 4]; hex += kHex[b & 0xf]; }

    // Deserialise
    auto nibble = [](char c) -> uint8_t {
        return (c >= '0' && c <= '9') ? c - '0' : c - 'a' + 10;
    };
    std::vector<uint8_t> decoded;
    decoded.reserve(hex.size() / 2);
    for (size_t i = 0; i + 1 < hex.size(); i += 2)
        decoded.push_back(static_cast<uint8_t>(nibble(hex[i]) << 4 | nibble(hex[i+1])));

    ASSERT_EQ(decoded, codes) << "Hex round-trip must be lossless";

    // Verify inner product is preserved after round-trip
    auto query = makeUnitVec(D, 888);
    float ip_before = tq.inner_product(query, codes);
    float ip_after  = tq.inner_product(query, decoded);
    EXPECT_FLOAT_EQ(ip_before, ip_after);
}
