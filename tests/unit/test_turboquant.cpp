#include <gtest/gtest.h>
#include "quantization/TurboQuantizer.h"
#include <cmath>
#include <random>

using namespace lettucecache::quantization;

static std::vector<float> makeUnitVec(size_t d, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v(d);
    float ns = 0.0f;
    for (auto& x : v) { x = nd(rng); ns += x*x; }
    float inv = 1.0f / std::sqrt(ns);
    for (auto& x : v) x *= inv;
    return v;
}

// ── Code size formula ─────────────────────────────────────────────────────────
TEST(TurboQuantizer, CodeSizeFormula) {
    // d=384, padded=512: 4 + ceil(512*3/8) + ceil(384/8) = 4+192+48 = 244
    TurboQuantizer tq(384);
    size_t padded = 512;
    EXPECT_EQ(tq.code_size(), 4 + (padded*3+7)/8 + (384+7)/8);
}

TEST(TurboQuantizer, SmallDimCodeSize) {
    // d=8 is power-of-2, padded=8: 4 + ceil(8*3/8) + ceil(8/8) = 8
    TurboQuantizer tq(8);
    EXPECT_EQ(tq.code_size(), 8u);
}

// ── MSE round-trip ─────────────────────────────────────────────────────────────
// Paper (Theorem 1): D_mse(b=3) <= sqrt(23*pi)*4^(-3) ~= 0.133 for any unit vector.
TEST(TurboQuantizer, EncodeDecodeRoundTrip_SmallDim) {
    TurboQuantizer tq(64);
    auto x = makeUnitVec(64, 1234);
    auto codes = tq.encode(x);
    auto x_hat = tq.decode(codes);
    ASSERT_EQ(x_hat.size(), 64u);
    float mse = 0.0f;
    for (size_t i = 0; i < 64; ++i) { float d = x[i]-x_hat[i]; mse += d*d; }
    EXPECT_LT(mse, 0.15f) << "MSE d=64: " << mse;
}

TEST(TurboQuantizer, EncodeDecodeRoundTrip_384Dim) {
    TurboQuantizer tq(384);
    auto x = makeUnitVec(384, 5678);
    auto codes = tq.encode(x);
    auto x_hat = tq.decode(codes);
    ASSERT_EQ(x_hat.size(), 384u);
    float mse = 0.0f;
    for (size_t i = 0; i < 384; ++i) { float d = x[i]-x_hat[i]; mse += d*d; }
    EXPECT_LT(mse, 0.15f) << "MSE d=384: " << mse;
}
