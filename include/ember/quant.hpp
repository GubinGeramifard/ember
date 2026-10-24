#pragma once
//
// Group-wise int8 quantization (the Q8_0 scheme). Each contiguous group of
// kGroupSize values shares one fp32 scale; a value is stored as round(v/scale)
// clamped to [-127, 127]. Weights are quantized once at load time; activations
// are quantized on the fly before each matmul. This cuts weight memory traffic
// ~4x, which is the real win for a memory-bandwidth-bound workload.
//
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ember {

// Group size. Must divide every matmul inner dimension and be a multiple of 32
// (one AVX2 register of int8). 32 divides all dims in the target models.
inline constexpr int kGroupSize = 32;

struct QTensor {
    std::vector<std::int8_t> q;  // quantized values, row-major
    std::vector<float> s;        // one scale per group (numel / kGroupSize)
};

// Quantize one group of `gs` floats in place into q[] (+ its scale).
inline void quantize_group(const float* src, std::int8_t* q, float& scale, int gs) {
    float amax = 0.0f;
    for (int j = 0; j < gs; ++j) amax = std::max(amax, std::fabs(src[j]));
    scale = amax / 127.0f;
    const float inv = scale != 0.0f ? 1.0f / scale : 0.0f;
    for (int j = 0; j < gs; ++j) {
        int qi = static_cast<int>(std::lround(src[j] * inv));
        qi = std::clamp(qi, -127, 127);
        q[j] = static_cast<std::int8_t>(qi);
    }
}

// Quantize a whole tensor. numel must be a multiple of gs, and gs must not cross
// a weight row (guaranteed here because every row length is a multiple of gs).
inline QTensor quantize(const float* src, std::size_t numel, int gs = kGroupSize) {
    QTensor t;
    t.q.resize(numel);
    t.s.resize(numel / gs);
    for (std::size_t g = 0; g < numel / gs; ++g)
        quantize_group(src + g * gs, t.q.data() + g * gs, t.s[g], gs);
    return t;
}

// Quantize an activation vector into caller-provided scratch buffers.
inline void quantize_into(const float* x, int n, std::int8_t* q, float* s, int gs = kGroupSize) {
    for (int g = 0; g < n / gs; ++g)
        quantize_group(x + g * gs, q + g * gs, s[g], gs);
}

// Reconstruct a quantized row back to fp32 (used for the embedding lookup).
inline void dequantize_row(const std::int8_t* q, const float* s, float* out, int n,
                           int gs = kGroupSize) {
    for (int g = 0; g < n / gs; ++g)
        for (int j = 0; j < gs; ++j) out[g * gs + j] = q[g * gs + j] * s[g];
}

// Horizontal sum of eight int32 lanes.
inline int hsum_i32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

// Quantized dot product: group-by-group int8*int8 -> int32, scaled back to float.
// AVX2 has no signed*signed byte multiply, so we use the standard trick: take
// |w| (unsigned) and transfer w's sign onto x, then maddubs (unsigned*signed).
inline float dot_q8(const std::int8_t* wq, const float* ws, const std::int8_t* xq,
                    const float* xs, int n, int gs = kGroupSize) {
    const __m256i ones = _mm256_set1_epi16(1);
    float out = 0.0f;
    for (int g = 0; g < n / gs; ++g) {
        __m256i acc = _mm256_setzero_si256();
        const std::int8_t* wp = wq + g * gs;
        const std::int8_t* xp = xq + g * gs;
        for (int j = 0; j < gs; j += 32) {
            const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp + j));
            const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xp + j));
            const __m256i aw = _mm256_sign_epi8(w, w);   // |w|, treated as unsigned
            const __m256i sx = _mm256_sign_epi8(x, w);   // x with w's sign applied
            __m256i p = _mm256_maddubs_epi16(aw, sx);    // 16 x int16 pair sums
            p = _mm256_madd_epi16(p, ones);              // widen+sum -> 8 x int32
            acc = _mm256_add_epi32(acc, p);
        }
        out += static_cast<float>(hsum_i32(acc)) * ws[g] * xs[g];
    }
    return out;
}

}  // namespace ember
