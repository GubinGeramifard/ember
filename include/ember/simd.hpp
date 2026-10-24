#pragma once
//
// Minimal AVX2 + FMA helpers. This header is the foundation of the engine's
// hot path: every matmul reduces to a dot product, and the dot product is
// where almost all inference time goes, so it is worth hand-vectorizing.
//
#include <immintrin.h>

#include <cstddef>

namespace ember {

// Horizontal sum of the 8 lanes of a __m256 (a + b + c + ... + h).
inline float hsum256(__m256 v) {
    // Add the high 128 bits to the low 128 bits, then reduce within 128 bits.
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);         // 4 partial sums
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum)); // fold 4 -> 2
    sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1)); // fold 2 -> 1
    return _mm_cvtss_f32(sum);
}

// Dot product of two float vectors using AVX2 + FMA, with a scalar tail.
// This is the scalar-free building block the whole engine leans on.
inline float dot(const float* a, const float* b, std::size_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    std::size_t i = 0;
    // Two accumulators hide FMA latency by keeping the pipeline busy.
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    float result = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < n; ++i) {  // scalar tail for the last < 8 elements
        result += a[i] * b[i];
    }
    return result;
}

}  // namespace ember
