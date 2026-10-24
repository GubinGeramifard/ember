#pragma once
//
// The transformer itself: owns the weight buffer, exposes a single forward()
// that advances the model by one token at sequence position `pos` and returns
// the logits over the vocabulary.
//
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "ember/config.hpp"
#include "ember/quant.hpp"
#include "ember/threadpool.hpp"

namespace ember {

// Which dot-product kernel the matmul uses. Scalar is the naive baseline;
// Simd is the AVX2 + FMA path. Selectable so the benchmark can compare them.
enum class Kernel { Scalar, Simd };

// A weight matrix as seen by matmul: either fp32 rows, or int8 rows with scales.
struct Linear {
    const float* f = nullptr;        // fp32 weights (row-major), or
    const std::int8_t* q = nullptr;  // int8 weights, with
    const float* s = nullptr;        // per-group scales
    [[nodiscard]] bool quantized() const noexcept { return q != nullptr; }
};

// Views into the loaded weight buffer. Nothing here owns memory; the backing
// store lives in Transformer::storage_. Pointers follow the exact on-disk order.
struct Weights {
    const float* token_embedding = nullptr;  // (vocab_size, dim)
    const float* rms_att = nullptr;          // (layer, dim)
    const float* wq = nullptr;               // (layer, dim, n_heads*head_size)
    const float* wk = nullptr;               // (layer, dim, n_kv_heads*head_size)
    const float* wv = nullptr;               // (layer, dim, n_kv_heads*head_size)
    const float* wo = nullptr;               // (layer, n_heads*head_size, dim)
    const float* rms_ffn = nullptr;          // (layer, dim)
    const float* w1 = nullptr;               // (layer, hidden_dim, dim)
    const float* w2 = nullptr;               // (layer, dim, hidden_dim)
    const float* w3 = nullptr;               // (layer, hidden_dim, dim)
    const float* rms_final = nullptr;        // (dim,)
    const float* wcls = nullptr;             // (vocab_size, dim); may alias embedding
};

// Scratch activations and the KV cache, allocated once and reused every step.
struct RunState {
    std::vector<float> x, xb, xb2;   // residual-stream buffers (dim)
    std::vector<float> hb, hb2;      // FFN hidden buffers (hidden_dim)
    std::vector<float> q;            // query (dim)
    std::vector<float> att;          // attention scores (n_heads, seq_len)
    std::vector<float> logits;       // output logits (vocab_size)
    std::vector<float> key_cache;    // (layer, seq_len, kv_dim)
    std::vector<float> value_cache;  // (layer, seq_len, kv_dim)

    std::vector<std::int8_t> xq;     // scratch: quantized activation
    std::vector<float> xs;           // scratch: activation group scales

    void allocate(const Config& c);
};

class Transformer {
public:
    // Load a checkpoint in the Llama-2 (llama2.c) binary format.
    //   n_threads: matmul parallelism; 0 means use all hardware threads.
    //   kernel:    fp32 dot-product implementation (SIMD by default).
    //   quantize:  if true, quantize weights to int8 and free the fp32 copy.
    explicit Transformer(const std::string& checkpoint_path, unsigned n_threads = 0,
                         Kernel kernel = Kernel::Simd, bool quantize = false);

    // Run one decode step for `token` at position `pos`; returns logits (vocab_size).
    // The span is valid until the next call to forward().
    [[nodiscard]] std::span<const float> forward(int token, int pos);

    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] unsigned threads() const noexcept { return pool_.size(); }
    [[nodiscard]] Kernel kernel() const noexcept { return kernel_; }
    [[nodiscard]] bool quantized() const noexcept { return quantized_; }

private:
    void map_weights(bool shared_classifier);
    void quantize_weights(bool shared_classifier);
    // xout(d) = W(d,n) @ x(n), parallelized across the pool.
    void matmul(float* xout, const Linear& w, const float* x, int n, int d);
    // Build a Linear view of one layer's weight block (fp32 or quantized).
    [[nodiscard]] Linear linear(const float* fbase, const QTensor& qt,
                                unsigned long long rows_before, int n) const;

    Config config_{};
    std::vector<std::byte> storage_;  // owns the fp32 weight buffer (freed after quantizing)
    Weights w_{};
    RunState s_{};
    Kernel kernel_;
    bool quantized_ = false;
    ThreadPool pool_;

    // Quantized weights (populated only when quantized_). The classifier shares
    // the token-embedding tensor when the checkpoint marks weights as shared.
    QTensor q_token_emb_, q_wq_, q_wk_, q_wv_, q_wo_, q_w1_, q_w2_, q_w3_, q_wcls_;
    const QTensor* q_cls_ = nullptr;
    std::vector<float> rms_att_f_, rms_ffn_f_, rms_final_f_;  // norms kept in fp32
};

}  // namespace ember
