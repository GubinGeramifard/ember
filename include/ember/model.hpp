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

namespace ember {

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

    void allocate(const Config& c);
};

class Transformer {
public:
    // Load a checkpoint in the Llama-2 (llama2.c) binary format.
    explicit Transformer(const std::string& checkpoint_path);

    // Run one decode step for `token` at position `pos`; returns logits (vocab_size).
    // The span is valid until the next call to forward().
    [[nodiscard]] std::span<const float> forward(int token, int pos);

    [[nodiscard]] const Config& config() const noexcept { return config_; }

private:
    void map_weights(bool shared_classifier);

    Config config_{};
    std::vector<std::byte> storage_;  // owns the entire weight buffer
    Weights w_{};
    RunState s_{};
};

}  // namespace ember
