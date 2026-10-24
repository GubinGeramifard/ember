#pragma once
//
// Model hyperparameters, read verbatim from the checkpoint header. The field
// order and types match the on-disk Llama-2 (llama2.c) format exactly: seven
// 32-bit integers, so this struct can be read directly from the file head.
//
#include <cstdint>

namespace ember {

struct Config {
    std::int32_t dim = 0;          // transformer embedding dimension
    std::int32_t hidden_dim = 0;   // FFN inner dimension
    std::int32_t n_layers = 0;     // number of transformer blocks
    std::int32_t n_heads = 0;      // number of query heads
    std::int32_t n_kv_heads = 0;   // number of key/value heads (<= n_heads for GQA)
    std::int32_t vocab_size = 0;   // vocabulary size
    std::int32_t seq_len = 0;      // max sequence length the model supports

    // Dimension of a single attention head.
    [[nodiscard]] int head_size() const noexcept { return dim / n_heads; }

    // Total width of the key/value projection (smaller than dim under GQA).
    [[nodiscard]] int kv_dim() const noexcept { return (dim * n_kv_heads) / n_heads; }

    // How many query heads share each key/value head (1 = full multi-head).
    [[nodiscard]] int kv_mul() const noexcept { return n_heads / n_kv_heads; }
};

static_assert(sizeof(Config) == 7 * sizeof(std::int32_t),
              "Config must match the 28-byte on-disk checkpoint header");

}  // namespace ember
