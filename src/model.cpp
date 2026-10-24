#include "ember/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>

#include "ember/simd.hpp"

namespace ember {
namespace {

// o = weight * (x / rms(x)), the Llama RMSNorm (no mean-subtraction).
void rmsnorm(float* o, const float* x, const float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; ++j) ss += x[j] * x[j];
    ss = 1.0f / std::sqrt(ss / static_cast<float>(size) + 1e-5f);
    for (int j = 0; j < size; ++j) o[j] = weight[j] * (ss * x[j]);
}

// In-place numerically-stable softmax over the first `size` elements.
void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) max_val = x[i] > max_val ? x[i] : max_val;
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; ++i) x[i] /= sum;
}

}  // namespace

void RunState::allocate(const Config& c) {
    const int kv_dim = c.kv_dim();
    x.assign(c.dim, 0.0f);
    xb.assign(c.dim, 0.0f);
    xb2.assign(c.dim, 0.0f);
    hb.assign(c.hidden_dim, 0.0f);
    hb2.assign(c.hidden_dim, 0.0f);
    q.assign(c.dim, 0.0f);
    att.assign(static_cast<std::size_t>(c.n_heads) * c.seq_len, 0.0f);
    logits.assign(c.vocab_size, 0.0f);
    key_cache.assign(static_cast<std::size_t>(c.n_layers) * c.seq_len * kv_dim, 0.0f);
    value_cache.assign(static_cast<std::size_t>(c.n_layers) * c.seq_len * kv_dim, 0.0f);
}

namespace {
unsigned resolve_threads(unsigned requested) {
    if (requested != 0) return requested;
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : hw;
}
}  // namespace

// xout(d) = W(d,n) @ x(n): each output row is one dot product. The d rows are
// split across the thread pool, and each uses the selected kernel.
void Transformer::matmul(float* xout, const float* x, const float* w, int n, int d) {
    const Kernel k = kernel_;
    pool_.parallel_for(d, [=](int begin, int end) {
        if (k == Kernel::Simd) {
            for (int i = begin; i < end; ++i)
                xout[i] = dot(w + static_cast<std::size_t>(i) * n, x, static_cast<std::size_t>(n));
        } else {
            for (int i = begin; i < end; ++i)
                xout[i] = dot_scalar(w + static_cast<std::size_t>(i) * n, x,
                                     static_cast<std::size_t>(n));
        }
    });
}

Transformer::Transformer(const std::string& checkpoint_path, unsigned n_threads, Kernel kernel)
    : kernel_(kernel), pool_(resolve_threads(n_threads)) {
    std::ifstream file(checkpoint_path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("ember: cannot open checkpoint " + checkpoint_path);
    const std::streamsize size = file.tellg();
    if (size < static_cast<std::streamsize>(sizeof(Config)))
        throw std::runtime_error("ember: checkpoint too small to contain a header");
    file.seekg(0);

    storage_.resize(static_cast<std::size_t>(size));
    if (!file.read(reinterpret_cast<char*>(storage_.data()), size))
        throw std::runtime_error("ember: failed to read checkpoint");

    std::memcpy(&config_, storage_.data(), sizeof(Config));
    // A negative vocab_size is the legacy flag for an unshared classifier matrix.
    const bool shared_classifier = config_.vocab_size > 0;
    config_.vocab_size = std::abs(config_.vocab_size);

    map_weights(shared_classifier);
    s_.allocate(config_);
}

void Transformer::map_weights(bool shared_classifier) {
    const Config& p = config_;
    const int head_size = p.head_size();
    const auto n_layers = static_cast<unsigned long long>(p.n_layers);

    // Weights begin immediately after the 28-byte config header.
    const float* ptr = reinterpret_cast<const float*>(storage_.data() + sizeof(Config));

    w_.token_embedding = ptr; ptr += static_cast<unsigned long long>(p.vocab_size) * p.dim;
    w_.rms_att = ptr;         ptr += n_layers * p.dim;
    w_.wq = ptr;              ptr += n_layers * p.dim * (p.n_heads * head_size);
    w_.wk = ptr;              ptr += n_layers * p.dim * (p.n_kv_heads * head_size);
    w_.wv = ptr;              ptr += n_layers * p.dim * (p.n_kv_heads * head_size);
    w_.wo = ptr;              ptr += n_layers * (p.n_heads * head_size) * p.dim;
    w_.rms_ffn = ptr;         ptr += n_layers * p.dim;
    w_.w1 = ptr;              ptr += n_layers * p.dim * p.hidden_dim;
    w_.w2 = ptr;              ptr += n_layers * p.hidden_dim * p.dim;
    w_.w3 = ptr;              ptr += n_layers * p.dim * p.hidden_dim;
    w_.rms_final = ptr;       ptr += p.dim;
    ptr += static_cast<unsigned long long>(p.seq_len) * head_size / 2;  // legacy RoPE real (unused)
    ptr += static_cast<unsigned long long>(p.seq_len) * head_size / 2;  // legacy RoPE imag (unused)
    w_.wcls = shared_classifier ? w_.token_embedding : ptr;
}

std::span<const float> Transformer::forward(int token, int pos) {
    const Config& p = config_;
    const int dim = p.dim;
    const int kv_dim = p.kv_dim();
    const int kv_mul = p.kv_mul();
    const int hidden_dim = p.hidden_dim;
    const int head_size = p.head_size();

    float* x = s_.x.data();
    std::memcpy(x, w_.token_embedding + static_cast<std::size_t>(token) * dim,
                dim * sizeof(float));

    for (unsigned long long l = 0; l < static_cast<unsigned long long>(p.n_layers); ++l) {
        rmsnorm(s_.xb.data(), x, w_.rms_att + l * dim, dim);

        // Key/value for this position live directly in the cache.
        const std::size_t loff = l * p.seq_len * kv_dim;
        float* k = s_.key_cache.data() + loff + static_cast<std::size_t>(pos) * kv_dim;
        float* v = s_.value_cache.data() + loff + static_cast<std::size_t>(pos) * kv_dim;

        matmul(s_.q.data(), s_.xb.data(), w_.wq + l * dim * dim, dim, dim);
        matmul(k, s_.xb.data(), w_.wk + l * dim * kv_dim, dim, kv_dim);
        matmul(v, s_.xb.data(), w_.wv + l * dim * kv_dim, dim, kv_dim);

        // RoPE: rotate (q, k) pairwise with position-dependent frequencies.
        for (int i = 0; i < dim; i += 2) {
            const int head_dim = i % head_size;
            const float freq = 1.0f / std::pow(10000.0f, head_dim / static_cast<float>(head_size));
            const float val = pos * freq;
            const float fcr = std::cos(val);
            const float fci = std::sin(val);
            const int rotn = i < kv_dim ? 2 : 1;  // rotate q (and k if within kv_dim)
            for (int r = 0; r < rotn; ++r) {
                float* vec = (r == 0) ? s_.q.data() : k;
                const float v0 = vec[i];
                const float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // Grouped-query attention, one head at a time.
        for (int h = 0; h < p.n_heads; ++h) {
            const float* q = s_.q.data() + h * head_size;
            float* att = s_.att.data() + static_cast<std::size_t>(h) * p.seq_len;
            for (int t = 0; t <= pos; ++t) {
                const float* kt = s_.key_cache.data() + loff +
                                  static_cast<std::size_t>(t) * kv_dim +
                                  (h / kv_mul) * head_size;
                att[t] = dot(q, kt, static_cast<std::size_t>(head_size)) /
                         std::sqrt(static_cast<float>(head_size));
            }
            softmax(att, pos + 1);

            float* xb = s_.xb.data() + h * head_size;
            std::memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; ++t) {
                const float* vt = s_.value_cache.data() + loff +
                                  static_cast<std::size_t>(t) * kv_dim +
                                  (h / kv_mul) * head_size;
                const float a = att[t];
                for (int i = 0; i < head_size; ++i) xb[i] += a * vt[i];
            }
        }

        matmul(s_.xb2.data(), s_.xb.data(), w_.wo + l * dim * dim, dim, dim);
        for (int i = 0; i < dim; ++i) x[i] += s_.xb2[i];

        // Feed-forward: w2( silu(w1 x) * w3 x ).
        rmsnorm(s_.xb.data(), x, w_.rms_ffn + l * dim, dim);
        matmul(s_.hb.data(), s_.xb.data(), w_.w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s_.hb2.data(), s_.xb.data(), w_.w3 + l * dim * hidden_dim, dim, hidden_dim);
        for (int i = 0; i < hidden_dim; ++i) {
            float val = s_.hb[i];
            val *= 1.0f / (1.0f + std::exp(-val));  // SiLU
            s_.hb[i] = val * s_.hb2[i];
        }
        matmul(s_.xb.data(), s_.hb.data(), w_.w2 + l * dim * hidden_dim, hidden_dim, dim);
        for (int i = 0; i < dim; ++i) x[i] += s_.xb[i];
    }

    rmsnorm(x, x, w_.rms_final, dim);
    matmul(s_.logits.data(), x, w_.wcls, p.dim, p.vocab_size);
    return {s_.logits.data(), static_cast<std::size_t>(p.vocab_size)};
}

}  // namespace ember
