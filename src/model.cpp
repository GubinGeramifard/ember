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

    // Activation-quantization scratch, sized to the widest matmul input.
    const int max_n = std::max(c.dim, c.hidden_dim);
    xq.assign(max_n, 0);
    xs.assign(max_n / kGroupSize, 0.0f);
}

namespace {
unsigned resolve_threads(unsigned requested) {
    if (requested != 0) return requested;
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : hw;
}
}  // namespace

Linear Transformer::linear(const float* fbase, const QTensor& qt, unsigned long long rows_before,
                           int n) const {
    if (quantized_)
        return Linear{nullptr, qt.q.data() + rows_before * n,
                      qt.s.data() + rows_before * (n / kGroupSize)};
    return Linear{fbase + rows_before * n, nullptr, nullptr};
}

// xout(d) = W(d,n) @ x(n): each output row is one dot product, split across the
// thread pool. Quantized weights use a dynamically-quantized activation.
void Transformer::matmul(float* xout, const Linear& w, const float* x, int n, int d) {
    if (w.quantized()) {
        quantize_into(x, n, s_.xq.data(), s_.xs.data());  // serial, O(n)
        const std::int8_t* xq = s_.xq.data();
        const float* xs = s_.xs.data();
        const std::int8_t* wq = w.q;
        const float* ws = w.s;
        pool_.parallel_for(d, [=](int begin, int end) {
            for (int i = begin; i < end; ++i)
                xout[i] = dot_q8(wq + static_cast<std::size_t>(i) * n,
                                 ws + static_cast<std::size_t>(i) * (n / kGroupSize), xq, xs, n);
        });
        return;
    }
    const Kernel k = kernel_;
    const float* wf = w.f;
    pool_.parallel_for(d, [=](int begin, int end) {
        if (k == Kernel::Simd) {
            for (int i = begin; i < end; ++i)
                xout[i] = dot(wf + static_cast<std::size_t>(i) * n, x, static_cast<std::size_t>(n));
        } else {
            for (int i = begin; i < end; ++i)
                xout[i] = dot_scalar(wf + static_cast<std::size_t>(i) * n, x,
                                     static_cast<std::size_t>(n));
        }
    });
}

Transformer::Transformer(const std::string& checkpoint_path, unsigned n_threads, Kernel kernel,
                         bool quantize)
    : kernel_(kernel), quantized_(quantize), pool_(resolve_threads(n_threads)) {
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
    if (quantized_) quantize_weights(shared_classifier);
}

// Quantize every large matrix to int8, copy the small norm weights out to fp32
// vectors, then release the fp32 buffer so only the int8 weights stay resident.
void Transformer::quantize_weights(bool shared_classifier) {
    const Config& p = config_;
    const auto L = static_cast<unsigned long long>(p.n_layers);
    const auto dim = static_cast<unsigned long long>(p.dim);
    const auto kv = static_cast<unsigned long long>(p.kv_dim());
    const auto hid = static_cast<unsigned long long>(p.hidden_dim);
    const auto vocab = static_cast<unsigned long long>(p.vocab_size);

    q_token_emb_ = quantize(w_.token_embedding, vocab * dim);
    q_wq_ = quantize(w_.wq, L * dim * dim);
    q_wk_ = quantize(w_.wk, L * dim * kv);
    q_wv_ = quantize(w_.wv, L * dim * kv);
    q_wo_ = quantize(w_.wo, L * dim * dim);
    q_w1_ = quantize(w_.w1, L * dim * hid);
    q_w2_ = quantize(w_.w2, L * hid * dim);
    q_w3_ = quantize(w_.w3, L * dim * hid);
    if (shared_classifier) {
        q_cls_ = &q_token_emb_;
    } else {
        q_wcls_ = quantize(w_.wcls, vocab * dim);
        q_cls_ = &q_wcls_;
    }

    // Norms stay in fp32 (tiny, applied elementwise); copy them out of storage_.
    rms_att_f_.assign(w_.rms_att, w_.rms_att + L * dim);
    rms_ffn_f_.assign(w_.rms_ffn, w_.rms_ffn + L * dim);
    rms_final_f_.assign(w_.rms_final, w_.rms_final + dim);

    // Release the fp32 weights; repoint the norm views at the kept copies and
    // null the rest so nothing accidentally reads the freed buffer.
    storage_.clear();
    storage_.shrink_to_fit();
    w_ = Weights{};
    w_.rms_att = rms_att_f_.data();
    w_.rms_ffn = rms_ffn_f_.data();
    w_.rms_final = rms_final_f_.data();
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
    if (quantized_) {
        dequantize_row(q_token_emb_.q.data() + static_cast<std::size_t>(token) * dim,
                       q_token_emb_.s.data() + static_cast<std::size_t>(token) * (dim / kGroupSize),
                       x, dim);
    } else {
        std::memcpy(x, w_.token_embedding + static_cast<std::size_t>(token) * dim,
                    dim * sizeof(float));
    }

    for (unsigned long long l = 0; l < static_cast<unsigned long long>(p.n_layers); ++l) {
        rmsnorm(s_.xb.data(), x, w_.rms_att + l * dim, dim);

        // Key/value for this position live directly in the cache.
        const std::size_t loff = l * p.seq_len * kv_dim;
        float* k = s_.key_cache.data() + loff + static_cast<std::size_t>(pos) * kv_dim;
        float* v = s_.value_cache.data() + loff + static_cast<std::size_t>(pos) * kv_dim;

        matmul(s_.q.data(), linear(w_.wq, q_wq_, l * dim, dim), s_.xb.data(), dim, dim);
        matmul(k, linear(w_.wk, q_wk_, l * kv_dim, dim), s_.xb.data(), dim, kv_dim);
        matmul(v, linear(w_.wv, q_wv_, l * kv_dim, dim), s_.xb.data(), dim, kv_dim);

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

        matmul(s_.xb2.data(), linear(w_.wo, q_wo_, l * dim, dim), s_.xb.data(), dim, dim);
        for (int i = 0; i < dim; ++i) x[i] += s_.xb2[i];

        // Feed-forward: w2( silu(w1 x) * w3 x ).
        rmsnorm(s_.xb.data(), x, w_.rms_ffn + l * dim, dim);
        matmul(s_.hb.data(), linear(w_.w1, q_w1_, l * hidden_dim, dim), s_.xb.data(), dim, hidden_dim);
        matmul(s_.hb2.data(), linear(w_.w3, q_w3_, l * hidden_dim, dim), s_.xb.data(), dim, hidden_dim);
        for (int i = 0; i < hidden_dim; ++i) {
            float val = s_.hb[i];
            val *= 1.0f / (1.0f + std::exp(-val));  // SiLU
            s_.hb[i] = val * s_.hb2[i];
        }
        matmul(s_.xb.data(), linear(w_.w2, q_w2_, l * dim, hidden_dim), s_.hb.data(), hidden_dim, dim);
        for (int i = 0; i < dim; ++i) x[i] += s_.xb[i];
    }

    rmsnorm(x, x, w_.rms_final, dim);
    const Linear cls = quantized_ ? Linear{nullptr, q_cls_->q.data(), q_cls_->s.data()}
                                  : Linear{w_.wcls, nullptr, nullptr};
    matmul(s_.logits.data(), cls, x, p.dim, p.vocab_size);
    return {s_.logits.data(), static_cast<std::size_t>(p.vocab_size)};
}

}  // namespace ember
