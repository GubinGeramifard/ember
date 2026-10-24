// ember test suite. No external framework: a few macros, a nonzero exit on
// failure. Pure tests (SIMD, quantization, thread pool) always run and are what
// CI exercises; tests that need model/tokenizer files skip cleanly when absent.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "ember/model.hpp"
#include "ember/quant.hpp"
#include "ember/simd.hpp"
#include "ember/threadpool.hpp"
#include "ember/tokenizer.hpp"

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond)                                                      \
    do {                                                                 \
        ++g_checks;                                                      \
        if (!(cond)) {                                                   \
            ++g_failures;                                                \
            std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
        }                                                                \
    } while (0)

#define CHECK_NEAR(a, b, eps) CHECK(std::fabs((a) - (b)) <= (eps))

int argmax(std::span<const float> v) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(v.size()); ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

// --- Pure tests (no data files) -------------------------------------------

void test_simd_dot() {
    std::printf("simd dot vs scalar\n");
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> d(-2.0f, 2.0f);
    for (int n : {1, 7, 8, 15, 16, 31, 100, 1000}) {
        std::vector<float> a(n), b(n);
        for (int i = 0; i < n; ++i) { a[i] = d(rng); b[i] = d(rng); }
        const float s = ember::dot_scalar(a.data(), b.data(), n);
        const float v = ember::dot(a.data(), b.data(), n);
        CHECK_NEAR(v, s, 1e-3f * (std::fabs(s) + 1.0f));
    }
}

void test_threadpool() {
    std::printf("thread pool parallel_for coverage\n");
    for (unsigned t : {1u, 2u, 4u, 8u}) {
        ember::ThreadPool pool(t);
        for (int n : {0, 1, 5, 33, 1000}) {
            std::vector<int> cover(static_cast<std::size_t>(n < 0 ? 0 : n), 0);
            pool.parallel_for(n, [&](int b, int e) {
                for (int i = b; i < e; ++i) cover[i]++;  // ranges are disjoint
            });
            bool ok = true;
            for (int c : cover) ok = ok && (c == 1);
            CHECK(ok);  // every index covered exactly once
        }
    }
}

void test_quantization() {
    std::printf("int8 quantization error + dot\n");
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-3.0f, 3.0f);
    const int n = 256;
    std::vector<float> w(n), x(n);
    for (int i = 0; i < n; ++i) { w[i] = d(rng); x[i] = d(rng); }

    // Dequantized values are within one quantization step of the originals.
    const ember::QTensor qw = ember::quantize(w.data(), n);
    std::vector<float> dw(n);
    ember::dequantize_row(qw.q.data(), qw.s.data(), dw.data(), n);
    for (int g = 0; g < n / ember::kGroupSize; ++g)
        for (int j = 0; j < ember::kGroupSize; ++j) {
            const int i = g * ember::kGroupSize + j;
            CHECK(std::fabs(dw[i] - w[i]) <= qw.s[g] + 1e-6f);
        }

    // Quantized dot is close to the exact fp32 dot (relative error a few %).
    std::vector<std::int8_t> xq(n);
    std::vector<float> xs(n / ember::kGroupSize);
    ember::quantize_into(x.data(), n, xq.data(), xs.data());
    const float exact = ember::dot_scalar(w.data(), x.data(), n);
    const float approx = ember::dot_q8(qw.q.data(), qw.s.data(), xq.data(), xs.data(), n);
    CHECK_NEAR(approx, exact, 0.03f * (std::fabs(exact) + 1.0f));
}

// --- Model tests (skip when weights are absent) ---------------------------

std::string find(const std::string& name) {
    for (const char* base : {"models/", "../models/", "../../models/"}) {
        std::string path = std::string(base) + name;
        if (std::filesystem::exists(path)) return path;
    }
    return {};
}

void test_tokenizer_roundtrip(const std::string& tok_path, int vocab) {
    std::printf("tokenizer round-trip\n");
    ember::Tokenizer tok(tok_path, vocab);
    const std::string text = "the little robot found a shiny key";
    const std::vector<int> ids = tok.encode(text, /*bos=*/true, /*eos=*/false);
    std::string back;
    for (std::size_t i = 1; i < ids.size(); ++i)  // skip BOS
        back += tok.decode(ids[i - 1], ids[i]);
    CHECK(back == text);
}

void test_generation(const std::string& model_path, const std::string& tok_path) {
    std::printf("fp32 generation + int8 agreement\n");
    const int steps = 64;
    const std::string prompt = "Once upon a time";

    // fp32: produce a reference sequence and sanity-check the text.
    std::vector<int> ref{};
    {
        ember::Transformer m(model_path, 0, ember::Kernel::Simd, false);
        ember::Tokenizer tok(tok_path, m.config().vocab_size);
        const auto pt = tok.encode(prompt, true, false);
        int token = pt[0], pos = 0;
        ref.push_back(token);
        std::string out;
        while (pos < steps) {
            const auto logits = m.forward(token, pos);
            int next = (pos < static_cast<int>(pt.size()) - 1) ? pt[pos + 1] : argmax(logits);
            ++pos;
            if (next == 1) break;
            out += tok.decode(token, next);
            ref.push_back(next);
            token = next;
        }
        CHECK(out.rfind("Once upon a time", 0) == 0);  // begins with the prompt
        CHECK(out.size() > 50);                         // produced real text
    }

    // int8: next-token agreement with fp32 over the reference sequence.
    {
        ember::Transformer m(model_path, 0, ember::Kernel::Simd, true);
        int matches = 0, total = 0;
        for (std::size_t pos = 0; pos + 1 < ref.size(); ++pos) {
            const auto logits = m.forward(ref[pos], static_cast<int>(pos));
            if (argmax(logits) == ref[pos + 1]) ++matches;
            ++total;
        }
        const double agree = total ? static_cast<double>(matches) / total : 0.0;
        std::printf("  int8 agreement with fp32: %.1f%%\n", 100.0 * agree);
        CHECK(agree >= 0.95);  // quantization keeps predictions nearly identical
    }
}

}  // namespace

int main() {
    std::printf("ember tests\n-----------\n");
    test_simd_dot();
    test_threadpool();
    test_quantization();

    const std::string model = find("stories15M.bin");
    const std::string tok = find("tokenizer.bin");
    if (!tok.empty()) {
        test_tokenizer_roundtrip(tok, 32000);
    } else {
        std::printf("tokenizer round-trip: SKIPPED (tokenizer.bin not found)\n");
    }
    if (!model.empty() && !tok.empty()) {
        test_generation(model, tok);
    } else {
        std::printf("generation: SKIPPED (model/tokenizer not found)\n");
    }

    std::printf("-----------\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
