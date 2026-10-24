// Benchmark harness: measures the optimization journey from a naive scalar
// baseline to the SIMD kernel to multithreaded matmul, and confirms that every
// configuration produces the *same* tokens (optimizing must never change output).
//
// Usage: bench <model.bin> <tokenizer.bin> [steps]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "ember/model.hpp"
#include "ember/tokenizer.hpp"

namespace {

int argmax(std::span<const float> v) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(v.size()); ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

struct Result {
    std::vector<int> tokens;
    double tok_per_s = 0.0;
};

// Greedily generate `steps` tokens from `prompt`, timing the whole loop.
Result run(ember::Transformer& model, const ember::Tokenizer& tok, const std::string& prompt,
           int steps) {
    const std::vector<int> prompt_tokens = tok.encode(prompt, /*bos=*/true, /*eos=*/false);
    Result r;
    int token = prompt_tokens[0];
    int pos = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (pos < steps) {
        const auto logits = model.forward(token, pos);
        int next = (pos < static_cast<int>(prompt_tokens.size()) - 1) ? prompt_tokens[pos + 1]
                                                                      : argmax(logits);
        ++pos;
        if (next == 1) break;
        r.tokens.push_back(next);
        token = next;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    r.tok_per_s = secs > 0.0 ? pos / secs : 0.0;
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.bin> <tokenizer.bin> [steps]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string tok_path = argv[2];
    const int steps = argc > 3 ? std::atoi(argv[3]) : 256;
    const std::string prompt = "Once upon a time";

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());

    // (label, threads, kernel). Build the thread-count ladder up to the machine's max.
    struct Cfg {
        std::string label;
        unsigned threads;
        ember::Kernel kernel;
    };
    std::vector<Cfg> cfgs;
    cfgs.push_back({"scalar  x1", 1, ember::Kernel::Scalar});
    cfgs.push_back({"simd    x1", 1, ember::Kernel::Simd});
    for (unsigned t : {2u, 4u, 6u, 8u, 12u, 16u}) {
        if (t <= hw) cfgs.push_back({"simd    x" + std::to_string(t), t, ember::Kernel::Simd});
    }
    if (cfgs.back().threads != hw)
        cfgs.push_back({"simd    x" + std::to_string(hw), hw, ember::Kernel::Simd});

    std::printf("ember benchmark  |  %d steps, prompt \"%s\"  |  %u hardware threads\n\n",
                steps, prompt.c_str(), hw);
    std::printf("%-14s %12s %10s %9s\n", "config", "tok/s", "speedup", "output");
    std::printf("%-14s %12s %10s %9s\n", "------", "-----", "-------", "------");

    double baseline = 0.0;
    std::vector<int> reference_tokens;
    for (const Cfg& c : cfgs) {
        ember::Transformer model(model_path, c.threads, c.kernel);
        ember::Tokenizer tok(tok_path, model.config().vocab_size);
        const Result r = run(model, tok, prompt, steps);

        if (reference_tokens.empty()) {
            reference_tokens = r.tokens;
            baseline = r.tok_per_s;
        }
        const bool same = (r.tokens == reference_tokens);
        std::printf("%-14s %12.1f %9.2fx %9s\n", c.label.c_str(), r.tok_per_s,
                    baseline > 0 ? r.tok_per_s / baseline : 0.0, same ? "ok" : "DIFFERS");
    }
    std::printf("\n(\"ok\" = produced byte-identical tokens to the scalar baseline)\n");
    return 0;
}
