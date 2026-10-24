// Benchmark harness: measures the optimization journey from a naive scalar
// baseline through the SIMD kernel, multithreading, and int8 quantization.
//
// Two numbers per configuration:
//   tok/s   - throughput, from free generation
//   agree%  - fraction of next-token predictions that match the fp32 model,
//             measured by teacher-forcing the fp32 reference sequence. fp32
//             configs are 100% by construction; int8 shows its (small) cost.
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

// Free greedy generation; returns the full token sequence fed (seq[0] = first
// prompt token, then each produced token) and the throughput in tokens/sec.
std::pair<std::vector<int>, double> generate(ember::Transformer& model,
                                             const ember::Tokenizer& tok,
                                             const std::string& prompt, int steps) {
    const std::vector<int> prompt_tokens = tok.encode(prompt, /*bos=*/true, /*eos=*/false);
    std::vector<int> seq{prompt_tokens[0]};
    int token = prompt_tokens[0];
    int pos = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (pos < steps) {
        const auto logits = model.forward(token, pos);
        int next = (pos < static_cast<int>(prompt_tokens.size()) - 1) ? prompt_tokens[pos + 1]
                                                                      : argmax(logits);
        ++pos;
        if (next == 1) break;
        seq.push_back(next);
        token = next;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return {seq, secs > 0.0 ? pos / secs : 0.0};
}

// Teacher-forced next-token agreement with a reference sequence.
double agreement(ember::Transformer& model, const std::vector<int>& ref) {
    int matches = 0, total = 0;
    for (std::size_t pos = 0; pos + 1 < ref.size(); ++pos) {
        const auto logits = model.forward(ref[pos], static_cast<int>(pos));
        if (argmax(logits) == ref[pos + 1]) ++matches;
        ++total;
    }
    return total ? 100.0 * matches / total : 0.0;
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

    // fp32 reference sequence (what every config is scored against).
    std::vector<int> reference;
    {
        ember::Transformer ref(model_path, hw, ember::Kernel::Simd, /*quantize=*/false);
        ember::Tokenizer tok(tok_path, ref.config().vocab_size);
        reference = generate(ref, tok, prompt, steps).first;
    }

    struct Cfg {
        std::string label;
        unsigned threads;
        ember::Kernel kernel;
        bool quantize;
    };
    const std::vector<Cfg> cfgs = {
        {"scalar  x1", 1, ember::Kernel::Scalar, false},
        {"simd    x1", 1, ember::Kernel::Simd, false},
        {"simd    x4", std::min(4u, hw), ember::Kernel::Simd, false},
        {"simd    x" + std::to_string(hw), hw, ember::Kernel::Simd, false},
        {"int8    x1", 1, ember::Kernel::Simd, true},
        {"int8    x4", std::min(4u, hw), ember::Kernel::Simd, true},
        {"int8    x" + std::to_string(hw), hw, ember::Kernel::Simd, true},
    };

    std::printf("ember benchmark  |  %d steps, prompt \"%s\"  |  %u hardware threads\n\n",
                steps, prompt.c_str(), hw);
    std::printf("%-12s %12s %10s %9s\n", "config", "tok/s", "speedup", "agree%");
    std::printf("%-12s %12s %10s %9s\n", "------", "-----", "-------", "------");

    double baseline = 0.0;
    for (const Cfg& c : cfgs) {
        ember::Transformer model(model_path, c.threads, c.kernel, c.quantize);
        ember::Tokenizer tok(tok_path, model.config().vocab_size);
        const double tps = generate(model, tok, prompt, steps).second;
        const double agree = agreement(model, reference);
        if (baseline == 0.0) baseline = tps;
        std::printf("%-12s %12.1f %9.2fx %8.1f%%\n", c.label.c_str(), tps,
                    baseline > 0 ? tps / baseline : 0.0, agree);
    }
    std::printf("\n(agree%% = next-token match vs the fp32 model; int8 trades a little for speed)\n");
    return 0;
}
