// ember - a CPU inference engine for transformer language models.
//
// Usage: ember <model.bin> <tokenizer.bin> [prompt] [steps]
// Greedy (argmax) decoding, so output is deterministic and can be checked
// token-for-token against the reference implementation.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>

#include "ember/model.hpp"
#include "ember/tokenizer.hpp"

namespace {

int argmax(std::span<const float> v) {
    int best = 0;
    for (int i = 1; i < static_cast<int>(v.size()); ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <model.bin> <tokenizer.bin> [prompt] [steps] [threads] [q]\n"
                     "  pass 'q' as the last argument to run the int8-quantized model\n",
                     argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string tok_path = argv[2];
    const std::string prompt = argc > 3 ? argv[3] : "";
    int steps = argc > 4 ? std::atoi(argv[4]) : 256;
    const unsigned threads = argc > 5 ? static_cast<unsigned>(std::atoi(argv[5])) : 0;
    const bool quantize = argc > 6 && std::string(argv[6]) == "q";

    ember::Transformer model(model_path, threads, ember::Kernel::Simd, quantize);
    ember::Tokenizer tok(tok_path, model.config().vocab_size);

    if (steps <= 0 || steps > model.config().seq_len) steps = model.config().seq_len;

    const std::vector<int> prompt_tokens = tok.encode(prompt, /*bos=*/true, /*eos=*/false);
    if (prompt_tokens.empty()) {
        std::fprintf(stderr, "ember: prompt produced no tokens\n");
        return 1;
    }

    // Standard autoregressive loop: feed the prompt, then sample greedily.
    int token = prompt_tokens[0];
    int pos = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (pos < steps) {
        const std::span<const float> logits = model.forward(token, pos);
        int next = (pos < static_cast<int>(prompt_tokens.size()) - 1) ? prompt_tokens[pos + 1]
                                                                      : argmax(logits);
        ++pos;
        if (next == 1) break;  // BOS doubles as the end-of-sequence delimiter here
        std::fputs(tok.decode(token, next).c_str(), stdout);
        std::fflush(stdout);
        token = next;
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::putchar('\n');

    const double secs = std::chrono::duration<double>(t1 - t0).count();
    if (secs > 0.0)
        std::fprintf(stderr, "\n[%d tokens in %.3f s = %.1f tok/s]\n", pos, secs, pos / secs);
    return 0;
}
