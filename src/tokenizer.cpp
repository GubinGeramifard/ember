#include "ember/tokenizer.hpp"

#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace ember {

Tokenizer::Tokenizer(const std::string& path, int vocab_size) : vocab_size_(vocab_size) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("ember: cannot open tokenizer " + path);

    auto read_i32 = [&](auto& out) {
        return static_cast<bool>(file.read(reinterpret_cast<char*>(&out), sizeof(out)));
    };

    if (!read_i32(max_token_length_)) throw std::runtime_error("ember: bad tokenizer header");

    vocab_.reserve(vocab_size);
    scores_.reserve(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        float score = 0.0f;
        int len = 0;
        if (!read_i32(score) || !read_i32(len) || len < 0)
            throw std::runtime_error("ember: truncated tokenizer file");
        std::string piece(static_cast<std::size_t>(len), '\0');
        if (len > 0 && !file.read(piece.data(), len))
            throw std::runtime_error("ember: truncated token string");
        scores_.push_back(score);
        index_.emplace(piece, i);  // first occurrence wins, matching vocab order
        vocab_.push_back(std::move(piece));
    }
}

int Tokenizer::lookup(const std::string& s) const {
    const auto it = index_.find(s);
    return it == index_.end() ? -1 : it->second;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool bos, bool eos) const {
    std::vector<int> tokens;
    if (bos) tokens.push_back(1);

    // SentencePiece prepends a space to non-empty input (the "dummy prefix").
    if (!text.empty()) {
        const int space = lookup(" ");
        if (space != -1) tokens.push_back(space);
    }

    // Greedily map each UTF-8 codepoint to a token, falling back to raw bytes.
    std::string cp;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if ((c & 0xC0) != 0x80) cp.clear();  // not a continuation byte: start fresh
        cp.push_back(static_cast<char>(c));
        const bool next_is_continuation =
            i + 1 < text.size() && (static_cast<unsigned char>(text[i + 1]) & 0xC0) == 0x80;
        if (next_is_continuation && cp.size() < 4) continue;

        const int id = lookup(cp);
        if (id != -1) {
            tokens.push_back(id);
        } else {
            // byte_fallback: the first 3 vocab slots are <unk>,<s>,</s>, so bytes start at +3.
            for (unsigned char b : cp) tokens.push_back(static_cast<int>(b) + 3);
        }
        cp.clear();
    }

    // Repeatedly merge the highest-scoring adjacent pair until none remain.
    for (;;) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;
        for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
            const int id = lookup(vocab_[tokens[i]] + vocab_[tokens[i + 1]]);
            if (id != -1 && scores_[id] > best_score) {
                best_score = scores_[id];
                best_id = id;
                best_idx = static_cast<int>(i);
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        tokens.erase(tokens.begin() + best_idx + 1);
    }

    if (eos) tokens.push_back(2);
    return tokens;
}

std::string Tokenizer::decode(int prev_token, int token) const {
    const std::string& piece = vocab_[token];
    // After BOS, SentencePiece strips a single leading space.
    std::size_t start = (prev_token == 1 && !piece.empty() && piece[0] == ' ') ? 1 : 0;

    // Raw-byte tokens look like "<0x0A>"; convert to the actual byte.
    unsigned int byte_val = 0;
    if (std::sscanf(piece.c_str(), "<0x%02X>", &byte_val) == 1)
        return std::string(1, static_cast<char>(byte_val));

    return piece.substr(start);
}

}  // namespace ember
