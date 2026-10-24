#pragma once
//
// SentencePiece-style BPE tokenizer in the llama2.c binary format:
//   [int max_token_length]
//   then vocab_size records of: [float score][int len][len bytes]
//
#include <string>
#include <unordered_map>
#include <vector>

namespace ember {

class Tokenizer {
public:
    // vocab_size comes from the model config (the token file does not store it).
    Tokenizer(const std::string& path, int vocab_size);

    // Encode text to token ids, optionally wrapping with BOS (1) / EOS (2).
    [[nodiscard]] std::vector<int> encode(const std::string& text, bool bos, bool eos) const;

    // Decode a single token to its text piece, given the previous token for the
    // SentencePiece leading-space rule. Raw-byte tokens (<0xNN>) become that byte.
    [[nodiscard]] std::string decode(int prev_token, int token) const;

private:
    [[nodiscard]] int lookup(const std::string& s) const;

    std::vector<std::string> vocab_;
    std::vector<float> scores_;
    std::unordered_map<std::string, int> index_;  // exact-string -> id
    int vocab_size_ = 0;
    unsigned int max_token_length_ = 0;
};

}  // namespace ember
