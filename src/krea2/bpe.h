// Qwen2/Qwen3 byte-level BPE tokenizer (vocab.json + merges.txt + added tokens).
#pragma once
#include <string>
#include <unordered_map>
#include <vector>

class QwenTokenizer {
public:
    // Loads the files embedded in the exe (resources 101..103), or from a folder if given.
    bool load(const std::string& folder, std::string& err);
    std::vector<int> encode(const std::string& utf8) const;
    int vocabSize() const { return (int)vocab_.size(); }
private:
    std::vector<int> encodeChunk(const std::string& text) const;
    std::vector<int> bpe(const std::string& mapped) const;
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<std::string, int> rank_;            // "a b" -> rank
    std::vector<std::pair<std::string, int>> added_;       // special/added tokens, longest first
    std::string b2u_[256];                                 // byte -> utf8 of mapped char
};

// The conditioning prompt used by Krea2 (Qwen-Image system template).
std::string krea2Template(const std::string& text);
