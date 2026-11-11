// SentencePiece (BPE with byte fallback) tokenizer read straight from a serialized ModelProto.
#pragma once
#include <string>
#include <unordered_map>
#include <vector>

class SentencePiece {
public:
    bool load(const std::string& modelProto, std::string& err);
    std::vector<int> encode(const std::string& utf8) const;
    int vocabSize() const { return (int)pieces_.size(); }
    // info for debugging
    int modelType = 0;                 // 1 unigram, 2 bpe
    bool addDummyPrefix = true, removeExtraWhitespaces = true, escapeWhitespaces = true, byteFallback = false;
    size_t charsmapBytes = 0;
private:
    struct Piece { std::string s; float score; int type; };
    std::vector<Piece> pieces_;
    std::unordered_map<std::string, int> index_;
    int unk_ = 0;
    int byteId_[256];
};
