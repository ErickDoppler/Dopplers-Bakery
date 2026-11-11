// T5Gemma encoder (text conditioning of Stable Audio 3) on ggml, with its SentencePiece tokenizer.
#pragma once
#include <string>
#include <vector>
#include "audio/spiece.h"

struct ggml_context; struct ggml_backend_buffer; struct ggml_tensor;
class StFile;

class T5GemmaEncoder {
public:
    ~T5GemmaEncoder() { unload(); }
    bool load(const std::string& path, std::string& err);       // t5gemma_b_b_ul2.safetensors
    void unload();
    bool loaded() const { return buf_ != nullptr; }
    std::vector<int> tokenize(const std::string& text) const { return sp_.encode(text); }
    // out: n x 768 (token-major), n = number of tokens
    bool encode(const std::vector<int>& ids, std::vector<float>& out, std::string& err);
    const SentencePiece& tokenizer() const { return sp_; }
private:
    struct Layer { ggml_tensor *n1, *n2, *n3, *n4, *q, *k, *v, *o, *gate, *up, *down; };
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer* buf_ = nullptr;
    StFile* st_ = nullptr;
    std::vector<Layer> layers_;
    ggml_tensor* norm_ = nullptr;
    SentencePiece sp_;
};
