// Krea2 conditioning: Qwen3-VL-4B text tower on ggml, 12 hidden-state taps.
#pragma once
#include <functional>
#include <string>
#include <vector>

struct ggml_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_tensor;

class Krea2TextEncoder {
public:
    ~Krea2TextEncoder() { unload(); }
    // Loads the scaled-FP8 safetensors and converts the linears to Q8_0 (about 4 GB RAM).
    bool load(const std::string& path, int threads, std::string& err, const std::function<void(float)>& progress = nullptr);
    void unload();
    bool loaded() const { return backend_ != nullptr; }

    // ids = tokens of the templated prompt. Output: tokens x (12*2560), tap-major per token, with the
    // system/user prefix stripped exactly like ComfyUI. Returns the number of kept tokens in `seq`.
    bool encode(const std::vector<int>& ids, std::vector<float>& out, int& seq, std::string& err);

    static constexpr int kLayers = 36, kHidden = 2560, kHeads = 32, kKvHeads = 8, kHeadDim = 128, kFF = 9728, kTaps = 12;
    static const int* tapLayers();

private:
    struct Layer { ggml_tensor *qn, *kn, *q, *k, *v, *o, *inLn, *postLn, *gate, *up, *down; };
    ggml_context* ctx_ = nullptr;
    ggml_backend* backend_ = nullptr;
    ggml_backend_buffer* buf_ = nullptr;
    std::vector<Layer> layers_;
    std::vector<float> embed_;          // not kept: embeddings are read from the mapped file
    void* st_ = nullptr;                // StFile* (kept open for embed_tokens lookups)
    int threads_ = 4;
};
