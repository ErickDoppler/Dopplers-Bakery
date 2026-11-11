// Stable Audio 3 diffusion transformer on ggml: 24 blocks of differential self-attention, differential cross-attention
// to the T5Gemma tokens, adaLN-style shared conditioning, 64 memory tokens, 256-channel latents (4096 samples each).
#pragma once
#include <functional>
#include <string>
#include <utility>
#include <vector>

struct ggml_context; struct ggml_backend_buffer; struct ggml_tensor;

class SaDiT {
public:
    ~SaDiT() { unload(); }
    bool load(const std::string& checkpointPath, std::string& err);
    void unload();
    bool loaded() const { return buf_ != nullptr; }

    // latent: T x 256 (time-major). text: n x 768 T5Gemma tokens. seconds: audio length used for the length conditioning.
    // out: velocity, same layout as latent.
    bool forward(const std::vector<float>& latent, int T, float sigma, const std::vector<float>& text, int n, float seconds,
                 std::vector<float>& out, std::string& err);

    bool debug = false;
    std::vector<std::pair<std::string, std::vector<float>>> dump;

    static constexpr int kDim = 1536, kHeads = 24, kHeadDim = 64, kDepth = 24, kMem = 64, kCh = 256;

private:
    struct Layer {
        ggml_tensor *preN, *crossN, *ffN, *qkv, *sOut, *sqn, *skn, *cq, *ckv, *cOut, *cqn, *ckn, *ff0w, *ff0b, *ff2w, *ff2b, *ssg, *local;
    };
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer* buf_ = nullptr;
    std::vector<Layer> layers_;
    ggml_tensor *preW_, *postW_, *cond0_, *cond2_, *glob0_, *glob2_, *time0w_, *time0b_, *time2w_, *time2b_;
    ggml_tensor *gc0w_, *gc0b_, *gc2w_, *gc2b_, *projIn_, *projOut_, *memTok_;
    std::vector<float> padding_;            // padding_embedding [768]
    std::vector<float> secW_, secB_;        // seconds_total embedder (768 x 256 + bias)
    std::vector<std::vector<float>> localConst_;
};
