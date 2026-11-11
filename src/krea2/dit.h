// Krea2 single-stream MMDiT on ggml (CPU). Weights: scaled FP8 -> Q8_0 at load.
#pragma once
#include <functional>
#include <string>
#include <vector>
#include <utility>

struct ggml_context; struct ggml_backend; struct ggml_backend_buffer; struct ggml_tensor; struct ggml_gallocr;

class Krea2DiT {
public:
    ~Krea2DiT() { unload(); }
    bool load(const std::string& path, int threads, std::string& err, const std::function<void(float)>& progress = nullptr,
              const std::string& loraPath = "", float loraStrength = 1.0f);   // LoRA is merged into the weights while loading
    int loraPatched = 0;
    void unload();
    bool loaded() const { return backend_ != nullptr; }

    // latent: 16 x H x W (channel-major, H and W multiples of 2 are padded by the caller),
    // ctx: seq x (12*2560) from Krea2TextEncoder. out: velocity, same layout as latent.
    bool forward(const std::vector<float>& latent, int H, int W, float sigma,
                 const std::vector<float>& ctx, int seq, std::vector<float>& out, std::string& err);

    bool debug = false;                                         // keep intermediate tensors in `dump`
    std::vector<std::pair<std::string, std::vector<float>>> dump;

    static constexpr int kDim = 6144, kHeads = 48, kKvHeads = 12, kHeadDim = 128, kBlocks = 28, kFF = 16384;
    static constexpr int kTxtDim = 2560, kTxtHeads = 20, kTxtFF = 6912, kTxtLayers = 12;

    struct Blk {
        ggml_tensor *mod = nullptr, *pre, *post, *qn, *kn, *wq, *wk, *wv, *gate, *wo, *mg, *mu, *md;
    };
private:
    ggml_context* ctx_ = nullptr;
    ggml_backend* backend_ = nullptr;
    ggml_backend_buffer* buf_ = nullptr;
    ggml_gallocr* ga_ = nullptr;
    void* st_ = nullptr;
    int threads_ = 8;
    std::vector<Blk> blocks_, layerwise_, refiner_;
    ggml_tensor *firstW, *firstB, *tm0W, *tm0B, *tm2W, *tm2B, *tpW, *tpB;
    ggml_tensor *projector, *txtNorm, *txt1W, *txt1B, *txt3W, *txt3B;
    ggml_tensor *lastNorm, *lastLin, *lastW, *lastB;
};
