// Krea2 single-stream MMDiT on ggml (CPU). Weights: scaled FP8 -> Q8_0 at load.
#pragma once
#include <cstdint>
#include <functional>
#include <map>
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
    int quantBits = 8;                  // 8 (default) or 4 (low-memory mode); set before load()
    int mlpChunk = 0;                   // > 0: run the MLP of the image blocks in chunks of this many tokens (small GPU memory)
    bool loaded() const { return backend_ != nullptr; }
    // (Off unless BAKERY_DIT_RESIDENT is set: when VRAM runs short Windows silently pages it out and steps get many times slower.)
    // The weights stay in host memory and the GPU reads them over PCIe on every step (about 0.7 s per step for the 13 GB of 8-bit weights). For the length of one job
    // the first blocks' matrices can be copied into GPU memory (at most maxBytes); forward() falls back to the host copies by itself if the graph then does not fit.
    int makeResident(size_t maxBytes);  // returns how many blocks are on the GPU
    void releaseResident();

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
    ggml_context* ctxDev_ = nullptr; ggml_backend_buffer* bufDev_ = nullptr; std::vector<Blk> devBlocks_; int devN_ = 0;     // GPU copies of the first devN_ blocks (see makeResident)
    ggml_tensor *firstW, *firstB, *tm0W, *tm0B, *tm2W, *tm2B, *tpW, *tpB;
    ggml_tensor *projector, *txtNorm, *txt1W, *txt1B, *txt3W, *txt3B;
    ggml_tensor *lastNorm, *lastLin, *lastW, *lastB;
    // The text half of the model (12-layer fusion, refiner, projection) depends only on the prompt, not on the picture or the step: it is computed once per prompt and reused.
    std::map<uint64_t, std::vector<float>> txCache_;
    bool textTokens(const std::vector<float>& ctx, int seq, const std::vector<float>*& out, std::string& err);
};
