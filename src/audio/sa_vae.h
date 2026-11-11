// Stable Audio 3 audio VAE, decoder side (SA3AudioVAE): 256-channel latents -> stereo 44.1 kHz, on ggml.
#pragma once
#include <string>
#include <vector>

struct ggml_context; struct ggml_backend_buffer; struct ggml_tensor;

class SaVae {
public:
    ~SaVae() { unload(); }
    bool load(const std::string& checkpointPath, std::string& err);
    void unload();
    bool loaded() const { return buf_ != nullptr; }
    static constexpr int kSampleRate = 44100, kSamplesPerLatent = 4096;

    // latent: T x 256 (time-major). pcm: T*4096 stereo frames, interleaved L,R. Long inputs are decoded in overlapping chunks.
    bool decode(const std::vector<float>& latent, int T, std::vector<float>& pcm, std::string& err);

private:
    bool decodeChunk(const float* lat, int Tc, std::vector<float>& left, std::vector<float>& right, std::string& err);
    struct Dyt { ggml_tensor *alpha = nullptr, *gamma = nullptr, *beta = nullptr; };
    struct Layer { Dyt pre, ff, qn, kn; ggml_tensor *qkv, *out, *ff0w, *ff0b, *ff2w, *ff2b; bool sin; };
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer* buf_ = nullptr;
    ggml_tensor *linW_ = nullptr, *linB_ = nullptr, *newTok_ = nullptr, *mapW_ = nullptr, *mapB_ = nullptr;
    std::vector<Layer> layers_;
    float runningStd_ = 1.0f;
};
