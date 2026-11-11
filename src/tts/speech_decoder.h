// Qwen3-TTS 12Hz speech tokenizer, decoder side: 16 RVQ codes per frame -> 24 kHz waveform (ggml).
#pragma once
#include <string>
#include <vector>

struct ggml_context; struct ggml_backend_buffer; struct ggml_tensor;

class SpeechDecoder {
public:
    ~SpeechDecoder() { unload(); }
    bool load(const std::string& path, std::string& err);       // <model>/speech_tokenizer/model.safetensors
    void unload();
    bool loaded() const { return buf_ != nullptr; }
    static constexpr int kSampleRate = 24000, kSamplesPerFrame = 1920, kGroups = 16;

    // codes: frames x 16 (frame-major). Chunked like the reference (300 frames, 25 frames of left context).
    bool decode(const std::vector<int>& codes, int frames, std::vector<float>& wav, std::string& err);

private:
    bool decodeChunk(const int* codes, int frames, std::vector<float>& wav, std::string& err);
    struct Layer { ggml_tensor *ln1, *q, *k, *v, *o, *ln2, *gate, *up, *down, *s1, *s2; };
    struct Up { ggml_tensor *tw, *tb, *dw, *db, *nw, *nb, *p1w, *p1b, *p2w, *p2b, *gamma; };
    struct Unit { ggml_tensor *a1, *b1, *c1w, *c1b, *a2, *b2, *c2w, *c2b; int dil; };
    struct Block { ggml_tensor *sa, *sb, *tw, *tb; int stride; Unit u[3]; };
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer* buf_ = nullptr;
    std::vector<float> cb_[16];            // dequantized codebooks, 2048 x 256 each
    ggml_tensor *opFirst_ = nullptr, *opRest_ = nullptr;
    ggml_tensor *preW_ = nullptr, *preB_ = nullptr, *inW_ = nullptr, *inB_ = nullptr, *normW_ = nullptr, *outW_ = nullptr, *outB_ = nullptr;
    std::vector<Layer> layers_;
    Up up_[2];
    ggml_tensor *d0w_ = nullptr, *d0b_ = nullptr, *fa_ = nullptr, *fb_ = nullptr, *flw_ = nullptr, *flb_ = nullptr;
    Block blocks_[4];
};
