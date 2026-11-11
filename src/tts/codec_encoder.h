// Qwen3-TTS 12Hz speech tokenizer, encoder side (a Mimi codec): 24 kHz waveform -> 16 codes per 80 ms frame.
// Used by voice cloning (Base model, in-context mode).
#pragma once
#include <string>
#include <vector>

struct ggml_context; struct ggml_backend_buffer; struct ggml_tensor;

class CodecEncoder {
public:
    ~CodecEncoder() { unload(); }
    bool load(const std::string& path, std::string& err);       // <model>/speech_tokenizer/model.safetensors
    void unload();
    bool loaded() const { return buf_ != nullptr; }
    // wav: mono 24 kHz. codes: frames x 16, frame-major.
    bool encode(const std::vector<float>& wav, std::vector<int>& codes, int& frames, std::string& err);
    std::vector<float> dbgEmb;                     // last pre-quantization embeddings [512, T2] (channel-major)

private:
    struct Conv { ggml_tensor *w = nullptr, *b = nullptr; };
    struct Res { Conv c1, c2; };
    struct Layer { ggml_tensor *n1w, *n1b, *n2w, *n2b, *q, *k, *v, *o, *fc1, *fc2, *s1, *s2; };
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer* buf_ = nullptr;
    Conv c0_, last_;
    Res res_[4];
    Conv down_[4];
    ggml_tensor* downsample_ = nullptr;
    std::vector<Layer> layers_;
    std::vector<float> inProj_[2];                  // semantic / acoustic input projections, 256 x 512
    std::vector<float> cb_[16];                     // dequantized codebooks, 2048 x 256 (1 semantic + 15 acoustic)
};
