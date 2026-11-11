// Qwen3-TTS speaker encoder (ECAPA-TDNN on a 128-bin mel spectrogram) for voice cloning, on ggml.
#pragma once
#include <string>
#include <vector>

struct ggml_context; struct ggml_backend_buffer; struct ggml_tensor;

// 24 kHz mono waveform -> 128 x frames log-mel (frame-major: frames x 128), exactly like the reference implementation
void qwenMelSpectrogram(const std::vector<float>& wav24k, std::vector<float>& mel, int& frames);

class SpeakerEncoder {
public:
    ~SpeakerEncoder() { unload(); }
    bool load(const std::string& baseModelSafetensors, std::string& err);   // reads the speaker_encoder.* tensors
    void unload();
    bool loaded() const { return buf_ != nullptr; }
    // wav24k: mono samples at 24000 Hz. emb: 2048 floats.
    bool embed(const std::vector<float>& wav24k, std::vector<float>& emb, std::string& err);
    bool embedFromMel(const std::vector<float>& mel, int frames, std::vector<float>& emb, std::string& err);

private:
    struct Conv { ggml_tensor *w = nullptr, *b = nullptr; int dil = 1; };
    struct SeRes { Conv t1, t2, r2[7], se1, se2; };
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer* buf_ = nullptr;
    Conv b0_, mfa_, asp_t_, asp_c_, fc_;
    SeRes se_[3];
};
