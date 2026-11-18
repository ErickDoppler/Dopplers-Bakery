// LTX-2.3 audio decoding on ggml: audio VAE decoder (causal 2D ResNet, latent -> stereo mel), BigVGAN-style vocoder with anti-aliased SnakeBeta
// activations (mel -> 16 kHz waveform) and the bandwidth-extension stage (16 kHz -> 48 kHz).
#pragma once
#include <functional>
#include <string>
#include <vector>
#include "ltx/ltx_weights.h"

struct LtxAudio {
    const LtxWeights* ckpt = nullptr;
    // normalized audio latent [8][T][16] -> stereo waveform [2][samples] at 48 kHz (samples = 4 * (4T - 3) * 160 * 3 / 4 ... see below)
    bool decode(const std::vector<float>& latent, int T, std::vector<float>& wave, int& samples, const std::function<void(const std::string&)>& progress, std::string& err);
    std::vector<float> lastMel;               // [2][Tm][64], kept for tests
    static constexpr int kSampleRate = 48000;
};

int runLtxAudioTest(const std::vector<std::string>& args);      // ltx-audio <ckpt> <reference dir> [T]
