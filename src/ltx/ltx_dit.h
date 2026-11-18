// LTX-2.3 audio+video transformer (48 blocks, video stream 4096 wide, audio stream 2048 wide) on ggml.
// Every block runs as one graph with its weights uploaded just before (F16); activations stay on the host between blocks.
// First version: explicit attention (small token counts), exact math checked against ComfyUI's LTXAVModel.
#pragma once
#include <functional>
#include <string>
#include <vector>
#include "ltx/ltx_weights.h"

struct LtxDitInput {
    int F = 0, h = 0, w = 0, T = 0;            // video latent frames / height / width, audio latent frames
    std::vector<float> video;                  // [128][F][h][w]
    std::vector<float> audio;                  // [8][T][16]
    std::vector<float> videoSigma;             // per latent frame (or one value): 0..1, a conditioned first frame has 0
    float audioSigma = 1.f;
    std::vector<float> vctx, actx;             // connector outputs: Lv x 4096, La x 2048
    int Lv = 0, La = 0;
    float frameRate = 24.f;
};

struct LtxDit {
    const LtxWeights* ckpt = nullptr;
    int blocks = 48;                           // fewer blocks only for tests
    // velocity predictions, same layouts as the inputs
    bool forward(const LtxDitInput& in, std::vector<float>& videoOut, std::vector<float>& audioOut, const std::function<void(int)>& progress, std::string& err);
};

int runLtxDitTest(const std::vector<std::string>& args);      // ltx-dit <ckpt> <reference dir> [blocks F h w audioTokens]
