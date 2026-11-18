// LTX-2.3 image/text-to-video pipeline as in VIDEO_WORKFLOW.json:
//   text -> Gemma + connectors -> stage 1 (half resolution, 8 Euler steps, distilled LoRA) -> x2 spatial upscaler -> stage 2 (3 Euler steps) -> VAE decode
// The first latent frame can be fixed to a start image (image-to-video); audio is sampled together with the video.
#pragma once
#include <functional>
#include <string>
#include <vector>
#include "ltx/ltx_dit.h"
#include "ltx/ltx_vae.h"
#include "util/image_io.h"

struct LtxFiles { std::string checkpoint, gemma, upscaler, lora; };

struct LtxGenParams {
    std::string prompt;
    int width = 768, height = 512;        // final size (multiples of 64); stage 1 runs at half of it
    float seconds = 4.f;
    int fps = 24;
    int seed = 42;
    bool upscale = true;                  // false: stop after stage 1 (half size)
    float loraStrength = 0.5f;
    int ditBlocks = 48;                   // fewer only for tests
    ImageRgb* startImage = nullptr;       // optional
    float stage1Strength = 1.0f, stage2Strength = 0.7f;
};

struct LtxResult {
    Vol frames;                           // [T][3][H][W] in [-1, 1]
    std::vector<float> audioLatent;       // [8][Ta][16] (decoded to sound in stage 6)
    int audioFrames = 0;
};

// progress(0..1, text); return false from `cancel` to stop
bool ltxGenerate(const LtxFiles& files, const LtxGenParams& p, LtxResult& out, const std::function<void(float, const std::string&)>& progress, std::string& err);

int runLtxGenTest(const std::vector<std::string>& args);      // ltx-gen <out dir> "<prompt>" W H seconds fps seed [image] [blocks]
