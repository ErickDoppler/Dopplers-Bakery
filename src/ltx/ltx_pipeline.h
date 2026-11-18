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

// The opened model files and the merged-weight cache; keep one alive between jobs (merging the LoRA takes minutes, the F16 copy of the transformer is ~37 GB of RAM).
struct LtxModels {
    LtxFiles files; float loraStrength = -1.f;
    LtxWeights ck, up, gemma; LtxLora lora;
    bool open(const LtxFiles& f, float strength, std::string& err);
    void release();
};

struct LtxGenParams {
    std::string prompt;
    int width = 768, height = 512;        // final size (multiples of 64); stage 1 runs at half of it
    float seconds = 4.f;
    int fps = 24;
    int seed = 42;
    bool upscale = true;                  // false: stop after stage 1 (half size)
    bool decodeAudio = true;              // decode the sound track (48 kHz stereo)
    float loraStrength = 0.5f;
    int ditBlocks = 48;                   // fewer only for tests
    ImageRgb* startImage = nullptr;       // optional
    float stage1Strength = 1.0f, stage2Strength = 0.7f;
    std::function<bool()> cancel;         // return true to stop
};

struct LtxResult {
    Vol frames;                           // [T][3][H][W] in [-1, 1]
    std::vector<float> audioLatent;       // [8][Ta][16] (decoded to sound in stage 6)
    int audioFrames = 0;
    std::vector<float> wave;              // [2][samples] at 48 kHz when decodeAudio
    int waveSamples = 0;
};

// progress(0..1, text); return false from `cancel` to stop
bool ltxGenerate(LtxModels& models, const LtxGenParams& p, LtxResult& out, const std::function<void(float, const std::string&)>& progress, std::string& err);

int runLtxGenTest(const std::vector<std::string>& args);      // ltx-gen <out dir> "<prompt>" W H seconds fps seed [image] [blocks]



