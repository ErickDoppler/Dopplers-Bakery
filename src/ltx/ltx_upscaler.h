// LTX-2.3 spatial latent upscaler x2 (3D conv ResNet with GroupNorm, a per-frame conv + pixel shuffle in the middle) on ggml.
#pragma once
#include <string>
#include <vector>
#include "ltx/ltx_vae.h"

struct LtxUpscaler {
    const LtxWeights* model = nullptr;        // ltx-2.3-spatial-upscaler-x2-1.1.safetensors
    const LtxWeights* ckpt = nullptr;         // the LTX checkpoint (VAE channel statistics)
    // normalized latent [T,128,h,w] -> normalized latent [T,128,2h,2w]
    bool run(const Vol& latent, Vol& out, std::string& err);
};

int runLtxUpscalerTest(const std::vector<std::string>& args);      // ltx-up <upscaler> <ckpt> <reference dir> [T h w]
