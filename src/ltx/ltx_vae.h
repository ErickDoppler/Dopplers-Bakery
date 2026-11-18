// LTX-2.3 video VAE (CausalVideoAutoencoder, pixel-norm, patch 4, 128 latent channels, 8x time / 32x space) on ggml.
// Volumes are kept on the host as float [T][C][H][W]; every block runs as one ggml graph (3D convs = three 2D convs over the frames).
#pragma once
#include <functional>
#include <string>
#include <vector>
#include "ltx/ltx_weights.h"

struct Vol {
    int T = 0, C = 0, H = 0, W = 0;
    std::vector<float> d;
    void alloc(int t, int c, int h, int w) { T = t; C = c; H = h; W = w; d.assign((size_t)t * c * h * w, 0.f); }
    float& at(int t, int c, int h, int w) { return d[(((size_t)t * C + c) * H + h) * W + w]; }
    float at(int t, int c, int h, int w) const { return d[(((size_t)t * C + c) * H + h) * W + w]; }
};

struct LtxVae {
    const LtxWeights* ckpt = nullptr;
    // latent (normalized, as the sampler sees it) [Tl,128,h,w] -> frames [8*(Tl-1)+1, 3, 32h, 32w] in [-1, 1]
    bool decode(const Vol& latent, Vol& frames, const std::function<void(const std::string&)>& progress, std::string& err);
    // image [1,3,H,W] in [-1, 1] (H, W multiples of 32) -> normalized latent [1,128,H/32,W/32]
    bool encode(const Vol& image, Vol& latent, std::string& err);
};

int runLtxVaeTest(const std::vector<std::string>& args);      // ltx-vae <ckpt> <reference dir>
