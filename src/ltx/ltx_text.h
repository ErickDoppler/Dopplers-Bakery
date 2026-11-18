// LTX-2.3 text conditioning on ggml: Gemma 3 12B (all 49 hidden states) -> per-token RMS + dual linear projection -> two 8-block 1D connectors
// (video 4096 wide, audio 2048 wide, 128 learnable registers). Gemma runs one layer at a time (weights dequantized to F16 on the fly), so only
// ~0.6 GB of its 12B parameters are on the GPU at once.
#pragma once
#include <functional>
#include <string>
#include <vector>
#include "audio/spiece.h"
#include "ltx/ltx_weights.h"

struct LtxTextEncoder {
    LtxWeights gemma, ckpt;
    SentencePiece sp;
    bool open(const std::string& gemmaPath, const std::string& ckptPath, std::string& err);
    // BOS + SentencePiece ids of the raw prompt (what ComfyUI's CLIPTextEncode feeds Gemma: no chat template).
    std::vector<int> tokenize(const std::string& text) const;
    // hidden[layer 0..48] each n x 3840 (token-major): embeddings, the input of every layer, the normed output of the last one
    bool gemmaHidden(const std::vector<int>& ids, std::vector<std::vector<float>>& hidden, const std::function<void(int)>& progress, std::string& err);
    // per-token RMS + dual linear: n x (4096 video | 2048 audio), the two halves are returned separately
    bool project(const std::vector<std::vector<float>>& hidden, int n, std::vector<float>& video, std::vector<float>& audio, std::string& err);
    // connectors: n x dim in, L x dim out (L = max(1024, n) rounded up to 128; registers fill the tail)
    bool connector(bool video, const std::vector<float>& in, int n, std::vector<float>& out, int& outLen, std::string& err);
};

int runLtxTextTest(const std::vector<std::string>& args);      // ltx-text <gemma> <ckpt> <ref dir> "<prompt>"
