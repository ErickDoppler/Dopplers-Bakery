#include "ltx/ltx_weights.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>

static bool endsWith(const std::string& s, const char* suf) { size_t n = strlen(suf); return s.size() >= n && s.compare(s.size() - n, n, suf) == 0; }

std::string LtxWeights::formatOf(const std::string& name) const {
    const StTensor* t = file.find(name);
    if (!t || !endsWith(name, ".weight")) return "none";
    const std::string base = name.substr(0, name.size() - 7);
    if (const StTensor* q = file.find(base + ".comfy_quant")) {
        std::string s((const char*)q->data, (size_t)q->bytes);
        if (s.find("nvfp4") != std::string::npos) return "nvfp4";
        if (s.find("float8") != std::string::npos) return "fp8";
    }
    if (t->dtype == "F8_E4M3" && file.find(base + ".weight_scale")) return "fp8";
    return "none";
}

bool LtxWeights::shapeOf(const std::string& name, std::vector<int64_t>& shape) const {
    const StTensor* t = file.find(name);
    if (!t) return false;
    shape = t->shape;
    if (formatOf(name) == "nvfp4" && !shape.empty()) shape.back() *= 2;
    return true;
}

// cuBLAS 128x4 tiles: output (row, col) lives at ((rb * nColBlocks + cb) * 512) + (r % 32) * 16 + (r / 32) * 4 + c, r = row % 128, c = col % 4
void ltxUnswizzleScales(const uint8_t* blocked, int64_t rows, int64_t cols, std::vector<uint8_t>& out) {
    const int64_t ncb = (cols + 3) / 4;
    out.resize((size_t)(rows * cols));
    for (int64_t row = 0; row < rows; row++) {
        const int64_t rb = row / 128, r = row % 128, i = r % 32, a = r / 32;
        for (int64_t col = 0; col < cols; col++) {
            const int64_t cb = col / 4, c = col % 4;
            out[(size_t)(row * cols + col)] = blocked[(rb * ncb + cb) * 512 + i * 16 + a * 4 + c];
        }
    }
}

bool LtxWeights::loadF32(const std::string& name, std::vector<float>& out, std::string& err) const {
    const StTensor* t = file.find(name);
    if (!t) { err = "tensor not found: " + name; return false; }
    const std::string fmt = formatOf(name);
    const std::string base = endsWith(name, ".weight") ? name.substr(0, name.size() - 7) : name;
    if (fmt == "none") {
        if (!file.toFloat(*t, out)) { err = "unsupported dtype " + t->dtype + " for " + name; return false; }
        return true;
    }
    float s1 = 1.0f;                                            // per-tensor scale (fp8: weight_scale, nvfp4: weight_scale_2)
    if (fmt == "fp8") {
        if (file.find(base + ".weight_scale") && !file.scalar(base + ".weight_scale", s1)) { err = "bad weight_scale for " + name; return false; }
        return file.toFloat(*t, out, s1);
    }
    // nvfp4
    const StTensor* bs = file.find(base + ".weight_scale");
    if (!bs || t->shape.size() != 2) { err = "nvfp4 layer without block scales: " + name; return false; }
    file.scalar(base + ".weight_scale_2", s1);
    const int64_t rows = t->shape[0], cols = t->shape[1] * 2, nb = cols / 16;
    if (bs->numel() < ((rows + 127) / 128) * 128 * (((nb + 3) / 4) * 4)) { err = "block scales too small for " + name; return false; }
    std::vector<uint8_t> sc;
    ltxUnswizzleScales(bs->data, rows, nb, sc);
    static const float e2m1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    float lut8[256]; for (int i = 0; i < 256; i++) lut8[i] = fp8e4m3ToF32((uint8_t)i);
    out.resize((size_t)(rows * cols));
    for (int64_t r = 0; r < rows; r++) {
        const uint8_t* q = t->data + r * t->shape[1];
        float* o = out.data() + r * cols;
        for (int64_t b = 0; b < nb; b++) {
            const float scale = s1 * lut8[sc[(size_t)(r * nb + b)]];
            for (int k = 0; k < 8; k++) {
                const uint8_t v = q[b * 8 + k];
                o[b * 16 + k * 2] = e2m1[v >> 4] * scale;           // high nibble = even element
                o[b * 16 + k * 2 + 1] = e2m1[v & 15] * scale;
            }
        }
    }
    return true;
}

// `DopplersBakery ltx-weights <checkpoint> <gemma>`: loads sample tensors of every format and prints checksums (compare with tools/ltx_ref.py)
int runLtxWeightsTest(const std::vector<std::string>& a) {
    if (a.size() < 2) { fprintf(stderr, "usage: ltx-weights <ltx checkpoint> <gemma file>\n"); return 2; }
    const char* names[2][6] = {
        {"model.diffusion_model.transformer_blocks.20.attn1.to_q.weight", "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight", "model.diffusion_model.patchify_proj.weight",
         "model.diffusion_model.transformer_blocks.47.ff.net.2.weight", "model.diffusion_model.transformer_blocks.20.scale_shift_table", "vae.decoder.conv_in.conv.weight"},
        {"model.layers.5.mlp.up_proj.weight", "model.layers.0.mlp.down_proj.weight", "model.layers.10.mlp.gate_proj.weight", "model.layers.47.self_attn.q_proj.weight", "model.embed_tokens.weight", "model.norm.weight"}};
    int bad = 0;
    for (int f = 0; f < 2; f++) {
        LtxWeights w; std::string err;
        if (!w.open(a[f], err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        printf("%s: %zu tensors\n", a[f].c_str(), w.file.tensors().size());
        for (const char* n : names[f]) {
            if (!w.has(n)) { printf("  %-70s missing\n", n); continue; }
            std::vector<float> v; std::vector<int64_t> sh; w.shapeOf(n, sh);
            if (!w.loadF32(n, v, err)) { printf("  %-70s ERROR %s\n", n, err.c_str()); bad++; continue; }
            double sum = 0, sa = 0; size_t nan = 0;
            for (float x : v) { if (std::isnan(x)) nan++; else { sum += x; sa += std::fabs(x); } }
            std::string shs; for (auto d : sh) shs += (shs.empty() ? "" : "x") + std::to_string(d);
            printf("  %-70s %-8s %-12s sum=%.6g sumabs=%.6g nan=%zu first=%.6g %.6g %.6g %.6g\n", n, w.formatOf(n).c_str(), shs.c_str(), sum, sa, nan, v[0], v[1], v[2], v[3]);
            if (nan) bad++;
        }
    }
    return bad ? 1 : 0;
}
