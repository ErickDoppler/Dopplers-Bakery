#include "krea2/vae.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gpu/compute.h"
#include "krea2/safetensors.h"

static const float kMean[16] = {-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
                                0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f};
static const float kStd[16] = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                               3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};

void Krea2Vae::latentToVae(std::vector<float>& z, int H, int W) {
    for (int c = 0; c < 16; c++) for (size_t i = 0; i < (size_t)H * W; i++) z[(size_t)c * H * W + i] = z[(size_t)c * H * W + i] * kStd[c] + kMean[c];
}

void Krea2Vae::unload() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
    buf_ = nullptr; ctx_ = nullptr; ups_.clear();
}

namespace {
struct Pending { ggml_tensor* t; std::string name; int mode; };   // 0 conv kernel (last temporal tap, f16), 1 f32 vector
}

bool Krea2Vae::load(const std::string& path, std::string& err) {
    unload();
    StFile st;
    if (!st.open(path, err)) return false;
    std::string e2;
    if (!Compute::get().init(0, e2)) { err = e2; return false; }
    ggml_init_params ip = {ggml_tensor_overhead() * 200, nullptr, true};
    ctx_ = ggml_init(ip);
    std::vector<Pending> jobs;
    bool ok = true;
    auto missing = [&](const std::string& n) { if (ok) err = "missing tensor " + n; ok = false; };

    auto kernel = [&](const std::string& n) -> ggml_tensor* {
        const StTensor* t = st.find(n);
        if (!t) { missing(n); return nullptr; }
        int64_t o = t->shape[0], i = t->shape[1], kh = t->shape.size() >= 4 ? t->shape[t->shape.size() - 2] : 1, kw = t->shape.back();
        ggml_tensor* g = ggml_new_tensor_4d(ctx_, GGML_TYPE_F16, kw, kh, i, o);
        ggml_set_name(g, n.c_str()); jobs.push_back({g, n, 0}); return g;
    };
    auto vec = [&](const std::string& n) -> ggml_tensor* {
        const StTensor* t = st.find(n);
        if (!t) { missing(n); return nullptr; }
        ggml_tensor* g = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, t->numel());
        ggml_set_name(g, n.c_str()); jobs.push_back({g, n, 1}); return g;
    };
    auto conv = [&](const std::string& p) { Conv c; c.w = kernel(p + ".weight"); c.b = vec(p + ".bias"); return c; };
    auto res = [&](const std::string& p, bool shortcut) {
        Res r;
        r.g1 = vec(p + ".residual.0.gamma"); r.c1 = conv(p + ".residual.2");
        r.g2 = vec(p + ".residual.3.gamma"); r.c2 = conv(p + ".residual.6");
        if (shortcut) r.sc = conv(p + ".shortcut");
        return r;
    };
    conv2_ = conv("conv2");
    conv1_ = conv("decoder.conv1");
    mid0_ = res("decoder.middle.0", false);
    attnG_ = vec("decoder.middle.1.norm.gamma"); qkv_ = conv("decoder.middle.1.to_qkv"); proj_ = conv("decoder.middle.1.proj");
    mid2_ = res("decoder.middle.2", false);
    // upsample stack: stages of 3 residual blocks, resample after stages 0..2 (indices 3, 7, 11)
    const int resIdx[12] = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14};
    for (int k = 0; k < 12; k++) ups_.push_back(res("decoder.upsamples." + std::to_string(resIdx[k]), resIdx[k] == 4));
    const int rsIdx[3] = {3, 7, 11};
    for (int k = 0; k < 3; k++) rs_[k] = conv("decoder.upsamples." + std::to_string(rsIdx[k]) + ".resample.1");
    headG_ = vec("decoder.head.0.gamma"); headConv_ = conv("decoder.head.2");
    if (!ok) { unload(); return false; }

    buf_ = Compute::get().allocDeviceWeights(ctx_);
    if (!buf_) { err = "cannot allocate VAE weights"; unload(); return false; }
    std::vector<float> f;
    for (auto& j : jobs) {
        const StTensor* t = st.find(j.name);
        st.toFloat(*t, f);
        if (j.mode == 1) { ggml_backend_tensor_set(j.t, f.data(), 0, f.size() * 4); continue; }
        // conv kernel: keep only the last temporal tap of 5D weights ([o,i,kd,kh,kw] -> [o,i,kh,kw])
        int64_t o = j.t->ne[3], i = j.t->ne[2], kh = j.t->ne[1], kw = j.t->ne[0];
        size_t plane = (size_t)kh * kw;
        int64_t kd = t->shape.size() == 5 ? t->shape[2] : 1;
        std::vector<ggml_fp16_t> h((size_t)o * i * plane);
        for (int64_t oc = 0; oc < o; oc++) for (int64_t ic = 0; ic < i; ic++) {
            const float* src = f.data() + (((size_t)oc * i + ic) * kd + (kd - 1)) * plane;
            ggml_fp32_to_fp16_row(src, h.data() + ((size_t)oc * i + ic) * plane, plane);
        }
        ggml_backend_tensor_set(j.t, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
    }
    return true;
}

// ---------------------------------------------------------------- graph pieces (tensors are [W, H, C, 1])
namespace {
using TT = ggml_tensor*;
TT convApply(ggml_context* c, TT w, TT b, TT x, int pad) {
    TT y = ggml_conv_2d(c, w, x, 1, 1, pad, pad, 1, 1);
    return ggml_add(c, y, ggml_reshape_4d(c, b, 1, 1, b->ne[0], 1));
}
// channel RMS norm: x / rms_c(x) * gamma   (== F.normalize(x, dim=1) * sqrt(C) * gamma)
TT rmsC(ggml_context* c, TT x, TT gamma) {
    TT t = ggml_cont(c, ggml_permute(c, x, 1, 2, 0, 3));           // [C, W, H]
    t = ggml_mul(c, ggml_rms_norm(c, t, 1e-12f), gamma);
    return ggml_cont(c, ggml_permute(c, t, 2, 0, 1, 3));           // back to [W, H, C]
}
}  // namespace

bool Krea2Vae::decodeTile(const std::vector<float>& z, int H, int W, std::vector<float>& rgb, std::string& err) {
    Compute& cm = Compute::get();
    const size_t nodes = 4096;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false) + (1 << 20));
    ggml_init_params ip = {meta.size(), meta.data(), true};
    ggml_context* c = ggml_init(ip);

    TT in = ggml_new_tensor_4d(c, GGML_TYPE_F32, W, H, 16, 1); ggml_set_input(in);
    TT x = convApply(c, conv2_.w, conv2_.b, in, 0);
    x = convApply(c, conv1_.w, conv1_.b, x, 1);

    auto resblock = [&](const Res& r, TT v) {
        TT h = ggml_silu(c, rmsC(c, v, r.g1));
        h = convApply(c, r.c1.w, r.c1.b, h, 1);
        h = ggml_silu(c, rmsC(c, h, r.g2));
        h = convApply(c, r.c2.w, r.c2.b, h, 1);
        TT sc = r.sc.w ? convApply(c, r.sc.w, r.sc.b, v, 0) : v;
        return ggml_add(c, h, sc);
    };

    x = resblock(mid0_, x);
    {   // single-head spatial attention over all pixels, queries processed in chunks to bound memory
        TT id = x;
        TT n = rmsC(c, x, attnG_);
        TT qkv = convApply(c, qkv_.w, qkv_.b, n, 0);                        // [W,H,3C]
        const int64_t C = x->ne[2], N = (int64_t)x->ne[0] * x->ne[1];
        auto part = [&](int k) { return ggml_view_3d(c, qkv, x->ne[0], x->ne[1], C, qkv->nb[1], qkv->nb[2], (size_t)k * C * qkv->nb[2]); };
        TT q = ggml_cont(c, ggml_permute(c, ggml_reshape_2d(c, ggml_cont(c, part(0)), N, C), 1, 0, 2, 3));   // [C, N]
        TT k = ggml_cont(c, ggml_permute(c, ggml_reshape_2d(c, ggml_cont(c, part(1)), N, C), 1, 0, 2, 3));   // [C, N]
        TT v = ggml_reshape_2d(c, ggml_cont(c, part(2)), N, C);                                              // [N, C]
        const int64_t chunk = 1024;
        TT out = nullptr;
        for (int64_t s = 0; s < N; s += chunk) {
            int64_t m = std::min(chunk, N - s);
            TT qc = ggml_view_2d(c, q, C, m, q->nb[1], (size_t)s * q->nb[1]);
            TT sc = ggml_mul_mat(c, k, qc);                                  // [N keys, m queries]
            sc = ggml_soft_max_ext(c, sc, nullptr, 1.0f / std::sqrt((float)C), 0.0f);
            TT o = ggml_mul_mat(c, v, sc);                                   // [C, m]
            out = out ? ggml_concat(c, out, o, 1) : o;
        }
        TT sp = ggml_reshape_4d(c, ggml_cont(c, ggml_permute(c, out, 1, 0, 2, 3)), x->ne[0], x->ne[1], C, 1);   // [N,C] -> [W,H,C]
        x = ggml_add(c, convApply(c, proj_.w, proj_.b, sp, 0), id);
    }
    x = resblock(mid2_, x);

    for (int stage = 0; stage < 4; stage++) {
        for (int b = 0; b < 3; b++) x = resblock(ups_[stage * 3 + b], x);
        if (stage < 3) {
            x = ggml_upscale(c, x, 2, GGML_SCALE_MODE_NEAREST);
            x = convApply(c, rs_[stage].w, rs_[stage].b, x, 1);
        }
    }
    x = ggml_silu(c, rmsC(c, x, headG_));
    x = convApply(c, headConv_.w, headConv_.b, x, 1);                          // [8W, 8H, 3]
    ggml_set_output(x);

    ggml_cgraph* g = ggml_new_graph_custom(c, nodes, false);
    ggml_build_forward_expand(g, x);
    ggml_backend_sched_reset(cm.sched);
    if (!ggml_backend_sched_alloc_graph(cm.sched, g)) { err = "cannot allocate VAE graph (image too large for GPU memory?)"; ggml_free(c); return false; }
    ggml_backend_tensor_set(in, z.data(), 0, z.size() * 4);
    if (ggml_backend_sched_graph_compute(cm.sched, g) != GGML_STATUS_SUCCESS) { err = "VAE compute failed"; ggml_free(c); return false; }

    const int OW = W * 8, OH = H * 8;
    std::vector<float> planar((size_t)OW * OH * 3);
    ggml_backend_tensor_get(x, planar.data(), 0, planar.size() * 4);
    rgb.assign((size_t)OW * OH * 3, 0.0f);
    for (int ch = 0; ch < 3; ch++) for (size_t i = 0; i < (size_t)OW * OH; i++) {
        float v = (planar[(size_t)ch * OW * OH + i] + 1.0f) * 0.5f;
        rgb[i * 3 + ch] = v < 0 ? 0.0f : (v > 1 ? 1.0f : v);
    }
    ggml_backend_sched_reset(cm.sched);
    ggml_free(c);
    return true;
}

bool Krea2Vae::decode(const std::vector<float>& z, int H, int W, std::vector<float>& rgb, std::string& err, int maxTileArea) {
    if (!loaded()) { err = "VAE not loaded"; return false; }
    if (H * W <= maxTileArea) return decodeTile(z, H, W, rgb, err);

    // overlapping tiles with linear feathering
    const int T = std::max(16, (int)std::sqrt((double)maxTileArea)), OV = std::max(4, T / 8), STEP = T - OV;
    const int OW = W * 8, OH = H * 8;
    std::vector<float> acc((size_t)OW * OH * 3, 0.0f), wsum((size_t)OW * OH, 0.0f);
    for (int ty = 0; ty < H; ty += STEP) {
        int th = std::min(T, H - ty); if (th < T && ty > 0) { ty = H - T; th = T; }
        for (int tx = 0; tx < W; tx += STEP) {
            int tw = std::min(T, W - tx); if (tw < T && tx > 0) { tx = W - T; tw = T; }
            std::vector<float> tile((size_t)16 * th * tw), out;
            for (int ch = 0; ch < 16; ch++) for (int y = 0; y < th; y++)
                memcpy(&tile[((size_t)ch * th + y) * tw], &z[((size_t)ch * H + ty + y) * W + tx], tw * sizeof(float));
            if (!decodeTile(tile, th, tw, out, err)) return false;
            const int pw = tw * 8, ph = th * 8, ramp = OV * 8;
            for (int y = 0; y < ph; y++) for (int x = 0; x < pw; x++) {
                float wy = 1.0f, wx = 1.0f;
                if (ty > 0) wy = std::min(wy, (y + 0.5f) / ramp);
                if (ty + th < H) wy = std::min(wy, (ph - y - 0.5f) / ramp);
                if (tx > 0) wx = std::min(wx, (x + 0.5f) / ramp);
                if (tx + tw < W) wx = std::min(wx, (pw - x - 0.5f) / ramp);
                float wgt = std::max(1e-4f, std::min(1.0f, wy) * std::min(1.0f, wx));
                size_t o = (size_t)(ty * 8 + y) * OW + (tx * 8 + x);
                for (int ch = 0; ch < 3; ch++) acc[o * 3 + ch] += out[((size_t)y * pw + x) * 3 + ch] * wgt;
                wsum[o] += wgt;
            }
            if (tx + tw >= W) break;
        }
        if (ty + th >= H) break;
    }
    rgb.resize(acc.size());
    for (size_t i = 0; i < wsum.size(); i++) for (int ch = 0; ch < 3; ch++) rgb[i * 3 + ch] = acc[i * 3 + ch] / wsum[i];
    return true;
}
