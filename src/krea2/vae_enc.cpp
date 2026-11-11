// Wan2.1 / Qwen-Image VAE encoder, single frame (every causal 3D conv reduces to its last temporal tap), on ggml.
#include <algorithm>
#include <cmath>
#include <cstring>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gpu/compute.h"
#include "krea2/safetensors.h"
#include "krea2/vae.h"

static const float kMean[16] = {-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
                                0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f};
static const float kStd[16] = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                               3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};

void Krea2VaeEncoder::unload() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
    buf_ = nullptr; ctx_ = nullptr; res_.clear();
}

namespace {
struct Pending { ggml_tensor* t; std::string name; int mode; };      // 0 conv kernel (last temporal tap, f16), 1 f32 vector
using TT = ggml_tensor*;
TT convApply(ggml_context* c, TT w, TT b, TT x, int stride, int pad) {
    TT y = ggml_conv_2d(c, w, x, stride, stride, pad, pad, 1, 1);
    return ggml_add(c, y, ggml_reshape_4d(c, b, 1, 1, b->ne[0], 1));
}
TT rmsC(ggml_context* c, TT x, TT gamma) {
    TT t = ggml_cont(c, ggml_permute(c, x, 1, 2, 0, 3));
    t = ggml_mul(c, ggml_rms_norm(c, t, 1e-12f), gamma);
    return ggml_cont(c, ggml_permute(c, t, 2, 0, 1, 3));
}
}  // namespace

bool Krea2VaeEncoder::load(const std::string& path, std::string& err) {
    unload();
    StFile st;
    if (!st.open(path, err)) return false;
    std::string e2;
    if (!Compute::get().init(0, e2)) { err = e2; return false; }
    ggml_init_params ip = {ggml_tensor_overhead() * 200, nullptr, true};
    ctx_ = ggml_init(ip);
    std::vector<Pending> jobs;
    bool ok = true;
    auto missing = [&](const std::string& n) { if (ok) err = "VAE file has no encoder (missing tensor " + n + ")"; ok = false; };
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
    conv1_ = conv("encoder.conv1");
    const int resIdx[8] = {0, 1, 3, 4, 6, 7, 9, 10};
    for (int k = 0; k < 8; k++) res_.push_back(res("encoder.downsamples." + std::to_string(resIdx[k]), resIdx[k] == 3 || resIdx[k] == 6));
    const int dnIdx[3] = {2, 5, 8};
    for (int k = 0; k < 3; k++) down_[k] = conv("encoder.downsamples." + std::to_string(dnIdx[k]) + ".resample.1");
    mid0_ = res("encoder.middle.0", false);
    attnG_ = vec("encoder.middle.1.norm.gamma"); qkv_ = conv("encoder.middle.1.to_qkv"); proj_ = conv("encoder.middle.1.proj");
    mid2_ = res("encoder.middle.2", false);
    headG_ = vec("encoder.head.0.gamma"); headConv_ = conv("encoder.head.2");
    quant_ = conv("conv1");
    if (!ok) { unload(); return false; }

    buf_ = Compute::get().allocDeviceWeights(ctx_);
    if (!buf_) { err = "cannot allocate VAE encoder weights"; unload(); return false; }
    std::vector<float> f;
    for (auto& j : jobs) {
        const StTensor* t = st.find(j.name);
        st.toFloat(*t, f);
        if (j.mode == 1) { ggml_backend_tensor_set(j.t, f.data(), 0, f.size() * 4); continue; }
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

bool Krea2VaeEncoder::encode(const std::vector<float>& rgb, int H, int W, std::vector<float>& lat, std::string& err) {
    if (!loaded()) { err = "VAE encoder not loaded"; return false; }
    if (H % 8 || W % 8) { err = "image size must be a multiple of 8"; return false; }
    Compute& cm = Compute::get();
    const size_t nodes = 4096;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false) + (1 << 20));
    ggml_init_params ip = {meta.size(), meta.data(), true};
    ggml_context* c = ggml_init(ip);

    TT in = ggml_new_tensor_4d(c, GGML_TYPE_F32, W, H, 3, 1); ggml_set_input(in);
    TT x = convApply(c, conv1_.w, conv1_.b, in, 1, 1);

    auto resblock = [&](const Res& r, TT v) {
        TT h = ggml_silu(c, rmsC(c, v, r.g1));
        h = convApply(c, r.c1.w, r.c1.b, h, 1, 1);
        h = ggml_silu(c, rmsC(c, h, r.g2));
        h = convApply(c, r.c2.w, r.c2.b, h, 1, 1);
        TT sc = r.sc.w ? convApply(c, r.sc.w, r.sc.b, v, 1, 0) : v;
        return ggml_add(c, h, sc);
    };
    for (int stage = 0; stage < 4; stage++) {
        x = resblock(res_[stage * 2], x);
        x = resblock(res_[stage * 2 + 1], x);
        if (stage < 3) {                                   // ZeroPad2d((0,1,0,1)) then 3x3 conv, stride 2, no padding
            x = ggml_pad(c, x, 1, 1, 0, 0);
            x = convApply(c, down_[stage].w, down_[stage].b, x, 2, 0);
        }
    }
    x = resblock(mid0_, x);
    {
        TT id = x;
        TT n = rmsC(c, x, attnG_);
        TT qkv = convApply(c, qkv_.w, qkv_.b, n, 1, 0);
        const int64_t C = x->ne[2], N = (int64_t)x->ne[0] * x->ne[1];
        auto part = [&](int k) { return ggml_view_3d(c, qkv, x->ne[0], x->ne[1], C, qkv->nb[1], qkv->nb[2], (size_t)k * C * qkv->nb[2]); };
        TT q = ggml_cont(c, ggml_permute(c, ggml_reshape_2d(c, ggml_cont(c, part(0)), N, C), 1, 0, 2, 3));
        TT k = ggml_cont(c, ggml_permute(c, ggml_reshape_2d(c, ggml_cont(c, part(1)), N, C), 1, 0, 2, 3));
        TT v = ggml_reshape_2d(c, ggml_cont(c, part(2)), N, C);
        const int64_t chunk = 1024;
        TT out = nullptr;
        for (int64_t s = 0; s < N; s += chunk) {
            int64_t m = std::min(chunk, N - s);
            TT qc = ggml_view_2d(c, q, C, m, q->nb[1], (size_t)s * q->nb[1]);
            TT sc = ggml_mul_mat(c, k, qc);
            sc = ggml_soft_max_ext(c, sc, nullptr, 1.0f / std::sqrt((float)C), 0.0f);
            TT o = ggml_mul_mat(c, v, sc);
            out = out ? ggml_concat(c, out, o, 1) : o;
        }
        TT sp = ggml_reshape_4d(c, ggml_cont(c, ggml_permute(c, out, 1, 0, 2, 3)), x->ne[0], x->ne[1], C, 1);
        x = ggml_add(c, convApply(c, proj_.w, proj_.b, sp, 1, 0), id);
    }
    x = resblock(mid2_, x);
    x = ggml_silu(c, rmsC(c, x, headG_));
    x = convApply(c, headConv_.w, headConv_.b, x, 1, 1);        // [w, h, 32]
    x = convApply(c, quant_.w, quant_.b, x, 1, 0);              // [w, h, 32]: mean (16) then log-variance (16)
    TT mu = ggml_cont(c, ggml_view_3d(c, x, x->ne[0], x->ne[1], 16, x->nb[1], x->nb[2], 0));
    ggml_set_output(mu);

    ggml_cgraph* g = ggml_new_graph_custom(c, nodes, false);
    ggml_build_forward_expand(g, mu);
    ggml_backend_sched_reset(cm.sched);
    if (!ggml_backend_sched_alloc_graph(cm.sched, g)) { err = "cannot allocate VAE encoder graph (image too large for GPU memory?)"; ggml_free(c); return false; }

    std::vector<float> planar((size_t)W * H * 3);
    for (size_t i = 0; i < (size_t)W * H; i++) for (int ch = 0; ch < 3; ch++) planar[(size_t)ch * W * H + i] = rgb[i * 3 + ch] * 2.0f - 1.0f;
    ggml_backend_tensor_set(in, planar.data(), 0, planar.size() * 4);
    if (ggml_backend_sched_graph_compute(cm.sched, g) != GGML_STATUS_SUCCESS) { err = "VAE encoder compute failed"; ggml_free(c); return false; }

    const int lw = W / 8, lh = H / 8;
    lat.resize((size_t)16 * lw * lh);
    ggml_backend_tensor_get(mu, lat.data(), 0, lat.size() * 4);
    for (int ch = 0; ch < 16; ch++) for (size_t i = 0; i < (size_t)lw * lh; i++) {
        float& v = lat[(size_t)ch * lw * lh + i];
        v = (v - kMean[ch]) / kStd[ch];
    }
    ggml_backend_sched_reset(cm.sched);
    ggml_free(c);
    return true;
}
