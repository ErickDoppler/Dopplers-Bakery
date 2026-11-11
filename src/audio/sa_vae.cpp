#include "audio/sa_vae.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gpu/compute.h"
#include "krea2/safetensors.h"

using TT = ggml_tensor*;
static const int kDim = 1536, kHeads = 24, kHD = 64, kStride = 16, kSub = 17, kDepth = 12, kSinBlocks = 8;

void SaVae::unload() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
    buf_ = nullptr; ctx_ = nullptr; layers_.clear();
}

bool SaVae::load(const std::string& path, std::string& err) {
    unload();
    StFile st;
    if (!st.open(path, err)) return false;
    std::string e2;
    if (!Compute::get().init(0, e2)) { err = e2; return false; }
    ggml_init_params ip = {ggml_tensor_overhead() * 400, nullptr, true};
    ctx_ = ggml_init(ip);
    struct Job { TT t; std::string name; int mode; };      // 0: matrix -> F16, 1: F32 vector/array
    std::vector<Job> jobs;
    bool ok = true;
    auto find = [&](const std::string& n) -> const StTensor* { const StTensor* t = st.find(n); if (!t && ok) { err = "missing tensor " + n; ok = false; } return t; };
    auto mat = [&](const std::string& n) -> TT {
        const StTensor* t = find(n); if (!t) return nullptr;
        TT g = ggml_new_tensor_2d(ctx_, GGML_TYPE_F16, t->shape[1], t->shape[0]); jobs.push_back({g, n, 0}); return g;
    };
    auto vec = [&](const std::string& n) -> TT {
        const StTensor* t = find(n); if (!t) return nullptr;
        TT g = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, t->numel()); jobs.push_back({g, n, 1}); return g;
    };
    auto dyt = [&](const std::string& p) { Dyt d; d.alpha = vec(p + ".alpha"); d.gamma = vec(p + ".gamma"); d.beta = vec(p + ".beta"); return d; };
    const std::string D = "pretransform.model.decoder.layers.";
    linW_ = mat(D + "1.weight"); linB_ = vec(D + "1.bias");
    newTok_ = vec(D + "3.new_tokens"); mapB_ = vec(D + "3.mapping.bias");
    mapW_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F16, kDim, 512);                          // weight-normed 1x1 conv, merged below
    for (int i = 0; i < kDepth; i++) {
        std::string p = D + "3.transformers." + std::to_string(i) + ".";
        Layer L;
        L.pre = dyt(p + "pre_norm"); L.ff = dyt(p + "ff_norm");
        L.qn = dyt(p + "self_attn.q_norm"); L.kn = dyt(p + "self_attn.k_norm");
        L.qkv = mat(p + "self_attn.to_qkv.weight"); L.out = mat(p + "self_attn.to_out.weight");
        L.ff0w = mat(p + "ff.ff.0.proj.weight"); L.ff0b = vec(p + "ff.ff.0.proj.bias");
        L.ff2w = mat(p + "ff.ff.2.weight"); L.ff2b = vec(p + "ff.ff.2.bias");
        L.sin = (kDepth - i) < kSinBlocks;                                               // sinusoidal feed-forward in the last blocks
        layers_.push_back(L);
    }
    const StTensor* rs = find("pretransform.model.bottleneck.running_std");
    const StTensor* mg = find(D + "3.mapping.weight_g");
    const StTensor* mv = find(D + "3.mapping.weight_v");
    if (!ok) { unload(); return false; }
    buf_ = Compute::get().allocDeviceWeights(ctx_);
    if (!buf_) { err = "cannot allocate audio VAE weights"; unload(); return false; }
    std::vector<float> f; std::vector<ggml_fp16_t> h;
    for (auto& j : jobs) {
        st.toFloat(*st.find(j.name), f);
        if (j.mode == 0) { h.resize(f.size()); ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t)f.size()); ggml_backend_tensor_set(j.t, h.data(), 0, h.size() * 2); }
        else ggml_backend_tensor_set(j.t, f.data(), 0, f.size() * 4);
    }
    {   // weight norm: w[o][:] = g[o] * v[o][:] / ||v[o][:]||
        std::vector<float> g, v; st.toFloat(*mg, g); st.toFloat(*mv, v);
        std::vector<float> w(v.size());
        for (int o = 0; o < 512; o++) {
            double nrm = 0; for (int i = 0; i < kDim; i++) nrm += (double)v[(size_t)o * kDim + i] * v[(size_t)o * kDim + i];
            float s = g[o] / (float)std::sqrt(nrm + 1e-30);
            for (int i = 0; i < kDim; i++) w[(size_t)o * kDim + i] = v[(size_t)o * kDim + i] * s;
        }
        h.resize(w.size()); ggml_fp32_to_fp16_row(w.data(), h.data(), (int64_t)w.size());
        ggml_backend_tensor_set(mapW_, h.data(), 0, h.size() * 2);
        std::vector<float> r; st.toFloat(*rs, r); runningStd_ = r[0];
    }
    return true;
}

namespace {
TT headsOf(ggml_context* c, TT x2d, int offset, int n) {
    TT v = ggml_view_2d(c, x2d, kDim, n, x2d->nb[1], (size_t)offset * sizeof(float));
    return ggml_reshape_3d(c, ggml_cont(c, v), kHD, kHeads, n);
}
TT attend(ggml_context* c, TT q, TT k, TT v, TT mask) {
    TT a = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), ggml_cast(c, ggml_permute(c, k, 0, 2, 1, 3), GGML_TYPE_F16),
                               ggml_cast(c, ggml_permute(c, v, 0, 2, 1, 3), GGML_TYPE_F16), mask, 0.125f, 0.0f, 0.0f);
    return ggml_reshape_2d(c, a, kDim, q->ne[2]);
}
}  // namespace

bool SaVae::decodeChunk(const float* lat, int Tc, std::vector<float>& left, std::vector<float>& right, std::string& err) {
    Compute& cm = Compute::get();
    const int N = Tc * kSub;
    const size_t nodes = 4096;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false) + (1 << 20));
    ggml_init_params ip = {meta.size(), meta.data(), true};
    ggml_context* c = ggml_init(ip);
    TT in = ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, Tc); ggml_set_input(in);
    TT pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, N); ggml_set_input(pos);
    const int padN = (N + 63) / 64 * 64;
    TT mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, N, padN); ggml_set_input(mask);

    TT x = ggml_add(c, ggml_mul_mat(c, linW_, in), linB_);                                 // [1536, Tc]
    // per latent: [latent, 16 copies of the learned new token]
    TT x3 = ggml_reshape_3d(c, x, kDim, 1, Tc);
    TT nt = ggml_repeat(c, ggml_reshape_3d(c, newTok_, kDim, 1, 1), ggml_new_tensor_3d(c, GGML_TYPE_F32, kDim, kStride, Tc));
    x = ggml_reshape_2d(c, ggml_concat(c, x3, nt, 1), kDim, N);                           // [1536, 17*Tc]

    auto dytf = [&](TT t, const Dyt& d) { return ggml_add(c, ggml_mul(c, ggml_tanh(c, ggml_mul(c, t, d.alpha)), d.gamma), d.beta); };
    auto rope = [&](TT t) { return ggml_rope_ext(c, t, pos, nullptr, 32, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f); };
    for (auto& L : layers_) {
        TT h = dytf(x, L.pre);
        TT qkv = ggml_mul_mat(c, L.qkv, h);                                                  // [7680, N]
        TT q = rope(dytf(headsOf(c, qkv, 0, N), L.qn)), k = rope(dytf(headsOf(c, qkv, kDim, N), L.kn)), v = headsOf(c, qkv, 2 * kDim, N);
        TT qd = rope(dytf(headsOf(c, qkv, 3 * kDim, N), L.qn)), kd = rope(dytf(headsOf(c, qkv, 4 * kDim, N), L.kn));
        TT a = ggml_sub(c, attend(c, q, k, v, mask), attend(c, qd, kd, v, mask));
        x = ggml_add(c, x, ggml_mul_mat(c, L.out, a));
        TT h2 = dytf(x, L.ff);
        TT p = ggml_add(c, ggml_mul_mat(c, L.ff0w, h2), L.ff0b);                             // [9216, N]
        TT xa = ggml_cont(c, ggml_view_2d(c, p, 4608, N, p->nb[1], 0));
        TT xg = ggml_cont(c, ggml_view_2d(c, p, 4608, N, p->nb[1], (size_t)4608 * sizeof(float)));
        TT act = L.sin ? ggml_sin(c, ggml_scale(c, xg, 3.14159265359f)) : ggml_silu(c, xg);
        x = ggml_add(c, x, ggml_add(c, ggml_mul_mat(c, L.ff2w, ggml_mul(c, xa, act)), L.ff2b));
    }
    // keep the last 16 tokens of every group of 17, then the 1x1 mapping 1536 -> 512
    TT xg3 = ggml_reshape_3d(c, x, kDim, kSub, Tc);
    TT keep = ggml_cont(c, ggml_view_3d(c, xg3, kDim, kStride, Tc, xg3->nb[1], xg3->nb[2], xg3->nb[1]));
    TT y = ggml_add(c, ggml_mul_mat(c, mapW_, ggml_reshape_2d(c, keep, kDim, kStride * Tc)), mapB_);   // [512, 16*Tc]
    ggml_set_output(y);

    ggml_cgraph* g = ggml_new_graph_custom(c, nodes, false);
    ggml_build_forward_expand(g, y);
    ggml_backend_sched_reset(cm.sched);
    if (!ggml_backend_sched_alloc_graph(cm.sched, g)) { err = "cannot allocate audio VAE graph (out of memory?)"; ggml_free(c); return false; }
    std::vector<float> scaled((size_t)Tc * 256);                                            // bottleneck decode: latent * running_std
    for (size_t i = 0; i < scaled.size(); i++) scaled[i] = lat[i] * runningStd_;
    ggml_backend_tensor_set(in, scaled.data(), 0, scaled.size() * 4);
    std::vector<int32_t> ps(N); std::iota(ps.begin(), ps.end(), 0);
    ggml_backend_tensor_set(pos, ps.data(), 0, (size_t)N * 4);
    std::vector<ggml_fp16_t> mk((size_t)N * padN, ggml_fp32_to_fp16(-INFINITY));
    ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    for (int i = 0; i < N; i++) for (int j = std::max(0, i - kSub); j <= std::min(N - 1, i + kSub); j++) mk[(size_t)i * N + j] = zero;
    ggml_backend_tensor_set(mask, mk.data(), 0, mk.size() * 2);
    if (ggml_backend_sched_graph_compute(cm.sched, g) != GGML_STATUS_SUCCESS) { err = "audio VAE compute failed"; ggml_free(c); return false; }
    const int L = kStride * Tc;                                                             // tokens, 256 samples per channel each
    std::vector<float> o((size_t)512 * L);
    ggml_backend_tensor_get(y, o.data(), 0, o.size() * 4);
    ggml_backend_sched_reset(cm.sched);
    ggml_free(c);
    left.assign((size_t)L * 256, 0.0f); right.assign((size_t)L * 256, 0.0f);
    for (int l = 0; l < L; l++) for (int s = 0; s < 256; s++) {                              // channel index = ch*256 + sample-in-token
        left[(size_t)l * 256 + s] = o[(size_t)l * 512 + s];
        right[(size_t)l * 256 + s] = o[(size_t)l * 512 + 256 + s];
    }
    return true;
}

bool SaVae::decode(const std::vector<float>& latent, int T, std::vector<float>& pcm, std::string& err) {
    if (!loaded()) { err = "audio VAE not loaded"; return false; }
    int core = 128; const int ctxN = 16;
    if (const char* e = getenv("BAKERY_VAE_CORE")) core = std::max(1, atoi(e));   // testing aid                         // overlap of 16 latents covers the 12-layer receptive field
    pcm.assign((size_t)T * kSamplesPerLatent * 2, 0.0f);
    for (int s = 0; s < T; s += core) {
        int e = std::min(T, s + core), cs = std::max(0, s - ctxN), ce = std::min(T, e + ctxN);
        std::vector<float> l, r;
        if (!decodeChunk(&latent[(size_t)cs * 256], ce - cs, l, r, err)) return false;
        for (int t = s; t < e; t++) for (int i = 0; i < kSamplesPerLatent; i++) {
            size_t src = (size_t)(t - cs) * kSamplesPerLatent + i, dst = ((size_t)t * kSamplesPerLatent + i) * 2;
            pcm[dst] = l[src]; pcm[dst + 1] = r[src];
        }
    }
    return true;
}
