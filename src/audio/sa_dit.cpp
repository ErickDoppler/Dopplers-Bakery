#include "audio/sa_dit.h"
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
static const char* kP = "model.model.";

void SaDiT::unload() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
    buf_ = nullptr; ctx_ = nullptr; layers_.clear();
}

bool SaDiT::load(const std::string& path, std::string& err) {
    unload();
    StFile st;
    if (!st.open(path, err)) return false;
    std::string e2;
    if (!Compute::get().init(0, e2)) { err = e2; return false; }
    ggml_init_params ip = {ggml_tensor_overhead() * 800, nullptr, true};
    ctx_ = ggml_init(ip);
    struct Job { TT t; std::string name; int mode; };      // 0: matrix -> F16, 1: F32 vector
    std::vector<Job> jobs;
    bool ok = true;
    auto find = [&](const std::string& n) -> const StTensor* { const StTensor* t = st.find(n); if (!t && ok) { err = "missing tensor " + n; ok = false; } return t; };
    auto mat = [&](const std::string& n) -> TT {          // trailing 1-sized dims (1x1 convs) are dropped
        const StTensor* t = find(n); if (!t) return nullptr;
        std::vector<int64_t> d(t->shape.begin(), t->shape.end());
        while (d.size() > 2 && d.back() == 1) d.pop_back();
        TT g = ggml_new_tensor_2d(ctx_, GGML_TYPE_F16, d[1], d[0]); ggml_set_name(g, n.c_str()); jobs.push_back({g, n, 0}); return g;
    };
    auto vec = [&](const std::string& n) -> TT {
        const StTensor* t = find(n); if (!t) return nullptr;
        TT g = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, t->numel()); ggml_set_name(g, n.c_str()); jobs.push_back({g, n, 1}); return g;
    };
    const std::string P = kP;
    preW_ = mat(P + "preprocess_conv.weight"); postW_ = mat(P + "postprocess_conv.weight");
    cond0_ = mat(P + "to_cond_embed.0.weight"); cond2_ = mat(P + "to_cond_embed.2.weight");
    glob0_ = mat(P + "to_global_embed.0.weight"); glob2_ = mat(P + "to_global_embed.2.weight");
    time0w_ = mat(P + "to_timestep_embed.0.weight"); time0b_ = vec(P + "to_timestep_embed.0.bias");
    time2w_ = mat(P + "to_timestep_embed.2.weight"); time2b_ = vec(P + "to_timestep_embed.2.bias");
    const std::string T = P + "transformer.";
    gc0w_ = mat(T + "global_cond_embedder.0.weight"); gc0b_ = vec(T + "global_cond_embedder.0.bias");
    gc2w_ = mat(T + "global_cond_embedder.2.weight"); gc2b_ = vec(T + "global_cond_embedder.2.bias");
    projIn_ = mat(T + "project_in.weight"); projOut_ = mat(T + "project_out.weight");
    {   // memory tokens [64, 1536] stay F32
        const StTensor* t = find(T + "memory_tokens");
        if (t) { memTok_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, t->shape[1], t->shape[0]); jobs.push_back({memTok_, T + "memory_tokens", 1}); }
    }
    for (int i = 0; i < kDepth; i++) {
        std::string p = T + "layers." + std::to_string(i) + ".";
        Layer L;
        L.preN = vec(p + "pre_norm.gamma"); L.crossN = vec(p + "cross_attend_norm.gamma"); L.ffN = vec(p + "ff_norm.gamma");
        L.qkv = mat(p + "self_attn.to_qkv.weight"); L.sOut = mat(p + "self_attn.to_out.weight");
        L.sqn = vec(p + "self_attn.q_norm.gamma"); L.skn = vec(p + "self_attn.k_norm.gamma");
        L.cq = mat(p + "cross_attn.to_q.weight"); L.ckv = mat(p + "cross_attn.to_kv.weight"); L.cOut = mat(p + "cross_attn.to_out.weight");
        L.cqn = vec(p + "cross_attn.q_norm.gamma"); L.ckn = vec(p + "cross_attn.k_norm.gamma");
        L.ff0w = mat(p + "ff.ff.0.proj.weight"); L.ff0b = vec(p + "ff.ff.0.proj.bias");
        L.ff2w = mat(p + "ff.ff.2.weight"); L.ff2b = vec(p + "ff.ff.2.bias");
        L.ssg = vec(p + "to_scale_shift_gate");
        // text-to-audio has no local conditioning input (zeros): to_local_embed collapses to a constant per layer
        std::vector<float> b1, w2, b2;
        const StTensor *tb1 = find(p + "to_local_embed.0.bias"), *tw2 = find(p + "to_local_embed.2.weight"), *tb2 = find(p + "to_local_embed.2.bias");
        L.local = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, kDim);
        if (tb1 && tw2 && tb2) {
            st.toFloat(*tb1, b1); st.toFloat(*tw2, w2); st.toFloat(*tb2, b2);
            std::vector<float> c(kDim);
            for (int o = 0; o < kDim; o++) {
                double acc = b2[o];
                for (int k = 0; k < kDim; k++) { double s = b1[k] / (1.0 + std::exp(-(double)b1[k])); acc += (double)w2[(size_t)o * kDim + k] * s; }
                c[o] = (float)acc;
            }
            localConst_.push_back(std::move(c));
        }
        layers_.push_back(L);
    }
    if (!ok) { unload(); return false; }
    buf_ = Compute::get().allocDeviceWeights(ctx_);
    if (!buf_) { err = "cannot allocate Stable Audio DiT weights (needs ~3 GB)"; unload(); return false; }
    std::vector<float> f; std::vector<ggml_fp16_t> h;
    for (auto& j : jobs) {
        st.toFloat(*st.find(j.name), f);
        if (j.mode == 0) { h.resize(f.size()); ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t)f.size()); ggml_backend_tensor_set(j.t, h.data(), 0, h.size() * 2); }
        else ggml_backend_tensor_set(j.t, f.data(), 0, f.size() * 4);
    }
    for (int i = 0; i < kDepth && i < (int)localConst_.size(); i++) ggml_backend_tensor_set(layers_[i].local, localConst_[i].data(), 0, kDim * 4);
    localConst_.clear();
    // conditioner pieces used on the host
    if (const StTensor* t = find("conditioner.conditioners.prompt.padding_embedding")) st.toFloat(*t, padding_);
    if (const StTensor* t = find("conditioner.conditioners.seconds_total.embedder.embedding.1.weight")) st.toFloat(*t, secW_);
    if (const StTensor* t = find("conditioner.conditioners.seconds_total.embedder.embedding.1.bias")) st.toFloat(*t, secB_);
    if (!ok) { unload(); return false; }
    return true;
}

namespace {
const float kRmsEps = 1.1920929e-7f;      // torch.finfo(float32).eps, what F.rms_norm uses without an explicit eps

// exponentially spaced Fourier features (dim 256), as in ExpoFourierFeatures(256, 0.5, 10000)
void expoFeatures(float t, float* out256) {
    const double lo = std::log(0.5), hi = std::log(10000.0);
    for (int i = 0; i < 128; i++) {
        double ramp = i / 127.0, f = std::exp(ramp * (hi - lo) + lo), a = (double)t * f * 2.0 * M_PI;
        out256[i] = (float)std::cos(a); out256[128 + i] = (float)std::sin(a);
    }
}

TT headsOf(ggml_context* c, TT x2d, int offset, int n) {           // columns [offset, offset+1536) of [W, n] -> [64, 24, n]
    TT v = ggml_view_2d(c, x2d, SaDiT::kDim, n, x2d->nb[1], (size_t)offset * sizeof(float));
    return ggml_reshape_3d(c, ggml_cont(c, v), SaDiT::kHeadDim, SaDiT::kHeads, n);
}
TT attend(ggml_context* c, TT q, TT k, TT v) {                     // all [64, 24, n]; returns [1536, nq]
    TT a = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), ggml_cast(c, ggml_permute(c, k, 0, 2, 1, 3), GGML_TYPE_F16),
                               ggml_cast(c, ggml_permute(c, v, 0, 2, 1, 3), GGML_TYPE_F16), nullptr, 0.125f, 0.0f, 0.0f);
    return ggml_reshape_2d(c, a, SaDiT::kDim, q->ne[2]);
}
}  // namespace

bool SaDiT::forward(const std::vector<float>& latent, int T, float sigma, const std::vector<float>& text, int n, float seconds,
                    std::vector<float>& out, std::string& err) {
    if (!loaded()) { err = "Stable Audio DiT not loaded"; return false; }
    if (padding_.size() != 768 || secW_.size() != 768 * 256) { err = "conditioner weights missing from the checkpoint"; return false; }
    Compute& cm = Compute::get();
    const int N = T + kMem;
    const int maxText = 256;
    const int nText = std::max(n, maxText);                 // tokens are padded up to 256 with padding_embedding
    const int Nc = nText + 1;                               // + the seconds token

    // ---- host-side conditioning inputs
    std::vector<float> ctx((size_t)Nc * 768);
    for (int i = 0; i < nText; i++) {
        const float* src = i < n ? &text[(size_t)i * 768] : padding_.data();
        memcpy(&ctx[(size_t)i * 768], src, 768 * 4);
    }
    float feat[256];
    std::vector<float> sec(768);
    {
        float norm = std::min(std::max(seconds, 0.0f), 384.0f) / 384.0f;
        expoFeatures(norm, feat);
        for (int o = 0; o < 768; o++) { double acc = secB_[o]; for (int i = 0; i < 256; i++) acc += (double)secW_[(size_t)o * 256 + i] * feat[i]; sec[o] = (float)acc; }
        memcpy(&ctx[(size_t)nText * 768], sec.data(), 768 * 4);
    }
    float tfeat[256]; expoFeatures(sigma, tfeat);

    const size_t nodes = 8192;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false) + (1 << 20));
    ggml_init_params ip = {meta.size(), meta.data(), true};
    ggml_context* c = ggml_init(ip);
    std::vector<std::pair<std::string, TT>> taps;
    auto tap = [&](const std::string& nm, TT x) { if (debug) { ggml_set_output(x); taps.push_back({nm, x}); } };

    TT xin = ggml_new_tensor_2d(c, GGML_TYPE_F32, kCh, T); ggml_set_input(xin);
    TT ctxIn = ggml_new_tensor_2d(c, GGML_TYPE_F32, 768, Nc); ggml_set_input(ctxIn);
    TT secIn = ggml_new_tensor_2d(c, GGML_TYPE_F32, 768, 1); ggml_set_input(secIn);
    TT tIn = ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, 1); ggml_set_input(tIn);
    TT pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, N); ggml_set_input(pos);
    TT lmask = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, N); ggml_set_input(lmask);

    // conditioning
    TT ce = ggml_mul_mat(c, cond2_, ggml_silu(c, ggml_mul_mat(c, cond0_, ctxIn)));                    // [1536, Nc]
    TT ge = ggml_mul_mat(c, glob2_, ggml_silu(c, ggml_mul_mat(c, glob0_, secIn)));                    // [1536, 1]
    TT te = ggml_add(c, ggml_mul_mat(c, time0w_, tIn), time0b_);
    te = ggml_add(c, ggml_mul_mat(c, time2w_, ggml_silu(c, te)), time2b_);                           // [1536, 1]
    TT gc = ggml_add(c, ggml_mul_mat(c, gc0w_, ggml_add(c, ge, te)), gc0b_);
    gc = ggml_add(c, ggml_mul_mat(c, gc2w_, ggml_silu(c, gc)), gc2b_);                               // [9216, 1]
    tap("cond_embed", ce); tap("global_cond", gc);

    // input path
    TT h0 = ggml_add(c, ggml_mul_mat(c, preW_, xin), xin);                                           // [256, T]
    TT x = ggml_concat(c, memTok_, ggml_mul_mat(c, projIn_, h0), 1);                                  // [1536, N]
    TT maskFull = ggml_repeat(c, lmask, ggml_new_tensor_2d(c, GGML_TYPE_F32, kDim, N));

    for (int i = 0; i < kDepth; i++) {
        Layer& L = layers_[i];
        TT ssg = ggml_add(c, gc, L.ssg);                                                              // [9216, 1]
        auto chunk = [&](int k) { return ggml_view_1d(c, ssg, kDim, (size_t)k * kDim * sizeof(float)); };
        TT scaleS = chunk(0), shiftS = chunk(1), gateS = chunk(2), scaleF = chunk(3), shiftF = chunk(4), gateF = chunk(5);

        // differential self-attention
        TT n1 = ggml_mul(c, ggml_rms_norm(c, x, kRmsEps), L.preN);
        n1 = ggml_add(c, ggml_mul(c, n1, ggml_scale_bias(c, scaleS, 1.0f, 1.0f)), shiftS);
        TT qkv = ggml_mul_mat(c, L.qkv, n1);                                                          // [7680, N]
        TT q = headsOf(c, qkv, 0, N), k = headsOf(c, qkv, kDim, N), v = headsOf(c, qkv, 2 * kDim, N);
        TT qd = headsOf(c, qkv, 3 * kDim, N), kd = headsOf(c, qkv, 4 * kDim, N);
        auto nrm = [&](TT t, TT g) { return ggml_mul(c, ggml_rms_norm(c, t, kRmsEps), g); };
        auto rope = [&](TT t) { return ggml_rope_ext(c, t, pos, nullptr, 32, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f); };
        q = rope(nrm(q, L.sqn)); k = rope(nrm(k, L.skn)); qd = rope(nrm(qd, L.sqn)); kd = rope(nrm(kd, L.skn));
        TT a = ggml_sub(c, attend(c, q, k, v), attend(c, qd, kd, v));
        a = ggml_mul_mat(c, L.sOut, a);
        a = ggml_mul(c, a, ggml_sigmoid(c, ggml_scale_bias(c, gateS, -1.0f, 1.0f)));
        x = ggml_add(c, a, x);

        // differential cross-attention (no gating, no rope)
        TT n2 = ggml_mul(c, ggml_rms_norm(c, x, kRmsEps), L.crossN);
        TT cqq = ggml_mul_mat(c, L.cq, n2);                                                           // [3072, N]
        TT ckv = ggml_mul_mat(c, L.ckv, ce);                                                          // [4608, Nc]
        TT q2 = nrm(headsOf(c, cqq, 0, N), L.cqn), q2d = nrm(headsOf(c, cqq, kDim, N), L.cqn);
        TT k2 = nrm(headsOf(c, ckv, 0, Nc), L.ckn), k2d = nrm(headsOf(c, ckv, kDim, Nc), L.ckn), v2 = headsOf(c, ckv, 2 * kDim, Nc);
        TT ca = ggml_sub(c, attend(c, q2, k2, v2), attend(c, q2d, k2d, v2));
        x = ggml_add(c, x, ggml_mul_mat(c, L.cOut, ca));

        // constant local embedding on the non-memory tokens
        x = ggml_add(c, x, ggml_mul(c, maskFull, L.local));

        // feed-forward (GLU with SiLU gate)
        TT n3 = ggml_mul(c, ggml_rms_norm(c, x, kRmsEps), L.ffN);
        n3 = ggml_add(c, ggml_mul(c, n3, ggml_scale_bias(c, scaleF, 1.0f, 1.0f)), shiftF);
        TT p = ggml_add(c, ggml_mul_mat(c, L.ff0w, n3), L.ff0b);                                      // [12288, N]
        TT xa = ggml_cont(c, ggml_view_2d(c, p, 6144, N, p->nb[1], 0));
        TT xg = ggml_cont(c, ggml_view_2d(c, p, 6144, N, p->nb[1], (size_t)6144 * sizeof(float)));
        TT f = ggml_add(c, ggml_mul_mat(c, L.ff2w, ggml_mul(c, xa, ggml_silu(c, xg))), L.ff2b);
        f = ggml_mul(c, f, ggml_sigmoid(c, ggml_scale_bias(c, gateF, -1.0f, 1.0f)));
        x = ggml_add(c, f, x);
        if (i == 0) tap("blk0", x); if (i == 1) tap("blk1", x); if (i == kDepth - 1) tap("blk23", x);
    }
    TT xt = ggml_view_2d(c, x, kDim, T, x->nb[1], (size_t)kMem * x->nb[1]);
    TT o = ggml_mul_mat(c, projOut_, xt);                                                             // [256, T]
    o = ggml_add(c, o, ggml_mul_mat(c, postW_, o));
    ggml_set_output(o);

    ggml_cgraph* g = ggml_new_graph_custom(c, nodes, false);
    ggml_build_forward_expand(g, o);
    for (auto& tp : taps) ggml_build_forward_expand(g, tp.second);
    ggml_backend_sched_reset(cm.sched);
    if (!ggml_backend_sched_alloc_graph(cm.sched, g)) { err = "cannot allocate Stable Audio DiT graph"; ggml_free(c); return false; }
    ggml_backend_tensor_set(xin, latent.data(), 0, (size_t)T * kCh * 4);
    ggml_backend_tensor_set(ctxIn, ctx.data(), 0, ctx.size() * 4);
    ggml_backend_tensor_set(secIn, sec.data(), 0, 768 * 4);
    ggml_backend_tensor_set(tIn, tfeat, 0, 256 * 4);
    std::vector<int32_t> ps(N); std::iota(ps.begin(), ps.end(), 0);
    ggml_backend_tensor_set(pos, ps.data(), 0, (size_t)N * 4);
    std::vector<float> lm(N, 1.0f); for (int i = 0; i < kMem; i++) lm[i] = 0.0f;
    ggml_backend_tensor_set(lmask, lm.data(), 0, (size_t)N * 4);
    if (ggml_backend_sched_graph_compute(cm.sched, g) != GGML_STATUS_SUCCESS) { err = "Stable Audio DiT compute failed"; ggml_free(c); return false; }
    dump.clear();
    for (auto& tp : taps) { std::vector<float> v((size_t)ggml_nelements(tp.second)); ggml_backend_tensor_get(tp.second, v.data(), 0, v.size() * 4); dump.push_back({tp.first, std::move(v)}); }
    out.resize((size_t)T * kCh);
    ggml_backend_tensor_get(o, out.data(), 0, out.size() * 4);
    ggml_backend_sched_reset(cm.sched);
    ggml_free(c);
    return true;
}
