#include "ltx/ltx_dit.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "ltx/ltx_ggml.h"
#include "util/sys.h"

using namespace ltxg;
namespace {
constexpr int kVD = 4096, kAD = 2048, kHeads = 32;

struct Att { TT wq, bq, wk, bk, wv, bv, wo, bo, qn, kn, wg, bg; };
struct Rope { TT cs, sn; int half; };
struct BlockW {                                    // everything one transformer block needs on the device
    WeightSet W; Att va1{}, va2{}, aa1{}, aa2{}, a2v{}, v2a{};
    TT fw1 = nullptr, fb1 = nullptr, fw2 = nullptr, fb2 = nullptr, gw1 = nullptr, gb1 = nullptr, gw2 = nullptr, gb2 = nullptr;
    TT tV = nullptr, tA = nullptr, tPV = nullptr, tPA = nullptr, tCaA = nullptr, tCaV = nullptr;
    bool loaded = false;
};

TT tab(WeightSet& W, const LtxWeights& ck, const std::string& n, std::string& err) {          // small 2D F32 table [rows][dim] -> ggml [dim, rows]
    std::vector<int64_t> sh; std::vector<float> d;
    if (!ck.shapeOf(n, sh) || !ck.loadF32(n, d, err)) { if (err.empty()) err = "missing " + n; return nullptr; }
    TT t = ggml_new_tensor_2d(W.ctx, GGML_TYPE_F32, sh[1], sh[0]);
    W.items.push_back({t, std::move(d), false});
    return t;
}
bool loadAtt(WeightSet& W, const LtxWeights& ck, const std::string& p, Att& a, std::string& err) {
    a.wq = W.mat(ck, p + "to_q.weight", err); a.bq = W.vec(ck, p + "to_q.bias", err);
    a.wk = W.mat(ck, p + "to_k.weight", err); a.bk = W.vec(ck, p + "to_k.bias", err);
    a.wv = W.mat(ck, p + "to_v.weight", err); a.bv = W.vec(ck, p + "to_v.bias", err);
    a.wo = W.mat(ck, p + "to_out.0.weight", err); a.bo = W.vec(ck, p + "to_out.0.bias", err);
    a.qn = W.vec(ck, p + "q_norm.weight", err); a.kn = W.vec(ck, p + "k_norm.weight", err);
    a.wg = W.mat(ck, p + "to_gate_logits.weight", err); a.bg = W.vec(ck, p + "to_gate_logits.bias", err);
    return err.empty();
}

TT ropeApply(ggml_context* c, TT t, const Rope& r) {             // t [hd, heads, N]; split-half rotation with per-head frequencies
    const int64_t hd = t->ne[0], heads = t->ne[1], N = t->ne[2]; const int half = (int)(hd / 2);
    TT x1 = ggml_view_3d(c, t, half, heads, N, t->nb[1], t->nb[2], 0), x2 = ggml_view_3d(c, t, half, heads, N, t->nb[1], t->nb[2], (size_t)half * 4);
    TT o1 = ggml_sub(c, ggml_mul(c, x1, r.cs), ggml_mul(c, x2, r.sn)), o2 = ggml_add(c, ggml_mul(c, x2, r.cs), ggml_mul(c, x1, r.sn));
    return ggml_concat(c, o1, o2, 0);
}

bool flashAttention() { static const bool on = getenv("LTX_NO_FLASH") == nullptr; return on; }

// gated multi-head attention: xq [dq, Nq] attends to xkv [dk, Nk]
TT attention(ggml_context* c, const Att& a, TT xq, TT xkv, int hd, const Rope* rq, const Rope* rk) {
    const int heads = kHeads; const int64_t Nq = xq->ne[1], Nk = xkv->ne[1];
    TT q = ggml_mul(c, ggml_rms_norm(c, ggml_add(c, mm(c, a.wq, xq), a.bq), 1e-5f), a.qn);
    TT k = ggml_mul(c, ggml_rms_norm(c, ggml_add(c, mm(c, a.wk, xkv), a.bk), 1e-5f), a.kn);
    TT v = ggml_add(c, mm(c, a.wv, xkv), a.bv);
    TT q3 = ggml_reshape_3d(c, q, hd, heads, Nq), k3 = ggml_reshape_3d(c, k, hd, heads, Nk), v3 = ggml_reshape_3d(c, v, hd, heads, Nk);
    if (rq) q3 = ropeApply(c, q3, *rq);
    if (rk) k3 = ropeApply(c, k3, *rk);
    TT o;
    if (flashAttention()) {                                                                            // fused kernel: no Nq x Nk score matrix in memory
        TT kf = ggml_cast(c, ggml_permute(c, k3, 0, 2, 1, 3), GGML_TYPE_F16), vf = ggml_cast(c, ggml_permute(c, v3, 0, 2, 1, 3), GGML_TYPE_F16);
        o = ggml_flash_attn_ext(c, ggml_permute(c, q3, 0, 2, 1, 3), kf, vf, nullptr, 1.0f / std::sqrt((float)hd), 0.f, 0.f);        // [hd, heads, Nq]
    } else {
        TT sc = ggml_soft_max_ext(c, ggml_mul_mat(c, ggml_permute(c, k3, 0, 2, 1, 3), ggml_permute(c, q3, 0, 2, 1, 3)), nullptr, 1.0f / std::sqrt((float)hd), 0.f);      // [Nk, Nq, heads]
        TT vp = ggml_cont(c, ggml_permute(c, v3, 1, 2, 0, 3));                                          // [Nk, hd, heads]
        o = ggml_cont(c, ggml_permute(c, ggml_mul_mat(c, vp, sc), 0, 2, 1, 3));                        // [hd, heads, Nq]
    }
    TT gate = ggml_scale(c, ggml_sigmoid(c, ggml_add(c, mm(c, a.wg, xq), a.bg)), 2.0f);   // [heads, Nq]
    o = ggml_mul(c, o, ggml_reshape_3d(c, gate, 1, heads, Nq));
    return ggml_add(c, mm(c, a.wo, ggml_reshape_2d(c, o, (int64_t)hd * heads, Nq)), a.bo);
}

TT feedForward(ggml_context* c, TT w1, TT b1, TT w2, TT b2, TT x) { return ggml_add(c, mm(c, w2, ggml_gelu(c, ggml_add(c, mm(c, w1, x), b1))), b2); }
TT affine(ggml_context* c, TT x, TT scale, TT shift) { return ggml_add(c, ggml_add(c, x, ggml_mul(c, x, scale)), shift); }      // x * (1 + scale) + shift
// row r of (table + ada): table [dim, rows], ada [rows * dim, F]  ->  [dim, 1, F]
TT modRow(ggml_context* c, TT table, TT ada, int r, int dim) {
    const int64_t F = ada->ne[1];
    TT a = ggml_cont(c, ggml_view_2d(c, ada, dim, F, ada->nb[1], (size_t)r * dim * 4));
    TT t = ggml_cont(c, ggml_view_2d(c, table, dim, 1, table->nb[1], (size_t)r * table->nb[1]));
    return ggml_reshape_3d(c, ggml_add(c, a, t), dim, 1, F);
}

// rotary tables [half, heads, N] for tokens with fractional positions frac[n * nd + d] in [0, 1] (already divided by the max position)
void makeRope(const std::vector<float>& frac, int nd, int N, int dim, std::vector<float>& cs, std::vector<float>& sn) {
    const int total = dim / 2, half = total / kHeads, nIdx = dim / (2 * nd), pad = total - nIdx * nd;
    std::vector<float> idx(nIdx);
    for (int k = 0; k < nIdx; k++) idx[k] = (float)(std::pow(10000.0, (double)k / (double)(nIdx - 1)) * 3.14159265358979323846 / 2.0);
    cs.assign((size_t)half * kHeads * N, 1.f); sn.assign(cs.size(), 0.f);
    for (int n = 0; n < N; n++) for (int j = 0; j < total; j++) {
        if (j < pad) continue;
        const int jj = j - pad, k = jj / nd, d = jj % nd;
        const float a = idx[k] * (frac[(size_t)n * nd + d] * 2.0f - 1.0f);
        const size_t o = ((size_t)n * kHeads + j / half) * half + j % half;
        cs[o] = std::cos(a); sn[o] = std::sin(a);
    }
}

// sinusoidal timestep embedding (256) -> linear_1 -> silu -> linear_2 = emb; modulation = linear(silu(emb))
bool adaln(const LtxWeights& ck, const std::string& p, const std::vector<float>& tsScaled, std::vector<float>& emb, std::vector<float>& mod, std::string& err) {
    const int n = (int)tsScaled.size();
    std::vector<float> sin256((size_t)n * 256);
    for (int i = 0; i < n; i++) for (int k = 0; k < 128; k++) {
        const float a = tsScaled[i] * std::exp(-std::log(10000.0f) * (float)k / 128.0f);
        sin256[(size_t)i * 256 + k] = std::cos(a); sin256[(size_t)i * 256 + 128 + k] = std::sin(a);          // flip_sin_to_cos: [cos, sin]
    }
    WeightSet W;
    TT w1 = W.mat(ck, p + "emb.timestep_embedder.linear_1.weight", err), b1 = W.vec(ck, p + "emb.timestep_embedder.linear_1.bias", err);
    TT w2 = W.mat(ck, p + "emb.timestep_embedder.linear_2.weight", err), b2 = W.vec(ck, p + "emb.timestep_embedder.linear_2.bias", err);
    TT wl = W.mat(ck, p + "linear.weight", err), bl = W.vec(ck, p + "linear.bias", err);
    if (!err.empty() || !W.upload(err)) return false;
    Graph G(64); ggml_context* c = G.c;
    TT x = ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, n); ggml_set_input(x);
    TT e = ggml_add(c, mm(c, w2, ggml_silu(c, ggml_add(c, mm(c, w1, x), b1))), b2);
    TT m = ggml_add(c, mm(c, wl, ggml_silu(c, e)), bl);
    ggml_set_output(e); ggml_set_output(m); ggml_build_forward_expand(G.g, e); ggml_build_forward_expand(G.g, m);
    if (!runGraph(G, err)) return false;
    ggml_backend_tensor_set(x, sin256.data(), 0, sin256.size() * 4);
    if (!computeGraph(G, err)) return false;
    emb.resize((size_t)ggml_nelements(e)); mod.resize((size_t)ggml_nelements(m));
    ggml_backend_tensor_get(e, emb.data(), 0, emb.size() * 4); ggml_backend_tensor_get(m, mod.data(), 0, mod.size() * 4);
    ggml_backend_sched_reset(Compute::get().sched);
    return true;
}

TT input2d(ggml_context* c, int64_t a, int64_t b) { TT t = ggml_new_tensor_2d(c, GGML_TYPE_F32, a, b); ggml_set_input(t); return t; }
void setT(TT t, const std::vector<float>& v) { ggml_backend_tensor_set(t, v.data(), 0, v.size() * 4); }
}  // namespace

bool LtxDit::forward(const LtxDitInput& in, std::vector<float>& videoOut, std::vector<float>& audioOut, const std::function<void(int)>& progress, std::string& err) {
    if (!Compute::get().init(0, err)) return false;
    const LtxWeights& ck = *ckpt; const std::string P = "model.diffusion_model.";
    const int F = in.F, hw = in.h * in.w, N = F * hw, Ta = in.T;
    // ---- tokens
    std::vector<float> vtok((size_t)N * 128), atok((size_t)Ta * 128);
    for (int c = 0; c < 128; c++) for (int f = 0; f < F; f++) for (int p = 0; p < hw; p++) vtok[((size_t)f * hw + p) * 128 + c] = in.video[((size_t)c * F + f) * hw + p];
    for (int c = 0; c < 8; c++) for (int t = 0; t < Ta; t++) for (int q = 0; q < 16; q++) atok[(size_t)t * 128 + c * 16 + q] = in.audio[((size_t)c * Ta + t) * 16 + q];
    auto project = [&](const std::string& name, const std::vector<float>& tok, int n, int outDim, std::vector<float>& out) -> bool {
        WeightSet W; TT w = W.mat(ck, P + name + ".weight", err), b = W.vec(ck, P + name + ".bias", err);
        if (!err.empty() || !W.upload(err)) return false;
        Graph G(16); ggml_context* c = G.c; TT x = input2d(c, 128, n); TT y = ggml_add(c, mm(c, w, x), b);
        ggml_set_output(y); ggml_build_forward_expand(G.g, y);
        if (!runGraph(G, err)) return false;
        setT(x, tok); if (!computeGraph(G, err)) return false;
        out.resize((size_t)n * outDim); ggml_backend_tensor_get(y, out.data(), 0, out.size() * 4); ggml_backend_sched_reset(Compute::get().sched); return true;
    };
    std::vector<float> vx, ax;
    if (!project("patchify_proj", vtok, N, kVD, vx) || !project("audio_patchify_proj", atok, Ta, kAD, ax)) return false;

    // ---- timesteps (all scaled by 1000); a single video value is shared by every frame
    std::vector<float> vts(F), vsc(F);
    for (int f = 0; f < F; f++) { vts[f] = in.videoSigma.size() == 1 ? in.videoSigma[0] : in.videoSigma[f]; vsc[f] = vts[f] * 1000.f; }
    const float vmax = *std::max_element(vsc.begin(), vsc.end()), asc = in.audioSigma * 1000.f;
    std::vector<float> vEmb, vMod, vPrompt, aEmb, aMod, aPrompt, aCa, vCa, a2vGate, v2aGate, tmp;
    if (!adaln(ck, P + "adaln_single.", vsc, vEmb, vMod, err) || !adaln(ck, P + "prompt_adaln_single.", {vmax}, tmp, vPrompt, err) ||
        !adaln(ck, P + "audio_adaln_single.", {asc}, aEmb, aMod, err) || !adaln(ck, P + "audio_prompt_adaln_single.", {asc}, tmp, aPrompt, err) ||
        !adaln(ck, P + "av_ca_audio_scale_shift_adaln_single.", {asc}, tmp, aCa, err) || !adaln(ck, P + "av_ca_video_scale_shift_adaln_single.", vsc, tmp, vCa, err) ||
        !adaln(ck, P + "av_ca_a2v_gate_adaln_single.", {asc}, tmp, a2vGate, err) || !adaln(ck, P + "av_ca_v2a_gate_adaln_single.", {vmax}, tmp, v2aGate, err)) return false;

    // ---- rotary tables
    std::vector<float> fv((size_t)N * 3), tv(N), fa(Ta), ta(Ta);
    for (int f = 0; f < F; f++) {
        auto pt = [&](int fr) { return std::max((float)fr * 8.f + 1.f - 8.f, 0.f); };
        const float mid = (pt(f) + pt(f + 1)) / 2.f / in.frameRate;
        for (int y = 0; y < in.h; y++) for (int x = 0; x < in.w; x++) {
            const size_t n = ((size_t)f * in.h + y) * in.w + x;
            fv[n * 3] = mid / 20.f; fv[n * 3 + 1] = ((float)y * 32.f + (float)(y + 1) * 32.f) / 2.f / 2048.f; fv[n * 3 + 2] = ((float)x * 32.f + (float)(x + 1) * 32.f) / 2.f / 2048.f;
            tv[n] = mid / 20.f;
        }
    }
    for (int t = 0; t < Ta; t++) {
        auto sec = [&](int l) { return std::max((float)l * 4.f + 1.f - 4.f, 0.f) * 160.f / 16000.f; };
        const float mid = (sec(t) + sec(t + 1)) / 2.f; fa[t] = mid / 20.f; ta[t] = mid / 20.f;
    }
    std::vector<float> vrC, vrS, arC, arS, vcC, vcS, acC, acS;
    makeRope(fv, 3, N, kVD, vrC, vrS); makeRope(fa, 1, Ta, kAD, arC, arS); makeRope(tv, 1, N, kAD, vcC, vcS); makeRope(ta, 1, Ta, kAD, acC, acS);

    // ---- blocks
    for (int b = 0; b < blocks; b++) {
        if (progress) progress(b);
        const std::string p = P + "transformer_blocks." + std::to_string(b) + ".";
        std::shared_ptr<BlockW> bw = (b < (int)resident_.size() && resident_[b]) ? std::static_pointer_cast<BlockW>(resident_[b]) : std::make_shared<BlockW>();
        WeightSet& W = bw->W;
        Att &va1 = bw->va1, &va2 = bw->va2, &aa1 = bw->aa1, &aa2 = bw->aa2, &a2v = bw->a2v, &v2a = bw->v2a;
        TT &fw1 = bw->fw1, &fb1 = bw->fb1, &fw2 = bw->fw2, &fb2 = bw->fb2, &gw1 = bw->gw1, &gb1 = bw->gb1, &gw2 = bw->gw2, &gb2 = bw->gb2;
        TT &tV = bw->tV, &tA = bw->tA, &tPV = bw->tPV, &tPA = bw->tPA, &tCaA = bw->tCaA, &tCaV = bw->tCaV;
        if (!bw->loaded) {
            loadAtt(W, ck, p + "attn1.", va1, err) && loadAtt(W, ck, p + "attn2.", va2, err) && loadAtt(W, ck, p + "audio_attn1.", aa1, err) && loadAtt(W, ck, p + "audio_attn2.", aa2, err) &&
                loadAtt(W, ck, p + "audio_to_video_attn.", a2v, err) && loadAtt(W, ck, p + "video_to_audio_attn.", v2a, err);
            fw1 = W.mat(ck, p + "ff.net.0.proj.weight", err); fb1 = W.vec(ck, p + "ff.net.0.proj.bias", err); fw2 = W.mat(ck, p + "ff.net.2.weight", err); fb2 = W.vec(ck, p + "ff.net.2.bias", err);
            gw1 = W.mat(ck, p + "audio_ff.net.0.proj.weight", err); gb1 = W.vec(ck, p + "audio_ff.net.0.proj.bias", err); gw2 = W.mat(ck, p + "audio_ff.net.2.weight", err); gb2 = W.vec(ck, p + "audio_ff.net.2.bias", err);
            tV = tab(W, ck, p + "scale_shift_table", err); tA = tab(W, ck, p + "audio_scale_shift_table", err); tPV = tab(W, ck, p + "prompt_scale_shift_table", err); tPA = tab(W, ck, p + "audio_prompt_scale_shift_table", err);
            tCaA = tab(W, ck, p + "scale_shift_table_a2v_ca_audio", err); tCaV = tab(W, ck, p + "scale_shift_table_a2v_ca_video", err);
            if (!err.empty() || !W.upload(err)) return false;
            bw->loaded = true;
        }
        // keep the block in video memory for the next steps while there is room (the rest of the memory is for the activations)
        const bool keepIt = !(b < (int)resident_.size() && resident_[b]) && Compute::get().gpuFreeMB() > (size_t)(4500 + N / 2);

        Graph G(2048); ggml_context* c = G.c;
        TT xv = input2d(c, kVD, N), xa = input2d(c, kAD, Ta);
        TT iVada = input2d(c, 9 * kVD, F), iAada = input2d(c, 9 * kAD, 1), iVp = input2d(c, 2 * kVD, 1), iAp = input2d(c, 2 * kAD, 1);
        TT iVca = input2d(c, 4 * kVD, F), iAca = input2d(c, 4 * kAD, 1), iGv = input2d(c, kVD, 1), iGa = input2d(c, kAD, 1);
        TT iVctx = input2d(c, kVD, in.Lv), iActx = input2d(c, kAD, in.La);
        auto ropeIn = [&](int half, int n) { Rope r; r.half = half; r.cs = ggml_new_tensor_3d(c, GGML_TYPE_F32, half, kHeads, n); r.sn = ggml_new_tensor_3d(c, GGML_TYPE_F32, half, kHeads, n); ggml_set_input(r.cs); ggml_set_input(r.sn); return r; };
        Rope rV = ropeIn(64, N), rA = ropeIn(32, Ta), rVc = ropeIn(32, N), rAc = ropeIn(32, Ta);

        TT vx3 = ggml_reshape_3d(c, xv, kVD, hw, F), ax3 = ggml_reshape_3d(c, xa, kAD, Ta, 1);
        auto as2 = [&](TT t3, int dim, int n) { return ggml_reshape_2d(c, t3, dim, n); };
        auto as3v = [&](TT t2) { return ggml_reshape_3d(c, t2, kVD, hw, F); };
        auto as3a = [&](TT t2) { return ggml_reshape_3d(c, t2, kAD, Ta, 1); };
        // video / audio self attention
        {   TT n1 = affine(c, ggml_rms_norm(c, vx3, 1e-6f), modRow(c, tV, iVada, 1, kVD), modRow(c, tV, iVada, 0, kVD));
            TT a = attention(c, va1, as2(n1, kVD, N), as2(n1, kVD, N), 128, &rV, &rV);
            vx3 = ggml_add(c, vx3, ggml_mul(c, as3v(a), modRow(c, tV, iVada, 2, kVD))); }
        {   TT n1 = affine(c, ggml_rms_norm(c, ax3, 1e-6f), modRow(c, tA, iAada, 1, kAD), modRow(c, tA, iAada, 0, kAD));
            TT a = attention(c, aa1, as2(n1, kAD, Ta), as2(n1, kAD, Ta), 64, &rA, &rA);
            ax3 = ggml_add(c, ax3, ggml_mul(c, as3a(a), modRow(c, tA, iAada, 2, kAD))); }
        // text cross attention with adaLN on the query and the keys / values
        {   TT q = affine(c, ggml_rms_norm(c, vx3, 1e-6f), modRow(c, tV, iVada, 7, kVD), modRow(c, tV, iVada, 6, kVD));
            TT sh = as2(modRow(c, tPV, iVp, 0, kVD), kVD, 1), sc = as2(modRow(c, tPV, iVp, 1, kVD), kVD, 1);
            TT kv = affine(c, iVctx, sc, sh);
            TT a = attention(c, va2, as2(q, kVD, N), kv, 128, nullptr, nullptr);
            vx3 = ggml_add(c, vx3, ggml_mul(c, as3v(a), modRow(c, tV, iVada, 8, kVD))); }
        {   TT q = affine(c, ggml_rms_norm(c, ax3, 1e-6f), modRow(c, tA, iAada, 7, kAD), modRow(c, tA, iAada, 6, kAD));
            TT sh = as2(modRow(c, tPA, iAp, 0, kAD), kAD, 1), sc = as2(modRow(c, tPA, iAp, 1, kAD), kAD, 1);
            TT kv = affine(c, iActx, sc, sh);
            TT a = attention(c, aa2, as2(q, kAD, Ta), kv, 64, nullptr, nullptr);
            ax3 = ggml_add(c, ax3, ggml_mul(c, as3a(a), modRow(c, tA, iAada, 8, kAD))); }
        // audio <-> video cross attention (both use the normalized streams from before the exchange)
        {   TT vn = ggml_rms_norm(c, vx3, 1e-6f), an = ggml_rms_norm(c, ax3, 1e-6f);
            TT vs0 = modRow(c, tCaV, iVca, 0, kVD), vs1 = modRow(c, tCaV, iVca, 1, kVD), vs2 = modRow(c, tCaV, iVca, 2, kVD), vs3 = modRow(c, tCaV, iVca, 3, kVD);
            TT as0 = modRow(c, tCaA, iAca, 0, kAD), as1 = modRow(c, tCaA, iAca, 1, kAD), as2_ = modRow(c, tCaA, iAca, 2, kAD), as3_ = modRow(c, tCaA, iAca, 3, kAD);
            TT vgate = ggml_reshape_3d(c, ggml_add(c, iGv, ggml_cont(c, ggml_view_2d(c, tCaV, kVD, 1, tCaV->nb[1], (size_t)4 * tCaV->nb[1]))), kVD, 1, 1);
            TT agate = ggml_reshape_3d(c, ggml_add(c, iGa, ggml_cont(c, ggml_view_2d(c, tCaA, kAD, 1, tCaA->nb[1], (size_t)4 * tCaA->nb[1]))), kAD, 1, 1);
            TT vq = affine(c, vn, vs0, vs1), ak = affine(c, an, as0, as1);
            TT o1 = attention(c, a2v, as2(vq, kVD, N), as2(ak, kAD, Ta), 64, &rVc, &rAc);
            TT aq = affine(c, an, as2_, as3_), vk = affine(c, vn, vs2, vs3);
            TT o2 = attention(c, v2a, as2(aq, kAD, Ta), as2(vk, kVD, N), 64, &rAc, &rVc);
            vx3 = ggml_add(c, vx3, ggml_mul(c, as3v(o1), vgate));
            ax3 = ggml_add(c, ax3, ggml_mul(c, as3a(o2), agate)); }
        // feed forward
        {   TT n = affine(c, ggml_rms_norm(c, vx3, 1e-6f), modRow(c, tV, iVada, 4, kVD), modRow(c, tV, iVada, 3, kVD));
            vx3 = ggml_add(c, vx3, ggml_mul(c, as3v(feedForward(c, fw1, fb1, fw2, fb2, as2(n, kVD, N))), modRow(c, tV, iVada, 5, kVD))); }
        {   TT n = affine(c, ggml_rms_norm(c, ax3, 1e-6f), modRow(c, tA, iAada, 4, kAD), modRow(c, tA, iAada, 3, kAD));
            ax3 = ggml_add(c, ax3, ggml_mul(c, as3a(feedForward(c, gw1, gb1, gw2, gb2, as2(n, kAD, Ta))), modRow(c, tA, iAada, 5, kAD))); }
        TT ov = ggml_cont(c, as2(vx3, kVD, N)), oa = ggml_cont(c, as2(ax3, kAD, Ta));
        ggml_set_output(ov); ggml_set_output(oa); ggml_build_forward_expand(G.g, ov); ggml_build_forward_expand(G.g, oa);
        if (!runGraph(G, err)) return false;
        setT(xv, vx); setT(xa, ax); setT(iVada, vMod); setT(iAada, aMod); setT(iVp, vPrompt); setT(iAp, aPrompt); setT(iVca, vCa); setT(iAca, aCa); setT(iGv, a2vGate); setT(iGa, v2aGate);
        setT(iVctx, in.vctx); setT(iActx, in.actx);
        setT(rV.cs, vrC); setT(rV.sn, vrS); setT(rA.cs, arC); setT(rA.sn, arS); setT(rVc.cs, vcC); setT(rVc.sn, vcS); setT(rAc.cs, acC); setT(rAc.sn, acS);
        if (!computeGraph(G, err)) return false;
        ggml_backend_tensor_get(ov, vx.data(), 0, vx.size() * 4); ggml_backend_tensor_get(oa, ax.data(), 0, ax.size() * 4);
        ggml_backend_sched_reset(Compute::get().sched);
        if (keepIt) { if (resident_.size() < (size_t)blocks) resident_.resize(blocks); resident_[b] = bw; }
    }

    // ---- output: LayerNorm (no affine) * (1 + scale) + shift, projection to 128 channels, unpatchify
    auto outStage = [&](const std::string& tabName, const std::string& projName, const std::vector<float>& x, const std::vector<float>& emb, int dim, int n, int per, std::vector<float>& out) -> bool {
        WeightSet W; TT t = tab(W, ck, P + tabName, err), w = W.mat(ck, P + projName + ".weight", err), b = W.vec(ck, P + projName + ".bias", err);
        if (!err.empty() || !W.upload(err)) return false;
        Graph G(64); ggml_context* c = G.c;
        TT xi = input2d(c, dim, n), ei = input2d(c, dim, (int64_t)emb.size() / dim);
        const int64_t F2 = ei->ne[1];
        TT shift = ggml_reshape_3d(c, ggml_add(c, ei, ggml_cont(c, ggml_view_2d(c, t, dim, 1, t->nb[1], 0))), dim, 1, F2);
        TT scale = ggml_reshape_3d(c, ggml_add(c, ei, ggml_cont(c, ggml_view_2d(c, t, dim, 1, t->nb[1], t->nb[1]))), dim, 1, F2);
        TT y = affine(c, ggml_norm(c, ggml_reshape_3d(c, xi, dim, per, F2), 1e-6f), scale, shift);
        y = ggml_add(c, mm(c, w, ggml_reshape_2d(c, y, dim, n)), b);
        ggml_set_output(y); ggml_build_forward_expand(G.g, y);
        if (!runGraph(G, err)) return false;
        setT(xi, x); setT(ei, emb); if (!computeGraph(G, err)) return false;
        out.resize((size_t)n * 128); ggml_backend_tensor_get(y, out.data(), 0, out.size() * 4); ggml_backend_sched_reset(Compute::get().sched); return true;
    };
    std::vector<float> vo, ao;
    if (!outStage("scale_shift_table", "proj_out", vx, vEmb, kVD, N, hw, vo) || !outStage("audio_scale_shift_table", "audio_proj_out", ax, aEmb, kAD, Ta, Ta, ao)) return false;
    videoOut.assign((size_t)128 * N, 0.f); audioOut.assign((size_t)8 * Ta * 16, 0.f);
    for (int c = 0; c < 128; c++) for (int f = 0; f < F; f++) for (int p = 0; p < hw; p++) videoOut[((size_t)c * F + f) * hw + p] = vo[((size_t)f * hw + p) * 128 + c];
    for (int c = 0; c < 8; c++) for (int t = 0; t < Ta; t++) for (int q = 0; q < 16; q++) audioOut[((size_t)c * Ta + t) * 16 + q] = ao[(size_t)t * 128 + c * 16 + q];
    return true;
}

// ---- self test against tools/ltx_ref_dit.py output
static bool readBin(const std::string& p, std::vector<float>& v) { std::string s; if (!sys::readFile(p, s)) return false; v.resize(s.size() / 4); memcpy(v.data(), s.data(), v.size() * 4); return true; }
static void compare(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) { printf("%-8s size mismatch: %zu vs reference %zu\n", what, a.size(), b.size()); return; }
    double md = 0, dot = 0, na = 0, nb = 0, mr = 0;
    for (size_t i = 0; i < a.size(); i++) { md = std::max(md, (double)std::fabs(a[i] - b[i])); dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; mr = std::max(mr, (double)std::fabs(b[i])); }
    printf("%-8s n=%zu max|diff|=%.5g (max|ref|=%.4g) cosine=%.6f\n", what, a.size(), md, mr, dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30));
}
int runLtxDitTest(const std::vector<std::string>& a) {
    if (a.size() < 2) { fprintf(stderr, "usage: ltx-dit <ltx checkpoint> <reference dir> [blocks F h w audioTokens]\n"); return 2; }
    LtxWeights w; std::string err; if (!w.open(a[0], err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    LtxLora lora; const float ls = a.size() > 7 ? (float)atof(a[7].c_str()) : 0.f;           // optional distilled LoRA strength (same file as the reference)
    if (ls > 0) { lora.strength = ls; if (!lora.open("c:\\workenv\\ComfyUI-Easy-Install\\ComfyUI-Easy-Install\\ComfyUI\\models\\loras\\ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors", err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; } w.lora = &lora; }
    w.cacheQ8 = true;
    LtxDit dit; dit.ckpt = &w; dit.blocks = a.size() > 2 ? atoi(a[2].c_str()) : 2;
    LtxDitInput in; in.F = a.size() > 3 ? atoi(a[3].c_str()) : 3; in.h = a.size() > 4 ? atoi(a[4].c_str()) : 4; in.w = a.size() > 5 ? atoi(a[5].c_str()) : 4; in.T = a.size() > 6 ? atoi(a[6].c_str()) : 10;
    std::vector<float> ctx, ro, rao;
    if (!readBin(a[1] + "\\dit_vx.bin", in.video) || !readBin(a[1] + "\\dit_ax.bin", in.audio) || !readBin(a[1] + "\\dit_ctx.bin", ctx) || !readBin(a[1] + "\\dit_vo.bin", ro) || !readBin(a[1] + "\\dit_ao.bin", rao)) { fprintf(stderr, "reference files missing (run tools/ltx_ref_dit.py)\n"); return 1; }
    in.Lv = in.La = (int)(ctx.size() / 6144);
    in.vctx.resize((size_t)in.Lv * 4096); in.actx.resize((size_t)in.La * 2048);
    for (int l = 0; l < in.Lv; l++) { memcpy(&in.vctx[(size_t)l * 4096], &ctx[(size_t)l * 6144], 4096 * 4); memcpy(&in.actx[(size_t)l * 2048], &ctx[(size_t)l * 6144 + 4096], 2048 * 4); }
    in.videoSigma = {0.7f}; in.audioSigma = 0.7f; in.frameRate = 24.f;
    std::vector<float> vo, ao;
    if (!dit.forward(in, vo, ao, [](int b) { fprintf(stderr, "  block %d\n", b); }, err)) { fprintf(stderr, "dit: %s\n", err.c_str()); return 1; }
    compare("video", vo, ro); compare("audio", ao, rao);
    return 0;
}






