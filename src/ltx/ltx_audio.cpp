#include "ltx/ltx_audio.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "ltx/ltx_ggml.h"
#include "util/sys.h"

using namespace ltxg;
namespace {
// ---------------------------------------------------------------- weights
TT w4(WeightSet& W, const LtxWeights& ck, const std::string& n, std::string& err) {                // conv2d [oc][ic][kh][kw] -> ggml (kw, kh, ic, oc), F16
    std::vector<int64_t> sh; std::vector<float> d;
    if (!ck.shapeOf(n, sh) || !ck.loadF32(n, d, err)) { if (err.empty()) err = "missing " + n; return nullptr; }
    TT t = ggml_new_tensor_4d(W.ctx, GGML_TYPE_F16, sh[3], sh[2], sh[1], sh[0]); W.items.push_back({t, std::move(d), true}); return t;
}
TT w3(WeightSet& W, const LtxWeights& ck, const std::string& n, bool transposed, std::string& err) {  // conv1d [oc][ic][k] -> (k, ic, oc); convT [ic][oc][k] -> (k, oc, ic)
    std::vector<int64_t> sh; std::vector<float> d;
    if (!ck.shapeOf(n, sh) || !ck.loadF32(n, d, err)) { if (err.empty()) err = "missing " + n; return nullptr; }
    TT t = ggml_new_tensor_3d(W.ctx, GGML_TYPE_F16, sh[2], sh[1], sh[0]); (void)transposed; W.items.push_back({t, std::move(d), true}); return t;
}
TT col(WeightSet& W, std::vector<float> d) { TT t = ggml_new_tensor_2d(W.ctx, GGML_TYPE_F32, 1, (int64_t)d.size()); W.items.push_back({t, std::move(d), false}); return t; }   // per-channel [1, C]
TT filt(WeightSet& W, const LtxWeights& ck, const std::string& n, bool reverse, std::string& err) {  // 12-tap filter [1,1,12] -> F16 [12,1,1]
    std::vector<float> d; if (!ck.loadF32(n, d, err)) return nullptr;
    if (reverse) std::reverse(d.begin(), d.end());
    TT t = ggml_new_tensor_3d(W.ctx, GGML_TYPE_F16, (int64_t)d.size(), 1, 1); W.items.push_back({t, std::move(d), true}); return t;
}

// ---------------------------------------------------------------- audio VAE decoder (2D: W = frequency, H = time, C = channels)
TT pixNorm6(ggml_context* c, TT x) {
    TT p = ggml_cont(c, ggml_permute(c, x, 1, 2, 0, 3));
    p = ggml_rms_norm(c, p, 1e-6f);
    return ggml_cont(c, ggml_permute(c, p, 2, 0, 1, 3));
}
struct C2 { TT w, b; };
bool loadC2(WeightSet& W, const LtxWeights& ck, const std::string& p, C2& cv, std::string& err) { cv.w = w4(W, ck, p + "conv.weight", err); cv.b = W.vec(ck, p + "conv.bias", err); return err.empty(); }
TT causalConv(ggml_context* c, const C2& cv, TT x) {                 // 3x3 (or 1x1) conv, two zero rows in front on the time axis, one zero column each side on the frequency axis
    const int kh = (int)cv.w->ne[1];
    TT xp = x;
    if (kh == 3) { TT z = ggml_scale(c, ggml_view_4d(c, x, x->ne[0], 1, x->ne[2], 1, x->nb[1], x->nb[2], x->nb[3], 0), 0.0f); xp = ggml_concat(c, z, ggml_concat(c, z, x, 1), 1); }
    TT y = ggml_conv_2d(c, cv.w, xp, 1, 1, kh == 3 ? 1 : 0, 0, 1, 1);
    return ggml_add(c, y, ggml_reshape_4d(c, cv.b, 1, 1, cv.b->ne[0], 1));
}
struct R2 { C2 c1, c2, sc; bool hasSc = false; };
bool loadR2(WeightSet& W, const LtxWeights& ck, const std::string& p, R2& r, std::string& err) {
    if (!loadC2(W, ck, p + "conv1.", r.c1, err) || !loadC2(W, ck, p + "conv2.", r.c2, err)) return false;
    if (ck.has(p + "nin_shortcut.conv.weight")) { r.hasSc = true; return loadC2(W, ck, p + "nin_shortcut.", r.sc, err); }
    return true;
}
TT res2(ggml_context* c, const R2& r, TT x) {
    TT h = causalConv(c, r.c1, ggml_silu(c, pixNorm6(c, x)));
    h = causalConv(c, r.c2, ggml_silu(c, pixNorm6(c, h)));
    return ggml_add(c, r.hasSc ? causalConv(c, r.sc, x) : x, h);
}

// ---------------------------------------------------------------- vocoder (1D: ne0 = time, ne1 = channels)
TT repPad(ggml_context* c, TT x, int l, int r) {
    const int64_t L = x->ne[0], C = x->ne[1];
    TT out = x;
    if (l) { TT first = ggml_view_2d(c, x, 1, C, x->nb[1], 0); out = ggml_concat(c, ggml_repeat(c, first, ggml_new_tensor_2d(c, GGML_TYPE_F32, l, C)), out, 0); }
    if (r) { TT last = ggml_view_2d(c, x, 1, C, x->nb[1], (size_t)(L - 1) * 4); out = ggml_concat(c, out, ggml_repeat(c, last, ggml_new_tensor_2d(c, GGML_TYPE_F32, r, C)), 0); }
    return out;
}
TT fir(ggml_context* c, TT x, TT f, int stride) {                    // the same 12-tap filter on every channel (channels as the batch of a 1-channel conv)
    TT x3 = ggml_reshape_3d(c, ggml_cont(c, x), x->ne[0], 1, x->ne[1]);
    TT y = ggml_conv_1d(c, f, x3, stride, 0, 1);
    return ggml_reshape_2d(c, y, y->ne[0], x->ne[1]);
}
struct Act { TT ia, ib, upF, dnF; };
bool loadAct(WeightSet& W, const LtxWeights& ck, const std::string& p, Act& a, std::string& err) {
    std::vector<float> al, be;
    if (!ck.loadF32(p + "act.alpha", al, err) || !ck.loadF32(p + "act.beta", be, err)) return false;
    for (auto& v : al) v = std::exp(v);                              // alpha, beta are stored in log scale
    for (auto& v : be) v = 1.0f / (std::exp(v) + 1e-9f);
    a.ia = col(W, al); a.ib = col(W, be);
    a.upF = filt(W, ck, p + "upsample.filter", true, err); a.dnF = filt(W, ck, p + "downsample.lowpass.filter", false, err);
    return err.empty();
}
TT act1d(ggml_context* c, const Act& a, TT x) {                      // upsample x2 -> SnakeBeta -> low-pass downsample x2
    const int64_t L = x->ne[0], C = x->ne[1];
    TT xp = repPad(c, x, 5, 5);
    TT x3 = ggml_reshape_3d(c, xp, L + 10, 1, C);
    TT z = ggml_concat(c, x3, ggml_scale(c, x3, 0.0f), 1);                                           // [L+10, 2, C]: zero stuffing
    TT s = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, z, 1, 0, 2, 3)), 2 * (L + 10), C);
    TT sv = ggml_cont(c, ggml_view_2d(c, s, 2 * L + 11, C, s->nb[1], 16));
    TT u = ggml_scale(c, fir(c, sv, a.upF, 1), 2.0f);                                                // [2L, C]
    u = ggml_add(c, u, ggml_mul(c, ggml_sqr(c, ggml_sin(c, ggml_mul(c, u, a.ia))), a.ib));
    return fir(c, repPad(c, u, 5, 6), a.dnF, 2);                                                      // [L, C]
}
struct Cv1 { TT w, b; int pad, dil; };
TT conv1(ggml_context* c, const Cv1& cv, TT x) {
    TT y = ggml_conv_1d(c, cv.w, x, 1, cv.pad, cv.dil);
    return cv.b ? ggml_add(c, y, ggml_reshape_2d(c, cv.b, 1, cv.b->ne[0])) : y;
}
struct Amp { Cv1 c1[3], c2[3]; Act a1[3], a2[3]; };
bool loadAmp(WeightSet& W, const LtxWeights& ck, const std::string& p, int k, const int dil[3], Amp& b, std::string& err) {
    for (int i = 0; i < 3; i++) {
        const std::string s = std::to_string(i);
        b.c1[i] = {w3(W, ck, p + "convs1." + s + ".weight", false, err), W.vec(ck, p + "convs1." + s + ".bias", err), (k * dil[i] - dil[i]) / 2, dil[i]};
        b.c2[i] = {w3(W, ck, p + "convs2." + s + ".weight", false, err), W.vec(ck, p + "convs2." + s + ".bias", err), (k - 1) / 2, 1};
        if (!loadAct(W, ck, p + "acts1." + s + ".", b.a1[i], err) || !loadAct(W, ck, p + "acts2." + s + ".", b.a2[i], err)) return false;
    }
    return err.empty();
}
TT amp(ggml_context* c, const Amp& b, TT x) {
    for (int i = 0; i < 3; i++) { TT t = conv1(c, b.c1[i], act1d(c, b.a1[i], x)); t = conv1(c, b.c2[i], act1d(c, b.a2[i], t)); x = ggml_add(c, x, t); }
    return x;
}

struct Sig { int C = 0, L = 0; std::vector<float> d; };              // [C][L]

template <class F> bool runSig(const Sig& in, Sig& out, F build, size_t nodes, std::string& err) {
    Graph G(nodes); ggml_context* c = G.c;
    TT x = ggml_new_tensor_2d(c, GGML_TYPE_F32, in.L, in.C); ggml_set_input(x);
    TT y = build(c, x); ggml_set_output(y); ggml_build_forward_expand(G.g, y);
    if (!runGraph(G, err)) return false;
    ggml_backend_tensor_set(x, in.d.data(), 0, in.d.size() * 4);
    if (!computeGraph(G, err)) return false;
    out.L = (int)y->ne[0]; out.C = (int)y->ne[1]; out.d.resize((size_t)out.L * out.C); ggml_backend_tensor_get(y, out.d.data(), 0, out.d.size() * 4);
    ggml_backend_sched_reset(Compute::get().sched);
    return true;
}

struct VocCfg { int initCh; std::vector<int> rates, kernels; bool clampFinal; };
bool vocoder(const LtxWeights& ck, const std::string& P, const VocCfg& cfg, const Sig& in, Sig& out, const std::function<void(const std::string&)>& say, std::string& err) {
    Sig x;
    {   WeightSet W; Cv1 cp{w3(W, ck, P + "conv_pre.weight", false, err), W.vec(ck, P + "conv_pre.bias", err), 3, 1};
        if (!err.empty() || !W.upload(err)) return false;
        if (!runSig(in, x, [&](ggml_context* c, TT t) { return conv1(c, cp, t); }, 256, err)) return false; }
    const int ks[3] = {3, 7, 11}; const int dil[3] = {1, 3, 5};
    for (size_t i = 0; i < cfg.rates.size(); i++) {
        if (say) say("upsample stage " + std::to_string(i + 1) + "/" + std::to_string(cfg.rates.size()));
        const int u = cfg.rates[i], k = cfg.kernels[i], ch = cfg.initCh >> (i + 1);
        WeightSet W; const std::string s = P + "ups." + std::to_string(i) + ".";
        TT uw = w3(W, ck, s + "weight", true, err), ub = W.vec(ck, s + "bias", err);
        Amp bl[3];
        for (int j = 0; j < 3; j++) if (!loadAmp(W, ck, P + "resblocks." + std::to_string(i * 3 + j) + ".", ks[j], dil, bl[j], err)) return false;
        if (!err.empty() || !W.upload(err)) return false;
        Sig y; if (!runSig(x, y, [&](ggml_context* c, TT t) {
                TT up = ggml_conv_transpose_1d(c, uw, t, u, 0, 1);
                const int pad = (k - u) / 2; const int64_t Lo = up->ne[0] - 2 * pad;
                up = ggml_cont(c, ggml_view_2d(c, up, Lo, up->ne[1], up->nb[1], (size_t)pad * 4));
                up = ggml_add(c, up, ggml_reshape_2d(c, ub, 1, ub->ne[0]));
                TT sum = nullptr; for (int j = 0; j < 3; j++) { TT r = amp(c, bl[j], up); sum = sum ? ggml_add(c, sum, r) : r; }
                (void)ch; return ggml_scale(c, sum, 1.0f / 3.0f); }, 16384, err)) return false;
        x = std::move(y);
    }
    {   if (say) say("output stage");
        WeightSet W; Act ap; if (!loadAct(W, ck, P + "act_post.", ap, err)) return false;
        Cv1 cq{w3(W, ck, P + "conv_post.weight", false, err), nullptr, 3, 1};
        if (!err.empty() || !W.upload(err)) return false;
        if (!runSig(x, out, [&](ggml_context* c, TT t) { TT y = conv1(c, cq, act1d(c, ap, t)); return cfg.clampFinal ? ggml_clamp(c, y, -1.f, 1.f) : y; }, 512, err)) return false; }
    return true;
}
}  // namespace

bool LtxAudio::decode(const std::vector<float>& latent, int T, std::vector<float>& wave, int& samples, const std::function<void(const std::string&)>& say, std::string& err) {
    if (!Compute::get().init(0, err)) return false;
    const LtxWeights& ck = *ckpt;
    std::vector<float> mean, stdv;
    if (!ck.loadF32("audio_vae.per_channel_statistics.mean-of-means", mean, err) || !ck.loadF32("audio_vae.per_channel_statistics.std-of-means", stdv, err)) return false;
    // ---- latent [8][T][16] (de-normalized in patch space: feature = channel * 16 + bin) -> mel [2][Tm][64]
    std::vector<float> z(latent.size());
    for (int c = 0; c < 8; c++) for (int t = 0; t < T; t++) for (int f = 0; f < 16; f++) { const size_t i = ((size_t)c * T + t) * 16 + f; z[i] = latent[i] * stdv[c * 16 + f] + mean[c * 16 + f]; }
    if (say) say("audio decoder");
    const int Tm = T * 4 - 3;
    std::vector<float> mel;
    {   const std::string D = "audio_vae.decoder.";
        WeightSet W; C2 cin, cout_; R2 mid1, mid2; std::vector<R2> blocks[3]; C2 upc[3];
        if (!loadC2(W, ck, D + "conv_in.", cin, err) || !loadR2(W, ck, D + "mid.block_1.", mid1, err) || !loadR2(W, ck, D + "mid.block_2.", mid2, err) || !loadC2(W, ck, D + "conv_out.", cout_, err)) return false;
        for (int lv = 0; lv < 3; lv++) { blocks[lv].resize(3); for (int b = 0; b < 3; b++) if (!loadR2(W, ck, D + "up." + std::to_string(lv) + ".block." + std::to_string(b) + ".", blocks[lv][b], err)) return false;
            if (lv > 0 && !loadC2(W, ck, D + "up." + std::to_string(lv) + ".upsample.conv.", upc[lv], err)) return false; }
        if (!W.upload(err)) return false;
        Graph G(4096); ggml_context* c = G.c;
        TT x = ggml_new_tensor_4d(c, GGML_TYPE_F32, 16, T, 8, 1); ggml_set_input(x);
        TT h = causalConv(c, cin, x);
        h = res2(c, mid2, res2(c, mid1, h));
        for (int lv = 2; lv >= 0; lv--) {
            for (int b = 0; b < 3; b++) h = res2(c, blocks[lv][b], h);
            if (lv > 0) {
                h = causalConv(c, upc[lv], ggml_upscale(c, h, 2, GGML_SCALE_MODE_NEAREST));
                h = ggml_cont(c, ggml_view_4d(c, h, h->ne[0], h->ne[1] - 1, h->ne[2], 1, h->nb[1], h->nb[2], h->nb[3], h->nb[1]));      // drop the first time row
            }
        }
        h = causalConv(c, cout_, ggml_silu(c, pixNorm6(c, h)));
        ggml_set_output(h); ggml_build_forward_expand(G.g, h);
        if (!runGraph(G, err)) return false;
        ggml_backend_tensor_set(x, z.data(), 0, z.size() * 4);
        if (!computeGraph(G, err)) return false;
        if ((int)h->ne[1] != Tm || h->ne[0] != 64) { err = "unexpected mel size " + std::to_string(h->ne[0]) + "x" + std::to_string(h->ne[1]); return false; }
        mel.resize((size_t)2 * Tm * 64); ggml_backend_tensor_get(h, mel.data(), 0, mel.size() * 4);
        ggml_backend_sched_reset(Compute::get().sched);
    }
    lastMel = mel;
    // ---- vocoder: channels = ch * 64 + bin over time
    Sig in; in.C = 128; in.L = Tm; in.d.resize((size_t)128 * Tm);
    for (int ch = 0; ch < 2; ch++) for (int t = 0; t < Tm; t++) for (int f = 0; f < 64; f++) in.d[((size_t)ch * 64 + f) * Tm + t] = mel[((size_t)ch * Tm + t) * 64 + f];
    Sig low;
    if (!vocoder(ck, "vocoder.vocoder.", {1536, {5, 2, 2, 2, 2, 2}, {11, 4, 4, 4, 4, 4}, true}, in, low, say, err)) return false;
    const int Tlow = low.L;                                           // 16 kHz samples per channel
    // ---- bandwidth extension: causal STFT -> log-mel of the 16 kHz signal
    if (say) say("bandwidth extension");
    const int hop = 80, nfft = 512, nfreq = 257, Lpad = (Tlow + hop - 1) / hop * hop, frames = Lpad / hop;
    std::vector<float> fb, mb; if (!ck.loadF32("vocoder.mel_stft.stft_fn.forward_basis", fb, err) || !ck.loadF32("vocoder.mel_stft.mel_basis", mb, err)) return false;
    Sig mel2; mel2.C = 128; mel2.L = frames; mel2.d.assign((size_t)128 * frames, 0.f);
    {   std::vector<float> mag((size_t)nfreq * frames);
        for (int ch = 0; ch < 2; ch++) {
            std::vector<float> sig((size_t)(nfft - hop) + Lpad, 0.f);        // causal left padding of win - hop zeros
            for (int i = 0; i < Tlow; i++) sig[(size_t)(nfft - hop) + i] = low.d[(size_t)ch * Tlow + i];
            for (int k = 0; k < frames; k++) for (int f = 0; f < nfreq; f++) {
                const float *re = &fb[(size_t)f * nfft], *im = &fb[(size_t)(f + nfreq) * nfft], *s = &sig[(size_t)k * hop];
                float a = 0, b = 0; for (int j = 0; j < nfft; j++) { a += re[j] * s[j]; b += im[j] * s[j]; }
                mag[(size_t)f * frames + k] = std::sqrt(a * a + b * b);
            }
            for (int m = 0; m < 64; m++) for (int k = 0; k < frames; k++) {
                float v = 0; for (int f = 0; f < nfreq; f++) v += mb[(size_t)m * nfreq + f] * mag[(size_t)f * frames + k];
                mel2.d[((size_t)ch * 64 + m) * frames + k] = std::log(std::max(v, 1e-5f));
            }
        }
    }
    Sig res;
    if (!vocoder(ck, "vocoder.bwe_generator.", {512, {6, 5, 2, 2, 2}, {12, 11, 4, 4, 4}, false}, mel2, res, say, err)) return false;
    // ---- skip connection: x3 resampler (hann-windowed sinc, replicate padding), then clamp(residual + skip)
    const int ratio = 3, width = 7, K = 2 * width * ratio + 1, padL = 2 * width * ratio, padR = K - ratio;
    std::vector<float> f(K);
    for (int i = 0; i < K; i++) {
        const double t = ((double)i / ratio - width) * 0.99, tc = std::max(-6.0, std::min(6.0, t));
        const double sinc = t == 0 ? 1.0 : std::sin(3.14159265358979323846 * t) / (3.14159265358979323846 * t), w = std::pow(std::cos(tc * 3.14159265358979323846 / 6.0 / 2.0), 2.0);
        f[i] = (float)(sinc * w * 0.99 / ratio);
    }
    samples = Tlow * ratio;
    wave.assign((size_t)2 * samples, 0.f);
    for (int ch = 0; ch < 2; ch++) {
        std::vector<float> xp((size_t)Tlow + 2 * width);
        for (int i = 0; i < (int)xp.size(); i++) xp[i] = low.d[(size_t)ch * Tlow + std::min(std::max(i - width, 0), Tlow - 1)];
        const int full = ((int)xp.size() - 1) * ratio + K;
        std::vector<float> y(full, 0.f);
        for (int n = 0; n < (int)xp.size(); n++) for (int j = 0; j < K; j++) y[(size_t)n * ratio + j] += xp[n] * f[j];
        for (int i = 0; i < samples; i++) {
            const float skip = (float)ratio * y[(size_t)padL + i], r = res.d[(size_t)ch * res.L + i];
            wave[(size_t)ch * samples + i] = std::max(-1.f, std::min(1.f, r + skip));
        }
        (void)padR;
    }
    return true;
}

// ---- self test against tools/ltx_ref_audio.py
static bool readBin(const std::string& p, std::vector<float>& v) { std::string s; if (!sys::readFile(p, s)) return false; v.resize(s.size() / 4); memcpy(v.data(), s.data(), v.size() * 4); return true; }
static void compare(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) { printf("%-6s size mismatch: %zu vs reference %zu\n", what, a.size(), b.size()); return; }
    double md = 0, dot = 0, na = 0, nb = 0, mr = 0;
    for (size_t i = 0; i < a.size(); i++) { md = std::max(md, (double)std::fabs(a[i] - b[i])); dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; mr = std::max(mr, (double)std::fabs(b[i])); }
    printf("%-6s n=%zu max|diff|=%.5g (max|ref|=%.4g) cosine=%.6f\n", what, a.size(), md, mr, dot / (std::sqrt(na * nb) + 1e-30));
}
int runLtxAudioTest(const std::vector<std::string>& a) {
    if (a.size() < 2) { fprintf(stderr, "usage: ltx-audio <ltx checkpoint> <reference dir> [T]\n"); return 2; }
    LtxWeights w; std::string err; if (!w.open(a[0], err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const int T = a.size() > 2 ? atoi(a[2].c_str()) : 12;
    std::vector<float> z, mel, wav; if (!readBin(a[1] + "\\au_z.bin", z) || !readBin(a[1] + "\\au_mel.bin", mel) || !readBin(a[1] + "\\au_wav.bin", wav)) { fprintf(stderr, "reference missing\n"); return 1; }
    LtxAudio au; au.ckpt = &w; std::vector<float> out; int n = 0;
    if (!au.decode(z, T, out, n, [](const std::string& s) { fprintf(stderr, "  %s\n", s.c_str()); }, err)) { fprintf(stderr, "audio: %s\n", err.c_str()); return 1; }
    compare("mel", au.lastMel, mel); compare("wave", out, wav);
    return 0;
}
