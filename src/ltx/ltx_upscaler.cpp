#include "ltx/ltx_upscaler.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include "ltx/ltx_ggml.h"
#include "util/sys.h"

using namespace ltxg;
namespace {
// GroupNorm(32) over channels AND all frames: fold the frames into the height so ggml's per-sample group norm sees them
TT groupNorm(ggml_context* c, TT x, TT w, TT b) {
    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], T = x->ne[3];
    TT p = ggml_reshape_4d(c, ggml_cont(c, ggml_permute(c, x, 0, 1, 3, 2)), W, H * T, C, 1);            // [W, H*T, C, 1]
    p = ggml_group_norm(c, p, 32, 1e-5f);
    TT y = ggml_cont(c, ggml_permute(c, ggml_reshape_4d(c, p, W, H, T, C), 0, 1, 3, 2));                // back to [W, H, C, T]
    return ggml_add(c, ggml_mul(c, y, ggml_reshape_4d(c, w, 1, 1, C, 1)), ggml_reshape_4d(c, b, 1, 1, C, 1));
}
struct Res { ConvW c1, c2; TT n1w, n1b, n2w, n2b; };
bool loadRes(WeightSet& W, const LtxWeights& m, const std::string& p, Res& r, std::string& err) {
    if (!loadConv(W, m, p + "conv1.", r.c1, err) || !loadConv(W, m, p + "conv2.", r.c2, err)) return false;
    r.n1w = W.vec(m, p + "norm1.weight", err); r.n1b = W.vec(m, p + "norm1.bias", err); r.n2w = W.vec(m, p + "norm2.weight", err); r.n2b = W.vec(m, p + "norm2.bias", err);
    return err.empty();
}
TT resBlock(ggml_context* c, const Res& r, TT x) {
    TT h = ggml_silu(c, groupNorm(c, conv3(c, r.c1, x, 1, 1, true), r.n1w, r.n1b));
    h = groupNorm(c, conv3(c, r.c2, h, 1, 1, true), r.n2w, r.n2b);
    return ggml_silu(c, ggml_add(c, h, x));
}
template <class F> bool runVol(const Vol& in, Vol& out, F build, std::string& err) {
    Graph G(8192); ggml_context* c = G.c;
    TT x = ggml_new_tensor_4d(c, GGML_TYPE_F32, in.W, in.H, in.C, in.T); ggml_set_input(x);
    TT y = build(c, x); ggml_set_output(y); ggml_build_forward_expand(G.g, y);
    if (!runGraph(G, err)) return false;
    ggml_backend_tensor_set(x, in.d.data(), 0, in.d.size() * 4);
    if (!computeGraph(G, err)) return false;
    out.T = (int)y->ne[3]; out.C = (int)y->ne[2]; out.H = (int)y->ne[1]; out.W = (int)y->ne[0];
    out.d.resize((size_t)out.T * out.C * out.H * out.W); ggml_backend_tensor_get(y, out.d.data(), 0, out.d.size() * 4);
    ggml_backend_sched_reset(Compute::get().sched);
    return true;
}
}  // namespace

bool LtxUpscaler::run(const Vol& latent, Vol& out, std::string& err) {
    if (!Compute::get().init(0, err)) return false;
    const LtxWeights& m = *model;
    std::vector<float> mean, stdv;
    if (!ckpt->loadF32("vae.per_channel_statistics.mean-of-means", mean, err) || !ckpt->loadF32("vae.per_channel_statistics.std-of-means", stdv, err)) return false;
    Vol x = latent;
    for (int t = 0; t < x.T; t++) for (int c = 0; c < x.C; c++) for (int h = 0; h < x.H; h++) for (int w = 0; w < x.W; w++) x.at(t, c, h, w) = x.at(t, c, h, w) * stdv[c] + mean[c];
    {   WeightSet W; ConvW ic; TT nw = nullptr, nb = nullptr; std::vector<Res> rs(4);
        if (!loadConv(W, m, "initial_conv.", ic, err)) return false;
        nw = W.vec(m, "initial_norm.weight", err); nb = W.vec(m, "initial_norm.bias", err);
        for (int i = 0; i < 4; i++) if (!loadRes(W, m, "res_blocks." + std::to_string(i) + ".", rs[i], err)) return false;
        if (!err.empty() || !W.upload(err)) return false;
        Vol y; if (!runVol(x, y, [&](ggml_context* c, TT in) { TT h = ggml_silu(c, groupNorm(c, conv3(c, ic, in, 1, 1, true), nw, nb)); for (auto& r : rs) h = resBlock(c, r, h); return h; }, err)) return false;
        x = std::move(y); }
    {   // per-frame Conv2d(1024 -> 4096) then a 2x2 pixel shuffle
        WeightSet W; std::vector<int64_t> sh; std::vector<float> d;
        if (!m.shapeOf("upsampler.0.weight", sh) || !m.loadF32("upsampler.0.weight", d, err)) return false;
        TT k = ggml_new_tensor_4d(W.ctx, GGML_TYPE_F16, 3, 3, sh[1], sh[0]); W.items.push_back({k, std::move(d), true});
        TT bias = W.vec(m, "upsampler.0.bias", err);
        if (!err.empty() || !W.upload(err)) return false;
        Vol y; if (!runVol(x, y, [&](ggml_context* c, TT in) { return ggml_add(c, ggml_conv_2d(c, k, in, 1, 1, 1, 1, 1, 1), ggml_reshape_4d(c, bias, 1, 1, (int64_t)sh[0], 1)); }, err)) return false;
        Vol s; s.alloc(y.T, y.C / 4, y.H * 2, y.W * 2);
        for (int t = 0; t < s.T; t++) for (int c = 0; c < s.C; c++) for (int h = 0; h < s.H; h++) for (int w = 0; w < s.W; w++) s.at(t, c, h, w) = y.at(t, (c * 2 + h % 2) * 2 + w % 2, h / 2, w / 2);
        x = std::move(s); }
    {   WeightSet W; ConvW fc; std::vector<Res> rs(4);
        for (int i = 0; i < 4; i++) if (!loadRes(W, m, "post_upsample_res_blocks." + std::to_string(i) + ".", rs[i], err)) return false;
        if (!loadConv(W, m, "final_conv.", fc, err) || !W.upload(err)) return false;
        Vol y; if (!runVol(x, y, [&](ggml_context* c, TT in) { TT h = in; for (auto& r : rs) h = resBlock(c, r, h); return conv3(c, fc, h, 1, 1, true); }, err)) return false;
        x = std::move(y); }
    for (int t = 0; t < x.T; t++) for (int c = 0; c < x.C; c++) for (int h = 0; h < x.H; h++) for (int w = 0; w < x.W; w++) x.at(t, c, h, w) = (x.at(t, c, h, w) - mean[c]) / stdv[c];
    out = std::move(x);
    return true;
}

// ---- self test against tools/ltx_ref_up.py
static bool readBin(const std::string& p, std::vector<float>& v) { std::string s; if (!sys::readFile(p, s)) return false; v.resize(s.size() / 4); memcpy(v.data(), s.data(), v.size() * 4); return true; }
static std::vector<float> swapAB(const std::vector<float>& a, int A, int B, size_t inner) {
    std::vector<float> o(a.size());
    for (int i = 0; i < A; i++) for (int j = 0; j < B; j++) memcpy(&o[((size_t)j * A + i) * inner], &a[((size_t)i * B + j) * inner], inner * 4);
    return o;
}
int runLtxUpscalerTest(const std::vector<std::string>& a) {
    if (a.size() < 3) { fprintf(stderr, "usage: ltx-up <upscaler> <ltx checkpoint> <reference dir> [T h w]\n"); return 2; }
    LtxWeights up, ck; std::string err;
    if (!up.open(a[0], err) || !ck.open(a[1], err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const int T = a.size() > 5 ? atoi(a[3].c_str()) : 3, h = a.size() > 5 ? atoi(a[4].c_str()) : 4, w = a.size() > 5 ? atoi(a[5].c_str()) : 4;
    std::vector<float> z, y; if (!readBin(a[2] + "\\up_z.bin", z) || !readBin(a[2] + "\\up_y.bin", y)) { fprintf(stderr, "reference missing\n"); return 1; }
    Vol in; in.T = T; in.C = 128; in.H = h; in.W = w; in.d = swapAB(z, 128, T, (size_t)h * w);
    LtxUpscaler u; u.model = &up; u.ckpt = &ck; Vol out;
    if (!u.run(in, out, err)) { fprintf(stderr, "upscaler: %s\n", err.c_str()); return 1; }
    const std::vector<float> mine = swapAB(out.d, out.T, 128, (size_t)out.H * out.W);
    if (mine.size() != y.size()) { printf("size mismatch %zu vs %zu\n", mine.size(), y.size()); return 1; }
    double md = 0, dot = 0, na = 0, nb = 0, mr = 0;
    for (size_t i = 0; i < y.size(); i++) { md = std::max(md, (double)std::fabs(mine[i] - y[i])); dot += (double)mine[i] * y[i]; na += (double)mine[i] * mine[i]; nb += (double)y[i] * y[i]; mr = std::max(mr, (double)std::fabs(y[i])); }
    printf("upscaler n=%zu max|diff|=%.5g (max|ref|=%.4g) cosine=%.6f\n", y.size(), md, mr, dot / (std::sqrt(na * nb) + 1e-30));
    return 0;
}
