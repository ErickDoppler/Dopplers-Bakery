#include "ltx/ltx_vae.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include "ltx/ltx_ggml.h"
#include "util/sys.h"

using namespace ltxg;
namespace {
struct Res { ConvW c1, c2; };
bool loadRes(WeightSet& W, const LtxWeights& ck, const std::string& p, Res& r, std::string& err) {
    return loadConv(W, ck, p + "conv1.conv.", r.c1, err) && loadConv(W, ck, p + "conv2.conv.", r.c2, err);
}
TT resBlock(ggml_context* c, const Res& r, TT x, int padF, int padB) {
    TT h = conv3(c, r.c1, ggml_silu(c, pixelNorm(c, x)), padF, padB);
    h = conv3(c, r.c2, ggml_silu(c, pixelNorm(c, h)), padF, padB);
    return ggml_add(c, h, x);
}

// ---- host-side rearranges
Vol depthToSpace(const Vol& y, int p1, int p2, int p3, bool dropFirst) {
    const int C = y.C / (p1 * p2 * p3), T = y.T * p1;
    Vol o; o.alloc(T - (dropFirst ? 1 : 0), C, y.H * p2, y.W * p3);
    for (int t = 0; t < o.T; t++) {
        const int st = (t + (dropFirst ? 1 : 0)) / p1, i1 = (t + (dropFirst ? 1 : 0)) % p1;
        for (int c = 0; c < C; c++) for (int h = 0; h < o.H; h++) for (int w = 0; w < o.W; w++)
            o.at(t, c, h, w) = y.at(st, ((c * p1 + i1) * p2 + h % p2) * p3 + w % p3, h / p2, w / p3);
    }
    return o;
}
Vol spaceToDepth(const Vol& x, int p1, int p2, int p3) {        // "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w"
    Vol o; o.alloc(x.T / p1, x.C * p1 * p2 * p3, x.H / p2, x.W / p3);
    for (int t = 0; t < x.T; t++) for (int c = 0; c < x.C; c++) for (int h = 0; h < x.H; h++) for (int w = 0; w < x.W; w++)
        o.at(t / p1, ((c * p1 + t % p1) * p2 + h % p2) * p3 + w % p3, h / p2, w / p3) = x.at(t, c, h, w);
    return o;
}
Vol unpatchify(const Vol& y, int q) {                           // "b (c r q) f h w -> b c f (h q) (w r)"
    Vol o; o.alloc(y.T, y.C / (q * q), y.H * q, y.W * q);
    for (int t = 0; t < o.T; t++) for (int c = 0; c < o.C; c++) for (int h = 0; h < o.H; h++) for (int w = 0; w < o.W; w++)
        o.at(t, c, h, w) = y.at(t, (c * q + w % q) * q + h % q, h / q, w / q);
    return o;
}
Vol patchify(const Vol& x, int q) {                             // "b c (h q) (w r) -> b (c r q) h w"
    Vol o; o.alloc(x.T, x.C * q * q, x.H / q, x.W / q);
    for (int t = 0; t < x.T; t++) for (int c = 0; c < x.C; c++) for (int h = 0; h < x.H; h++) for (int w = 0; w < x.W; w++)
        o.at(t, (c * q + w % q) * q + h % q, h / q, w / q) = x.at(t, c, h, w);
    return o;
}

// run `build` (x: input volume as a [W,H,C,T] tensor) as one graph and return the result volume
template <class F> bool runVol(const Vol& in, Vol& out, int outC, int outT, F build, std::string& err) {
    Graph G(16384); ggml_context* c = G.c;
    TT x = ggml_new_tensor_4d(c, GGML_TYPE_F32, in.W, in.H, in.C, in.T); ggml_set_input(x);
    TT y = build(c, x);
    ggml_set_output(y); ggml_build_forward_expand(G.g, y);
    if (!runGraph(G, err)) return false;
    ggml_backend_tensor_set(x, in.d.data(), 0, in.d.size() * 4);
    if (!computeGraph(G, err)) return false;
    out.T = (int)y->ne[3]; out.C = (int)y->ne[2]; out.H = (int)y->ne[1]; out.W = (int)y->ne[0];
    (void)outC; (void)outT;
    out.d.resize((size_t)out.T * out.C * out.H * out.W);
    ggml_backend_tensor_get(y, out.d.data(), 0, out.d.size() * 4);
    ggml_backend_sched_reset(Compute::get().sched);
    return true;
}
int countRes(const LtxWeights& ck, const std::string& p) { int n = 0; while (ck.has(p + "res_blocks." + std::to_string(n) + ".conv1.conv.weight")) n++; return n; }
}  // namespace

bool LtxVae::decode(const Vol& latent, Vol& frames, const std::function<void(const std::string&)>& progress, std::string& err) {
    if (!Compute::get().init(0, err)) return false;
    std::vector<float> mean, stdv;
    if (!ckpt->loadF32("vae.per_channel_statistics.mean-of-means", mean, err) || !ckpt->loadF32("vae.per_channel_statistics.std-of-means", stdv, err)) return false;
    Vol x = latent;
    for (int t = 0; t < x.T; t++) for (int c = 0; c < x.C; c++) for (int h = 0; h < x.H; h++) for (int w = 0; w < x.W; w++) x.at(t, c, h, w) = x.at(t, c, h, w) * stdv[c] + mean[c];
    const int F = 1, B = 1;                                           // non-causal decoder: replicate one frame at each end
    auto say = [&](const std::string& s) { if (progress) progress(s); };
    {   say("conv_in");
        WeightSet W; ConvW cw; if (!loadConv(W, *ckpt, "vae.decoder.conv_in.conv.", cw, err) || !W.upload(err)) return false;
        Vol y; if (!runVol(x, y, 0, 0, [&](ggml_context* c, TT in) { return conv3(c, cw, in, F, B); }, err)) return false; x = std::move(y);
    }
    struct Up { int p1, p2, p3; };
    const Up ups[9] = {{0,0,0}, {2,2,2}, {0,0,0}, {2,2,2}, {0,0,0}, {2,1,1}, {0,0,0}, {1,2,2}, {0,0,0}};
    for (int i = 0; i < 9; i++) {
        const std::string p = "vae.decoder.up_blocks." + std::to_string(i) + ".";
        say("up block " + std::to_string(i));
        WeightSet W;
        if (ups[i].p1 == 0) {
            std::vector<Res> rs((size_t)countRes(*ckpt, p));
            for (size_t k = 0; k < rs.size(); k++) if (!loadRes(W, *ckpt, p + "res_blocks." + std::to_string(k) + ".", rs[k], err)) return false;
            if (!W.upload(err)) return false;
            Vol y; if (!runVol(x, y, 0, 0, [&](ggml_context* c, TT in) { TT h = in; for (auto& r : rs) h = resBlock(c, r, h, F, B); return h; }, err)) return false; x = std::move(y);
        } else {
            ConvW cw; if (!loadConv(W, *ckpt, p + "conv.conv.", cw, err) || !W.upload(err)) return false;
            Vol y; if (!runVol(x, y, 0, 0, [&](ggml_context* c, TT in) { return conv3(c, cw, in, F, B); }, err)) return false;
            x = depthToSpace(y, ups[i].p1, ups[i].p2, ups[i].p3, ups[i].p1 == 2);
        }
    }
    say("conv_out");
    {   WeightSet W; ConvW cw; if (!loadConv(W, *ckpt, "vae.decoder.conv_out.conv.", cw, err) || !W.upload(err)) return false;
        Vol y; if (!runVol(x, y, 0, 0, [&](ggml_context* c, TT in) { return conv3(c, cw, ggml_silu(c, pixelNorm(c, in)), F, B); }, err)) return false;
        frames = unpatchify(y, 4);
    }
    return true;
}

bool LtxVae::encode(const Vol& image, Vol& latent, std::string& err) {
    if (!Compute::get().init(0, err)) return false;
    if (image.T != 1 || image.C != 3 || image.H % 32 || image.W % 32) { err = "the encoder takes one RGB frame whose sides are multiples of 32"; return false; }
    std::vector<float> mean, stdv;
    if (!ckpt->loadF32("vae.per_channel_statistics.mean-of-means", mean, err) || !ckpt->loadF32("vae.per_channel_statistics.std-of-means", stdv, err)) return false;
    struct F32Scope { F32Scope() { convF32() = true; } ~F32Scope() { convF32() = false; } } f32scope;     // the encoder has 1024-channel convolutions: accumulate in single precision
    const int F = 2, B = 0;                                           // causal encoder: two replicated frames in front, none behind
    Vol x = patchify(image, 4);
    {   WeightSet W; ConvW cw; if (!loadConv(W, *ckpt, "vae.encoder.conv_in.conv.", cw, err) || !W.upload(err)) return false;
        Vol y; if (!runVol(x, y, 0, 0, [&](ggml_context* c, TT in) { return conv3(c, cw, in, F, B); }, err)) return false; x = std::move(y);
        if (getenv("LTX_DEBUG")) { size_t bad = 0; for (float v : x.d) if (!std::isfinite(v)) bad++; fprintf(stderr, "  encoder %s: %dx%dx%dx%d, %zu invalid of %zu\n", "conv_in", x.T, x.C, x.H, x.W, bad, x.d.size()); }
    }
    struct Dn { int p1, p2, p3, out; };
    const Dn dn[9] = {{0,0,0,0}, {1,2,2,256}, {0,0,0,0}, {2,1,1,512}, {0,0,0,0}, {2,2,2,1024}, {0,0,0,0}, {2,2,2,1024}, {0,0,0,0}};
    for (int i = 0; i < 9; i++) {
        const std::string p = "vae.encoder.down_blocks." + std::to_string(i) + ".";
        WeightSet W;
        if (dn[i].p1 == 0) {
            std::vector<Res> rs((size_t)countRes(*ckpt, p));
            for (size_t k = 0; k < rs.size(); k++) if (!loadRes(W, *ckpt, p + "res_blocks." + std::to_string(k) + ".", rs[k], err)) return false;
            if (!W.upload(err)) return false;
            Vol y; if (!runVol(x, y, 0, 0, [&](ggml_context* c, TT in) { TT h = in; for (auto& r : rs) h = resBlock(c, r, h, F, B); return h; }, err)) return false; x = std::move(y);
        if (getenv("LTX_DEBUG")) { size_t bad = 0; for (float v : x.d) if (!std::isfinite(v)) bad++; fprintf(stderr, "  encoder %s: %dx%dx%dx%d, %zu invalid of %zu\n", ("res group " + std::to_string(i)).c_str(), x.T, x.C, x.H, x.W, bad, x.d.size()); }
        } else {
            // SpaceToDepthDownsample: first frame duplicated for a time stride, conv, shuffle, plus a group-mean skip connection
            Vol xin = x;
            if (dn[i].p1 == 2) { Vol d; d.alloc(x.T + 1, x.C, x.H, x.W); const size_t fs = (size_t)x.C * x.H * x.W; memcpy(d.d.data(), x.d.data(), fs * 4); memcpy(d.d.data() + fs, x.d.data(), x.d.size() * 4); xin = std::move(d); }
            Vol sk = spaceToDepth(xin, dn[i].p1, dn[i].p2, dn[i].p3);
            const int g = x.C * dn[i].p1 * dn[i].p2 * dn[i].p3 / dn[i].out;
            Vol skip; skip.alloc(sk.T, sk.C / g, sk.H, sk.W);
            for (int t = 0; t < sk.T; t++) for (int c = 0; c < skip.C; c++) for (int h = 0; h < sk.H; h++) for (int w = 0; w < sk.W; w++) {
                float s = 0; for (int k = 0; k < g; k++) s += sk.at(t, c * g + k, h, w);
                skip.at(t, c, h, w) = s / g;
            }
            ConvW cw; if (!loadConv(W, *ckpt, p + "conv.conv.", cw, err) || !W.upload(err)) return false;
            Vol y; if (!runVol(xin, y, 0, 0, [&](ggml_context* c, TT in) { return conv3(c, cw, in, F, B); }, err)) return false;
            y = spaceToDepth(y, dn[i].p1, dn[i].p2, dn[i].p3);
            for (size_t k = 0; k < y.d.size(); k++) y.d[k] += skip.d[k];
            x = std::move(y);
        if (getenv("LTX_DEBUG")) { size_t bad = 0; for (float v : x.d) if (!std::isfinite(v)) bad++; fprintf(stderr, "  encoder %s: %dx%dx%dx%d, %zu invalid of %zu\n", ("downsample " + std::to_string(i)).c_str(), x.T, x.C, x.H, x.W, bad, x.d.size()); }
        }
    }
    {   WeightSet W; ConvW cw; if (!loadConv(W, *ckpt, "vae.encoder.conv_out.conv.", cw, err) || !W.upload(err)) return false;
        Vol y; if (!runVol(x, y, 0, 0, [&](ggml_context* c, TT in) { return conv3(c, cw, ggml_silu(c, pixelNorm(c, in)), F, B); }, err)) return false;
        latent.alloc(y.T, 128, y.H, y.W);
        for (int t = 0; t < y.T; t++) for (int c = 0; c < 128; c++) for (int h = 0; h < y.H; h++) for (int w = 0; w < y.W; w++) latent.at(t, c, h, w) = (y.at(t, c, h, w) - mean[c]) / stdv[c];
    }
    return true;
}

// ---- self test against tools/ltx_ref_vae.py output
static bool readBin(const std::string& p, std::vector<float>& v) { std::string s; if (!sys::readFile(p, s)) return false; v.resize(s.size() / 4); memcpy(v.data(), s.data(), v.size() * 4); return true; }
static void compare(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) { printf("%-10s size mismatch: %zu vs reference %zu\n", what, a.size(), b.size()); return; }
    double md = 0, dot = 0, na = 0, nb = 0, mr = 0;
    for (size_t i = 0; i < a.size(); i++) { md = std::max(md, (double)std::fabs(a[i] - b[i])); dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; mr = std::max(mr, (double)std::fabs(b[i])); }
    printf("%-10s n=%zu max|diff|=%.5g (max|ref|=%.4g) cosine=%.6f\n", what, a.size(), md, mr, dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30));
}
static std::vector<float> swapAB(const std::vector<float>& a, int A, int B, size_t inner) {      // [A][B][inner] -> [B][A][inner]
    std::vector<float> o(a.size());
    for (int i = 0; i < A; i++) for (int j = 0; j < B; j++) memcpy(&o[((size_t)j * A + i) * inner], &a[((size_t)i * B + j) * inner], inner * 4);
    return o;
}
int runLtxVaeTest(const std::vector<std::string>& a) {
    if (a.size() < 2) { fprintf(stderr, "usage: ltx-vae <ltx checkpoint> <reference dir> [T h w]\n"); return 2; }
    LtxWeights w; std::string err; if (!w.open(a[0], err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    LtxVae vae; vae.ckpt = &w;
    int T = a.size() > 4 ? atoi(a[2].c_str()) : 3, h = a.size() > 4 ? atoi(a[3].c_str()) : 6, ww = a.size() > 4 ? atoi(a[4].c_str()) : 6;
    std::vector<float> z, y, x, e;
    if (readBin(a[1] + "\\vae_z.bin", z) && readBin(a[1] + "\\vae_y.bin", y)) {
        Vol lat; lat.T = T; lat.C = 128; lat.H = h; lat.W = ww; lat.d = swapAB(z, 128, T, (size_t)h * ww);       // torch stores [C][T][H][W], Vol is [T][C][H][W]
        Vol out; if (!vae.decode(lat, out, [](const std::string& s) { fprintf(stderr, "  %s\n", s.c_str()); }, err)) { fprintf(stderr, "decode: %s\n", err.c_str()); return 1; }
        printf("decoded %d frames %dx%d\n", out.T, out.W, out.H);
        compare("decode", swapAB(out.d, out.T, 3, (size_t)out.H * out.W), y);
    }
    if (readBin(a[1] + "\\vae_x.bin", x) && readBin(a[1] + "\\vae_e.bin", e)) {
        Vol img; img.T = 1; img.C = 3; img.H = 64; img.W = 64; img.d = x;
        Vol lat; if (!vae.encode(img, lat, err)) { fprintf(stderr, "encode: %s\n", err.c_str()); return 1; }
        compare("encode", lat.d, e);
    }
    return 0;
}


