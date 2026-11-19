#include "ltx/ltx_vae.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <thread>
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

// ---- host-side rearranges (the volumes are gigabytes at full size: every one runs on all cores, one (frame, channel) row of output per task)
template <class F> void parallelRows(size_t n, F fn) {
    const size_t nt = std::max<size_t>(1, std::min<size_t>(n, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (size_t t = 0; t < nt; t++) th.emplace_back([&, t] { for (size_t i = t; i < n; i += nt) fn(i); });
    for (auto& x : th) x.join();
}
Vol depthToSpace(const Vol& y, int p1, int p2, int p3, bool dropFirst) {
    const int C = y.C / (p1 * p2 * p3), T = y.T * p1, drop = dropFirst ? 1 : 0;
    Vol o; o.T = T - drop; o.C = C; o.H = y.H * p2; o.W = y.W * p3; o.d.resize((size_t)o.T * o.C * o.H * o.W);
    parallelRows((size_t)o.T * o.C, [&](size_t row) {
        const int t = (int)(row / C), c = (int)(row % C), st = (t + drop) / p1, i1 = (t + drop) % p1;
        float* dst = &o.d[row * (size_t)o.H * o.W];
        for (int h = 0; h < o.H; h++) {
            const int base = ((c * p1 + i1) * p2 + h % p2) * p3;
            for (int w = 0; w < o.W; w++) dst[(size_t)h * o.W + w] = y.d[(((size_t)st * y.C + base + w % p3) * y.H + h / p2) * y.W + w / p3];
        }
    });
    return o;
}
Vol spaceToDepth(const Vol& x, int p1, int p2, int p3) {        // "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w"
    Vol o; o.T = x.T / p1; o.C = x.C * p1 * p2 * p3; o.H = x.H / p2; o.W = x.W / p3; o.d.resize((size_t)o.T * o.C * o.H * o.W);
    parallelRows((size_t)o.T * o.C, [&](size_t row) {
        const int t = (int)(row / o.C), ch = (int)(row % o.C), i3 = ch % p3, i2 = (ch / p3) % p2, i1 = (ch / (p3 * p2)) % p1, c = ch / (p3 * p2 * p1);
        float* dst = &o.d[row * (size_t)o.H * o.W];
        for (int h = 0; h < o.H; h++) for (int w = 0; w < o.W; w++) dst[(size_t)h * o.W + w] = x.d[(((size_t)(t * p1 + i1) * x.C + c) * x.H + h * p2 + i2) * x.W + w * p3 + i3];
    });
    return o;
}
Vol unpatchify(const Vol& y, int q) {                           // "b (c r q) f h w -> b c f (h q) (w r)"
    Vol o; o.T = y.T; o.C = y.C / (q * q); o.H = y.H * q; o.W = y.W * q; o.d.resize((size_t)o.T * o.C * o.H * o.W);
    parallelRows((size_t)o.T * o.C, [&](size_t row) {
        const int t = (int)(row / o.C), c = (int)(row % o.C);
        float* dst = &o.d[row * (size_t)o.H * o.W];
        for (int h = 0; h < o.H; h++) for (int w = 0; w < o.W; w++) dst[(size_t)h * o.W + w] = y.d[(((size_t)t * y.C + (c * q + w % q) * q + h % q) * y.H + h / q) * y.W + w / q];
    });
    return o;
}
Vol patchify(const Vol& x, int q) {                             // "b c (h q) (w r) -> b (c r q) h w"
    Vol o; o.T = x.T; o.C = x.C * q * q; o.H = x.H / q; o.W = x.W / q; o.d.resize((size_t)o.T * o.C * o.H * o.W);
    parallelRows((size_t)o.T * o.C, [&](size_t row) {
        const int t = (int)(row / o.C), ch = (int)(row % o.C), qi = ch % q, ri = (ch / q) % q, c = ch / (q * q);
        float* dst = &o.d[row * (size_t)o.H * o.W];
        for (int h = 0; h < o.H; h++) for (int w = 0; w < o.W; w++) dst[(size_t)h * o.W + w] = x.d[(((size_t)t * x.C + c) * x.H + h * q + qi) * x.W + w * q + ri];
    });
    return o;
}

struct VolTimes { double build = 0, alloc = 0, set = 0, compute = 0, get = 0; };
inline VolTimes& volTimes() { static VolTimes t; return t; }
inline double nowSec() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
template <class F> bool runVolOnce(const Vol& in, Vol& out, F build, std::string& err) {
    VolTimes& vt = volTimes(); double t0 = nowSec();
    Graph G(16384); ggml_context* c = G.c;
    TT x = ggml_new_tensor_4d(c, GGML_TYPE_F32, in.W, in.H, in.C, in.T); ggml_set_input(x);
    TT y = build(c, x);
    ggml_set_output(y); ggml_build_forward_expand(G.g, y);
    double t1 = nowSec(); vt.build += t1 - t0;
    if (!runGraph(G, err)) return false;
    double t2 = nowSec(); vt.alloc += t2 - t1;
    ggml_backend_tensor_set(x, in.d.data(), 0, in.d.size() * 4);
    double t3 = nowSec(); vt.set += t3 - t2;
    if (!computeGraph(G, err)) return false;
    double t4 = nowSec(); vt.compute += t4 - t3;
    out.T = (int)y->ne[3]; out.C = (int)y->ne[2]; out.H = (int)y->ne[1]; out.W = (int)y->ne[0];
    out.d.resize((size_t)out.T * out.C * out.H * out.W);
    ggml_backend_tensor_get(y, out.d.data(), 0, out.d.size() * 4);
    ggml_backend_sched_reset(Compute::get().sched);
    vt.get += nowSec() - t4;
    return true;
}// Same, but long videos go through the graph in time chunks. Every convolution looks one frame ahead and behind, so a chunk is given `halo` extra frames on
// each side (the number of convolutions in `build`) and only its exact middle is kept: the result equals one big graph, while the tensors stay small enough
// for the video memory (a whole 10 s video at full resolution is 8 GB per tensor and silently spills to the CPU). outC: channels of the biggest tensor if not the input's.
template <class F> bool runVol(const Vol& in, Vol& out, int halo, int outC, F build, std::string& err) {
    const size_t freeMB = Compute::get().gpuFreeMB();
    const size_t budget = std::min<size_t>(1u << 30, std::max<size_t>(256u << 20, (freeMB << 20) / 10));          // bytes of one tensor
    const size_t frameBytes = (size_t)std::max(in.C, outC) * in.H * in.W * 4;
    const int maxFrames = (int)std::max<size_t>((size_t)(2 * halo + 2), budget / std::max<size_t>(frameBytes, 1));
    if (halo <= 0 || in.T <= maxFrames) return runVolOnce(in, out, build, err);
    const int core = maxFrames - 2 * halo;
    const size_t inFrame = (size_t)in.C * in.H * in.W;
    for (int t0 = 0; t0 < in.T; t0 += core) {
        const int a = std::max(0, t0 - halo), b = std::min(in.T, t0 + core + halo);
        Vol sub; sub.T = b - a; sub.C = in.C; sub.H = in.H; sub.W = in.W;
        sub.d.assign(in.d.begin() + (size_t)a * inFrame, in.d.begin() + (size_t)b * inFrame);
        Vol ys; if (!runVolOnce(sub, ys, build, err)) return false;
        if (ys.T != sub.T) { err = "internal error: a time-chunked graph changed the number of frames"; return false; }
        if (t0 == 0) { out.T = in.T; out.C = ys.C; out.H = ys.H; out.W = ys.W; out.d.assign((size_t)out.T * out.C * out.H * out.W, 0.f); }
        const size_t outFrame = (size_t)out.C * out.H * out.W;
        const int keep0 = t0 - a, keepN = std::min(core, in.T - t0);
        memcpy(out.d.data() + (size_t)t0 * outFrame, ys.d.data() + (size_t)keep0 * outFrame, (size_t)keepN * outFrame * 4);
    }
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
    auto say = [&](const std::string& s) {
        if (getenv("LTX_DEBUG")) { VolTimes& t = volTimes(); fprintf(stderr, "      [previous block: build %.1f s, graph alloc %.1f s, upload %.1f s, compute %.1f s, download %.1f s]\n", t.build, t.alloc, t.set, t.compute, t.get); t = VolTimes(); }
        if (progress) progress(s);
    };
    {   say("conv_in");
        WeightSet W; ConvW cw; if (!loadConv(W, *ckpt, "vae.decoder.conv_in.conv.", cw, err) || !W.upload(err)) return false;
        Vol y; if (!runVol(x, y, 1, 0, [&](ggml_context* c, TT in) { return conv3(c, cw, in, F, B); }, err)) return false; x = std::move(y);
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
            Vol y; if (!runVol(x, y, 2 * (int)rs.size(), 0, [&](ggml_context* c, TT in) { TT h = in; for (auto& r : rs) h = resBlock(c, r, h, F, B); return h; }, err)) return false; x = std::move(y);
        } else {
            ConvW cw; if (!loadConv(W, *ckpt, p + "conv.conv.", cw, err) || !W.upload(err)) return false;
            Vol y; if (!runVol(x, y, 1, cw.oc, [&](ggml_context* c, TT in) { return conv3(c, cw, in, F, B); }, err)) return false;
            x = depthToSpace(y, ups[i].p1, ups[i].p2, ups[i].p3, ups[i].p1 == 2);
        }
    }
    say("conv_out");
    {   WeightSet W; ConvW cw; if (!loadConv(W, *ckpt, "vae.decoder.conv_out.conv.", cw, err) || !W.upload(err)) return false;
        Vol y; if (!runVol(x, y, 1, 0, [&](ggml_context* c, TT in) { return conv3(c, cw, ggml_silu(c, pixelNorm(c, in)), F, B); }, err)) return false;
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




// `DopplersBakery ltx-vae-bench <ckpt> T h w`: decodes a random latent of that size and prints the time of every block (no reference needed)
int runLtxVaeBench(const std::vector<std::string>& a) {
    if (a.size() < 4) { fprintf(stderr, "usage: ltx-vae-bench <ltx checkpoint> T h w\n"); return 2; }
    LtxWeights w; std::string err; if (!w.open(a[0], err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    LtxVae vae; vae.ckpt = &w;
    Vol lat; lat.alloc(atoi(a[1].c_str()), 128, atoi(a[2].c_str()), atoi(a[3].c_str()));
    unsigned s = 12345u; for (auto& v : lat.d) { s = s * 1664525u + 1013904223u; v = ((float)(s >> 8) / 16777216.f - 0.5f) * 3.f; }
    auto t0 = std::chrono::steady_clock::now(); double last = 0; Vol out;
    if (!vae.decode(lat, out, [&](const std::string& m) { const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); fprintf(stderr, "  %7.1f s  (+%.1f)  %s\n", t, t - last, m.c_str()); last = t; }, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    printf("decoded %d frames %dx%d in %.1f s\n", out.T, out.W, out.H, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
}



// `DopplersBakery ltx-conv-bench <ckpt> [frames]`: times one 128-channel 3D convolution of the last decoder stage (320 x 192 pixels) done three ways
int runLtxConvBench(const std::vector<std::string>& a) {
    if (a.size() < 1) { fprintf(stderr, "usage: ltx-conv-bench <ltx checkpoint> [frames]\n"); return 2; }
    LtxWeights w; std::string err; if (!w.open(a[0], err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    if (!Compute::get().init(0, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const int n = a.size() > 1 ? atoi(a[1].c_str()) : 16, W = 320, H = 192, C = 128;
    const std::string name = "vae.decoder.up_blocks.8.res_blocks.0.conv1.conv.";
    std::vector<float> in((size_t)W * H * C * n); unsigned s = 7u; for (auto& v : in) { s = s * 1664525u + 1013904223u; v = ((float)(s >> 8) / 16777216.f - 0.5f); }
    WeightSet ws; ConvW cw; if (!loadConv(ws, w, name, cw, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    // the same weights as one 3D kernel for ggml_conv_3d: file order [oc][ic][kd][kh][kw] is already ggml [kw, kh, kd, oc*ic]
    std::vector<float> raw; w.loadF32(name + "weight", raw, err);
    TT k3 = ggml_new_tensor_4d(ws.ctx, GGML_TYPE_F16, 3, 3, 3, (int64_t)C * C); ws.items.push_back({k3, raw, true});
    if (!ws.upload(err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::vector<float> outA, outB;
    for (int variant = 0; variant < 4; variant++) {
        double best = 1e9;
        for (int rep = 0; rep < 3; rep++) {
            Graph G(4096); ggml_context* c = G.c;
            TT x = ggml_new_tensor_4d(c, GGML_TYPE_F32, W, H, C, n); ggml_set_input(x);
            TT y;
            if (variant == 0) y = conv3(c, cw, x, 1, 1);
            else if (variant == 2) y = ggml_conv_2d(c, cw.w[0], x, 1, 1, 1, 1, 1, 1);                                  // one tap: im2col + matrix product
            else if (variant == 3) y = ggml_im2col(c, cw.w[0], x, 1, 1, 1, 1, 1, 1, true, GGML_TYPE_F16);                // only the im2col of that tap
            else {
                TT xt = ggml_cont(c, ggml_permute(c, x, 0, 1, 3, 2));                                        // [W,H,T,C]
                TT first = ggml_view_4d(c, xt, W, H, 1, C, xt->nb[1], xt->nb[2], xt->nb[3], 0), last = ggml_view_4d(c, xt, W, H, 1, C, xt->nb[1], xt->nb[2], xt->nb[3], (size_t)(n - 1) * xt->nb[2]);
                TT xp = ggml_concat(c, ggml_concat(c, first, xt, 2), last, 2);
                TT r = ggml_conv_3d(c, k3, xp, C, 1, 1, 1, 1, 1, 0, 1, 1, 1);                                // [W,H,T,OC]
                y = ggml_add(c, ggml_cont(c, ggml_permute(c, r, 0, 1, 3, 2)), ggml_reshape_4d(c, cw.b, 1, 1, C, 1));
            }
            ggml_set_output(y); ggml_build_forward_expand(G.g, y);
            if (!runGraph(G, err)) { fprintf(stderr, "variant %d: %s\n", variant, err.c_str()); break; }
            ggml_backend_tensor_set(x, in.data(), 0, in.size() * 4);
            auto t0 = std::chrono::steady_clock::now();
            if (!computeGraph(G, err)) { fprintf(stderr, "%s\n", err.c_str()); break; }
            std::vector<float> o((size_t)ggml_nbytes(y) / 4 + 1); ggml_backend_tensor_get(y, o.data(), 0, ggml_nbytes(y));       // (the im2col variant is F16)
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            (variant == 0 ? outA : outB) = o;
            ggml_backend_sched_reset(Compute::get().sched);
        }
        const double tflop = 2.0 * (variant >= 2 ? 9 : 27) * C * C * (double)W * H * n / 1e12;
        printf("%-26s %7.3f s   %.2f TFLOPS\n", variant == 0 ? "3 taps (current)" : variant == 1 ? "ggml_conv_3d" : variant == 2 ? "one tap conv_2d" : "one tap im2col only", best, tflop / best);
    }
    if (!outA.empty() && outA.size() == outB.size()) { double dot = 0, na = 0, nb = 0; for (size_t i = 0; i < outA.size(); i++) { dot += (double)outA[i] * outB[i]; na += (double)outA[i] * outA[i]; nb += (double)outB[i] * outB[i]; } printf("agreement (cosine): %.6f\n", dot / std::sqrt(na * nb)); }
    return 0;
}




