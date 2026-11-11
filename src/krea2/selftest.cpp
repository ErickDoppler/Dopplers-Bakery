// `DopplersBakery selftest <ref.json> [file.safetensors ...]`: compares stage A against dumps from ComfyUI.
#include <cmath>
#include <cstdio>
#include <map>
#include "krea2/bpe.h"
#include "krea2/safetensors.h"
#include "krea2/sampling.h"
#include "util/json.h"
#include "util/sys.h"

static int g_fail = 0;
static void check(bool ok, const char* what, const std::string& detail = "") {
    printf("[%s] %s %s\n", ok ? "ok  " : "FAIL", what, detail.c_str());
    if (!ok) g_fail++;
}
static double maxDiff(const std::vector<float>& a, const Json& b) {
    if (a.size() != b.a.size()) return 1e9;
    double m = 0;
    for (size_t i = 0; i < a.size(); i++) m = std::fmax(m, std::fabs(a[i] - b.a[i].num()));
    return m;
}

int runSelfTest(const std::vector<std::string>& args) {
    std::string txt;
    if (args.empty() || !sys::readFile(args[0], txt)) { printf("usage: selftest <ref.json> [model.safetensors ...]\n"); return 2; }
    bool ok; Json ref = Json::parse(txt, &ok);
    if (!ok) { printf("bad reference json\n"); return 2; }

    // --- sigma schedules
    for (int steps : {10, 8, 20}) {
        auto s = k2::betaSigmas(steps);
        std::string key = "beta" + std::to_string(steps);
        double d = maxDiff(s, ref.get(key));
        char b[96]; snprintf(b, sizeof b, "(n=%zu, max diff %.2e)", s.size(), d);
        check(d < 2e-5, key.c_str(), b);
    }
    { auto s = k2::simpleSigmas(10); double d = maxDiff(s, ref.get("simple10")); char b[64]; snprintf(b, sizeof b, "(max diff %.2e)", d); check(d < 2e-5, "simple10", b); }

    // --- ER-SDE with the same toy denoiser and noise as ComfyUI
    {
        auto sig = k2::betaSigmas(10);
        std::vector<float> x; for (auto& v : ref.get("er_x0").a) x.push_back((float)v.num());
        for (auto& v : x) v *= sig[0];
        int ni = 0;
        auto den = [&](const std::vector<float>& xx, float s, std::vector<float>& out) {
            for (size_t i = 0; i < xx.size(); i++) out[i] = xx[i] * 0.9f + 0.05f * s;
            return true;
        };
        auto noise = [&](std::vector<float>& n) {
            const Json& src = ref.get("er_noises").a[ni++];
            for (size_t i = 0; i < n.size(); i++) n[i] = (float)src.a[i].num();
        };
        k2::erSde(x, sig, 1.15, den, noise);
        double d = maxDiff(x, ref.get("er_result"));
        char b[64]; snprintf(b, sizeof b, "(max diff %.2e)", d);
        check(d < 5e-4, "er_sde 10 steps", b);
    }

    // --- tokenizer
    {
        QwenTokenizer tk; std::string err;
        check(tk.load("", err), "tokenizer loads from exe resources", err);
        for (auto& kv : ref.get("tokens").o) {
            std::string text = kv.first;
            std::vector<int> got = tk.encode(kv.first.rfind("<|im_start|>", 0) == 0 ? text : krea2Template(text));
            bool same = got.size() == kv.second.a.size();
            for (size_t i = 0; same && i < got.size(); i++) same = got[i] == (int)kv.second.a[i].num();
            std::string label = text.substr(0, 28);
            for (auto& c : label) if (c == '\n' || c == '\t') c = ' ';
            check(same, ("tokens: " + label).c_str(), same ? "" : "got " + std::to_string(got.size()) + " ids vs " + std::to_string(kv.second.a.size()));
        }
    }

    // --- safetensors
    for (size_t i = 1; i < args.size(); i++) {
        StFile f; std::string err;
        if (!f.open(args[i], err)) { check(false, "open", err); continue; }
        std::map<std::string, int> dt;
        for (auto& t : f.tensors()) dt[t.dtype]++;
        std::string d; for (auto& kv : dt) d += kv.first + ":" + std::to_string(kv.second) + " ";
        check(true, ("safetensors " + args[i].substr(args[i].find_last_of("\\/") + 1)).c_str(), "tensors=" + std::to_string(f.tensors().size()) + "  " + d);
        // sanity: dequantize one fp8 matrix and report its range
        for (auto& t : f.tensors()) {
            if (t.dtype != "F8_E4M3") continue;
            float sc = 1; f.scalar(t.name.substr(0, t.name.size() - 7) + ".weight_scale", sc);
            std::vector<float> v; f.toFloat(t, v, sc);
            double mn = 1e30, mx = -1e30, ss = 0; for (float x : v) { mn = std::fmin(mn, x); mx = std::fmax(mx, x); ss += (double)x * x; }
            char b[160]; snprintf(b, sizeof b, "%s scale=%.5g range=[%.4f, %.4f] rms=%.5f", t.name.c_str(), sc, mn, mx, std::sqrt(ss / v.size()));
            check(std::isfinite(mn) && std::isfinite(mx), "fp8 dequant", b);
            break;
        }
    }
    printf(g_fail ? "\n%d check(s) FAILED\n" : "\nall checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}

#include <chrono>
#include "krea2/text_encoder.h"
#include "gpu/compute.h"

// selftest-te <qwen3vl.safetensors> <ref_prefix>   (ref_prefix.bin / ref_prefix.json from ComfyUI)
int runTextEncoderTest(const std::vector<std::string>& a) {
    if (a.size() < 2) { printf("usage: selftest-te <qwen3vl_4b_fp8.safetensors> <ref_prefix>\n"); return 2; }
    std::string js, bin;
    if (!sys::readFile(a[1] + ".json", js) || !sys::readFile(a[1] + ".bin", bin)) { printf("reference files not found\n"); return 2; }
    bool ok; Json meta = Json::parse(js, &ok);
    std::string text = meta.str("text");
    int seqRef = meta.get("shape").a[1].integer(), dim = meta.get("shape").a[2].integer();
    const float* ref = (const float*)bin.data();

    QwenTokenizer tk; std::string err;
    if (!tk.load("", err)) { printf("tokenizer: %s\n", err.c_str()); return 1; }
    auto ids = tk.encode(krea2Template(text));

    auto t0 = std::chrono::steady_clock::now();
    Krea2TextEncoder te;
    if (!te.load(a[0], 8, err, [](float f) { printf("\rloading %3d%%", (int)(f * 100)); fflush(stdout); })) { printf("\nload failed: %s\n", err.c_str()); return 1; }
    auto t1 = std::chrono::steady_clock::now();
    std::vector<float> out; int seq = 0;
    if (!te.encode(ids, out, seq, err)) { printf("\nencode failed: %s\n", err.c_str()); return 1; }
    auto t2 = std::chrono::steady_clock::now();
    printf("\nload %.1fs, encode %.1fs, tokens %zu -> seq %d (ref %d), dim %d\n",
           std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count(), ids.size(), seq, seqRef, dim);
    check(seq == seqRef && dim == Krea2TextEncoder::kTaps * Krea2TextEncoder::kHidden, "output shape matches ComfyUI");
    if (seq != seqRef) return 1;
    for (int t = 0; t < Krea2TextEncoder::kTaps; t++) {
        double dot = 0, na = 0, nb = 0, md = 0, mref = 0;
        for (int i = 0; i < seq; i++) for (int j = 0; j < Krea2TextEncoder::kHidden; j++) {
            double x = out[((size_t)i * Krea2TextEncoder::kTaps + t) * Krea2TextEncoder::kHidden + j];
            double y = ref[(size_t)i * dim + (size_t)t * Krea2TextEncoder::kHidden + j];
            dot += x * y; na += x * x; nb += y * y; md = std::fmax(md, std::fabs(x - y)); mref = std::fmax(mref, std::fabs(y));
        }
        double cs = dot / std::sqrt(na * nb + 1e-30);
        char b[160]; snprintf(b, sizeof b, "tap layer %2d: cos %.5f  max|diff| %.4f (max|ref| %.2f)", Krea2TextEncoder::tapLayers()[t], cs, md, mref);
        check(cs > 0.995, "text encoder", b);
    }
    printf(g_fail ? "\n%d check(s) FAILED\n" : "\nall checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}

#include "krea2/dit.h"

// selftest-dit <dit.safetensors> <refdir>   (dit_x.bin, dit_y.bin, dit.json, te_short.bin from ComfyUI)
int runDiTTest(const std::vector<std::string>& a) {
    if (a.size() < 2) { printf("usage: selftest-dit <dit.safetensors> <refdir>\n"); return 2; }
    std::string js, xb, yb, cb;
    if (!sys::readFile(a[1] + "/dit.json", js) || !sys::readFile(a[1] + "/dit_x.bin", xb) || !sys::readFile(a[1] + (sys::exists(a[1] + "/dit_y32.bin") ? "/dit_y32.bin" : "/dit_y.bin"), yb) || !sys::readFile(a[1] + "/te_short.bin", cb)) { printf("reference files missing\n"); return 2; }
    bool ok; Json m = Json::parse(js, &ok);
    int H = m.integer("H", 0), W = m.integer("W", 0), seq = m.integer("seq", 0); float sigma = (float)m.num("sigma", 0.5);
    std::vector<float> x((const float*)xb.data(), (const float*)(xb.data() + xb.size()));
    std::vector<float> ctx((const float*)cb.data(), (const float*)(cb.data() + cb.size()));
    // apply_model returns the denoised sample x - v*sigma; recover v
    std::vector<float> refV(x.size());
    for (size_t i = 0; i < x.size(); i++) refV[i] = (x[i] - ((const float*)yb.data())[i]) / sigma;
    const float* ref = refV.data();

    Krea2DiT dit; std::string err;
    dit.debug = true;
    auto t0 = std::chrono::steady_clock::now();
    if (!dit.load(a[0], 16, err, [](float f) { printf("\rloading %3d%%", (int)(f * 100)); fflush(stdout); })) { printf("\nload failed: %s\n", err.c_str()); return 1; }
    auto t1 = std::chrono::steady_clock::now();
    std::vector<float> y;
    if (!dit.forward(x, H, W, sigma, ctx, seq, y, err)) { printf("\nforward failed: %s\n", err.c_str()); return 1; }
    auto t2 = std::chrono::steady_clock::now();
    printf("\nload %.1fs, forward %.1fs (%d image tokens + %d text)\n", std::chrono::duration<double>(t1 - t0).count(),
           std::chrono::duration<double>(t2 - t1).count(), H * W / 4, seq);
    extern void dumpCompare(Krea2DiT&, const std::string&, int);
    dumpCompare(dit, a[1], seq);
    double dot = 0, na = 0, nb = 0, md = 0, ss = 0;
    for (size_t i = 0; i < y.size(); i++) { dot += (double)y[i] * ref[i]; na += (double)y[i] * y[i]; nb += (double)ref[i] * ref[i]; md = std::fmax(md, std::fabs(y[i] - ref[i])); ss += (double)(y[i] - ref[i]) * (y[i] - ref[i]); }
    double cs = dot / std::sqrt(na * nb + 1e-30);
    char b[200]; snprintf(b, sizeof b, "cos %.5f  rel-rms err %.4f  max|diff| %.4f  (rms ref %.4f, ours %.4f)", cs, std::sqrt(ss / (nb + 1e-30)), md, std::sqrt(nb / y.size()), std::sqrt(na / y.size()));
    check(cs > 0.99, "DiT forward", b);
    return g_fail ? 1 : 0;
}

// compares a dumped intermediate to the ComfyUI hook output
static void cmpDbg(Krea2DiT& dit, const std::string& dir, const char* name, size_t refOffsetFloats = 0) {
    std::string rb;
    if (!sys::readFile(dir + "/dbg_" + name + ".bin", rb)) { printf("  (no ref for %s)\n", name); return; }
    const float* r = (const float*)rb.data(); size_t rn = rb.size() / 4;
    for (auto& kv : dit.dump) if (kv.first == name) {
        const auto& v = kv.second;
        size_t n = std::min(v.size(), rn - refOffsetFloats);
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < n; i++) { double x = v[i], y = r[refOffsetFloats + i]; dot += x * y; na += x * x; nb += y * y; }
        printf("  %-7s n=%zu  cos %.5f  rms ours %.4f ref %.4f\n", name, n, dot / std::sqrt(na * nb + 1e-30), std::sqrt(na / n), std::sqrt(nb / n));
        if (false && std::string(name).rfind("blk", 0) == 0) {
            const size_t D = 6144;
            for (int part = 0; part < 2; part++) {
                size_t b0 = part == 0 ? 0 : 13 * D, b1 = part == 0 ? 13 * D : n;
                double d2 = 0, a2 = 0, c2 = 0;
                for (size_t i = b0; i < b1; i++) { double x = v[i], y = r[i]; d2 += x * y; a2 += x * x; c2 += y * y; }
                printf("      %s tokens: cos %.5f\n", part == 0 ? "text " : "image", d2 / std::sqrt(a2 * c2 + 1e-30));
            }
        }
    }
}
void dumpCompare(Krea2DiT& dit, const std::string& dir, int seq) {
    for (const char* n : {"t", "tvec", "fusion", "txt", "img0", "blk0", "blk1", "blk3", "blk7", "blk11", "blk15", "blk19", "blk23", "blk27"}) cmpDbg(dit, dir, n);
    (void)seq;
}

// bench-dit <dit.safetensors> <H> <W> : one forward pass on random data, prints the time
int runDiTBench(const std::vector<std::string>& a) {
    if (a.size() < 3) { printf("usage: bench-dit <dit.safetensors> <latentH> <latentW>\n"); return 2; }
    int H = atoi(a[1].c_str()), W = atoi(a[2].c_str()), seq = 40;
    Krea2DiT dit; std::string err;
    if (!dit.load(a[0], 16, err)) { printf("load failed: %s\n", err.c_str()); return 1; }
    k2::Rng rng(1);
    std::vector<float> x((size_t)16 * H * W), ctx((size_t)seq * 12 * 2560), y;
    rng.normal(x); rng.normal(ctx);
    for (int i = 0; i < 2; i++) {
        auto t0 = std::chrono::steady_clock::now();
        if (!dit.forward(x, H, W, 0.8f, ctx, seq, y, err)) { printf("forward failed: %s\n", err.c_str()); return 1; }
        printf("forward %d: %.1fs for %d image tokens\n", i, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), H * W / 4);
    }
    return 0;
}

#include "krea2/vae.h"

// selftest-vae <qwen_image_vae.safetensors> <refdir>
int runVaeTest(const std::vector<std::string>& a) {
    if (a.size() < 2) { printf("usage: selftest-vae <vae.safetensors> <refdir>\n"); return 2; }
    Krea2Vae vae; std::string err;
    if (!vae.load(a[0], err)) { printf("load failed: %s\n", err.c_str()); return 1; }
    printf("backend: %s\n", Compute::get().description().c_str());
    struct Case { const char* name; int h, w; int tile; } cases[] = {{"small", 24, 32, 1 << 20}, {"mid", 48, 40, 1 << 20}, {"mid", 48, 40, 24 * 24}};
    for (auto& cs : cases) {
        std::string zb, yb;
        if (!sys::readFile(a[1] + "/vae_" + cs.name + "_z.bin", zb) || !sys::readFile(a[1] + "/vae_" + cs.name + "_y.bin", yb)) { printf("reference missing\n"); return 2; }
        std::vector<float> z((const float*)zb.data(), (const float*)(zb.data() + zb.size())), rgb;
        auto t0 = std::chrono::steady_clock::now();
        if (!vae.decode(z, cs.h, cs.w, rgb, err, cs.tile)) { printf("decode failed: %s\n", err.c_str()); return 1; }
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const float* ref = (const float*)yb.data();
        size_t n = rgb.size();
        if (n != yb.size() / 4) { check(false, "VAE output size", std::to_string(n) + " vs " + std::to_string(yb.size() / 4)); continue; }
        double dot = 0, na = 0, nb = 0, md = 0, ss = 0;
        for (size_t i = 0; i < n; i++) { double x = rgb[i], y = ref[i]; dot += x * y; na += x * x; nb += y * y; md = std::fmax(md, std::fabs(x - y)); ss += (x - y) * (x - y); }
        char b[200]; snprintf(b, sizeof b, "%s %dx%d%s: cos %.6f  rms err %.5f  max|diff| %.4f  (%.2fs)", cs.name, cs.w * 8, cs.h * 8, cs.tile < 1000000 ? " tiled" : "", dot / std::sqrt(na * nb), std::sqrt(ss / n), md, sec);
        check(cs.tile < 1000000 ? std::sqrt(ss / n) < 0.02 : std::sqrt(ss / n) < 0.01, "VAE decode", b);
    }
    return g_fail ? 1 : 0;
}

#include "tts/speech_decoder.h"

// selftest-tts-dec <speech_tokenizer/model.safetensors> <refdir>   (tts_codes.bin, tts_wav.bin, tts.json)
int runTtsDecoderTest(const std::vector<std::string>& a) {
    if (a.size() < 2) { printf("usage: selftest-tts-dec <speech_tokenizer.safetensors> <refdir>\n"); return 2; }
    std::string cb, wb, js;
    if (!sys::readFile(a[1] + "/tts_codes.bin", cb) || !sys::readFile(a[1] + "/tts_wav.bin", wb) || !sys::readFile(a[1] + "/tts.json", js)) { printf("reference missing\n"); return 2; }
    bool ok; Json m = Json::parse(js, &ok);
    int T = m.integer("T", 0);
    std::vector<int> codes((const int*)cb.data(), (const int*)(cb.data() + cb.size()));
    const float* ref = (const float*)wb.data(); size_t refN = wb.size() / 4;
    SpeechDecoder dec; std::string err;
    if (!dec.load(a[0], err)) { printf("load failed: %s\n", err.c_str()); return 1; }
    printf("backend: %s\n", Compute::get().description().c_str());
    std::vector<float> wav;
    auto t0 = std::chrono::steady_clock::now();
    if (!dec.decode(codes, T, wav, err)) { printf("decode failed: %s\n", err.c_str()); return 1; }
    printf("decoded %zu samples (ref %zu) in %.2fs\n", wav.size(), refN, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    size_t n = std::min(wav.size(), refN);
    double dot = 0, na = 0, nb = 0, md = 0, ss = 0;
    for (size_t i = 0; i < n; i++) { double x = wav[i], y = ref[i]; dot += x * y; na += x * x; nb += y * y; md = std::fmax(md, std::fabs(x - y)); ss += (x - y) * (x - y); }
    char b[200]; snprintf(b, sizeof b, "cos %.6f  rms err %.5f  max|diff| %.4f  (rms ref %.4f)", dot / std::sqrt(na * nb + 1e-30), std::sqrt(ss / n), md, std::sqrt(nb / n));
    check(dot / std::sqrt(na * nb + 1e-30) > 0.99, "speech decoder", b);
    return g_fail ? 1 : 0;
}

#include "tts/talker.h"

// selftest-talker <model dir> <refdir>   greedy generation compared with the ComfyUI run (tts_codes.bin, tts_prefill.bin ...)
int runTalkerTest(const std::vector<std::string>& a) {
    if (a.size() < 2) { printf("usage: selftest-talker <Qwen3-TTS model dir> <refdir>\n"); return 2; }
    std::string cb, js, pre, tr;
    if (!sys::readFile(a[1] + "/tts_codes.bin", cb) || !sys::readFile(a[1] + "/tts.json", js) || !sys::readFile(a[1] + "/tts_prefill.bin", pre) || !sys::readFile(a[1] + "/tts_trailing.bin", tr)) { printf("reference missing\n"); return 2; }
    bool ok; Json m = Json::parse(js, &ok);
    const int* ref = (const int*)cb.data(); int refT = m.integer("T", 0);
    QwenTalker t; std::string err;
    auto t0 = std::chrono::steady_clock::now();
    if (!t.load(a[0], 8, err, [](float f) { printf("\rloading %3d%%", (int)(f * 100)); fflush(stdout); })) { printf("\nload failed: %s\n", err.c_str()); return 1; }
    printf("\nloaded in %.1fs, backend %s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), Compute::get().description().c_str());
    QwenTalker::Params p; p.text = m.str("text"); p.speaker = "ryan"; p.language = "English"; p.sample = false; p.maxFrames = 70;
    std::vector<int> codes; int frames = 0;
    auto t1 = std::chrono::steady_clock::now();
    if (!t.generate(p, codes, frames, err)) { printf("generate failed: %s\n", err.c_str()); return 1; }
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    printf("generated %d frames (ref %d) in %.2fs = %.1f ms/frame\n", frames, refT, sec, sec * 1000 / std::max(frames, 1));

    auto cmp = [&](const std::vector<float>& ours, const std::string& rb, const char* name) {
        const float* r = (const float*)rb.data(); size_t n = std::min(ours.size(), rb.size() / 4);
        double dot = 0, na = 0, nb = 0; for (size_t i = 0; i < n; i++) { dot += (double)ours[i] * r[i]; na += (double)ours[i] * ours[i]; nb += (double)r[i] * r[i]; }
        char b[160]; snprintf(b, sizeof b, "%s: %zu vs %zu floats, cos %.6f", name, ours.size(), rb.size() / 4, dot / std::sqrt(na * nb + 1e-30));
        check(ours.size() == rb.size() / 4 && dot / std::sqrt(na * nb + 1e-30) > 0.999, "prompt", b);
    };
    cmp(t.dbgPrefill, pre, "prefill embeddings"); cmp(t.dbgTrailing, tr, "trailing text");

    {   // teacher-forced run: errors cannot compound, so this measures per-step accuracy
        std::vector<int> refCodes(ref, ref + (size_t)refT * 16);
        QwenTalker::Params q = p; q.forced = &refCodes; q.maxFrames = refT + 2;
        std::vector<int> c2; int f2 = 0;
        if (!t.generate(q, c2, f2, err)) { printf("forced run failed: %s\n", err.c_str()); return 1; }
        char b2[200]; snprintf(b2, sizeof b2, "code0 argmax agrees %d/%d, sub-codes %d/%d", t.forcedAgree0, t.forcedTotal0, t.forcedAgreeSub, t.forcedTotalSub);
        check(t.forcedAgree0 >= t.forcedTotal0 * 0.9 && t.forcedAgreeSub >= t.forcedTotalSub * 0.9, "teacher forced agreement", b2);
    }
    int same = 0, firstBad = -1;
    for (int f = 0; f < std::min(frames, refT); f++) {
        bool eq = true; for (int g = 0; g < 16; g++) if (codes[f * 16 + g] != ref[f * 16 + g]) eq = false;
        if (eq) same++; else if (firstBad < 0) firstBad = f;
    }
    int firstCode0Bad = -1; for (int f = 0; f < std::min(frames, refT); f++) if (codes[f * 16] != ref[f * 16]) { firstCode0Bad = f; break; }
    char b[200]; snprintf(b, sizeof b, "%d/%d frames identical, first differing frame %d, first code0 mismatch %d", same, refT, firstBad, firstCode0Bad);
    check(frames == refT && same == refT, "greedy codes identical", b);
    return g_fail ? 1 : 0;
}

#include "audio/t5gemma.h"

// selftest-sa-te <t5gemma.safetensors> <refdir>   (sa_te.json, sa_te_a.bin, sa_te_b.bin)
int runSaTextTest(const std::vector<std::string>& a) {
    if (a.size() < 2) { printf("usage: selftest-sa-te <t5gemma.safetensors> <refdir>\n"); return 2; }
    std::string js;
    if (!sys::readFile(a[1] + "/sa_te.json", js)) { printf("reference missing\n"); return 2; }
    bool ok; Json meta = Json::parse(js, &ok);
    T5GemmaEncoder te; std::string err;
    if (!te.load(a[0], err)) { printf("load failed: %s\n", err.c_str()); return 1; }
    const auto& sp = te.tokenizer();
    printf("spiece: %d pieces, model_type %d, dummy_prefix %d, remove_extra_ws %d, escape_ws %d, byte_fallback %d, charsmap %zu bytes\n",
           sp.vocabSize(), sp.modelType, sp.addDummyPrefix, sp.removeExtraWhitespaces, sp.escapeWhitespaces, sp.byteFallback, sp.charsmapBytes);
    for (auto& kv : meta.o) {
        std::vector<int> ref; for (auto& v : kv.second.get("ids").a) ref.push_back(v.integer());
        auto ids = te.tokenize(kv.second.str("text"));
        bool same = ids == ref;
        std::string d = same ? "" : "got:"; if (!same) { for (size_t i = 0; i < std::min<size_t>(ids.size(), 12); i++) d += " " + std::to_string(ids[i]); d += " ref:"; for (size_t i = 0; i < std::min<size_t>(ref.size(), 12); i++) d += " " + std::to_string(ref[i]); }
        check(same, ("tokens '" + kv.first + "' (" + std::to_string(ref.size()) + " ids)").c_str(), d);
        std::vector<float> out;
        if (!te.encode(same ? ids : ref, out, err)) { printf("encode failed: %s\n", err.c_str()); return 1; }
        std::string rb;
        if (!sys::readFile(a[1] + "/sa_te_" + kv.first + ".bin", rb)) continue;
        const float* r = (const float*)rb.data(); size_t n = std::min(out.size(), rb.size() / 4);
        double dot = 0, na = 0, nb = 0, md = 0; for (size_t i = 0; i < n; i++) { dot += (double)out[i] * r[i]; na += (double)out[i] * out[i]; nb += (double)r[i] * r[i]; md = std::fmax(md, std::fabs(out[i] - r[i])); }
        char b[160]; snprintf(b, sizeof b, "%zu vs %zu floats, cos %.6f, max|diff| %.4f (rms %.3f)", out.size(), rb.size() / 4, dot / std::sqrt(na * nb + 1e-30), md, std::sqrt(nb / n));
        check(out.size() == rb.size() / 4 && dot / std::sqrt(na * nb + 1e-30) > 0.999, ("T5Gemma output '" + kv.first + "'").c_str(), b);
    }
    return g_fail ? 1 : 0;
}

#include "tts/speaker_encoder.h"
#include "util/wav.h"

// selftest-base-spk <Base model.safetensors> <refdir>   (clone_ref.wav, clone_mel.bin, clone_spk.bin)
int runSpeakerTest(const std::vector<std::string>& a) {
    if (a.size() < 2) { printf("usage: selftest-base-spk <Base model.safetensors> <refdir>\n"); return 2; }
    std::vector<float> wav; int sr = 0; std::string err;
    if (!readWavMono(a[1] + "/clone_ref.wav", wav, sr, err)) { printf("%s\n", err.c_str()); return 2; }
    std::string mb, sb;
    if (!sys::readFile(a[1] + "/clone_mel.bin", mb) || !sys::readFile(a[1] + "/clone_spk.bin", sb)) { printf("reference missing\n"); return 2; }
    std::vector<float> mel; int T = 0;
    qwenMelSpectrogram(wav, mel, T);
    {   // reference mel is [128, T] channel-major; ours is frame-major
        const float* r = (const float*)mb.data(); size_t n = mb.size() / 4;
        double dot = 0, na = 0, nb = 0, md = 0;
        if ((size_t)T * 128 == n) for (int t = 0; t < T; t++) for (int m = 0; m < 128; m++) { double x = mel[(size_t)t * 128 + m], y = r[(size_t)m * T + t]; dot += x * y; na += x * x; nb += y * y; md = std::fmax(md, std::fabs(x - y)); }
        char b[160]; snprintf(b, sizeof b, "%d frames (ref %zu), cos %.7f, max|diff| %.5f", T, n / 128, dot / std::sqrt(na * nb + 1e-30), md);
        check((size_t)T * 128 == n && dot / std::sqrt(na * nb + 1e-30) > 0.99999, "mel spectrogram", b);
    }
    SpeakerEncoder enc;
    if (!enc.load(a[0], err)) { printf("load failed: %s\n", err.c_str()); return 1; }
    std::vector<float> emb;
    if (!enc.embedFromMel(mel, T, emb, err)) { printf("embed failed: %s\n", err.c_str()); return 1; }
    const float* r = (const float*)sb.data(); size_t n = std::min(emb.size(), sb.size() / 4);
    double dot = 0, na = 0, nb = 0, md = 0; for (size_t i = 0; i < n; i++) { dot += (double)emb[i] * r[i]; na += (double)emb[i] * emb[i]; nb += (double)r[i] * r[i]; md = std::fmax(md, std::fabs(emb[i] - r[i])); }
    char b[160]; snprintf(b, sizeof b, "%zu vs %zu floats, cos %.6f, max|diff| %.4f (rms %.3f)", emb.size(), sb.size() / 4, dot / std::sqrt(na * nb + 1e-30), md, std::sqrt(nb / n));
    check(emb.size() == sb.size() / 4 && dot / std::sqrt(na * nb + 1e-30) > 0.999, "speaker embedding", b);
    return g_fail ? 1 : 0;
}

// clone-sim <Base model.safetensors> <a.wav> <b.wav> ...   cosine similarity of speaker embeddings between all pairs
int runCloneSim(const std::vector<std::string>& a) {
    if (a.size() < 3) { printf("usage: clone-sim <Base model.safetensors> <a.wav> <b.wav> [...]\n"); return 2; }
    SpeakerEncoder enc; std::string err;
    if (!enc.load(a[0], err)) { printf("load failed: %s\n", err.c_str()); return 1; }
    std::vector<std::vector<float>> embs;
    for (size_t i = 1; i < a.size(); i++) {
        std::vector<float> w; int sr = 0, ok = 0;
        if (!readWavMono(a[i], w, sr, err)) { printf("%s: %s\n", a[i].c_str(), err.c_str()); return 1; }
        w = resampleSinc(w, sr, 24000);
        std::vector<float> e; if (!enc.embed(w, e, err)) { printf("embed: %s\n", err.c_str()); return 1; }
        embs.push_back(e); (void)ok;
    }
    for (size_t i = 0; i < embs.size(); i++) for (size_t j = i + 1; j < embs.size(); j++) {
        double dot = 0, na = 0, nb = 0; for (size_t k = 0; k < embs[i].size(); k++) { dot += (double)embs[i][k] * embs[j][k]; na += (double)embs[i][k] * embs[i][k]; nb += (double)embs[j][k] * embs[j][k]; }
        printf("sim(%zu,%zu) = %.4f   %s  vs  %s\n", i + 1, j + 1, dot / std::sqrt(na * nb), a[i + 1].substr(a[i + 1].find_last_of("\/") + 1).c_str(), a[j + 1].substr(a[j + 1].find_last_of("\/") + 1).c_str());
    }
    return 0;
}

#include "tts/codec_encoder.h"

// selftest-codec <speech_tokenizer/model.safetensors> <refdir>   (clone_ref.wav, clone_codes.bin)
int runCodecTest(const std::vector<std::string>& a) {
    if (a.size() < 2) { printf("usage: selftest-codec <speech_tokenizer.safetensors> <refdir>\n"); return 2; }
    std::vector<float> wav; int sr = 0; std::string err, cb;
    if (!readWavMono(a[1] + "/clone_ref.wav", wav, sr, err) || !sys::readFile(a[1] + (sys::exists(a[1] + "/clone_codes32.bin") ? "/clone_codes32.bin" : "/clone_codes.bin"), cb)) { printf("reference missing\n"); return 2; }
    CodecEncoder enc;
    if (!enc.load(a[0], err)) { printf("load failed: %s\n", err.c_str()); return 1; }
    std::vector<int> codes; int frames = 0;
    auto t0 = std::chrono::steady_clock::now();
    if (!enc.encode(wav, codes, frames, err)) { printf("encode failed: %s\n", err.c_str()); return 1; }
    printf("encoded %d frames in %.2fs on %s\n", frames, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), Compute::get().description().c_str());
    {   // continuous embeddings before quantization
        std::string eb;
        if (sys::readFile(a[1] + "/clone_emb32.bin", eb)) {
            const float* r = (const float*)eb.data(); size_t n = std::min(enc.dbgEmb.size(), eb.size() / 4);
            double dot = 0, na = 0, nb = 0, md = 0; for (size_t i = 0; i < n; i++) { dot += (double)enc.dbgEmb[i] * r[i]; na += (double)enc.dbgEmb[i] * enc.dbgEmb[i]; nb += (double)r[i] * r[i]; md = std::fmax(md, std::fabs(enc.dbgEmb[i] - r[i])); }
            char b0[160]; snprintf(b0, sizeof b0, "%zu vs %zu floats, cos %.7f, max|diff| %.5f (rms %.3f)", enc.dbgEmb.size(), eb.size() / 4, dot / std::sqrt(na * nb), md, std::sqrt(nb / n));
            check(dot / std::sqrt(na * nb) > 0.9999, "encoder embeddings", b0);
            const int T2 = (int)(enc.dbgEmb.size() / 512); std::string pf;
            for (int t = 0; t < T2; t += (T2 > 12 ? T2 / 12 : 1)) { double d2 = 0, a2 = 0, b2 = 0; for (int ch = 0; ch < 512; ch++) { double x = enc.dbgEmb[(size_t)ch * T2 + t], y = r[(size_t)ch * T2 + t]; d2 += x * y; a2 += x * x; b2 += y * y; } char bb[32]; snprintf(bb, sizeof bb, "t%d:%.5f ", t, d2 / std::sqrt(a2 * b2)); pf += bb; }
            printf("      per-frame cos: %s\n", pf.c_str());
        }
    }
    const int* ref = (const int*)cb.data(); int refT = (int)(cb.size() / 64);
    check(frames == refT, "frame count", std::to_string(frames) + " vs " + std::to_string(refT));
    int n = std::min(frames, refT), eq = 0, eqG[16] = {0};
    for (int t = 0; t < n; t++) for (int g = 0; g < 16; g++) if (codes[t * 16 + g] == ref[t * 16 + g]) { eq++; eqG[g]++; }
    std::string d; for (int g = 0; g < 16; g++) d += std::to_string(eqG[g] * 100 / std::max(n, 1)) + "% ";
    char b[300]; snprintf(b, sizeof b, "%.1f%% of codes identical; per group: %s", 100.0 * eq / (n * 16), d.c_str());
    check(100.0 * eq / (n * 16) > 80.0, "codec codes vs ComfyUI", b);
    return g_fail ? 1 : 0;
}

#include "audio/sa_dit.h"

// selftest-sa-dit <stable_audio_3_medium.safetensors> <refdir>   (sa_dit.json, sa_dit_x/ctx/v.bin, sadbg_*.bin from ComfyUI fp32)
int runSaDiTTest(const std::vector<std::string>& a) {
    if (a.size() < 2) { printf("usage: selftest-sa-dit <checkpoint> <refdir>\n"); return 2; }
    std::string js, xb, cb, vb;
    if (!sys::readFile(a[1] + "/sa_dit.json", js) || !sys::readFile(a[1] + "/sa_dit_x.bin", xb) || !sys::readFile(a[1] + "/sa_dit_ctx.bin", cb) || !sys::readFile(a[1] + "/sa_dit_v.bin", vb)) { printf("reference missing\n"); return 2; }
    bool ok; Json m = Json::parse(js, &ok);
    const int T = m.integer("T", 0), n = m.integer("n_text", 0); const float sigma = (float)m.num("sigma", 0.5), secs = (float)m.num("seconds", 6);
    const float* xr = (const float*)xb.data(); const float* vr = (const float*)vb.data();
    std::vector<float> x((size_t)T * 256);
    for (int t = 0; t < T; t++) for (int c = 0; c < 256; c++) x[(size_t)t * 256 + c] = xr[(size_t)c * T + t];       // channel-major -> time-major
    std::vector<float> text((const float*)cb.data(), (const float*)cb.data() + (size_t)n * 768);
    SaDiT dit; std::string err;
    auto t0 = std::chrono::steady_clock::now();
    if (!dit.load(a[0], err)) { printf("load failed: %s\n", err.c_str()); return 1; }
    printf("loaded in %.1fs, backend %s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), Compute::get().description().c_str());
    dit.debug = true;
    std::vector<float> y;
    auto t1 = std::chrono::steady_clock::now();
    if (!dit.forward(x, T, sigma, text, n, secs, y, err)) { printf("forward failed: %s\n", err.c_str()); return 1; }
    printf("forward %.2fs\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count());
    for (auto& kv : dit.dump) {
        std::string rb;
        if (!sys::readFile(a[1] + "/sadbg_" + kv.first + ".bin", rb)) continue;
        const float* r = (const float*)rb.data(); size_t cnt = std::min(kv.second.size(), rb.size() / 4);
        double dot = 0, na = 0, nb = 0; for (size_t i = 0; i < cnt; i++) { dot += (double)kv.second[i] * r[i]; na += (double)kv.second[i] * kv.second[i]; nb += (double)r[i] * r[i]; }
        printf("  %-12s n=%zu (ref %zu)  cos %.6f  rms ours %.4f ref %.4f\n", kv.first.c_str(), kv.second.size(), rb.size() / 4, dot / std::sqrt(na * nb + 1e-30), std::sqrt(na / cnt), std::sqrt(nb / cnt));
    }
    double dot = 0, na = 0, nb = 0, ss = 0;
    for (int t = 0; t < T; t++) for (int c = 0; c < 256; c++) { double p = y[(size_t)t * 256 + c], q = vr[(size_t)c * T + t]; dot += p * q; na += p * p; nb += q * q; ss += (p - q) * (p - q); }
    char b[200]; snprintf(b, sizeof b, "cos %.6f  rel-rms err %.4f  (rms ref %.4f, ours %.4f)", dot / std::sqrt(na * nb), std::sqrt(ss / nb), std::sqrt(nb / y.size()), std::sqrt(na / y.size()));
    check(dot / std::sqrt(na * nb) > 0.99, "Stable Audio DiT forward", b);
    return g_fail ? 1 : 0;
}

#include "audio/sa_vae.h"

// selftest-sa-vae <stable_audio_3_medium.safetensors> <refdir>   (sa_vae_z24/40.bin, sa_vae_y24/40.bin from ComfyUI)
int runSaVaeTest(const std::vector<std::string>& a) {
    if (a.size() < 2) { printf("usage: selftest-sa-vae <checkpoint> <refdir>\n"); return 2; }
    SaVae vae; std::string err;
    auto t0 = std::chrono::steady_clock::now();
    if (!vae.load(a[0], err)) { printf("load failed: %s\n", err.c_str()); return 1; }
    printf("loaded in %.1fs, backend %s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), Compute::get().description().c_str());
    struct Case { int T; const char* core; } cases[] = {{24, "128"}, {40, "128"}, {40, "16"}};
    for (auto& cs : cases) {
        std::string zb, yb;
        if (!sys::readFile(a[1] + "/sa_vae_z" + std::to_string(cs.T) + ".bin", zb) || !sys::readFile(a[1] + "/sa_vae_y" + std::to_string(cs.T) + ".bin", yb)) { printf("reference missing\n"); return 2; }
        const float* z = (const float*)zb.data(); const float* y = (const float*)yb.data();
        const int T = cs.T;
        std::vector<float> lat((size_t)T * 256);
        for (int t = 0; t < T; t++) for (int c = 0; c < 256; c++) lat[(size_t)t * 256 + c] = z[(size_t)c * T + t];
        _putenv_s("BAKERY_VAE_CORE", cs.core);
        std::vector<float> pcm;
        auto t1 = std::chrono::steady_clock::now();
        if (!vae.decode(lat, T, pcm, err)) { printf("decode failed: %s\n", err.c_str()); return 1; }
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        const size_t S = (size_t)T * 4096;
        double dot = 0, na = 0, nb = 0, ss = 0, md = 0;
        for (size_t i = 0; i < S; i++) for (int ch = 0; ch < 2; ch++) {
            double p = pcm[i * 2 + ch], q = y[(size_t)ch * S + i];
            dot += p * q; na += p * p; nb += q * q; ss += (p - q) * (p - q); md = std::fmax(md, std::fabs(p - q));
        }
        char b[200]; snprintf(b, sizeof b, "T=%d core=%s: cos %.6f  rel-rms err %.4f  max|diff| %.4f (rms %.4f)  %.2fs", T, cs.core, dot / std::sqrt(na * nb), std::sqrt(ss / nb), md, std::sqrt(nb / (S * 2)), sec);
        check(dot / std::sqrt(na * nb) > 0.99, "audio VAE decode", b);
    }
    _putenv_s("BAKERY_VAE_CORE", "");
    return g_fail ? 1 : 0;
}
