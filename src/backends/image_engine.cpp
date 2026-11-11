// Text to image: Krea2 (Qwen3-VL text tower -> single-stream DiT -> Wan VAE) on ggml.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <thread>
#include "backends/engines.h"
#include "gpu/compute.h"
#include "krea2/bpe.h"
#include "krea2/dit.h"
#include "krea2/sampling.h"
#include "krea2/text_encoder.h"
#include "krea2/vae.h"
#include "util/image_io.h"
#include "util/png.h"
#include "util/sys.h"

namespace {
// Models stay loaded between jobs while their files do not change.
struct Engines {
    std::string tePath, ditKey, vaePath;
    Krea2TextEncoder te;
    Krea2DiT dit;
    Krea2Vae vae;
    Krea2VaeEncoder venc;
    std::string vencPath;
    QwenTokenizer tok;
    bool tokOk = false;
} g;

double secs(std::chrono::steady_clock::time_point a) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count(); }
std::string trim(const std::string& s) { size_t a = s.find_first_not_of(" \r\n\t"), b = s.find_last_not_of(" \r\n\t"); return a == std::string::npos ? "" : s.substr(a, b - a + 1); }
}  // namespace

// models are tied to the backend: drop them when the backend is switched
static const bool g_hookImage = (Compute::get().addResetHook([] { g.te.unload(); g.dit.unload(); g.vae.unload(); g.venc.unload(); g.tePath.clear(); g.ditKey.clear(); g.vaePath.clear(); g.vencPath.clear(); }), true);

bool runImage(const Job& job, JobCtl& ctl, std::string& err) {
    const Json& p = job.params;
    std::string ditPath = p.str("model"), tePath = p.str("clip"), vaePath = p.str("vae"), loraPath = p.str("lora");
    if (ditPath.empty() || tePath.empty() || vaePath.empty()) { err = "Select the diffusion model, text encoder and VAE in [Settings]."; return false; }

    // ---- parameters
    std::string pre = trim(p.str("pre_prompt")), main = trim(p.str("prompt")), neg = trim(p.str("negative"));
    std::string prompt = pre.empty() ? main : (main.empty() ? pre : pre + " " + main);
    if (prompt.empty()) { err = "Prompt is empty."; return false; }
    int W = p.integer("width", 720), H = p.integer("height", 720);
    // pick the backend first: Vulkan is fast but renders noise when the DiT sees more than ~6500 tokens in total (cap kept a little lower), CUDA has no such limit
    std::string backendKind;
    int seqEst = 0;
    {
        std::string eb;
        const int thr0 = std::max(4u, std::thread::hardware_concurrency());
        if (!Compute::get().use(p.str("backend"), thr0, eb)) { err = eb; return false; }
        backendKind = Compute::get().kind();
        if (backendKind == "vulkan") {
            if (!g.tokOk) { if (!g.tok.load("", err)) return false; g.tokOk = true; }
            seqEst = (int)g.tok.encode(krea2Template(prompt)).size();      // text tokens count against the same budget
        }
    }
    const auto capToBackend = [&](int& w, int& h) {
        if (backendKind != "vulkan") return;
        const double budget = 6450.0 - seqEst - 24;                        // image tokens: one per 16x16 pixels
        const double toks = std::ceil(w / 16.0) * std::ceil(h / 16.0);
        if (toks <= budget) return;
        const double s = std::sqrt(budget / toks);
        w = std::max(64, (int)(w * s) / 16 * 16); h = std::max(64, (int)(h * s) / 16 * 16);
        ctl.progress(0.0f, "size reduced to " + std::to_string(w) + "x" + std::to_string(h) + " (the Vulkan backend cannot render larger; use CUDA for big images)");
    };
    capToBackend(W, H);
    // image-to-image: the picture sets the aspect ratio; its longest side becomes max(width, height)
    const std::string initPath = trim(p.str("init_image"));
    const float strength = (float)std::min(1.0, std::max(0.02, p.num("denoise", 0.6)));
    ImageRgb initImg; std::vector<float> initPix;
    if (!initPath.empty()) {
        if (!loadImage(initPath, initImg, err)) return false;
        fitSize8(initImg.w, initImg.h, std::max(W, H), W, H);
        capToBackend(W, H);
        resizeToFloat(initImg, W, H, initPix);
    }
    int latW = W / 8, latH = H / 8;
    ctl.info(std::to_string(W) + "x" + std::to_string(H));
    if (latW < 8 || latH < 8) { err = "Image is too small (minimum 64x64)."; return false; }
    int steps = std::min(std::max(p.integer("steps", 10), 1), 150);
    float cfg = (float)p.num("cfg", 1.0);
    std::string sampler = p.str("sampler", "ER-SDE"), sched = p.str("scheduler", "beta");
    long long seed = (long long)p.num("seed", -1);
    if (seed < 0) seed = (long long)(std::random_device{}() & 0x7fffffff);
    float loraStrength = (float)p.num("lora_strength", 1.0);
    if (loraPath.empty()) loraStrength = 0;
    const bool useCfg = cfg > 1.001f;
    const int threads = std::max(4u, std::thread::hardware_concurrency());

    // ---- backend + models
    std::string e2;
    if (!Compute::get().init(threads, e2)) { err = e2; return false; }
    auto t0 = std::chrono::steady_clock::now();
    if (!g.tokOk) { if (!g.tok.load("", err)) return false; g.tokOk = true; }

    if (!g.te.loaded() || g.tePath != tePath) {
        ctl.progress(0.00f, "loading text encoder");
        if (!ctl.checkpoint()) { err = "cancelled"; return false; }
        if (!g.te.load(tePath, threads, err)) return false;
        g.tePath = tePath;
    }
    char key[64]; snprintf(key, sizeof key, "|%.4f", loraStrength);
    std::string ditKey = ditPath + "|" + loraPath + key;
    if (!g.dit.loaded() || g.ditKey != ditKey) {
        ctl.progress(0.03f, loraPath.empty() ? "loading diffusion model" : "loading diffusion model + LoRA");
        if (!ctl.checkpoint()) { err = "cancelled"; return false; }
        g.dit.unload();
        if (!g.dit.load(ditPath, threads, err, [&](float f) { ctl.progress(0.03f + 0.10f * f, "loading diffusion model"); }, loraPath, loraStrength)) return false;
        g.ditKey = ditKey;
    }
    if (!g.vae.loaded() || g.vaePath != vaePath) {
        ctl.progress(0.13f, "loading VAE");
        if (!g.vae.load(vaePath, err)) return false;
        g.vaePath = vaePath;
    }
    std::vector<float> initLat;
    if (!initPath.empty()) {
        ctl.progress(0.13f, "encoding the input image");
        if (!g.venc.loaded() || g.vencPath != vaePath) {
            g.venc.unload();
            if (!g.venc.load(vaePath, err)) return false;
            g.vencPath = vaePath;
        }
        if (!g.venc.encode(initPix, H, W, initLat, err)) return false;
    }
    const double loadSec = secs(t0);

    // ---- conditioning
    if (!ctl.checkpoint()) { err = "cancelled"; return false; }
    ctl.progress(0.14f, "encoding prompt");
    std::vector<float> ctxPos, ctxNeg; int seqPos = 0, seqNeg = 0;
    if (!g.te.encode(g.tok.encode(krea2Template(prompt)), ctxPos, seqPos, err)) return false;
    if (useCfg && !g.te.encode(g.tok.encode(krea2Template(neg)), ctxNeg, seqNeg, err)) return false;

    // ---- sampling in latent space (DiT needs even latent sizes; extra border is cropped afterwards)
    const int Hp = latH + (latH & 1), Wp = latW + (latW & 1);
    std::vector<float> x((size_t)16 * Hp * Wp);
    k2::Rng rng((unsigned long long)seed);
    rng.normal(x);
    std::vector<float> sigmas;
    const bool i2i = !initLat.empty() && strength < 0.999f;
    const int total = i2i ? std::max(steps, (int)std::lround(steps / strength)) : steps;       // schedule for full denoise, keep its tail
    if (sched == "simple") sigmas = k2::simpleSigmas(total);
    else sigmas = k2::betaSigmas(total);
    if (i2i && (int)sigmas.size() > steps + 1) sigmas.erase(sigmas.begin(), sigmas.end() - (steps + 1));
    if (sched != "beta" && sched != "simple") ctl.progress(0.15f, "scheduler '" + sched + "' not available, using beta");
    const int nSteps = (int)sigmas.size() - 1;
    for (auto& v : x) v *= sigmas[0];                                   // CONST noise scaling: sigma*noise + (1-sigma)*latent
    if (!initLat.empty()) {
        for (int c = 0; c < 16; c++) for (int y = 0; y < Hp; y++) for (int xx = 0; xx < Wp; xx++)
            x[((size_t)c * Hp + y) * Wp + xx] += (1.0f - sigmas[0]) * initLat[((size_t)c * latH + std::min(y, latH - 1)) * latW + std::min(xx, latW - 1)];
    }

    auto tSample = std::chrono::steady_clock::now();
    std::vector<float> vPos, vNeg;
    auto denoise = [&](const std::vector<float>& xin, float sigma, std::vector<float>& out) -> bool {
        if (!ctl.checkpoint()) return false;
        std::string e;
        if (!g.dit.forward(xin, Hp, Wp, sigma, ctxPos, seqPos, vPos, e)) { err = e; return false; }
        if (useCfg) {
            if (!ctl.checkpoint()) return false;
            if (!g.dit.forward(xin, Hp, Wp, sigma, ctxNeg, seqNeg, vNeg, e)) { err = e; return false; }
            for (size_t i = 0; i < vPos.size(); i++) vPos[i] = vNeg[i] + cfg * (vPos[i] - vNeg[i]);
        }
        out.resize(xin.size());
        for (size_t i = 0; i < xin.size(); i++) out[i] = xin[i] - vPos[i] * sigma;       // denoised = x - v*sigma
        return true;
    };
    auto onStep = [&](int i, int total) {
        double per = secs(tSample) / (i + 1);
        char b[96]; snprintf(b, sizeof b, "step %d/%d  (%.1f s/step)", i + 1, total, per);
        ctl.progress(0.16f + 0.74f * (float)(i + 1) / total, b);
    };

    bool ok = false;
    if (sampler == "Euler") {
        std::vector<float> den;
        ok = true;
        for (int i = 0; i < nSteps && ok; i++) {
            ok = denoise(x, sigmas[i], den);
            if (!ok) break;
            if (sigmas[i + 1] == 0) x = den;
            else { float dt = sigmas[i + 1] - sigmas[i]; for (size_t k = 0; k < x.size(); k++) x[k] += (x[k] - den[k]) / sigmas[i] * dt; }
            onStep(i, nSteps);
        }
    } else {
        if (sampler != "ER-SDE") ctl.progress(0.15f, "sampler '" + sampler + "' not available, using ER-SDE");
        ok = k2::erSde(x, sigmas, 1.15, denoise, [&](std::vector<float>& n) { rng.normal(n); }, onStep);
    }
    if (!ok) { if (err.empty()) err = "cancelled"; return false; }

    // ---- decode
    if (!ctl.checkpoint()) { err = "cancelled"; return false; }
    ctl.progress(0.91f, "decoding image");
    std::vector<float> lat((size_t)16 * latH * latW);
    for (int c = 0; c < 16; c++) for (int y = 0; y < latH; y++)
        memcpy(&lat[((size_t)c * latH + y) * latW], &x[((size_t)c * Hp + y) * Wp], latW * sizeof(float));
    Krea2Vae::latentToVae(lat, latH, latW);
    std::vector<float> rgb;
    if (!g.vae.decode(lat, latH, latW, rgb, err)) return false;

    const int OW = latW * 8, OH = latH * 8;
    std::vector<uint8_t> px(rgb.size());
    for (size_t i = 0; i < rgb.size(); i++) px[i] = (uint8_t)std::min(255.0f, std::max(0.0f, rgb[i] * 255.0f + 0.5f));
    std::string file = outputPath("image", job.id, ".png");
    char meta[1400];
    snprintf(meta, sizeof meta, "%.1000s\nSteps: %d, Sampler: %s, Scheduler: %s, CFG: %g, Seed: %lld, Size: %dx%d, Model: Krea2%s",
             prompt.c_str(), nSteps, sampler.c_str(), sched.c_str(), cfg, seed, OW, OH, loraPath.empty() ? "" : ", LoRA");
    if (!writePngRgb8(file, px.data(), OW, OH, "parameters", meta)) { err = "cannot write " + file; return false; }
    ctl.addOutput(file);
    char done[160];
    snprintf(done, sizeof done, "%dx%d in %.0f s (load %.0f s) on %s, seed %lld%s", OW, OH, secs(t0), loadSec, Compute::get().description().c_str(), seed,
             loraPath.empty() ? "" : (" , LoRA on " + std::to_string(g.dit.loraPatched) + " layers").c_str());
    ctl.progress(1.0f, done);
    return true;
}
