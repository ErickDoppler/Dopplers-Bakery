#include "ltx/ltx_pipeline.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include "ltx/ltx_audio.h"
#include "ltx/ltx_text.h"
#include "ltx/ltx_upscaler.h"
#include "util/mp4.h"
#include "util/png.h"
#include "util/sys.h"

namespace {
std::vector<float> swapAB(const std::vector<float>& a, int A, int B, size_t inner) {            // [A][B][inner] -> [B][A][inner]
    std::vector<float> o(a.size());
    for (int i = 0; i < A; i++) for (int j = 0; j < B; j++) memcpy(&o[((size_t)j * A + i) * inner], &a[((size_t)i * B + j) * inner], inner * 4);
    return o;
}
void gaussian(std::vector<float>& v, size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed); std::normal_distribution<float> nd(0.f, 1.f);
    v.resize(n); for (auto& x : v) x = nd(rng);
}

// Euler sampler of ComfyUI (flow matching: denoised = x - v * sigma) with a per-frame denoise mask for the video and the clean latent_image outside it
bool eulerStage(LtxDit& dit, LtxDitInput base, std::vector<float>& vid, std::vector<float>& aud, const std::vector<float>& vLI, const std::vector<float>& aLI,
                const std::vector<float>& mask, const std::vector<float>& sigmas, uint64_t seed, const std::function<void(int, int, int)>& step, const std::function<bool()>& stop, std::string& err) {
    const int F = base.F, hw = base.h * base.w;
    std::vector<float> nv, na; gaussian(nv, vLI.size(), seed); gaussian(na, aLI.size(), seed + 1);
    const float s0 = sigmas[0];
    vid.resize(vLI.size()); aud.resize(aLI.size());
    for (size_t i = 0; i < vid.size(); i++) vid[i] = s0 * nv[i] + (1.f - s0) * vLI[i];
    for (size_t i = 0; i < aud.size(); i++) aud[i] = s0 * na[i] + (1.f - s0) * aLI[i];
    const int steps = (int)sigmas.size() - 1;
    std::vector<float> xin, vo, ao;
    for (int i = 0; i < steps; i++) {
        if (stop && stop()) { err = "cancelled"; return false; }
        const float s = sigmas[i], sn = sigmas[i + 1];
        xin = vid;
        for (int c = 0; c < 128; c++) for (int f = 0; f < F; f++) if (mask[f] < 1.f) for (int p = 0; p < hw; p++) {
            const size_t k = ((size_t)c * F + f) * hw + p; xin[k] = vid[k] * mask[f] + vLI[k] * (1.f - mask[f]);
        }
        base.video = xin; base.audio = aud; base.audioSigma = s;
        base.videoSigma.resize(F); for (int f = 0; f < F; f++) base.videoSigma[f] = mask[f] * s;
        if (step) step(i, steps, -1);
        if (!dit.forward(base, vo, ao, [&](int b) { if (step) step(i, steps, b); }, err)) return false;
        for (int c = 0; c < 128; c++) for (int f = 0; f < F; f++) for (int p = 0; p < hw; p++) {
            const size_t k = ((size_t)c * F + f) * hw + p;
            float den = xin[k] - vo[k] * s; den = den * mask[f] + vLI[k] * (1.f - mask[f]);
            vid[k] = sn == 0.f ? den : vid[k] + (vid[k] - den) / s * (sn - s);
        }
        for (size_t k = 0; k < aud.size(); k++) { const float den = aud[k] - ao[k] * s; aud[k] = sn == 0.f ? den : aud[k] + (aud[k] - den) / s * (sn - s); }
    }
    return true;
}

// encode the start image at the latent's pixel size -> normalized latent frame [128][h][w]
bool encodeStart(const LtxVae& vae, const ImageRgb& img, int lh, int lw, std::vector<float>& frame, std::string& err) {
    // like the workflow (ResizeImageMaskNode, "center"): keep the aspect ratio, crop the overflow
    ImageRgb crop = img; const double want = (double)lw / lh, have = (double)img.w / img.h;
    if (std::fabs(want - have) > 1e-3) {
        int cw = img.w, ch = img.h; if (have > want) cw = (int)std::lround(img.h * want); else ch = (int)std::lround(img.w / want);
        const int x0 = (img.w - cw) / 2, y0 = (img.h - ch) / 2; crop.w = cw; crop.h = ch; crop.px.resize((size_t)cw * ch * 3);
        for (int y = 0; y < ch; y++) memcpy(&crop.px[(size_t)y * cw * 3], &img.px[((size_t)(y0 + y) * img.w + x0) * 3], (size_t)cw * 3);
    }
    std::vector<float> px; resizeToFloat(crop, lw * 32, lh * 32, px);
    Vol v; v.alloc(1, 3, lh * 32, lw * 32);
    for (int y = 0; y < v.H; y++) for (int x = 0; x < v.W; x++) for (int c = 0; c < 3; c++) v.at(0, c, y, x) = px[((size_t)y * v.W + x) * 3 + c] * 2.f - 1.f;
    Vol lat; if (!const_cast<LtxVae&>(vae).encode(v, lat, err)) return false;
    frame = lat.d;                                           // [1][128][h][w]
    return true;
}
}  // namespace

bool LtxModels::open(const LtxFiles& f, float strength, std::string& err) {
    if (f.checkpoint == files.checkpoint && f.upscaler == files.upscaler && f.lora == files.lora && strength == loraStrength && !files.checkpoint.empty()) return true;
    release();
    if (!ck.open(f.checkpoint, err) || !up.open(f.upscaler, err)) { release(); return false; }
    if (!f.lora.empty() && strength > 0) { lora.strength = strength; if (!lora.open(f.lora, err)) { release(); return false; } ck.lora = &lora; }
    ck.cacheF16 = true; files = f; loraStrength = strength;
    return true;
}
void LtxModels::release() { ck.f16cache.clear(); ck.lora = nullptr; ck = LtxWeights(); up = LtxWeights(); lora = LtxLora(); files = LtxFiles(); loraStrength = -1.f; }

bool ltxGenerate(LtxModels& models, const LtxGenParams& p, LtxResult& out, const std::function<void(float, const std::string&)>& progress, std::string& err) {
    auto say = [&](float f, const std::string& s) { if (progress) progress(f, s); };
    if (p.width % 64 || p.height % 64) { err = "width and height must be multiples of 64"; return false; }
    LtxWeights& ck = models.ck; LtxWeights& up = models.up; const LtxFiles& files = models.files;
    const int length = (int)std::lround(p.seconds * p.fps) + 1, F = (length - 1) / 8 + 1;
    const int Ta = (int)std::ceil((float)length / p.fps * 25.f);
    const int lh = p.height / 64, lw = p.width / 64;                  // stage 1 latent size (half resolution / 32)

    // ---- text
    say(0.0f, "encoding the prompt");
    LtxDitInput base; base.frameRate = (float)p.fps;
    {   LtxTextEncoder te; if (!te.open(files.gemma, files.checkpoint, err)) return false;
        const std::vector<int> ids = te.tokenize(p.prompt); const int n = (int)ids.size();
        std::vector<std::vector<float>> hid;
        if (!te.gemmaHidden(ids, hid, [&](int l) { say(0.0f + 0.05f * l / 48.f, "text encoder: layer " + std::to_string(l + 1) + "/48"); }, err)) return false;
        std::vector<float> pv, pa; if (!te.project(hid, n, pv, pa, err)) return false;
        if (!te.connector(true, pv, n, base.vctx, base.Lv, err) || !te.connector(false, pa, n, base.actx, base.La, err)) return false;
    }
    // ---- start image and shared pieces
    LtxVae vae; vae.ckpt = &ck;
    std::vector<float> start1, start2;
    if (p.startImage) { say(0.05f, "encoding the start image"); if (!encodeStart(vae, *p.startImage, lh, lw, start1, err)) return false; }
    LtxDit dit; dit.ckpt = &ck; dit.blocks = p.ditBlocks;

    // ---- stage 1
    base.F = F; base.h = lh; base.w = lw; base.T = Ta;
    std::vector<float> vLI((size_t)128 * F * lh * lw, 0.f), aLI((size_t)8 * Ta * 16, 0.f), mask(F, 1.f), vid, aud;
    if (p.startImage) {
        for (int c = 0; c < 128; c++) for (size_t q = 0; q < (size_t)lh * lw; q++) vLI[((size_t)c * F + 0) * lh * lw + q] = start1[(size_t)c * lh * lw + q];
        mask[0] = 1.f - p.stage1Strength;
    }
    const std::vector<float> sig1 = {1.0f, 0.99375f, 0.9875f, 0.98125f, 0.975f, 0.909375f, 0.725f, 0.421875f, 0.0f};
    auto stepCb = [&](float lo, float hi, const char* name) { return [=, &say](int i, int n, int b) { say(lo + (hi - lo) * ((i + std::max(b, 0) / 48.f) / n), std::string(name) + ": step " + std::to_string(i + 1) + "/" + std::to_string(n) + (b >= 0 ? ", block " + std::to_string(b + 1) : "")); }; };
    if (!eulerStage(dit, base, vid, aud, vLI, aLI, mask, sig1, (uint64_t)p.seed, stepCb(0.1f, 0.55f, "stage 1"), p.cancel, err)) return false;

    // ---- upscale and stage 2
    int fh = lh, fw = lw;
    if (p.upscale) {
        say(0.55f, "upscaling the latent");
        LtxUpscaler u; u.model = &up; u.ckpt = &ck;
        Vol in; in.T = F; in.C = 128; in.H = lh; in.W = lw; in.d = swapAB(vid, 128, F, (size_t)lh * lw);
        Vol big; if (!u.run(in, big, err)) return false;
        fh = lh * 2; fw = lw * 2;
        vid = swapAB(big.d, F, 128, (size_t)fh * fw);
        std::vector<float> vLI2 = vid, mask2(F, 1.f);
        if (p.startImage) {
            if (!encodeStart(vae, *p.startImage, fh, fw, start2, err)) return false;
            for (int c = 0; c < 128; c++) for (size_t q = 0; q < (size_t)fh * fw; q++) vLI2[((size_t)c * F + 0) * fh * fw + q] = start2[(size_t)c * fh * fw + q];
            mask2[0] = 1.f - p.stage2Strength;
        }
        base.h = fh; base.w = fw;
        const std::vector<float> sig2 = {0.85f, 0.725f, 0.421875f, 0.0f};
        std::vector<float> v2, a2;
        if (!eulerStage(dit, base, v2, a2, vLI2, aud, mask2, sig2, 42, stepCb(0.6f, 0.9f, "stage 2"), p.cancel, err)) return false;
        vid = std::move(v2); aud = std::move(a2);
    }

    // ---- decode
    say(0.92f, "decoding the video");
    Vol lat; lat.T = F; lat.C = 128; lat.H = fh; lat.W = fw; lat.d = swapAB(vid, 128, F, (size_t)fh * fw);
    if (!vae.decode(lat, out.frames, [&](const std::string& s) { say(0.95f, "video decoder: " + s); }, err)) return false;
    out.audioLatent = aud; out.audioFrames = Ta;
    if (p.decodeAudio) {
        say(0.97f, "decoding the sound");
        LtxAudio au; au.ckpt = &ck;
        if (!au.decode(aud, Ta, out.wave, out.waveSamples, nullptr, err)) return false;
    }
    say(1.0f, "done");
    return true;
}

// ---- test command: renders frames as PNG files
int runLtxGenTest(const std::vector<std::string>& a) {
    if (a.size() < 7) { fprintf(stderr, "usage: ltx-gen <out dir> \"<prompt>\" W H seconds fps seed [image|-] [blocks] [upscale 0|1]\n"); return 2; }
    const std::string M = "c:\\workenv\\ComfyUI-Easy-Install\\ComfyUI-Easy-Install\\ComfyUI\\models\\";
    LtxFiles f; f.checkpoint = M + "checkpoints\\ltx-2.3-22b-dev-fp8.safetensors"; f.gemma = M + "text_encoders\\gemma_3_12B_it_fp4_mixed.safetensors";
    f.upscaler = M + "latent_upscale_models\\ltx-2.3-spatial-upscaler-x2-1.1.safetensors"; f.lora = M + "loras\\ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors";
    LtxGenParams p; p.prompt = a[1]; p.width = atoi(a[2].c_str()); p.height = atoi(a[3].c_str()); p.seconds = (float)atof(a[4].c_str()); p.fps = atoi(a[5].c_str()); p.seed = atoi(a[6].c_str());
    ImageRgb img; std::string err;
    if (a.size() > 7 && a[7] != "-") { if (!loadImage(a[7], img, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; } p.startImage = &img; }
    if (a.size() > 8) p.ditBlocks = atoi(a[8].c_str());
    if (a.size() > 9) p.upscale = atoi(a[9].c_str()) != 0;
    sys::makeDirs(a[0]);
    LtxModels models; if (!models.open(f, p.loraStrength, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    LtxResult r; auto t0 = std::chrono::steady_clock::now();
    float last = -1;
    if (!ltxGenerate(models, p, r, [&](float fr, const std::string& s) { if (fr - last > 0.002f || fr >= 1.f) { last = fr; fprintf(stderr, "[%5.1f%%] %5.0fs %s\n", fr * 100.f, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), s.c_str()); } }, err)) { fprintf(stderr, "failed: %s\n", err.c_str()); return 1; }
    for (int t = 0; t < r.frames.T; t += std::max(1, r.frames.T / 8)) {
        std::vector<uint8_t> px((size_t)r.frames.W * r.frames.H * 3);
        for (int y = 0; y < r.frames.H; y++) for (int x = 0; x < r.frames.W; x++) for (int c = 0; c < 3; c++)
            px[((size_t)y * r.frames.W + x) * 3 + c] = (uint8_t)std::clamp((r.frames.at(t, c, y, x) + 1.f) * 127.5f, 0.f, 255.f);
        writePngRgb8(a[0] + "\\frame_" + std::to_string(t) + ".png", px.data(), r.frames.W, r.frames.H);
    }
    {   std::vector<std::vector<uint8_t>> store(r.frames.T); std::vector<const uint8_t*> ptrs;
        for (int t = 0; t < r.frames.T; t++) { store[t].resize((size_t)r.frames.W * r.frames.H * 3);
            for (int y = 0; y < r.frames.H; y++) for (int x = 0; x < r.frames.W; x++) for (int c = 0; c < 3; c++) store[t][((size_t)y * r.frames.W + x) * 3 + c] = (uint8_t)std::clamp((r.frames.at(t, c, y, x) + 1.f) * 127.5f, 0.f, 255.f);
            ptrs.push_back(store[t].data()); }
        std::vector<float> inter((size_t)r.waveSamples * 2); for (int i = 0; i < r.waveSamples; i++) { inter[(size_t)i * 2] = r.wave[i]; inter[(size_t)i * 2 + 1] = r.wave[(size_t)r.waveSamples + i]; }
        if (!writeMp4(a[0] + "\\clip.mp4", ptrs, r.frames.W, r.frames.H, p.fps, inter, 2, LtxAudio::kSampleRate, err)) { fprintf(stderr, "mp4: %s\n", err.c_str()); return 1; }
        std::vector<uint8_t> rgb; int w = 0, h = 0; double secs = 0; bool aud = false;
        if (!readMp4Frame(a[0] + "\\clip.mp4", r.frames.T / 2, rgb, w, h, secs, aud, err)) { fprintf(stderr, "readback: %s\n", err.c_str()); return 1; }
        writePngRgb8(a[0] + "\\readback.png", rgb.data(), w, h);
        printf("clip.mp4: %dx%d, %.2f s, audio stream: %s\n", w, h, secs, aud ? "yes" : "no");
    }
    printf("%d frames %dx%d, audio latent %d frames\n", r.frames.T, r.frames.W, r.frames.H, r.audioFrames);
    return 0;
}


