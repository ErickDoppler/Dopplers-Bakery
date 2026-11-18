// Text/Image to video: LTX-2.3 (audio + video DiT, Gemma 3 text encoder, video / audio VAE, vocoder, x2 latent upscaler) on ggml.
// The generator itself is in src/ltx; this file adapts it to the job queue (parameters, progress, cancel, output file).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include "backends/engines.h"
#include "gpu/compute.h"
#include "ltx/ltx_audio.h"
#include "ltx/ltx_pipeline.h"
#include "util/image_io.h"
#include "util/mp4.h"
#include "util/sys.h"

namespace {
LtxModels g_models;                       // stays open between jobs: the merged weights are expensive to rebuild
int roundTo64(double v, int lo, int hi) { return std::min(hi, std::max(lo, (int)std::lround(v / 64.0) * 64)); }
}

bool runVideo(const Job& job, JobCtl& ctl, std::string& err) {
    const Json& p = job.params;
    LtxFiles f; f.checkpoint = p.str("model"); f.gemma = p.str("clip"); f.upscaler = p.str("upscaler"); f.lora = p.str("lora");
    if (f.checkpoint.empty() || f.gemma.empty() || f.upscaler.empty() || f.lora.empty()) { err = "Select the LTX-2.3 checkpoint, the Gemma text encoder, the x2 upscaler and the distilled LoRA in [Settings]."; return false; }
    std::string prompt = p.str("prompt");
    const size_t a = prompt.find_first_not_of(" \r\n\t");
    if (a == std::string::npos) { err = "Prompt is empty."; return false; }
    prompt = prompt.substr(a, prompt.find_last_not_of(" \r\n\t") - a + 1);

    LtxGenParams gp; gp.prompt = prompt;
    gp.width = roundTo64(p.num("width", 768), 128, 1920); gp.height = roundTo64(p.num("height", 512), 128, 1920);
    gp.fps = std::min(std::max(p.integer("fps", 24), 8), 50);
    gp.seconds = (float)std::min(std::max(p.num("seconds", 4), 1.0), 20.0);
    long long seed = (long long)p.num("seed", -1);
    if (seed < 0) seed = (long long)(std::chrono::steady_clock::now().time_since_epoch().count() & 0x7fffffff);
    gp.seed = (int)seed; gp.upscale = p.boolean("upscale", true); gp.decodeAudio = p.boolean("audio", true);
    ImageRgb img; std::string imgPath = p.str("init_image");
    if (!imgPath.empty()) { if (!loadImage(imgPath, img, err)) return false; gp.startImage = &img; }
    gp.cancel = [&] { return !ctl.checkpoint(); };
    { char b[64]; snprintf(b, sizeof b, "%dx%d, %.0f s", gp.upscale ? gp.width : gp.width / 2, gp.upscale ? gp.height : gp.height / 2, (double)gp.seconds); ctl.info(b); }

    const auto t0 = std::chrono::steady_clock::now();
    ctl.progress(0.0f, "opening the model files");
    std::string cerr;
    if (!Compute::get().use("auto", 0, cerr)) { err = cerr; return false; }
    if (!g_models.open(f, gp.loraStrength, err)) return false;
    LtxResult r;
    if (!ltxGenerate(g_models, gp, r, [&](float fr, const std::string& s) { ctl.progress(fr, s); }, err)) return false;

    ctl.progress(0.98f, "writing the MP4");
    std::vector<std::vector<uint8_t>> store(r.frames.T); std::vector<const uint8_t*> ptrs;
    for (int t = 0; t < r.frames.T; t++) {
        store[t].resize((size_t)r.frames.W * r.frames.H * 3);
        for (int y = 0; y < r.frames.H; y++) for (int x = 0; x < r.frames.W; x++) for (int c = 0; c < 3; c++)
            store[t][((size_t)y * r.frames.W + x) * 3 + c] = (uint8_t)std::clamp((r.frames.at(t, c, y, x) + 1.f) * 127.5f, 0.f, 255.f);
        ptrs.push_back(store[t].data());
    }
    std::vector<float> inter;
    if (!r.wave.empty()) { inter.resize((size_t)r.waveSamples * 2); for (int i = 0; i < r.waveSamples; i++) { inter[(size_t)i * 2] = r.wave[i]; inter[(size_t)i * 2 + 1] = r.wave[(size_t)r.waveSamples + i]; } }
    const std::string file = outputPath("video", job.id, ".mp4");
    if (!writeMp4(file, ptrs, r.frames.W, r.frames.H, gp.fps, inter, 2, LtxAudio::kSampleRate, err)) return false;
    ctl.addOutput(file);
    char done[200];
    snprintf(done, sizeof done, "%dx%d, %d frames in %.0f s on %s, seed %lld", r.frames.W, r.frames.H, r.frames.T, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), Compute::get().description().c_str(), seed);
    ctl.progress(1.0f, done);
    return true;
}
