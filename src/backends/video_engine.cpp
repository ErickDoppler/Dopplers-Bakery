// Text/Image to video: LTX-2.3 (audio + video DiT, Gemma 3 text encoder, video / audio VAE, vocoder, x2 latent upscaler) on ggml.
// The generator itself is in src/ltx; this file adapts it to the job queue (parameters, progress, cancel, output file).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include "backends/engines.h"
#include "core/settings.h"
#include "gpu/compute.h"
#include "ltx/ltx_audio.h"
#include "ltx/ltx_pipeline.h"
#include "util/crashlog.h"
#include "util/image_io.h"
#include "util/mp4.h"
#include "util/sys.h"

namespace {
// ---- optional prompt expansion with the chat model (scene, camera, lighting, sound: what LTX-2.3 prompts need)
struct CaptureCtl : JobCtl {
    JobCtl& up; std::string text;
    explicit CaptureCtl(JobCtl& u) : up(u) {}
    void progress(float f, const std::string& s) override { up.progress(0.01f + 0.01f * f, "expanding the prompt: " + s); }
    void addOutput(const std::string&) override {}
    void setText(const std::string& t) override { text = t; }
    bool checkpoint() override { return up.checkpoint(); }
};
const char* kVideoSystemPrompt =
    "You write prompts for a text-to-video model that also generates sound. Rewrite the user idea as ONE detailed paragraph of 4 to 7 sentences, "
    "plain text, present tense. Describe, in the order they happen: the subject and what it does over time, the setting, the camera (shot size, angle, "
    "movement), the lighting and colors, the mood and fine visual details (textures, weather, particles, reflections), and the sounds (ambience, effects, "
    "music, and any speech in quotes). Keep everything the user asked for and do not contradict it. No titles, lists, comments or quotation marks around "
    "the answer: output only the prompt.";
// 0 = expanded, 1 = skipped (note says why), 2 = cancelled
int expandVideoPrompt(const Json& p, const Job& job, const std::string& prompt, double seconds, JobCtl& ctl, std::string& out, std::string& note) {
    std::string model = p.str("llm");
    if (model.size() < 5) model.clear(); else { std::string ext = model.substr(model.size() - 5); for (auto& c : ext) c = (char)tolower((unsigned char)c); if (ext != ".gguf") model.clear(); }
    if (model.empty()) model = Settings::get().resolvedSelection(Kind::LLM, "model");
    if (model.empty()) { note = "prompt not expanded: no GGUF chat model selected (LLM tab or Model settings)"; return 1; }
    Job j; j.id = job.id; j.kind = Kind::LLM;
    Json q = Json::object();
    q["model"] = model;
    q["prompt"] = std::string(kVideoSystemPrompt) + "\n\nIdea: " + prompt + "\nVideo length: " + std::to_string((int)std::lround(seconds)) + " seconds.\nPrompt:";
    q["max_tokens"] = 400; q["ctx"] = 4096; q["temperature"] = 0.7; q["top_k"] = 64; q["top_p"] = 0.95; q["min_p"] = 0.05; q["repeat_penalty"] = 1.05;
    q["seed"] = p.num("seed", -1); q["no_output"] = true;
    j.params = q;
    CaptureCtl cap(ctl); std::string err;
    if (!runLLM(j, cap, err)) { if (err == "cancelled") return 2; note = "prompt not expanded: " + err; return 1; }
    const size_t a = cap.text.find_first_not_of(" \r\n\t");
    if (a == std::string::npos) { note = "prompt not expanded: the chat model returned nothing"; return 1; }
    out = cap.text.substr(a, cap.text.find_last_not_of(" \r\n\t") - a + 1);
    return 0;
}

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

    std::string expandNote;                                                   // what happened to the "expand the prompt" option, shown when the job is done
    if (p.boolean("enhance", true)) {
        ctl.progress(0.01f, "expanding the prompt");
        std::string expanded, note;
        const int r = expandVideoPrompt(p, job, prompt, std::min(std::max(p.num("seconds", 4), 1.0), 20.0), ctl, expanded, note);
        if (r == 2) { err = "cancelled"; return false; }
        if (r == 0) { prompt = expanded; expandNote = " (prompt expanded)"; }
        else { ctl.progress(0.01f, note); expandNote = " (" + note + ")"; logLine("video: " + note); }
    } else expandNote = " (prompt not expanded: the option is off)";
    ctl.setText(prompt);                                                      // the prompt that actually conditioned the video (shown in the job details)

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
    if (!g_models.open(f, gp.loraStrength, err)) return false;
    // Auto = CUDA when its libraries are installed (measured about 30 % faster than Vulkan on an RTX 3080 Ti), otherwise Vulkan
    std::string want = p.str("backend", "auto");
    for (auto& c : want) c = (char)tolower((unsigned char)c);
    std::vector<std::string> order;
    if (want == "auto" || want.empty()) { if (Compute::hasCuda()) order.push_back("cuda"); order.push_back("auto"); } else order.push_back(want);
    LtxResult r; bool ok = false;
    for (size_t i = 0; i < order.size() && !ok; i++) {
        if (!Compute::get().use(order[i], 0, cerr)) { err = cerr; if (i + 1 < order.size()) continue; return false; }
        err.clear();
        ok = ltxGenerate(g_models, gp, r, [&](float fr, const std::string& s) { ctl.progress(fr, s); }, err);
        if (!ok) {
            if (err == "cancelled" || i + 1 >= order.size()) return false;
            logLine("video on " + order[i] + " failed (" + err + "): trying again on the other backend");           // a CUDA problem: Vulkan still works
            ctl.progress(0.0f, "CUDA failed - continuing on Vulkan");
        }
    }

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
    ctl.progress(1.0f, std::string(done) + expandNote);
    return true;
}

// `DopplersBakery video-expand <model.gguf> "<prompt>"`: what the "expand the prompt" option produces (a check without the video model)
int runVideoExpandTest(const std::vector<std::string>& a) {
    if (a.size() < 2) { fprintf(stderr, "usage: video-expand <model.gguf> \"<prompt>\"\n"); return 2; }
    struct Quiet : JobCtl {
        std::string text;
        void progress(float, const std::string&) override {}
        void addOutput(const std::string&) override {}
        void setText(const std::string& t) override { text = t; }
        bool checkpoint() override { return true; }
    } ctl;
    Json p = Json::object(); p["llm"] = a[0]; p["seed"] = 1;
    Job job; job.id = 0;
    std::string out, note;
    const int r = expandVideoPrompt(p, job, a[1], 4.0, ctl, out, note);
    printf("%s\n", r == 0 ? out.c_str() : ("NOT EXPANDED: " + note).c_str());
    return r == 0 ? 0 : 1;
}




