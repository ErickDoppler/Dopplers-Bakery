// Text to audio: Stable Audio 3 (T5Gemma text encoder -> DiT with the LCM sampler -> audio VAE) on ggml.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>
#include "audio/sa_dit.h"
#include "audio/sa_vae.h"
#include "audio/t5gemma.h"
#include <windows.h>
#include "backends/engines.h"
#include "core/settings.h"
#include "gpu/compute.h"
#include "krea2/sampling.h"
#include "util/sys.h"
#include "util/mp3.h"
#include "util/wav.h"

namespace {
struct AudioEngines {
    std::string tePath, ckptPath;
    T5GemmaEncoder te;
    SaDiT dit;
    SaVae vae;
} g;
std::string lowerStr(std::string s) { for (auto& c : s) c = (char)tolower((unsigned char)c); return s; }
double secs(std::chrono::steady_clock::time_point a) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count(); }

// ---- optional prompt expansion with an LLM (the workflow's TextGenerate node, with its per-mode system prompts)
struct CaptureCtl : JobCtl {
    JobCtl& up; std::string text;
    explicit CaptureCtl(JobCtl& u) : up(u) {}
    void progress(float f, const std::string& s) override { up.progress(0.11f + 0.04f * f, "expanding prompt: " + s); }
    void addOutput(const std::string&) override {}
    void setText(const std::string& t) override { text = t; }
    bool checkpoint() override { return up.checkpoint(); }
};

std::string systemPrompt(const std::string& mode) {
    HRSRC r = FindResourceW(nullptr, MAKEINTRESOURCEW(104), RT_RCDATA);
    if (!r) return "";
    HGLOBAL h = LoadResource(nullptr, r);
    bool ok; Json j = Json::parse(std::string((const char*)LockResource(h), SizeofResource(nullptr, r)), &ok);
    if (!ok) return "";
    std::string s = j.str(mode);
    return s.empty() ? j.str("Music") : s;
}

// 0 = expanded, 1 = skipped (note says why), 2 = cancelled
int expandPrompt(const Json& p, const Job& job, const std::string& prompt, int lengthSec, JobCtl& ctl, std::string& out, std::string& note) {
    std::string model = p.str("llm");
    if (model.size() < 5 || lowerStr(model.substr(model.size() - 5)) != ".gguf") model = Settings::get().resolvedSelection(Kind::LLM, "model");
    if (model.empty()) { note = "prompt not expanded: no GGUF LLM selected (LLM tab or Model settings)"; return 1; }
    std::string sys = systemPrompt(p.str("mode", "Music"));
    if (sys.empty()) { note = "prompt not expanded: system prompts missing"; return 1; }
    Job j; j.id = job.id; j.kind = Kind::LLM;
    Json q = Json::object();
    q["model"] = model;
    q["prompt"] = sys + "\n\nInput: " + prompt + "\nTarget audio length: " + std::to_string(lengthSec) + " seconds.\nOutput:";
    q["max_tokens"] = 256; q["ctx"] = 6144; q["temperature"] = 0.7; q["top_k"] = 64; q["top_p"] = 0.95; q["min_p"] = 0.05; q["repeat_penalty"] = 1.05;
    q["seed"] = p.num("seed", -1); q["no_output"] = true;
    j.params = q;
    CaptureCtl cap(ctl);
    std::string err;
    if (!runLLM(j, cap, err)) { if (err == "cancelled") return 2; note = "prompt not expanded: " + err; return 1; }
    size_t a = cap.text.find_first_not_of(" \r\n\t");
    if (a == std::string::npos) { note = "prompt not expanded: the LLM returned nothing"; return 1; }
    out = cap.text.substr(a, cap.text.find_last_not_of(" \r\n\t") - a + 1);
    return 0;
}

// ModelSamplingDiscreteFlow(shift 2.0) sigma table and the "simple" scheduler
std::vector<float> simpleSigmas(int steps) {
    std::vector<float> table(1000);
    for (int i = 0; i < 1000; i++) { double t = (i + 1) / 1000.0; table[i] = (float)(2.0 * t / (1.0 + t)); }
    std::vector<float> s;
    double ss = 1000.0 / steps;
    for (int x = 0; x < steps; x++) s.push_back(table[999 - (int)(x * ss)]);
    s.push_back(0.0f);
    return s;
}
}  // namespace

bool runAudio(const Job& job, JobCtl& ctl, std::string& err) {
    const Json& p = job.params;
    std::string ckpt = p.str("model"), tePath = p.str("clip");
    if (ckpt.empty() || tePath.empty()) { err = "Select the audio checkpoint and the T5Gemma text encoder in [Settings]."; return false; }
    std::string prompt = p.str("prompt");
    size_t a = prompt.find_first_not_of(" \r\n\t");
    if (a == std::string::npos) { err = "Prompt is empty."; return false; }
    prompt = prompt.substr(a, prompt.find_last_not_of(" \r\n\t") - a + 1);

    const double seconds = std::min(std::max(p.num("length", 60), 1.0), 380.0);
    const int T = std::max(2, (int)std::lround(seconds * SaVae::kSampleRate / SaVae::kSamplesPerLatent));
    const float secondsTotal = (float)(int)(T / 10.7666);                     // what the conditioner is given (ComfyUI: int(T / 10.7666))
    int steps = std::min(std::max(p.integer("steps", 8), 1), 100);
    std::string sampler = p.str("sampler", "lcm");
    long long seed = (long long)p.num("seed", -1);
    if (seed < 0) seed = (long long)(std::random_device{}() & 0x7fffffff);
    const int threads = std::max(4u, std::thread::hardware_concurrency());

    std::string e2;
    if (!Compute::get().init(threads, e2)) { err = e2; return false; }
    auto t0 = std::chrono::steady_clock::now();
    if (!g.te.loaded() || g.tePath != tePath) {
        ctl.progress(0.0f, "loading text encoder");
        if (!ctl.checkpoint()) { err = "cancelled"; return false; }
        if (!g.te.load(tePath, err)) return false;
        g.tePath = tePath;
    }
    if (!g.dit.loaded() || !g.vae.loaded() || g.ckptPath != ckpt) {
        ctl.progress(0.04f, "loading audio model");
        if (!ctl.checkpoint()) { err = "cancelled"; return false; }
        g.dit.unload(); g.vae.unload();
        if (!g.dit.load(ckpt, err)) return false;
        ctl.progress(0.10f, "loading audio decoder");
        if (!g.vae.load(ckpt, err)) return false;
        g.ckptPath = ckpt;
    }
    const double loadSec = secs(t0);

    // ---- optional: let an LLM rewrite the prompt the way the workflow does (genre, instruments, BPM, length ...)
    std::string note;
    if (p.boolean("enhance", true)) {
        ctl.progress(0.11f, "expanding prompt");
        std::string expanded;
        int r = expandPrompt(p, job, prompt, (int)std::lround(seconds), ctl, expanded, note);
        if (r == 2) { err = "cancelled"; return false; }
        if (r == 0) { prompt = expanded; note = " (prompt expanded)"; } else note = " (" + note + ")";
    }
    ctl.setText(prompt);                                                       // the prompt that actually conditioned the audio

    // ---- conditioning
    ctl.progress(0.12f, "encoding prompt");
    std::vector<int> ids = g.te.tokenize(prompt);
    std::vector<float> text;
    if (!g.te.encode(ids, text, err)) return false;

    // ---- sampling
    std::vector<float> sigmas = simpleSigmas(steps);
    const int nSteps = (int)sigmas.size() - 1;
    k2::Rng rng((unsigned long long)seed);
    std::vector<float> x((size_t)T * 256), noise(x.size()), v;
    rng.normal(x);
    for (auto& e : x) e *= sigmas[0];                                          // CONST noise scaling
    auto tSample = std::chrono::steady_clock::now();
    for (int i = 0; i < nSteps; i++) {
        if (!ctl.checkpoint()) { err = "cancelled"; return false; }
        if (!g.dit.forward(x, T, sigmas[i], text, (int)ids.size(), secondsTotal, v, err)) return false;
        const float s = sigmas[i], sn = sigmas[i + 1];
        if (sampler == "euler") {
            for (size_t k = 0; k < x.size(); k++) { float den = x[k] - v[k] * s; x[k] = sn == 0 ? den : x[k] + (x[k] - den) / s * (sn - s); }
        } else {                                                               // LCM: jump to the denoised sample, re-noise to sigma_next
            if (sn > 0) rng.normal(noise);
            for (size_t k = 0; k < x.size(); k++) { float den = x[k] - v[k] * s; x[k] = sn > 0 ? sn * noise[k] + (1.0f - sn) * den : den; }
        }
        char b[96]; snprintf(b, sizeof b, "step %d/%d  (%.1f s/step)", i + 1, nSteps, secs(tSample) / (i + 1));
        ctl.progress(0.15f + 0.65f * (float)(i + 1) / nSteps, b);
    }

    // ---- decode and save
    if (!ctl.checkpoint()) { err = "cancelled"; return false; }
    ctl.progress(0.82f, "decoding audio");
    std::vector<float> pcm;
    if (!g.vae.decode(x, T, pcm, err)) return false;
    {   // same loudness handling as ComfyUI's VAEDecodeAudio: divide by 5*std when that exceeds 1
        double sum = 0, sq = 0; for (float f : pcm) { sum += f; sq += (double)f * f; }
        double mean = sum / pcm.size(), sd = std::sqrt(std::max(0.0, sq / pcm.size() - mean * mean)) * 5.0;
        if (sd > 1.0) for (auto& f : pcm) f = (float)(f / sd);
    }
    std::string file;
    if (p.str("format", "mp3") == "mp3") {                                      // like the workflow's SaveAudioMP3 (V0 ~ 256 kbps)
        std::string q = p.str("mp3_quality", "V0");
        int kbps = q == "128k" ? 128 : (q == "320k" ? 320 : 256), n = 0;
        std::string merr;
        file = outputPath("audio", job.id, ".mp3");
        if (!writeMp3(file, pcm, 2, SaVae::kSampleRate, kbps, merr)) { note += " (MP3 failed: " + merr + "; saved as WAV)"; file.clear(); }
        (void)n;
    }
    if (file.empty()) {
        file = outputPath("audio", job.id, ".wav");
        if (!writeWavPcm16(file, pcm, 2, SaVae::kSampleRate)) { err = "cannot write " + file; return false; }
    }
    ctl.addOutput(file);
    char done[200];
    snprintf(done, sizeof done, "%.1f s of audio in %.0f s (load %.0f s) on %s, seed %lld%s", pcm.size() / 2.0 / SaVae::kSampleRate, secs(t0), loadSec, Compute::get().description().c_str(), seed, note.c_str());
    ctl.progress(1.0f, done);
    return true;
}
