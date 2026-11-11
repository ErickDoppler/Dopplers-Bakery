// Text to speech: Qwen3-TTS 12Hz (talker + code predictor + speech decoder) on ggml.
#include <algorithm>
#include <chrono>
#include <random>
#include <thread>
#include "backends/engines.h"
#include "gpu/compute.h"
#include "tts/codec_encoder.h"
#include "tts/speaker_encoder.h"
#include "tts/speech_decoder.h"
#include "tts/talker.h"
#include "util/json.h"
#include "util/sys.h"
#include "util/mp3.h"
#include "util/wav.h"

namespace {
struct TtsEngines {
    std::string dir;
    QwenTalker talker;
    SpeechDecoder decoder;
    SpeakerEncoder spk;
    std::string spkDir;
    CodecEncoder codec;
    std::string codecDir;
} g;

// Reference audio -> 16 codec codes per frame (Mimi-based codec encoder).
bool encodeReferenceCodes(const std::string& dir, const std::vector<float>& wav, std::vector<int>& out, std::string& err) {
    if (const char* f = getenv("BAKERY_REF_CODES")) {        // development hook: codes dumped from ComfyUI
        std::string d;
        if (sys::readFile(f, d) && d.size() % 64 == 0 && !d.empty()) { out.assign((const int*)d.data(), (const int*)(d.data() + d.size())); return true; }
    }
    if (g.codecDir != dir || !g.codec.loaded()) {
        if (!g.codec.load(sys::join(sys::join(dir, "speech_tokenizer"), "model.safetensors"), err)) return false;
        g.codecDir = dir;
    }
    int frames = 0;
    return g.codec.encode(wav, out, frames, err);
}
double secs(std::chrono::steady_clock::time_point a) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count(); }
}  // namespace

bool runTTS(const Job& job, JobCtl& ctl, std::string& err) {
    const Json& p = job.params;
    std::string dir = p.str("model");
    if (dir.empty()) { err = "No TTS model folder selected (open [Settings] on this tab)."; return false; }
    std::string text = p.str("text");
    size_t a = text.find_first_not_of(" \r\n\t");
    if (a == std::string::npos) { err = "Text is empty."; return false; }
    text = text.substr(a, text.find_last_not_of(" \r\n\t") - a + 1);

    // model flavour
    std::string cfgText, type;
    if (sys::readFile(sys::join(dir, "config.json"), cfgText)) { bool ok; Json c = Json::parse(cfgText, &ok); if (ok) type = c.str("tts_model_type"); }
    std::string instruct = p.str("instruct"), speaker = p.str("speaker", "ryan");
    // voice cloning (Base model): reference voice -> speaker embedding (+ codec frames for the in-context mode)
    std::vector<float> spkEmbed;
    std::vector<int> refCodes;
    std::string refText = p.str("ref_text"), note;
    if (type == "base") {
        std::string refPath = p.str("ref_audio");
        if (refPath.empty()) { err = "The Base model clones a voice: pick a reference .wav in 'Reference voice to clone' (3-20 seconds of clean speech works best)."; return false; }
        std::vector<float> wav; int sr = 0;
        ctl.progress(0.0f, "reading reference voice");
        if (!readWavMono(refPath, wav, sr, err)) return false;
        wav = resampleSinc(wav, sr, SpeechDecoder::kSampleRate);
        const size_t maxSamples = (size_t)SpeechDecoder::kSampleRate * 30;
        if (wav.size() > maxSamples) wav.resize(maxSamples);                    // the first 30 s are plenty
        if (wav.size() < (size_t)SpeechDecoder::kSampleRate / 2) { err = "The reference audio is shorter than half a second."; return false; }
        if (g.spkDir != dir || !g.spk.loaded()) {
            ctl.progress(0.01f, "loading speaker encoder");
            if (!g.spk.load(sys::join(dir, "model.safetensors"), err)) return false;
            g.spkDir = dir;
        }
        if (!g.spk.embed(wav, spkEmbed, err)) return false;
        if (!refText.empty()) {
            if (!encodeReferenceCodes(dir, wav, refCodes, err)) {               // in-context mode needs the codec encoder
                note = " (voice-only mode: " + err + ")"; err.clear(); refCodes.clear();
            }
        }
    }
    if (type == "voice_design") {
        speaker.clear();
        if (instruct.empty()) { err = "The VoiceDesign model needs a voice description in 'Style instruction'."; return false; }
    }

    const int threads = std::max(4u, std::thread::hardware_concurrency());
    std::string e2;
    if (!Compute::get().init(threads, e2)) { err = e2; return false; }
    auto t0 = std::chrono::steady_clock::now();
    if (!g.talker.loaded() || g.dir != dir) {
        ctl.progress(0.0f, "loading speech model");
        if (!ctl.checkpoint()) { err = "cancelled"; return false; }
        if (!g.talker.load(dir, threads, err, [&](float f) { ctl.progress(0.05f * f, "loading speech model"); })) return false;
        if (!g.decoder.load(sys::join(sys::join(dir, "speech_tokenizer"), "model.safetensors"), err)) return false;
        g.dir = dir;
    }
    const double loadSec = secs(t0);

    QwenTalker::Params q;
    q.text = text; q.speaker = speaker; q.language = p.str("language", "auto"); q.instruct = instruct;
    q.temperature = (float)p.num("temperature", 0.9); q.subTemperature = q.temperature;
    q.topK = std::max(1, p.integer("top_k", 50)); q.subTopK = q.topK;
    q.sample = q.temperature > 0.0f;
    long long seed = (long long)p.num("seed", -1);
    if (seed < 0) seed = (long long)(std::random_device{}() & 0x7fffffff);
    q.seed = (unsigned long long)seed;
    q.maxFrames = 4000;
    if (type == "base") {
        q.speakerEmbed = &spkEmbed;
        if (!refCodes.empty()) { q.refCodes = &refCodes; q.refText = refText; }
    }

    const double estFrames = std::max<double>(25.0, (double)text.size() * 0.9);
    std::vector<int> codes; int frames = 0;
    auto tGen = std::chrono::steady_clock::now();
    bool ok = g.talker.generate(q, codes, frames, err, [&](int f) {
        char b[96]; snprintf(b, sizeof b, "generating speech: %.1f s of audio", f / 12.5);
        ctl.progress(0.05f + 0.85f * (float)std::min(0.97, f / estFrames), b);
        return ctl.checkpoint();
    });
    if (!ok) return false;
    if (frames == 0) { err = "The model produced no audio for this text."; return false; }
    const double genSec = secs(tGen);

    ctl.progress(0.92f, "decoding audio");
    std::vector<float> wav;
    if (!g.decoder.decode(codes, frames, wav, err)) return false;

    std::string file;
    if (p.str("format", "wav") == "mp3") {
        std::string merr;
        file = outputPath("tts", job.id, ".mp3");
        if (!writeMp3(file, resampleSinc(wav, SpeechDecoder::kSampleRate, 44100), 1, 44100, 128, merr)) { note += " (MP3 failed: " + merr + "; saved as WAV)"; file.clear(); }
    }
    if (file.empty()) {
        file = outputPath("tts", job.id, ".wav");
        if (!writeWavMono16(file, wav, SpeechDecoder::kSampleRate)) { err = "cannot write " + file; return false; }
    }
    ctl.addOutput(file);
    { char b[32]; snprintf(b, sizeof b, "%.1f s", wav.size() / (double)SpeechDecoder::kSampleRate); ctl.info(b); }
    char done[200];
    snprintf(done, sizeof done, "%.1f s of audio in %.1f s (load %.1f s) on %s, seed %lld%s", wav.size() / (double)SpeechDecoder::kSampleRate, secs(t0), loadSec, Compute::get().description().c_str(), seed, note.c_str());
    (void)genSec;
    ctl.progress(1.0f, done);
    return true;
}
