#include "core/forms.h"
#include "core/settings.h"

const std::vector<Field>& fieldsOf(Kind k) {
    static const std::vector<Field> llm = {
        {"prompt", "Prompt", FT::Multi, 0, 0, 0, "", "", 6},
        {"system", "System prompt (optional)", FT::Multi, 0, 0, 0, "", "", 2},
        {"max_tokens", "Max new tokens", FT::Int, 512, 16, 8192, "", "", 0},
        {"ctx", "Context size", FT::Int, 4096, 512, 32768, "", "", 0},
        {"temperature", "Temperature (0 = greedy)", FT::Float, 0.7, 0, 2, "", "", 0},
        {"top_k", "Top-K", FT::Int, 40, 0, 1000, "", "", 0},
        {"top_p", "Top-P", FT::Float, 0.95, 0, 1, "", "", 0},
        {"min_p", "Min-P", FT::Float, 0.05, 0, 1, "", "", 0},
        {"repeat_penalty", "Repeat penalty", FT::Float, 1.05, 1, 2, "", "", 0},
        {"seed", "Seed (-1 = random)", FT::Int, -1, -1, 2147483647, "", "", 0},
    };
    static const std::vector<Field> tts = {
        {"text", "Text to speak", FT::Multi, 0, 0, 0, "", "", 7},
        {"instruct", "Style instruction / voice description (optional)", FT::Text, 0, 0, 0, "", "", 0},
        {"ref_audio", "Reference voice to clone - .wav file (Base model only)", FT::File, 0, 0, 0, "", "", 0},
        {"ref_text", "What the reference says (optional; better cloning with the Base model)", FT::Text, 0, 0, 0, "", "", 0},
        {"speaker", "Speaker", FT::Combo, 0, 0, 0, "Ryan", "Ryan|Aiden|Vivian|Serena|Uncle_Fu|Dylan|Eric|Ono_Anna|Sohee", 0},
        {"language", "Language", FT::Combo, 0, 0, 0, "Auto", "Auto|English|Chinese|Japanese|Korean|German|French|Russian|Portuguese|Spanish|Italian", 0},
        {"temperature", "Temperature", FT::Float, 0.9, 0.1, 2, "", "", 0},
        {"top_k", "Top-K", FT::Int, 50, 0, 1000, "", "", 0},
        {"seed", "Seed (-1 = random)", FT::Int, -1, -1, 2147483647, "", "", 0},
        {"format", "File format", FT::Combo, 0, 0, 0, "wav", "wav|mp3", 0},
    };
    static const std::vector<Field> img = {
        {"pre_prompt", "Pre prompt", FT::Multi, 0, 0, 0, "", "", 2},
        {"prompt", "Prompt", FT::Multi, 0, 0, 0, "", "", 5},
        {"negative", "Negative prompt", FT::Multi, 0, 0, 0, "", "", 3},
        {"init_image", "Input image for image-to-image (optional; its aspect ratio is kept, longest side = Width/Height)", FT::File, 0, 0, 0, "", "", 0},
        {"denoise", "Denoise strength for the input image (1 = ignore it, 0.5 = keep composition)", FT::Float, 0.6, 0.02, 1, "", "", 0},
        {"width", "Width", FT::Slider, 720, 50, 5000, "", "", 0},
        {"height", "Height", FT::Slider, 720, 50, 5000, "", "", 0},
        {"steps", "Steps", FT::Int, 10, 1, 150, "", "", 0},
        {"cfg", "CFG", FT::Float, 1, 0, 30, "", "", 0},
        {"sampler", "Sampler", FT::Combo, 0, 0, 0, "ER-SDE", "ER-SDE|Euler", 0},
        {"scheduler", "Scheduler", FT::Combo, 0, 0, 0, "beta", "beta|simple", 0},
        {"seed", "Seed (-1 = random)", FT::Int, -1, -1, 2147483647, "", "", 0},
        {"lora_strength", "LoRA strength", FT::Float, 0.9, -2, 2, "", "", 0},
    };
    static const std::vector<Field> aud = {
        {"prompt", "Prompt", FT::Multi, 0, 0, 0, "", "", 5},
        {"mode", "Kind of audio", FT::Combo, 0, 0, 0, "Music", "Music|Instrument|SFX|One-shot", 0},
        {"length", "Length (seconds)", FT::Slider, 60, 1, 300, "", "", 0},
        {"steps", "Steps", FT::Int, 8, 1, 100, "", "", 0},
        {"cfg", "CFG", FT::Float, 1, 0, 30, "", "", 0},
        {"sampler", "Sampler", FT::Combo, 0, 0, 0, "lcm", "lcm|euler", 0},
        {"scheduler", "Scheduler", FT::Combo, 0, 0, 0, "simple", "simple", 0},
        {"seed", "Seed (-1 = random)", FT::Int, -1, -1, 2147483647, "", "", 0},
        {"enhance", "Expand the prompt with an LLM first (needs a GGUF model, see Model settings)", FT::Check, 1, 0, 1, "", "", 0},
        {"format", "File format", FT::Combo, 0, 0, 0, "mp3", "mp3|wav", 0},
        {"mp3_quality", "MP3 quality (V0 = 256 kbps)", FT::Combo, 0, 0, 0, "V0", "V0|128k|320k", 0},
    };
    switch (k) { case Kind::LLM: return llm; case Kind::TTS: return tts; case Kind::Image: return img; default: return aud; }
}

Json defaultsOf(Kind k) {
    Json j = Json::object();
    for (auto& f : fieldsOf(k)) {
        switch (f.type) {
        case FT::Text: case FT::File: case FT::Multi: case FT::Combo: j[f.key] = std::string(f.sdef); break;
        case FT::Check: j[f.key] = f.def != 0; break;
        default: j[f.key] = f.def;
        }
    }
    return j;
}

std::string jobTitle(Kind k, const Json& p) {
    std::string s = k == Kind::TTS ? p.str("text") : p.str("prompt");
    for (auto& c : s) if (c == '\r' || c == '\n' || c == '\t') c = ' ';
    if (s.size() > 60) { s.resize(60); while (!s.empty() && (s.back() & 0xC0) == 0x80) s.pop_back(); s += "..."; }
    return s.empty() ? "(empty)" : s;
}

Json makeJobParams(Kind k, const Json& form) {
    Json p = defaultsOf(k);
    for (auto& kv : form.o) p[kv.first] = kv.second;
    for (auto& r : rolesOf(k)) p[r.key] = Settings::get().resolvedSelection(k, r.key);
    return p;
}
