// Shared vocabulary: job kinds, model roles per operation tab.
#pragma once
#include <string>
#include <vector>

enum class Kind { LLM = 0, TTS, Image, Audio };
constexpr int kKindCount = 4;

inline const char* kindKey(Kind k) {
    switch (k) { case Kind::LLM: return "llm"; case Kind::TTS: return "tts"; case Kind::Image: return "image"; default: return "audio"; }
}
inline const char* kindTitle(Kind k) {
    switch (k) { case Kind::LLM: return "LLM"; case Kind::TTS: return "TTS"; case Kind::Image: return "Text to image"; default: return "Audio"; }
}
inline Kind kindFromKey(const std::string& s) {
    for (int i = 0; i < kKindCount; i++) if (s == kindKey((Kind)i)) return (Kind)i;
    return Kind::LLM;
}

// One model file (or folder) the user picks in the tab's [Settings] panel.
struct Role {
    const char* key;
    const char* label;
    const char* folders;   // '|' separated keywords matched against path components ("" = anywhere)
    const char* exts;      // '|' separated extensions incl. dot, or "dir" for a model folder
    bool required;
};

inline const std::vector<Role>& rolesOf(Kind k) {
    static const std::vector<Role> llm = {
        {"model", "Model (GGUF)", "", ".gguf", true},
        {"lora", "LoRA adapter (optional)", "lora", ".gguf", false},
    };
    static const std::vector<Role> tts = {
        {"model", "TTS model folder", "qwen-tts|tts", "dir", true},
    };
    static const std::vector<Role> img = {
        {"model", "Diffusion model", "diffusion_models|unet|checkpoints", ".safetensors|.gguf", true},
        {"clip", "Text encoder (CLIP)", "text_encoders|clip", ".safetensors|.gguf", true},
        {"vae", "VAE", "vae", ".safetensors", true},
        {"lora", "LoRA (optional)", "loras", ".safetensors", false},
    };
    static const std::vector<Role> aud = {
        {"model", "Audio checkpoint", "checkpoints", ".safetensors", true},
        {"clip", "Text encoder (T5Gemma)", "text_encoders|clip", ".safetensors", true},
        {"llm", "Prompt enhancer LLM, GGUF (optional; the LLM tab's model is used if empty)", "LLM|gguf", ".gguf", false},
    };
    switch (k) { case Kind::LLM: return llm; case Kind::TTS: return tts; case Kind::Image: return img; default: return aud; }
}
