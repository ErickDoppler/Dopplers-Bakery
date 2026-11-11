// Dispatch from job kind to the engine that bakes it.
//
// Job params contract (all built by the UI form or the CLI, see ui/forms.cpp):
//   common   model paths already resolved to absolute: "model", "clip", "vae", "lora", "llm"
//   llm      prompt, system, max_tokens, temperature, top_k, top_p, min_p, repeat_penalty, seed, ctx, lora_scale
//   tts      text, speaker, language, instruct, temperature, top_k, seed
//   image    pre_prompt, prompt, negative, width, height, steps, cfg, sampler, scheduler, seed, lora_strength
//   audio    prompt, mode, length, steps, cfg, sampler, scheduler, seed, enhance, format
#include "backends/engines.h"

BackendFn backendFor(Kind k) {
    switch (k) {
    case Kind::LLM: return runLLM;
    case Kind::TTS: return runTTS;
    case Kind::Image: return runImage;
    default: return runAudio;
    }
}
