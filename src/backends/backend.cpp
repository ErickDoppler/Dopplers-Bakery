// Dispatch from job kind to the engine that bakes it.
//
// Job params contract (all built by the UI form or the CLI, see ui/forms.cpp):
//   common   model paths already resolved to absolute: "model", "clip", "vae", "lora", "llm"
//   llm      prompt, system, max_tokens, temperature, top_k, top_p, min_p, repeat_penalty, seed, ctx, lora_scale
//   tts      text, speaker, language, instruct, temperature, top_k, seed
//   image    pre_prompt, prompt, negative, width, height, steps, cfg, sampler, scheduler, seed, lora_strength
//   audio    prompt, mode, length, steps, cfg, sampler, scheduler, seed, enhance, format
//   all but llm: "backend" ("auto" / "vulkan" / "cuda" / "cpu")
#include "backends/engines.h"
#include "gpu/compute.h"
#include "util/crashlog.h"

// A graph that does not fit into the GPU memory (small or shared-memory GPUs such as Intel integrated graphics) fails the job with
// "cannot allocate ... graph". Instead of giving up, run the same job once more on the CPU: slower, but it finishes.
static BackendFn withCpuRetry(BackendFn fn) {
    return [fn](const Job& job, JobCtl& ctl, std::string& err) -> bool {
        if (fn(job, ctl, err)) return true;
        if (err.find(" graph") == std::string::npos || !Compute::get().usingGpu()) return false;
        const std::string first = err;
        logLine("GPU memory is not enough (" + first + "): retrying the job on the CPU");
        ctl.progress(0.0f, "the GPU memory is not enough - continuing on the CPU (slower)");
        Job retry = job; retry.params["backend"] = "cpu";
        err.clear();
        if (fn(retry, ctl, err)) return true;
        err = (err.empty() ? std::string("failed") : err) + " (the GPU attempt failed first: " + first + ")";
        return false;
    };
}

BackendFn backendFor(Kind k) {
    switch (k) {
    case Kind::LLM: return runLLM;
    case Kind::TTS: return withCpuRetry(runTTS);
    case Kind::Image: return withCpuRetry(runImage);
    default: return withCpuRetry(runAudio);
    }
}
