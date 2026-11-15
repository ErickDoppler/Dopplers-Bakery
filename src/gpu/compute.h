// Backend selection for the ggml engines: CUDA or Vulkan when a GPU backend DLL is present, CPU otherwise.
// Weights stay in (pinned) host memory; the ggml scheduler streams them to the GPU per op, so models
// larger than VRAM still run. The backend can be switched while the app runs (engines drop their models first).
#pragma once
#include <functional>
#include <string>
#include <vector>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "util/crashlog.h"

struct Compute {
    static Compute& get();
    // Loads the backend DLLs next to the exe (once) and notes which GPU backends exist. Cheap; call at start-up.
    static void probe();
    static bool hasCuda();
    static bool hasVulkan();
    static bool onlyIntegratedGpu();                              // the only GPU is an integrated one (Intel / APU: shares the system RAM)
    static void setSafeMode(bool on);                             // after a crash inside a GPU job: "auto" means CPU
    static bool safeMode();
    // Longest image side the backend can render correctly: Vulkan renders noise above ~6500 DiT tokens (about 1280 px square), CUDA and CPU do not.
    static int imageSideLimit(const std::string& pref);          // "cuda", "vulkan" or "" (= the default backend)

    // pref: "auto", "cuda", "vulkan" or "cpu" (env BAKERY_BACKEND overrides). Safe to call repeatedly.
    // "auto" prefers Vulkan (faster for normal sizes), then CUDA, then CPU.
    bool init(int threads, std::string& err, const std::string& pref = "auto");
    // Make `pref` the active backend; when it differs from the current one every registered engine drops its models first.
    bool use(const std::string& pref, int threads, std::string& err);
    void addResetHook(std::function<void()> fn) { hooks.push_back(std::move(fn)); }
    std::string kind() const { return activeKind; }                // "cuda", "vulkan" or "cpu"

    ggml_backend_buffer_t allocWeights(ggml_context* ctx);
    ggml_backend_buffer_t allocDeviceWeights(ggml_context* ctx);   // small weights: on the GPU when there is one
    std::string description() const { return desc; }
    bool usingGpu() const { return gpu != nullptr; }
    bool gpuIntegrated = false;                                        // the active GPU is an integrated one (Intel / APU)
    size_t gpuFreeMB() const { size_t f = 0, t = 0; if (gpuDev) ggml_backend_dev_memory(gpuDev, &f, &t); return f >> 20; }

    ggml_backend_t gpu = nullptr, cpu = nullptr;
    ggml_backend_dev_t gpuDev = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_sched_t schedCpu = nullptr;           // CPU-only scheduler for steps where GPU rounding matters (codec encoder)
    ggml_backend_buffer_t allocHostWeights(ggml_context* ctx);
    std::string desc, activeKind = "cpu";
    bool inited = false;
private:
    void shutdown();
    std::vector<std::function<void()>> hooks;
};
