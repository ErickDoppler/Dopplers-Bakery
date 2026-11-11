// Backend selection for the ggml engines: CUDA or Vulkan when a GPU backend DLL is present, CPU otherwise.
// Weights stay in (pinned) host memory; the ggml scheduler streams them to the GPU per op, so models
// larger than VRAM still run.
#pragma once
#include <string>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

struct Compute {
    static Compute& get();
    // pref: "auto", "cuda", "vulkan" or "cpu" (env BAKERY_BACKEND overrides). Safe to call repeatedly.
    bool init(int threads, std::string& err, const std::string& pref = "auto");
    ggml_backend_buffer_t allocWeights(ggml_context* ctx);
    ggml_backend_buffer_t allocDeviceWeights(ggml_context* ctx);   // small weights: on the GPU when there is one
    std::string description() const { return desc; }
    bool usingGpu() const { return gpu != nullptr; }

    ggml_backend_t gpu = nullptr, cpu = nullptr;
    ggml_backend_dev_t gpuDev = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_sched_t schedCpu = nullptr;           // CPU-only scheduler for steps where GPU rounding matters (codec encoder)
    ggml_backend_buffer_t allocHostWeights(ggml_context* ctx);
    std::string desc;
    bool inited = false;
};
