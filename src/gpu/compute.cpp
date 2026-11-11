#include "gpu/compute.h"
#include <cstdlib>
#include <cstring>
#include <thread>
#include "util/sys.h"

Compute& Compute::get() { static Compute c; return c; }

static std::string lowerStr(std::string s) { for (auto& c : s) c = (char)tolower((unsigned char)c); return s; }

bool Compute::init(int threads, std::string& err, const std::string& prefIn) {
    if (inited) return true;
    std::string pref = prefIn;
    if (const char* e = getenv("BAKERY_BACKEND")) pref = lowerStr(e);

    ggml_backend_load_all_from_path(sys::exeDir().c_str());

    ggml_backend_dev_t cpuDev = nullptr, best = nullptr;
    int bestScore = -1;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        auto type = ggml_backend_dev_type(d);
        std::string name = lowerStr(ggml_backend_dev_name(d));
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU) { if (!cpuDev) cpuDev = d; continue; }
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        int score = name.find("cuda") != std::string::npos ? 2 : (name.find("vulkan") != std::string::npos ? 1 : 0);
        if (pref == "cpu") continue;
        if (pref == "cuda" && score != 2) continue;
        if (pref == "vulkan" && score != 1) continue;
        if (score > bestScore) { bestScore = score; best = d; }
    }
    if (!cpuDev) { err = "no CPU backend found (ggml-cpu-*.dll missing next to the exe?)"; return false; }
    cpu = ggml_backend_dev_init(cpuDev, nullptr);
    if (!cpu) { err = "cannot create CPU backend"; return false; }
    if (ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(cpuDev)) {
        auto fn = (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (fn) fn(cpu, threads > 0 ? threads : (int)std::thread::hardware_concurrency());
    }
    if (best) {
        gpu = ggml_backend_dev_init(best, nullptr);
        if (gpu) {
            gpuDev = best;
            desc = std::string(ggml_backend_dev_description(best)) + " (" + ggml_backend_dev_name(best) + ")";
        }
    }
    if (!gpu) desc = "CPU";
    ggml_backend_t list[2]; int n = 0;
    if (gpu) list[n++] = gpu;
    list[n++] = cpu;
    sched = ggml_backend_sched_new(list, nullptr, n, 16384, false, true);
    if (!sched) { err = "cannot create backend scheduler"; return false; }
    ggml_backend_t cpuList[1] = {cpu};
    schedCpu = ggml_backend_sched_new(cpuList, nullptr, 1, 16384, false, false);
    inited = true;
    return true;
}

ggml_backend_buffer_t Compute::allocWeights(ggml_context* ctx) {
    ggml_backend_buffer_t b = nullptr;
    if (gpuDev) if (ggml_backend_buffer_type_t hb = ggml_backend_dev_host_buffer_type(gpuDev)) b = ggml_backend_alloc_ctx_tensors_from_buft(ctx, hb);
    if (!b) b = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    if (b) ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);   // lets the scheduler offload ops using these weights to the GPU
    return b;
}

#include <cstdio>
// `DopplersBakery backends`: list what ggml found next to the exe
int runBackendsList() {
    ggml_backend_load_all_from_path(sys::exeDir().c_str());
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        size_t freeMem = 0, total = 0; ggml_backend_dev_memory(d, &freeMem, &total);
        printf("%zu: %-10s type=%d  %s  mem %zu/%zu MB\n", i, ggml_backend_dev_name(d), (int)ggml_backend_dev_type(d), ggml_backend_dev_description(d), freeMem >> 20, total >> 20);
    }
    std::string err; Compute& c = Compute::get();
    if (c.init(8, err)) printf("selected: %s\n", c.description().c_str()); else printf("init failed: %s\n", err.c_str());
    return 0;
}

ggml_backend_buffer_t Compute::allocDeviceWeights(ggml_context* ctx) {
    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(ctx, gpu ? gpu : cpu);
    if (b) ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return b;
}

ggml_backend_buffer_t Compute::allocHostWeights(ggml_context* ctx) { return ggml_backend_alloc_ctx_tensors(ctx, cpu); }
