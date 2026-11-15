#include "gpu/compute.h"
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include "util/crashlog.h"
#include "util/sys.h"

Compute& Compute::get() { static Compute c; return c; }

static std::string lowerStr(std::string s) { for (auto& c : s) c = (char)tolower((unsigned char)c); return s; }

static bool g_hasCuda = false, g_hasVulkan = false, g_onlyIntegrated = false, g_safe = false;
// integrated GPUs (Intel, AMD APUs) are reported as GGML_BACKEND_DEVICE_TYPE_IGPU, not _GPU: both are usable
static bool isGpuType(enum ggml_backend_dev_type t) { return t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU; }
void Compute::probe() {
    static std::once_flag once;
    std::call_once(once, [] {
        ggml_backend_load_all_from_path(sys::exeDir().c_str());
        bool discrete = false, integrated = false;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            size_t freeMem = 0, total = 0; ggml_backend_dev_memory(d, &freeMem, &total);
            logLine(std::string("device ") + std::to_string(i) + ": " + ggml_backend_dev_name(d) + " | " + ggml_backend_dev_description(d) + " | type " +
                    std::to_string((int)ggml_backend_dev_type(d)) + " | memory " + std::to_string(freeMem >> 20) + "/" + std::to_string(total >> 20) + " MB");
            if (!isGpuType(ggml_backend_dev_type(d))) continue;
            (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_IGPU ? integrated : discrete) = true;
            std::string n = lowerStr(ggml_backend_dev_name(d));
            if (n.find("cuda") != std::string::npos) g_hasCuda = true;
            if (n.find("vulkan") != std::string::npos) g_hasVulkan = true;
        }
        g_onlyIntegrated = integrated && !discrete;
    });
}
bool Compute::hasCuda() { probe(); return g_hasCuda; }
bool Compute::hasVulkan() { probe(); return g_hasVulkan; }
bool Compute::onlyIntegratedGpu() { probe(); return g_onlyIntegrated; }
void Compute::setSafeMode(bool on) { g_safe = on; }
bool Compute::safeMode() { return g_safe; }
int Compute::imageSideLimit(const std::string& prefIn) {
    std::string p = lowerStr(prefIn);
    if (const char* e = getenv("BAKERY_BACKEND")) p = lowerStr(e);
    bool vk = p == "vulkan" || ((p.empty() || p == "auto") && hasVulkan());
    if (p == "cuda" && !hasCuda()) vk = hasVulkan();
    return vk ? 1264 : 5000;
}

void Compute::shutdown() {
    if (schedCpu) ggml_backend_sched_free(schedCpu);
    if (sched) ggml_backend_sched_free(sched);
    if (gpu) ggml_backend_free(gpu);
    if (cpu) ggml_backend_free(cpu);
    schedCpu = sched = nullptr; gpu = cpu = nullptr; gpuDev = nullptr; gpuIntegrated = false; inited = false; desc.clear(); activeKind = "cpu";
}

bool Compute::use(const std::string& prefIn, int threads, std::string& err) {
    std::string want = prefIn.empty() ? "auto" : lowerStr(prefIn);
    if (g_safe && want == "auto") want = "cpu";        // the last GPU job crashed the app: stay on the CPU unless a backend is chosen explicitly
    if (inited) {
        if (want == "auto" || want == activeKind || getenv("BAKERY_BACKEND")) return true;
        if ((want == "cuda" && !hasCuda()) || (want == "vulkan" && !hasVulkan())) return true;      // not available: keep what we have
        for (auto& h : hooks) h();                       // engines free their models (they are tied to the old backend)
        shutdown();
    }
    return init(threads, err, want);
}

bool Compute::init(int threads, std::string& err, const std::string& prefIn) {
    if (inited) return true;
    std::string pref = prefIn;
    if (g_safe && (pref == "auto" || pref.empty())) pref = "cpu";
    if (const char* e = getenv("BAKERY_BACKEND")) pref = lowerStr(e);

    probe();

    ggml_backend_dev_t cpuDev = nullptr, best = nullptr;
    int bestScore = -1;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        auto type = ggml_backend_dev_type(d);
        std::string name = lowerStr(ggml_backend_dev_name(d));
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU) { if (!cpuDev) cpuDev = d; continue; }
        if (!isGpuType(type)) continue;
        int base = name.find("vulkan") != std::string::npos ? 2 : (name.find("cuda") != std::string::npos ? 1 : 0);      // auto: Vulkan first
        if (pref == "cpu") continue;
        if (pref == "cuda" && base != 1) continue;
        if (pref == "vulkan" && base != 2) continue;
        int score = base * 10 + (type == GGML_BACKEND_DEVICE_TYPE_GPU ? 5 : 0);       // a discrete GPU wins over the integrated one (laptops have both)
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
            gpuIntegrated = ggml_backend_dev_type(best) == GGML_BACKEND_DEVICE_TYPE_IGPU;
            desc = std::string(ggml_backend_dev_description(best)) + " (" + ggml_backend_dev_name(best) + ")";
            logLine("using " + desc + (gpuIntegrated ? " (integrated, shares the system memory)" : "") + ", free memory " + std::to_string(gpuFreeMB()) + " MB");
        }
    }
    if (!gpu) desc = "CPU";
    activeKind = !gpu ? "cpu" : (lowerStr(ggml_backend_dev_name(best)).find("cuda") != std::string::npos ? "cuda" : "vulkan");
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
    Compute::probe();
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
