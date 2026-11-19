// Memory policy between tabs. Every engine keeps its models loaded after a job (reloading takes seconds to minutes), but together they do not fit:
// the video models alone are about 80 GB, the image models 18 GB, audio 10 GB, speech 4 GB. The limits depend on the RAM of the PC (see tierFor).
#include "backends/engines.h"
#include "gpu/compute.h"
#include "util/crashlog.h"
#include "util/sys.h"

namespace {

struct Holder { Kind kind; const char* name; size_t (*mb)(); void (*release)(); };
const Holder kHolders[] = {
    {Kind::Video, "video", videoModelsMB, releaseVideoModels},
    {Kind::Image, "image", imageModelsMB, releaseImageModels},
    {Kind::Audio, "audio", audioModelsMB, releaseAudioModels},
    {Kind::TTS,   "speech", ttsModelsMB,  releaseTtsModels},
    {Kind::LLM,   "chat",  llmModelMB,    releaseLLM},
};

// what the job will add to RAM when its models are not loaded yet
size_t needMB(Kind k, const Json& p) {
    const Holder* h = nullptr; for (auto& x : kHolders) if (x.kind == k) h = &x;
    if (!h || h->mb() > 0) return 0;                  // already loaded (a changed file is reloaded by the engine itself)
    switch (k) {
    case Kind::Video: return 85000;
    case Kind::Image: return (size_t)((sys::fileSize(p.str("model")) + sys::fileSize(p.str("clip"))) >> 20) + 1500;
    case Kind::Audio: return (size_t)((sys::fileSize(p.str("model")) + sys::fileSize(p.str("clip"))) >> 20) + 1000;
    case Kind::TTS:   return (size_t)(sys::fileSize(sys::join(p.str("model"), "model.safetensors")) >> 20) + 1000;
    default:          return (size_t)(sys::fileSize(p.str("model")) >> 20) + 500;
    }
}
}  // namespace

size_t freeRamMB() {
    if (const char* e = getenv("BAKERY_MEM_FREE_MB")) return (size_t)atoll(e);                 // test hook
    MEMORYSTATUSEX ms = {sizeof ms}; GlobalMemoryStatusEx(&ms); return (size_t)(ms.ullAvailPhys >> 20);
}
size_t totalRamMB() {
    if (const char* e = getenv("BAKERY_MEM_TOTAL_MB")) return (size_t)atoll(e);                // test hook
    MEMORYSTATUSEX ms = {sizeof ms}; GlobalMemoryStatusEx(&ms); return (size_t)(ms.ullTotalPhys >> 20);
}

// The rule, by the amount of RAM in the PC. On a job of another tab, cached models of the other tabs that are bigger than `bigMB` are freed, and when free RAM
// is below `floorMB` (counting what this job still has to load) the others are freed, biggest first, until it is not.
struct RamTier { size_t bigMB, floorMB; };
RamTier tierFor(size_t totalMB) {
    if (totalMB <= 24 * 1024)  return {4608, 4096};              // 16 GB:  bigger than 4.5 GB, or free below 4 GB
    if (totalMB <= 48 * 1024)  return {8192, 8192};              // 32 GB:  bigger than 8 GB,   or free below 8 GB
    if (totalMB <= 96 * 1024)  return {20480, 16384};            // 64 GB:  bigger than 20 GB,  or free below 16 GB
    return {32768, 16384};                                       // 128 GB: bigger than 32 GB,  or free below 16 GB
}

void prepareForJob(Kind k, const Json& params) {
    const RamTier t = tierFor(totalRamMB());
    auto release = [&](const Holder& h, size_t mb, const char* why) {
        logLine(std::string("memory: ") + h.name + " models released (" + std::to_string(mb >> 10) + " GB, " + why + ") for the " + kindKey(k) + " job; free RAM " + std::to_string(freeRamMB() >> 10) + " GB");
        h.release();
    };
    // 1) every other tab's models over the size limit go (a job of the same tab keeps its own: reloading them is the slow part)
    for (auto& h : kHolders) { if (h.kind == k) continue; const size_t m = h.mb(); if (m > t.bigMB) release(h, m, "bigger than the limit for this PC"); }
    // 2) free RAM below the floor (with what this job still has to load): the rest of the other tabs' models, biggest first
    const size_t need = needMB(k, params);
    while (freeRamMB() < t.floorMB + need) {
        const Holder* big = nullptr; size_t bigMB = 0;
        for (auto& h : kHolders) { if (h.kind == k) continue; const size_t m = h.mb(); if (m > bigMB) { big = &h; bigMB = m; } }
        if (!big) break;
        release(*big, bigMB, "free RAM too low");
    }
}
