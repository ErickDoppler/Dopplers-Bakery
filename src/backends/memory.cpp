// Memory policy between tabs. Every engine keeps its models loaded after a job (reloading takes seconds to minutes), but together they do not fit:
// the video models alone are about 80 GB, the image models 18 GB, audio 10 GB, speech 4 GB, and a chat model sits in video memory. Before a job starts,
// the cached models of the OTHER tabs that are bigger than 4 GB are freed when RAM would run short (largest first); a chat model over 4 GB also leaves
// video memory before any other tab's job. Small models stay: they cost nothing to keep.
#include <algorithm>
#include <windows.h>
#include "backends/engines.h"
#include "gpu/compute.h"
#include "util/crashlog.h"
#include "util/sys.h"

namespace {
constexpr size_t kBigMB = 4096;                       // models up to this size are never unloaded for a transition
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

size_t freeRamMB() { MEMORYSTATUSEX ms = {sizeof ms}; GlobalMemoryStatusEx(&ms); return (size_t)(ms.ullAvailPhys >> 20); }

void prepareForJob(Kind k, const Json& params) {
    MEMORYSTATUSEX ms = {sizeof ms}; GlobalMemoryStatusEx(&ms);
    size_t headroom = std::max<size_t>(12000, (size_t)(ms.ullTotalPhys >> 20) / 10);          // the system, the activations and the file cache
    if (const char* e = getenv("BAKERY_MEM_HEADROOM_MB")) headroom = (size_t)atoll(e);       // test hook: pretend RAM is scarce
    const size_t need = needMB(k, params);
    // 1) RAM: free the biggest other model sets until the job fits
    for (;;) {
        if (need == 0 || freeRamMB() >= need + headroom) break;
        const Holder* big = nullptr; size_t bigMB = 0;
        for (auto& h : kHolders) { if (h.kind == k) continue; const size_t m = h.mb(); if (m > kBigMB && m > bigMB) { big = &h; bigMB = m; } }
        if (!big) break;
        logLine(std::string("memory: ") + big->name + " models released (" + std::to_string(bigMB >> 10) + " GB) to make room for the " + kindKey(k) + " job; free RAM was " + std::to_string(freeRamMB() >> 10) + " GB");
        big->release();
    }
    // 2) video memory: a big chat model keeps its layers on the GPU, which the other tabs need
    if (k != Kind::LLM && llmModelMB() > kBigMB) { logLine("memory: chat model released (it was in video memory)"); releaseLLM(); }
}
