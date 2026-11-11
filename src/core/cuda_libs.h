// Optional CUDA runtime: downloaded from the llama.cpp release the app was built against (b12282), unpacked next to the exe.
// Vulkan is always available; CUDA adds renders above 1264 px. Needs a restart of the app to be picked up.
#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include "core/download.h"

bool cudaLibsInstalled();                // ggml-cuda.dll, cudart64_13.dll, cublas64_13.dll, cublasLt64_13.dll next to the exe
const unsigned long long kCudaLibsBytes = 449852079ull;     // size of the two archives

class CudaInstaller {
public:
    static CudaInstaller& get();
    void start();                                  // background; safe to call again after a failure
    bool running() const { return phase_ == 1 || phase_ == 2; }
    bool done() const { return phase_ == 3; }
    bool failed() const { return phase_ == 4; }
    std::string status();                          // "Downloading 120 / 450 MB", "Unpacking...", "Done", or the error
    float progress();                              // 0..1 while downloading
private:
    ~CudaInstaller();
    Downloader dl_;
    std::thread th_;
    std::atomic<int> phase_{0};                    // 0 idle, 1 downloading, 2 unpacking, 3 done, 4 failed
    std::mutex m_;
    std::string error_;
};
