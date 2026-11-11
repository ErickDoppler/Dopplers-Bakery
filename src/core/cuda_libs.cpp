#include "core/cuda_libs.h"
#include <windows.h>
#include <shellapi.h>
#include "util/sys.h"

namespace {
const char* kTag = "b12282";
const char* kWanted[] = {"cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll"};

std::string lowerS(std::string s) { for (auto& c : s) c = (char)tolower((unsigned char)c); return s; }

bool runHidden(const std::wstring& cmdLine, DWORD& exitCode) {
    STARTUPINFOW si = {sizeof si}; PROCESS_INFORMATION pi = {};
    std::wstring cl = cmdLine;
    if (!CreateProcessW(nullptr, &cl[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) return false;
    WaitForSingleObject(pi.hProcess, INFINITE);
    GetExitCodeProcess(pi.hProcess, &exitCode);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    return true;
}

// removes a folder this module created itself (path is built from the exe folder, never from user input)
void removeTree(const std::string& dir) {
    std::wstring w = sys::widen(dir); w.push_back(0); w.push_back(0);
    SHFILEOPSTRUCTW op = {}; op.wFunc = FO_DELETE; op.pFrom = w.c_str(); op.fFlags = FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI;
    SHFileOperationW(&op);
}

bool unzip(const std::string& zip, const std::string& dest, std::string& err) {
    sys::makeDirs(dest);
    wchar_t sysdir[MAX_PATH]; GetSystemDirectoryW(sysdir, MAX_PATH);
    std::wstring cmd = std::wstring(L"\"") + sysdir + L"\\tar.exe\" -xf \"" + sys::widen(zip) + L"\" -C \"" + sys::widen(dest) + L"\"";   // tar ships with Windows 10 1803+
    DWORD code = 1;
    if (!runHidden(cmd, code)) { err = "cannot run tar.exe (needs Windows 10 version 1803 or newer)"; return false; }
    if (code != 0) { err = "could not unpack " + zip.substr(zip.find_last_of("\\/") + 1); return false; }
    return true;
}

// first file called `name` under `root` (any depth), "" if none
std::string findFile(const std::string& root, const std::string& name) {
    std::vector<std::string> files; sys::listFiles(root, files, 4);
    for (auto& f : files) {
        std::string l = lowerS(f), n = lowerS(name);
        if (l == n || (l.size() > n.size() && l.compare(l.size() - n.size() - 1, n.size() + 1, "/" + n) == 0)) return sys::join(root, f);
    }
    return "";
}
}  // namespace

bool cudaLibsInstalled() {
    for (const char* n : {"ggml-cuda.dll", "cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll"})
        if (!sys::exists(sys::join(sys::exeDir(), n))) return false;
    return true;
}

CudaInstaller& CudaInstaller::get() { static CudaInstaller c; return c; }
CudaInstaller::~CudaInstaller() { if (th_.joinable()) { dl_.st.cancel = true; th_.join(); } }

float CudaInstaller::progress() {
    unsigned long long t = dl_.st.totalBytes;
    return t ? (float)((double)dl_.st.doneBytes / (double)t) : 0.0f;
}

std::string CudaInstaller::status() {
    switch (phase_) {
    case 1: return "Downloading CUDA libraries: " + std::to_string(dl_.st.doneBytes / 1000000) + " / " + std::to_string(dl_.st.totalBytes / 1000000) + " MB";
    case 2: return "Unpacking...";
    case 3: return "CUDA libraries installed - restart the app to use them.";
    case 4: { std::lock_guard<std::mutex> g(m_); return "Failed: " + error_; }
    default: return "";
    }
}

void CudaInstaller::start() {
    if (running()) return;
    if (th_.joinable()) th_.join();
    phase_ = 1;
    const std::string base = std::string("https://github.com/ggml-org/llama.cpp/releases/download/") + kTag + "/";
    CatEntry e; e.id = "cuda-libs"; e.label = "CUDA libraries";
    e.files.push_back({base + "llama-" + kTag + "-bin-win-cuda-13.5-x64.zip", "@exe/cuda_download/llama-cuda.zip", 171031017ull});
    e.files.push_back({base + "cudart-llama-bin-win-cuda-13.5-x64.zip", "@exe/cuda_download/cudart.zip", 278821062ull});
    dl_.start({e});
    th_ = std::thread([this] {
        dl_.join();
        const std::string work = sys::join(sys::exeDir(), "cuda_download");
        auto fail = [&](const std::string& why) { { std::lock_guard<std::mutex> g(m_); error_ = why; } phase_ = 4; };
        if (!dl_.st.ok) { std::lock_guard<std::mutex> g(dl_.st.m); fail(dl_.st.error.empty() ? "download cancelled" : dl_.st.error); return; }
        phase_ = 2;
        std::string err;
        const std::string t1 = sys::join(work, "x_llama"), t2 = sys::join(work, "x_cudart");
        if (!unzip(sys::join(work, "llama-cuda.zip"), t1, err) || !unzip(sys::join(work, "cudart.zip"), t2, err)) { fail(err); return; }
        struct Item { std::string from, to; };
        std::vector<Item> items;
        std::string ggml = findFile(t1, "ggml-cuda.dll");
        if (ggml.empty()) { fail("ggml-cuda.dll is not in the downloaded archive"); return; }
        items.push_back({ggml, sys::join(sys::exeDir(), "ggml-cuda.dll")});
        for (const char* n : kWanted) {
            std::string f = findFile(t2, n);
            if (f.empty()) { fail(std::string(n) + " is not in the downloaded archive"); return; }
            items.push_back({f, sys::join(sys::exeDir(), n)});
        }
        for (auto& it : items) {
            std::wstring a = sys::widen(it.from), b = sys::widen(it.to);
            if (!MoveFileExW(a.c_str(), b.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED)) { fail("cannot write " + it.to + " (is the folder writable?)"); return; }
        }
        removeTree(work);
        phase_ = 3;
    });
}
