#include "core/download.h"
#include <windows.h>
#include <winhttp.h>
#include <cstdio>
#include <map>
#include "core/settings.h"
#include "util/json.h"
#include "util/sys.h"

// ---------------------------------------------------------------- catalog

static CatEntry hf(const char* id, const char* label, Kind k, const char* repo, const char* dest, unsigned long long approx, const char* note = "") {
    CatEntry e; e.id = id; e.label = label; e.kind = k; e.hfRepo = repo; e.destDir = dest; e.approxBytes = approx; e.note = note; return e;
}
static CatEntry one(const char* id, const char* label, Kind k, const char* url, const char* rel, unsigned long long size, const char* note = "") {
    CatEntry e; e.id = id; e.label = label; e.kind = k; e.note = note; e.approxBytes = size;
    e.files.push_back({url, rel, size}); return e;
}

std::vector<CatEntry> loadCatalog() {
    std::vector<CatEntry> c;
    // LLM (Qwen)
    c.push_back(one("llm-qwen3-4b", "Qwen3 4B Instruct 2507 (Q4_K_M)", Kind::LLM,
        "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "LLM/Qwen3-4B-Instruct-2507-Q4_K_M.gguf", 2497000000ull));
    // TTS (Qwen3-TTS, same models the ComfyUI qwen3-tts node uses)
    c.push_back(hf("tts-qwen3-custom", "Qwen3-TTS 12Hz 1.7B CustomVoice", Kind::TTS,
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "qwen-tts/Qwen3-TTS-12Hz-1.7B-CustomVoice", 4400000000ull));
    // Audio (Stable Audio 3 medium; links from the AUDIO3-MEDIUM workflow)
    c.push_back(one("audio-sa3-medium", "Stable Audio 3 medium", Kind::Audio,
        "https://huggingface.co/Comfy-Org/stable-audio-3/resolve/main/checkpoints/stable_audio_3_medium.safetensors",
        "checkpoints/stable_audio_3_medium.safetensors", 8795000000ull));
    c.push_back(one("audio-t5gemma", "T5Gemma b-b UL2 text encoder", Kind::Audio,
        "https://huggingface.co/Comfy-Org/stable-audio-3/resolve/main/text_encoders/t5gemma_b_b_ul2.safetensors",
        "text_encoders/t5gemma_b_b_ul2.safetensors", 1132000000ull));
    c.push_back(one("audio-qwen35-2b", "Qwen3.5 2B (prompt enhancer)", Kind::Audio,
        "https://huggingface.co/Comfy-Org/Qwen3.5/resolve/main/text_encoders/qwen3.5_2b_bf16.safetensors",
        "text_encoders/qwen3.5_2b_bf16.safetensors", 4338000000ull));
    // Image (Krea2 pieces whose location is not yet confirmed come from catalog.json)
    c.push_back(one("image-qwen-vae", "Qwen-Image VAE", Kind::Image,
        "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
        "vae/qwen_image_vae.safetensors", 242000000ull));

    // Krea2 pieces without a verified download location: found by name through "Search in folder"
    c.push_back(one("image-krea2", "Krea2 turbo diffusion model (FP8)", Kind::Image,
        "https://huggingface.co/AlperKTS/Krea2_FP8/resolve/main/krea2_turbo_fp8.safetensors",
        "diffusion_models/krea2_turbo_fp8.safetensors", 12900096996ull));
    c.push_back(one("image-qwen3vl", "Qwen3-VL 4B text encoder for Krea2 (FP8)", Kind::Image,
        "https://huggingface.co/AlperKTS/Krea2_FP8/resolve/main/qwen3vl_4b_fp8_scaled.safetensors",
        "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", 5242467968ull));
    {   // which [Settings] slot each built-in entry fills
        static const char* roles[][2] = {{"llm-qwen3-4b", "model"}, {"tts-qwen3-custom", "model"}, {"audio-sa3-medium", "model"}, {"audio-t5gemma", "clip"},
                                         {"audio-qwen35-2b", "llm"}, {"image-qwen-vae", "vae"}, {"image-krea2", "model"}, {"image-qwen3vl", "clip"}};
        for (auto& e : c) for (auto& r : roles) if (e.id == r[0]) e.role = r[1];
    }

    // user catalog overrides / additions
    std::string txt;
    if (sys::readFile(sys::join(sys::exeDir(), "catalog.json"), txt)) {
        bool ok; Json j = Json::parse(txt, &ok);
        if (ok) for (auto& x : j.a) {
            CatEntry e; e.id = x.str("id"); e.label = x.str("label", e.id);
            e.kind = kindFromKey(x.str("kind")); e.note = x.str("note");
            e.hfRepo = x.str("hf_repo"); e.destDir = x.str("dest");
            e.approxBytes = (unsigned long long)x.num("size", 0); e.role = x.str("role");
            for (auto& f : x.get("files").a) e.files.push_back({f.str("url"), f.str("rel"), (unsigned long long)f.num("size", 0)});
            if (e.id.empty()) continue;
            bool replaced = false;
            for (auto& old : c) if (old.id == e.id) { old = e; replaced = true; }
            if (!replaced) c.push_back(e);
        }
    }
    return c;
}

std::string catalogLocation(const CatEntry& e) {
    std::string f = Settings::get().found(e.id);
    if (!f.empty() && sys::exists(f)) return f;
    auto here = [](const std::string& rel) -> std::string {
        std::string p = sys::join(sys::modelsDir(), rel);
        if (sys::exists(p)) return p;
        for (auto& r : Settings::get().extraRoots()) { p = sys::join(r, rel); if (sys::exists(p)) return p; }
        return "";
    };
    if (!e.hfRepo.empty()) { std::string c = here(e.destDir + "/config.json"); return c.empty() ? "" : c.substr(0, c.find_last_of("/\\")); }
    if (e.files.empty()) return "";
    std::string first;
    for (auto& fl : e.files) { std::string p = here(fl.rel); if (p.empty()) return ""; if (first.empty()) first = p; }
    return first;
}

bool catalogInstalled(const CatEntry& e) { return !catalogLocation(e).empty(); }

static std::string lowerStr(std::string s) { for (auto& c : s) c = (char)tolower((unsigned char)c); return s; }
static std::string baseName(const std::string& p) { size_t k = p.find_last_of("/\\"); return k == std::string::npos ? p : p.substr(k + 1); }

int scanFolderForCatalog(const std::string& folder, const std::vector<CatEntry>& cat) {
    std::vector<std::string> files, dirs;
    sys::listFiles(folder, files, 12);
    sys::listDirs(folder, dirs, 12);
    std::map<std::string, std::string> byName;           // lower-case file name -> relative path (first hit wins)
    for (auto& f : files) byName.emplace(lowerStr(baseName(f)), f);
    std::map<std::string, std::string> dirByName;
    for (auto& d : dirs) dirByName.emplace(lowerStr(baseName(d)), d);

    int found = 0;
    for (auto& e : cat) {
        std::string path;
        if (!e.hfRepo.empty()) {
            auto it = dirByName.find(lowerStr(baseName(e.destDir)));
            if (it != dirByName.end()) {
                std::string d = sys::join(folder, it->second);
                if (sys::exists(sys::join(d, "config.json"))) path = d;
            }
        } else if (!e.files.empty()) {
            bool all = true;
            for (auto& f : e.files) if (!byName.count(lowerStr(baseName(f.rel)))) all = false;
            if (all) path = sys::join(folder, byName[lowerStr(baseName(e.files[0].rel))]);
        }
        if (path.empty()) continue;
        for (auto& c : path) if (c == '/') c = '\\';
        Settings::get().setFound(e.id, path);
        if (!e.role.empty()) Settings::get().setSelection(e.kind, e.role, path);
        found++;
    }
    return found;
}

// ---------------------------------------------------------------- WinHTTP

namespace {
struct Handle {
    HINTERNET h = nullptr;
    ~Handle() { if (h) WinHttpCloseHandle(h); }
};

struct Conn {
    Handle session, connect, req;
    bool open(const std::string& url, const wchar_t* verb, const std::wstring& extraHeaders, std::string& err) {
        std::wstring w = sys::widen(url);
        URL_COMPONENTSW uc = {}; uc.dwStructSize = sizeof uc;
        wchar_t host[256], path[4096];
        uc.lpszHostName = host; uc.dwHostNameLength = 256; uc.lpszUrlPath = path; uc.dwUrlPathLength = 4096;
        if (!WinHttpCrackUrl(w.c_str(), 0, 0, &uc)) { err = "bad url"; return false; }
        session.h = WinHttpOpen(L"DopplersBakery/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, nullptr, nullptr, 0);
        if (!session.h) { err = "WinHttpOpen failed"; return false; }
        WinHttpSetTimeouts(session.h, 15000, 15000, 30000, 60000);
        connect.h = WinHttpConnect(session.h, host, uc.nPort, 0);
        if (!connect.h) { err = "connect failed"; return false; }
        DWORD flags = uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0;
        req.h = WinHttpOpenRequest(connect.h, verb, path, nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
        if (!req.h) { err = "open request failed"; return false; }
        DWORD redir = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
        WinHttpSetOption(req.h, WINHTTP_OPTION_REDIRECT_POLICY, &redir, sizeof redir);
        if (!WinHttpSendRequest(req.h, extraHeaders.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : extraHeaders.c_str(),
                                extraHeaders.empty() ? 0 : (DWORD)-1L, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
            !WinHttpReceiveResponse(req.h, nullptr)) {
            char b[64]; snprintf(b, sizeof b, "request failed (%lu)", GetLastError()); err = b; return false;
        }
        return true;
    }
    DWORD status() {
        DWORD s = 0, n = sizeof s;
        WinHttpQueryHeaders(req.h, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, nullptr, &s, &n, nullptr);
        return s;
    }
    unsigned long long contentLength() {
        wchar_t buf[64]; DWORD n = sizeof buf;
        if (WinHttpQueryHeaders(req.h, WINHTTP_QUERY_CONTENT_LENGTH, nullptr, buf, &n, nullptr)) return _wcstoui64(buf, nullptr, 10);
        return 0;
    }
};
}  // namespace

bool httpGetString(const std::string& url, std::string& body, std::string& err) {
    Conn c;
    if (!c.open(url, L"GET", L"", err)) return false;
    if (c.status() != 200) { err = "HTTP " + std::to_string(c.status()); return false; }
    body.clear();
    for (;;) {
        DWORD n = 0;
        if (!WinHttpQueryDataAvailable(c.req.h, &n) || n == 0) break;
        size_t off = body.size(); body.resize(off + n);
        DWORD got = 0;
        if (!WinHttpReadData(c.req.h, &body[off], n, &got)) { err = "read failed"; return false; }
        body.resize(off + got);
    }
    return true;
}

// Download to dest, resuming <dest>.part. Reports bytes via st.doneBytes.
static bool downloadFile(const std::string& url, const std::string& dest, unsigned long long expected, DlState& st, std::string& err) {
    std::string part = dest + ".part";
    sys::makeDirs(dest.substr(0, dest.find_last_of("\\/")));
    unsigned long long have = sys::fileSize(part);
    std::wstring hdr;
    if (have > 0) hdr = L"Range: bytes=" + std::to_wstring(have) + L"-\r\n";
    Conn c;
    if (!c.open(url, L"GET", hdr, err)) return false;
    DWORD code = c.status();
    if (code == 416) { have = 0; DeleteFileW(sys::widen(part).c_str()); err = "range rejected, retry"; return false; }
    if (code != 200 && code != 206) { err = "HTTP " + std::to_string(code); return false; }
    if (code == 200 && have > 0) { have = 0; DeleteFileW(sys::widen(part).c_str()); }   // server ignored Range
    unsigned long long total = c.contentLength() + have;
    if (total == 0) total = expected;
    st.doneBytes += have;
    FILE* f = _wfopen(sys::widen(part).c_str(), have ? L"ab" : L"wb");
    if (!f) { err = "cannot write " + part; return false; }
    std::vector<char> buf(1 << 20);
    unsigned long long got = have;
    bool ok = true;
    while (!st.cancel) {
        DWORD n = 0;
        if (!WinHttpQueryDataAvailable(c.req.h, &n)) { err = "connection lost"; ok = false; break; }
        if (n == 0) break;
        if (n > buf.size()) n = (DWORD)buf.size();
        DWORD r = 0;
        if (!WinHttpReadData(c.req.h, buf.data(), n, &r)) { err = "read failed"; ok = false; break; }
        if (fwrite(buf.data(), 1, r, f) != r) { err = "disk write failed (disk full?)"; ok = false; break; }
        got += r; st.doneBytes += r;
    }
    fclose(f);
    if (st.cancel) { err = "cancelled"; return false; }
    if (!ok) return false;
    if (total && got < total) { err = "incomplete download"; return false; }
    DeleteFileW(sys::widen(dest).c_str());
    if (!MoveFileW(sys::widen(part).c_str(), sys::widen(dest).c_str())) { err = "rename failed"; return false; }
    return true;
}

void Downloader::start(std::vector<CatEntry> entries) {
    st.cancel = false; st.finished = false; st.ok = false; st.doneBytes = 0; st.totalBytes = 0;
    th_ = std::thread([this, entries]() mutable {
        std::vector<CatFile> todo;
        std::string err;
        for (auto& e : entries) {
            if (!e.hfRepo.empty()) {
                { std::lock_guard<std::mutex> g(st.m); st.current = "Listing " + e.hfRepo; }
                std::string body;
                if (!httpGetString("https://huggingface.co/api/models/" + e.hfRepo + "/tree/main?recursive=1", body, err)) {
                    std::lock_guard<std::mutex> g(st.m); st.error = e.hfRepo + ": " + err; st.finished = true; return;
                }
                bool ok; Json j = Json::parse(body, &ok);
                for (auto& x : j.a) {
                    if (x.str("type") != "file") continue;
                    std::string p = x.str("path");
                    if (p == ".gitattributes" || p.find("overview.png") != std::string::npos) continue;
                    todo.push_back({"https://huggingface.co/" + e.hfRepo + "/resolve/main/" + p, e.destDir + "/" + p, (unsigned long long)x.num("size", 0)});
                }
            } else for (auto& f : e.files) if (!f.url.empty()) todo.push_back(f);
        }
        // skip finished files, total the rest
        std::vector<CatFile> need;
        unsigned long long total = 0;
        for (auto& f : todo) {
            std::string dest = sys::join(sys::modelsDir(), f.rel);
            if (sys::exists(dest) && (f.size == 0 || sys::fileSize(dest) == f.size)) continue;
            need.push_back(f); total += f.size;
        }
        st.totalBytes = total;
        { std::lock_guard<std::mutex> g(st.m); st.fileCount = (int)need.size(); }
        int idx = 0;
        for (auto& f : need) {
            { std::lock_guard<std::mutex> g(st.m); st.current = f.rel; st.fileIndex = ++idx; }
            std::string dest = sys::join(sys::modelsDir(), f.rel);
            for (char& c : dest) if (c == '/') c = '\\';
            bool ok = false;
            for (int attempt = 0; attempt < 3 && !ok && !st.cancel; attempt++) {
                unsigned long long before = st.doneBytes;
                ok = downloadFile(f.url, dest, f.size, st, err);
                if (!ok && err != "cancelled") st.doneBytes = before;   // retry resumes from .part and re-adds it
                if (!ok && attempt == 0) Sleep(1500);
            }
            if (!ok) {
                std::lock_guard<std::mutex> g(st.m);
                st.error = f.rel + ": " + err; st.finished = true; return;
            }
        }
        st.ok = true; st.finished = true;
    });
}
