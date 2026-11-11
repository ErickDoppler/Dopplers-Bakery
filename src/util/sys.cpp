#include "util/sys.h"
#include <cstdio>
#include <algorithm>

namespace sys {

std::wstring widen(const std::string& s) {
    if (s.empty()) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), &w[0], n);
    return w;
}
std::string narrow(const std::wstring& w) {
    if (w.empty()) return "";
    int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, 0);
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), &s[0], n, nullptr, nullptr);
    return s;
}
std::string exeDir() {
    static std::string d;
    if (d.empty()) {
        wchar_t buf[MAX_PATH * 4];
        DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH * 4);
        std::wstring p(buf, n);
        size_t k = p.find_last_of(L"\\/");
        d = narrow(p.substr(0, k));
    }
    return d;
}
std::string join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    char c = a.back();
    return (c == '\\' || c == '/') ? a + b : a + "\\" + b;
}
std::string modelsDir() { return join(exeDir(), "models"); }
std::string outputDir() { return join(exeDir(), "output"); }
bool exists(const std::string& p) { return GetFileAttributesW(widen(p).c_str()) != INVALID_FILE_ATTRIBUTES; }
bool isDir(const std::string& p) {
    DWORD a = GetFileAttributesW(widen(p).c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}
bool makeDirs(const std::string& p) {
    std::wstring w = widen(p);
    for (size_t i = 1; i <= w.size(); i++) {
        if (i == w.size() || w[i] == L'\\' || w[i] == L'/') {
            std::wstring part = w.substr(0, i);
            if (part.size() == 2 && part[1] == L':') continue;
            CreateDirectoryW(part.c_str(), nullptr);
        }
    }
    return isDir(p);
}
bool readFile(const std::string& p, std::string& out) {
    FILE* f = _wfopen(widen(p).c_str(), L"rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    out.resize(n > 0 ? n : 0);
    if (n > 0) fread(&out[0], 1, n, f);
    fclose(f); return true;
}
bool writeFile(const std::string& p, const std::string& data) {
    size_t k = p.find_last_of("\\/");
    if (k != std::string::npos) makeDirs(p.substr(0, k));
    FILE* f = _wfopen(widen(p).c_str(), L"wb");
    if (!f) return false;
    bool ok = fwrite(data.data(), 1, data.size(), f) == data.size();
    fclose(f); return ok;
}
unsigned long long fileSize(const std::string& p) {
    WIN32_FILE_ATTRIBUTE_DATA d;
    if (!GetFileAttributesExW(widen(p).c_str(), GetFileExInfoStandard, &d)) return 0;
    return ((unsigned long long)d.nFileSizeHigh << 32) | d.nFileSizeLow;
}
std::string timestamp() {
    SYSTEMTIME t; GetLocalTime(&t);
    char b[32];
    snprintf(b, sizeof b, "%04d%02d%02d-%02d%02d%02d", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
    return b;
}
double now() {
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}
std::string humanSize(unsigned long long n) {
    char b[32];
    if (n >= (1ull << 30)) snprintf(b, sizeof b, "%.2f GB", n / 1073741824.0);
    else if (n >= (1ull << 20)) snprintf(b, sizeof b, "%.1f MB", n / 1048576.0);
    else if (n >= 1024) snprintf(b, sizeof b, "%.1f KB", n / 1024.0);
    else snprintf(b, sizeof b, "%llu B", n);
    return b;
}

static void walk(const std::wstring& root, const std::wstring& rel, std::vector<std::string>& out, int depth, bool dirs) {
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW((root + (rel.empty() ? L"" : L"\\" + rel) + L"\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        std::wstring name = fd.cFileName;
        if (name == L"." || name == L"..") continue;
        std::wstring r = rel.empty() ? name : rel + L"/" + name;
        bool isd = fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY;
        if (isd) {
            if (name[0] == L'.') continue;  // .cache etc.
            if (dirs) out.push_back(narrow(r));
            if (depth > 0) walk(root, (rel.empty() ? L"" : rel + L"\\") + name, out, depth - 1, dirs);
        } else if (!dirs) out.push_back(narrow(r));
    } while (FindNextFileW(h, &fd));
    FindClose(h);
}
void listFiles(const std::string& root, std::vector<std::string>& out, int maxDepth) { walk(widen(root), L"", out, maxDepth, false); }
void listDirs(const std::string& root, std::vector<std::string>& out, int maxDepth) { walk(widen(root), L"", out, maxDepth, true); }

}  // namespace sys
