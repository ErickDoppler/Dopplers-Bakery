#include "util/crashlog.h"
#include <windows.h>
#include <cstdio>
#include <ctime>
#include <mutex>
#include "ggml.h"
#include "util/sys.h"

namespace {
std::mutex g_m;
std::string g_context;
std::string logPath() { return sys::join(sys::exeDir(), "bakery.log"); }
std::string markerPath() { return sys::join(sys::exeDir(), "crash_guard.txt"); }

void appendRaw(const std::string& line) {
    std::lock_guard<std::mutex> g(g_m);
    std::string p = logPath();
    if (sys::fileSize(p) > (1u << 20)) { MoveFileExW(sys::widen(p).c_str(), sys::widen(p + ".old").c_str(), MOVEFILE_REPLACE_EXISTING); }
    FILE* f = _wfopen(sys::widen(p).c_str(), L"ab");
    if (!f) return;
    time_t t = time(nullptr); tm tmv; localtime_s(&tmv, &t);
    char ts[32]; strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S ", &tmv);
    fputs(ts, f); fputs(line.c_str(), f); if (line.empty() || line.back() != '\n') fputc('\n', f);
    fclose(f);
}

void report(const std::string& title, const std::string& detail) {
    appendRaw(title + ": " + detail + " | doing: " + g_context);
    std::string msg = title + "\n\n" + detail + (g_context.empty() ? "" : "\n\nWhat was running: " + g_context) +
                      "\n\nThe details are in bakery.log next to the program. Start the app again; if it keeps happening, choose Backend = CPU on the Image tab.";
    MessageBoxW(nullptr, sys::widen(msg).c_str(), L"Dopplers Bakery stopped", MB_OK | MB_ICONERROR | MB_TOPMOST | MB_SETFOREGROUND);
}

void onAbort(const char* message) { report("The AI engine stopped", message ? message : "(no message)"); }

LONG WINAPI onException(EXCEPTION_POINTERS* ep) {
    char b[160];
    snprintf(b, sizeof b, "exception 0x%08lX at %p", ep->ExceptionRecord->ExceptionCode, ep->ExceptionRecord->ExceptionAddress);
    report("The app crashed", b);
    return EXCEPTION_EXECUTE_HANDLER;
}
}  // namespace

void logLine(const std::string& text) { appendRaw(text); }

void crashContext(const std::string& what) { std::lock_guard<std::mutex> g(g_m); g_context = what; }

void crashLogInstall() {
    ggml_set_abort_callback(onAbort);
    ggml_log_set([](ggml_log_level level, const char* text, void*) { if (level >= GGML_LOG_LEVEL_WARN && text) appendRaw(std::string("[ggml] ") + text); }, nullptr);
    SetUnhandledExceptionFilter(onException);
    appendRaw("---- started");
}

void crashGuardBegin(const std::string& what) { sys::writeFile(markerPath(), what); }
void crashGuardEnd() { DeleteFileW(sys::widen(markerPath()).c_str()); }
bool crashGuardRecover(std::string& what) {
    if (!sys::readFile(markerPath(), what)) return false;
    crashGuardEnd();
    appendRaw("previous run died inside: " + what);
    return true;
}
