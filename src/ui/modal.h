// Tiny modal-window helper on plain Win32 (no resource templates).
#pragma once
#include <windows.h>
#include <string>

HFONT uiFont();
HWND uiMake(HWND parent, const wchar_t* cls, const std::string& text, DWORD style, int id = 0, DWORD ex = 0);
void uiPlace(HWND h, int x, int y, int w, int hgt);
void uiBatchBegin(HWND parent, int n);       // uiPlace calls on children of `parent` are applied together at uiBatchEnd
void uiBatchEnd();
std::string uiText(HWND h);
void uiSetText(HWND h, const std::string& s);

struct Modal {
    HWND hwnd = nullptr;
    bool done = false;
    virtual ~Modal() {}
    virtual void onCreate() = 0;
    virtual void onCommand(int id, int code) { (void)id; (void)code; }
    virtual void onTimer() {}
    virtual bool onClose() { return true; }     // return false to veto closing
    void close() { done = true; }
};
// Shows `m` centred over parent and runs a nested message loop until it closes.
void runModal(Modal& m, HWND parent, const std::string& title, int w, int h);

// Common little dialogs
void showTextViewer(HWND parent, const std::string& title, const std::string& text);
bool pickFile(HWND parent, std::string& path, const wchar_t* filter);
bool pickFolder(HWND parent, std::string& path, const wchar_t* title);
void clipboardSetText(HWND owner, const std::string& text);
void clipboardSetFile(HWND owner, const std::string& path);
