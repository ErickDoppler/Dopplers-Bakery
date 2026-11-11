#include "ui/modal.h"
#include <commctrl.h>
#include <shlobj.h>
#include <commdlg.h>
#include "util/sys.h"

static HFONT g_font = nullptr;
HFONT uiFont() {
    if (!g_font) {
        NONCLIENTMETRICSW ncm = {}; ncm.cbSize = sizeof ncm;
        SystemParametersInfoW(SPI_GETNONCLIENTMETRICS, sizeof ncm, &ncm, 0);
        g_font = CreateFontIndirectW(&ncm.lfMessageFont);
    }
    return g_font;
}

HWND uiMake(HWND parent, const wchar_t* cls, const std::string& text, DWORD style, int id, DWORD ex) {
    HWND h = CreateWindowExW(ex, cls, sys::widen(text).c_str(), WS_CHILD | WS_VISIBLE | style, 0, 0, 10, 10,
                             parent, (HMENU)(INT_PTR)id, GetModuleHandleW(nullptr), nullptr);
    SendMessageW(h, WM_SETFONT, (WPARAM)uiFont(), TRUE);
    return h;
}
void uiPlace(HWND h, int x, int y, int w, int hgt) { MoveWindow(h, x, y, w, hgt, TRUE); }
std::string uiText(HWND h) {
    int n = GetWindowTextLengthW(h);
    std::wstring w(n + 1, 0);
    GetWindowTextW(h, &w[0], n + 1);
    w.resize(n);
    return sys::narrow(w);
}
void uiSetText(HWND h, const std::string& s) { SetWindowTextW(h, sys::widen(s).c_str()); }

static LRESULT CALLBACK modalProc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    Modal* m = (Modal*)GetWindowLongPtrW(h, GWLP_USERDATA);
    switch (msg) {
    case WM_NCCREATE:
        SetWindowLongPtrW(h, GWLP_USERDATA, (LONG_PTR)((CREATESTRUCTW*)lp)->lpCreateParams);
        break;
    case WM_CREATE:
        m = (Modal*)((CREATESTRUCTW*)lp)->lpCreateParams;
        m->hwnd = h;
        m->onCreate();
        return 0;
    case WM_COMMAND:
        if (m) m->onCommand(LOWORD(wp), HIWORD(wp));
        return 0;
    case WM_TIMER:
        if (m) m->onTimer();
        return 0;
    case WM_CLOSE:
        if (m && !m->onClose()) return 0;
        if (m) m->done = true;
        return 0;
    }
    return DefWindowProcW(h, msg, wp, lp);
}

void runModal(Modal& m, HWND parent, const std::string& title, int w, int h) {
    static bool reg = false;
    if (!reg) {
        WNDCLASSW wc = {};
        wc.lpfnWndProc = modalProc; wc.hInstance = GetModuleHandleW(nullptr);
        wc.hCursor = LoadCursorW(nullptr, IDC_ARROW); wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
        wc.lpszClassName = L"BakeryModal";
        RegisterClassW(&wc); reg = true;
    }
    RECT pr = {0, 0, 1280, 800};
    if (parent) GetWindowRect(parent, &pr);
    RECT r = {0, 0, w, h};
    AdjustWindowRect(&r, WS_CAPTION | WS_SYSMENU, FALSE);
    int ww = r.right - r.left, wh = r.bottom - r.top;
    int x = pr.left + ((pr.right - pr.left) - ww) / 2, y = pr.top + ((pr.bottom - pr.top) - wh) / 2;
    if (parent) EnableWindow(parent, FALSE);
    HWND hw = CreateWindowExW(WS_EX_DLGMODALFRAME, L"BakeryModal", sys::widen(title).c_str(), WS_CAPTION | WS_SYSMENU | WS_VISIBLE,
                              x, y, ww, wh, parent, nullptr, GetModuleHandleW(nullptr), &m);
    MSG msg;
    while (!m.done && GetMessageW(&msg, nullptr, 0, 0) > 0) {
        if (!IsDialogMessageW(hw, &msg)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
    }
    if (parent) EnableWindow(parent, TRUE);
    DestroyWindow(hw);
    if (parent) SetForegroundWindow(parent);
}

// ------------------------------------------------------------ text viewer
namespace {
struct Viewer : Modal {
    std::string text; HWND edit = nullptr;
    void onCreate() override {
        RECT rc; GetClientRect(hwnd, &rc);
        edit = uiMake(hwnd, L"EDIT", text, ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | WS_VSCROLL | WS_TABSTOP, 10, WS_EX_CLIENTEDGE);
        // normalise line endings for the edit control
        std::string t; for (size_t i = 0; i < text.size(); i++) { if (text[i] == '\n' && (i == 0 || text[i - 1] != '\r')) t += '\r'; t += text[i]; }
        uiSetText(edit, t);
        uiPlace(edit, 10, 10, rc.right - 20, rc.bottom - 54);
        uiPlace(uiMake(hwnd, L"BUTTON", "Copy text", BS_PUSHBUTTON | WS_TABSTOP, 11), 10, rc.bottom - 34, 100, 26);
        uiPlace(uiMake(hwnd, L"BUTTON", "Close", BS_DEFPUSHBUTTON | WS_TABSTOP, IDCANCEL), rc.right - 110, rc.bottom - 34, 100, 26);
    }
    void onCommand(int id, int) override {
        if (id == 11) clipboardSetText(hwnd, text);
        if (id == IDCANCEL) close();
    }
};
}  // namespace

void showTextViewer(HWND parent, const std::string& title, const std::string& text) {
    Viewer v; v.text = text;
    runModal(v, parent, title, 640, 460);
}

bool pickFile(HWND parent, std::string& path, const wchar_t* filter) {
    wchar_t buf[MAX_PATH * 2] = {};
    OPENFILENAMEW o = {}; o.lStructSize = sizeof o; o.hwndOwner = parent; o.lpstrFilter = filter;
    o.lpstrFile = buf; o.nMaxFile = MAX_PATH * 2; o.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (!GetOpenFileNameW(&o)) return false;
    path = sys::narrow(buf); return true;
}
bool pickFolder(HWND parent, std::string& path, const wchar_t* title) {
    BROWSEINFOW bi = {}; bi.hwndOwner = parent; bi.lpszTitle = title; bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
    PIDLIST_ABSOLUTE pidl = SHBrowseForFolderW(&bi);
    if (!pidl) return false;
    wchar_t buf[MAX_PATH * 2];
    bool ok = SHGetPathFromIDListW(pidl, buf);
    CoTaskMemFree(pidl);
    if (ok) path = sys::narrow(buf);
    return ok;
}

void clipboardSetText(HWND owner, const std::string& text) {
    std::wstring w = sys::widen(text);
    if (!OpenClipboard(owner)) return;
    EmptyClipboard();
    HGLOBAL g = GlobalAlloc(GMEM_MOVEABLE, (w.size() + 1) * sizeof(wchar_t));
    memcpy(GlobalLock(g), w.c_str(), (w.size() + 1) * sizeof(wchar_t));
    GlobalUnlock(g);
    SetClipboardData(CF_UNICODETEXT, g);
    CloseClipboard();
}
void clipboardSetFile(HWND owner, const std::string& path) {
    std::wstring w = sys::widen(path);
    size_t bytes = sizeof(DROPFILES) + (w.size() + 2) * sizeof(wchar_t);
    HGLOBAL g = GlobalAlloc(GMEM_MOVEABLE | GMEM_ZEROINIT, bytes);
    DROPFILES* df = (DROPFILES*)GlobalLock(g);
    df->pFiles = sizeof(DROPFILES); df->fWide = TRUE;
    memcpy((char*)df + sizeof(DROPFILES), w.c_str(), w.size() * sizeof(wchar_t));
    GlobalUnlock(g);
    if (!OpenClipboard(owner)) { GlobalFree(g); return; }
    EmptyClipboard();
    SetClipboardData(CF_HDROP, g);
    CloseClipboard();
}
