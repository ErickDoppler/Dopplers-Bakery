// Main window: operation tabs on the left, queue manager on the right.
#include "ui/ui.h"
#include <objbase.h>
#include <commctrl.h>
#include <cstring>
#include <shellapi.h>
#include <mmsystem.h>
#include <algorithm>
#include <array>
#include <cmath>
#include "gpu/compute.h"
#include "ui/chat_view.h"
#include "web/server.h"
#include "core/download.h"
#include "core/forms.h"
#include "core/queue.h"
#include "core/settings.h"
#include "ui/dialogs.h"
#include "ui/modal.h"
#include "ui/player.h"
#include "util/image_io.h"
#include "util/crashlog.h"
#include "util/sys.h"

namespace {

enum {
    IDC_CREATE = 1001, IDC_SETTINGS, IDC_OUTLIST, IDC_PREVIEW, IDC_COPY, IDC_DELETE, IDC_PL_PREV, IDC_PL_PLAY, IDC_PL_STOP, IDC_PL_NEXT, IDC_PL_TRK, IDC_EXTEND,
    IDC_FIELD0 = 2000, IDC_BROWSE0 = 2500,
    IDC_QLIST = 3001, IDC_QUP, IDC_QDOWN, IDC_QREMOVE, IDC_QPAUSE, IDC_QCANCEL, IDC_QCLEAR, IDC_QDETAILS, IDC_QPROG, IDC_QSTAT, IDC_QTOGGLE, IDC_QDETAIL,

    IDM_OPENOUT = 4001, IDM_EXIT, IDM_DOWNLOAD, IDM_ADDROOT, IDM_ABOUT, IDM_WEB, IDM_WEBLAN, IDM_QUEUE,
};
std::string g_recovered;              // what the previous run was doing when it died (empty: normal exit)
const UINT WM_FIRSTRUN = WM_APP + 1;
const int kQueueMinW = 340;

struct FieldCtl {
    const Field* f; HWND label = nullptr, ctl = nullptr, btn = nullptr;
    int x = 0, ly = 0, cy = 0, w = 0, ch = 0;      // logical layout (before scrolling)
    int ext = 0;                                   // value above the trackbar range (Audio length set with Extend...), 0 = none
};
struct OutItem { int jobId; std::string path, text, when; };

struct Page {
    Kind kind;
    HWND hwnd = nullptr, form = nullptr, warn = nullptr, bCreate = nullptr, bSettings = nullptr;
    int actY = 0;                        // logical y of the Create row inside the scrolling form
    HWND list = nullptr, bPreview = nullptr, bCopy = nullptr, bDelete = nullptr;
    std::vector<FieldCtl> fc;
    int contentH = 0, scroll = 0;
    bool ready = true;
    std::vector<OutItem> outs;
    std::string sig;
    // audio tabs: a small player under the list (Prev / Play-Pause / Stop / Next, position bar, what is playing)
    HWND bPrev = nullptr, bPlay = nullptr, bStop = nullptr, bNext = nullptr, trk = nullptr, lblNow = nullptr;
    bool seeking = false;
    HWND bExtend = nullptr;              // Audio tab: "Extend..." next to the length trackbar (lengths beyond five minutes)
};
inline bool pagePlays(const Page& p) { return p.trk != nullptr; }

HWND g_main, g_tab, g_qTitle, g_qList, g_qStat, g_qProg, g_qToggle = nullptr, g_qDetail = nullptr;
HWND g_qBtn[8];
Page g_pages[kKindCount];
std::vector<int> g_qIds;          // job id per queue row
std::string g_qSig;

Page& pageOf(HWND h) { return *(Page*)GetWindowLongPtrW(h, GWLP_USERDATA); }
bool smallField(const Field& f) { return f.type == FT::Int || f.type == FT::Float || f.type == FT::Combo || f.type == FT::Check; }

// ----------------------------------------------------------------- Create button (owner drawn)
HFONT btnFont(bool bold) {
    static HFONT f[2] = {};
    if (!f[bold]) f[bold] = CreateFontW(bold ? -19 : -15, 0, 0, 0, bold ? FW_SEMIBOLD : FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
    return f[bold];
}
LRESULT CALLBACK hoverProc(HWND h, UINT m, WPARAM wp, LPARAM lp, UINT_PTR, DWORD_PTR) {
    if (m == WM_MOUSEMOVE) {
        if (!GetPropW(h, L"hot")) { SetPropW(h, L"hot", (HANDLE)1); TRACKMOUSEEVENT t = {sizeof t, TME_LEAVE, h, 0}; TrackMouseEvent(&t); InvalidateRect(h, nullptr, FALSE); }
    } else if (m == WM_MOUSELEAVE) { RemovePropW(h, L"hot"); InvalidateRect(h, nullptr, FALSE); }
    else if (m == WM_NCDESTROY) { RemovePropW(h, L"hot"); RemoveWindowSubclass(h, hoverProc, 1); }
    return DefSubclassProc(h, m, wp, lp);
}
void drawCreateButton(DRAWITEMSTRUCT* d) {
    HDC dc = d->hDC; RECT r = d->rcItem;
    bool dis = (d->itemState & ODS_DISABLED) != 0, down = (d->itemState & ODS_SELECTED) != 0, hot = GetPropW(d->hwndItem, L"hot") != nullptr;
    FillRect(dc, &r, GetSysColorBrush(COLOR_BTNFACE));
    COLORREF top = dis ? RGB(166, 173, 186) : down ? RGB(30, 64, 175) : hot ? RGB(96, 165, 250) : RGB(59, 130, 246);
    COLORREF bot = dis ? RGB(140, 148, 162) : down ? RGB(30, 58, 138) : hot ? RGB(37, 99, 235) : RGB(29, 78, 216);
    HRGN rg = CreateRoundRectRgn(r.left, r.top, r.right + 1, r.bottom + 1, 16, 16);
    SelectClipRgn(dc, rg);
    TRIVERTEX v[2] = {{r.left, r.top, (COLOR16)(GetRValue(top) << 8), (COLOR16)(GetGValue(top) << 8), (COLOR16)(GetBValue(top) << 8), 0},
                      {r.right, r.bottom, (COLOR16)(GetRValue(bot) << 8), (COLOR16)(GetGValue(bot) << 8), (COLOR16)(GetBValue(bot) << 8), 0}};
    GRADIENT_RECT gr = {0, 1};
    GradientFill(dc, v, 2, &gr, 1, GRADIENT_FILL_RECT_V);
    SelectClipRgn(dc, nullptr); DeleteObject(rg);
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, dis ? RGB(236, 239, 244) : RGB(255, 255, 255));
    HGDIOBJ old = SelectObject(dc, btnFont(true));
    RECT tr = r; if (down) OffsetRect(&tr, 1, 1);
    wchar_t cap[32] = L"Create"; GetWindowTextW(d->hwndItem, cap, 32);
    DrawTextW(dc, cap, -1, &tr, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
    SelectObject(dc, old);
    if (d->itemState & ODS_FOCUS) { RECT fr = r; InflateRect(&fr, -4, -4); DrawFocusRect(dc, &fr); }
}

// ----------------------------------------------------------------- form values
std::string fmtNum(double v) { char b[32]; snprintf(b, sizeof b, "%.6g", v); return b; }

Json readForm(Page& p) {
    Json j = Json::object();
    for (auto& c : p.fc) {
        const Field& f = *c.f;
        switch (f.type) {
        case FT::Text: case FT::File: case FT::Multi: case FT::Combo: {
            std::string s = uiText(c.ctl);
            std::string n; for (size_t i = 0; i < s.size(); i++) { if (s[i] == '\r' && i + 1 < s.size() && s[i + 1] == '\n') continue; n += s[i]; }
            j[f.key] = n; break;
        }
        case FT::Check: j[f.key] = SendMessageW(c.ctl, BM_GETCHECK, 0, 0) == BST_CHECKED; break;
        case FT::Slider: j[f.key] = c.ext > 0 ? (double)c.ext : (double)SendMessageW(c.ctl, TBM_GETPOS, 0, 0); break;
        case FT::Int: case FT::Float: {
            std::string s = uiText(c.ctl); for (auto& ch : s) if (ch == ',') ch = '.';
            double v = s.empty() ? f.def : atof(s.c_str());
            v = std::min(std::max(v, f.mn), f.mx);
            if (f.type == FT::Int) v = std::round(v);
            j[f.key] = v; break;
        }
        }
    }
    return j;
}

void sliderLabel(FieldCtl& c) {
    int v = c.ext > 0 ? c.ext : (int)SendMessageW(c.ctl, TBM_GETPOS, 0, 0);
    uiSetText(c.label, std::string(c.f->label) + ": " + std::to_string(v) + (c.ext > 0 ? "  (extended beyond the trackbar)" : ""));
}

// Width/Height can only go as high as the chosen backend renders correctly (Vulkan 1264 px, CUDA much more).
void applyImageLimits(Page& p, bool fromSelection = false) {
    if (p.kind != Kind::Image) return;
    std::string pref;
    for (auto& c : p.fc) if (!strcmp(c.f->key, "backend")) {
        int s = (int)SendMessageW(c.ctl, CB_GETCURSEL, 0, 0);
        if (fromSelection && s >= 0) { wchar_t b[32] = {}; SendMessageW(c.ctl, CB_GETLBTEXT, s, (LPARAM)b); pref = sys::narrow(b); }
        else pref = uiText(c.ctl);
    }
    for (auto& ch : pref) ch = (char)tolower((unsigned char)ch);
    int lim = Compute::imageSideLimit(pref);
    for (auto& c : p.fc) if (!strcmp(c.f->key, "width") || !strcmp(c.f->key, "height")) {
        SendMessageW(c.ctl, TBM_SETRANGE, TRUE, MAKELONG((int)c.f->mn, lim));
        if ((int)SendMessageW(c.ctl, TBM_GETPOS, 0, 0) > lim) SendMessageW(c.ctl, TBM_SETPOS, TRUE, lim);
        sliderLabel(c);
    }
}

void writeForm(Page& p, const Json& saved) {
    for (auto& c : p.fc) {
        const Field& f = *c.f;
        const Json& v = saved.get(f.key);
        switch (f.type) {
        case FT::Text: case FT::File: case FT::Multi: case FT::Combo: {
            std::string s = v.t == Json::Str ? v.s : std::string(f.sdef);
            std::string n; for (char ch : s) { if (ch == '\n') n += '\r'; n += ch; }
            uiSetText(c.ctl, n); break;
        }
        case FT::Check: SendMessageW(c.ctl, BM_SETCHECK, v.isNull() ? f.def != 0 : v.boolean(), 0); break;
        case FT::Slider: {
            double want = v.isNull() ? f.def : v.num();
            c.ext = (!strcmp(f.key, "length") && want > f.mx) ? (int)want : 0;                 // an extended Audio length survives a restart
            SendMessageW(c.ctl, TBM_SETPOS, TRUE, (LPARAM)(c.ext > 0 ? f.mx : want)); sliderLabel(c); break;
        }
        case FT::Int: uiSetText(c.ctl, std::to_string((long long)std::llround(v.isNull() ? f.def : v.num()))); break;
        case FT::Float: uiSetText(c.ctl, fmtNum(v.isNull() ? f.def : v.num())); break;
        }
    }
}

// ----------------------------------------------------------------- page layout
bool isMainPrompt(const Field& f) { return !strcmp(f.key, "prompt") || !strcmp(f.key, "text"); }

void layoutForm(Page& p) {
    RECT rc; GetClientRect(p.form, &rc);
    int W = std::max<int>(rc.right, 200), M = 10, gap = 8;
    int sbw = GetSystemMetrics(SM_CXVSCROLL);
    int usable = W - (p.contentH > rc.bottom ? sbw : 0);
    int colw = (usable - 2 * M - gap) / 2, y = M, col = 0, rowH = 0;
    for (auto& c : p.fc) {
        const Field& f = *c.f;
        if (smallField(f)) {
            int h = (f.type == FT::Check ? 22 : 24), total = (f.type == FT::Check ? 22 : 18 + h) + 8;
            c.w = colw; c.x = col == 0 ? M : M + colw + gap;
            c.ly = y; c.cy = f.type == FT::Check ? y : y + 18; c.ch = h;
            rowH = std::max(rowH, total);
            if (col == 0) col = 1; else { y += rowH; rowH = 0; col = 0; }
        } else {
            if (col == 1) { y += rowH; rowH = 0; col = 0; }
            c.x = M; c.w = usable - 2 * M; c.ly = y; c.cy = y + 18;
            c.ch = f.type == FT::Multi ? f.lines * 17 + 10 : (f.type == FT::Slider ? 26 : 24);
            y += 18 + c.ch + 8;
            if (isMainPrompt(f)) { p.actY = y; y += 52; }          // Create + Settings row right under the main prompt
        }
    }
    if (col == 1) y += rowH;
    p.contentH = y + M;
    int maxScroll = std::max<int>(0, p.contentH - rc.bottom);
    p.scroll = std::min(std::max(p.scroll, 0), maxScroll);
    SCROLLINFO si = {sizeof si, SIF_RANGE | SIF_PAGE | SIF_POS, 0, p.contentH, (UINT)rc.bottom, p.scroll, 0};
    SetScrollInfo(p.form, SB_VERT, &si, TRUE);
    {
        int createW = (usable - 2 * M - gap) * 68 / 100;
        uiPlace(p.bCreate, M, p.actY - p.scroll, createW, 44);
        uiPlace(p.bSettings, M + createW + gap, p.actY - p.scroll, usable - 2 * M - gap - createW, 44);
    }
    for (auto& c : p.fc) {
        if (c.label) uiPlace(c.label, c.x, c.ly - p.scroll, c.w, 18);
        uiPlace(c.ctl, c.x, c.cy - p.scroll, c.btn ? c.w - 34 : c.w, c.f->type == FT::Combo ? 260 : c.ch);
        if (c.btn) uiPlace(c.btn, c.x + c.w - 30, c.cy - p.scroll, 30, c.ch);
        if (p.bExtend && !strcmp(c.f->key, "length")) {                   // Extend... sits right of the (shortened) trackbar
            const int bw = 100;
            uiPlace(c.ctl, c.x, c.cy - p.scroll, c.w - bw - 8, c.ch);
            uiPlace(p.bExtend, c.x + c.w - bw, c.cy - p.scroll, bw, 24);
        }
    }
}

void layoutPage(Page& p) {
    RECT rc; GetClientRect(p.hwnd, &rc);
    int W = rc.right, H = rc.bottom;
    const int playerH = pagePlays(p) ? 58 : 0;
    int warnH = p.ready ? 0 : 36, actH = 0, outH = 120, outBtnH = 32;
    ShowWindow(p.warn, p.ready ? SW_HIDE : SW_SHOW);
    uiPlace(p.warn, 6, 4, W - 12, warnH);
    int top = warnH ? warnH + 6 : 0;
    int formH = std::max(60, H - top - actH - outH - outBtnH - playerH - 6);
    uiPlace(p.form, 0, top, W, formH);
    int y = top + formH + 4;
    y += actH;
    uiPlace(p.list, 8, y, W - 16, outH);
    ListView_SetColumnWidth(p.list, 1, std::max(120, W - 16 - 130 - 22));
    y += outH + 4;
    int bw = pagePlays(p) ? 150 : 96;
    uiPlace(p.bPreview, 8, y, bw, 26);
    uiPlace(p.bCopy, 8 + bw + 6, y, bw, 26);
    uiPlace(p.bDelete, 8 + 2 * (bw + 6), y, bw, 26);
    if (pagePlays(p)) {
        y += 32;
        uiPlace(p.lblNow, 8, y, W - 16, 18);
        y += 20;
        uiPlace(p.bPrev, 8, y, 60, 26); uiPlace(p.bPlay, 74, y, 80, 26); uiPlace(p.bStop, 160, y, 60, 26); uiPlace(p.bNext, 226, y, 60, 26);
        uiPlace(p.trk, 296, y, std::max(60, W - 296 - 8), 26);
    }
    layoutForm(p);
}

void refreshReady(Page& p) {
    std::string why;
    p.ready = Settings::get().tabReady(p.kind, &why);
    if (!p.ready) uiSetText(p.warn, "Warning: " + why + " Create is inactive until you select a proper model in [Settings] for this tab.");
    EnableWindow(p.bCreate, p.ready);
    layoutPage(p);
}

// ----------------------------------------------------------------- outputs list
int outIndexOf(const Page& p, const std::string& path);
void refreshOutputs(Page& p) {
    auto jobs = JobQueue::get().snapshot();
    std::vector<OutItem> items;
    std::string sig;
    for (auto it = jobs.rbegin(); it != jobs.rend(); ++it) {
        if (it->kind != p.kind || it->state != JobState::Done) continue;
        for (auto& o : it->outputs) {
            OutItem oi; oi.jobId = it->id; oi.path = o;
            if (p.kind == Kind::LLM) oi.text = it->text;
            oi.when = "job " + std::to_string(it->id);
            items.push_back(oi); sig += std::to_string(it->id) + o + "|";
        }
    }
    if (sig == p.sig) return;
    // what was selected (by file, the list is newest first so row numbers shift) and which file was the newest before
    int selRow = ListView_GetNextItem(p.list, -1, LVNI_SELECTED);
    const std::string selPath = selRow >= 0 && selRow < (int)p.outs.size() ? p.outs[selRow].path : std::string();
    const std::string oldNewest = p.outs.empty() ? std::string() : p.outs[0].path;
    p.sig = sig; p.outs = items;
    ListView_DeleteAllItems(p.list);
    for (size_t i = 0; i < items.size(); i++) {
        LVITEMW it = {}; it.mask = LVIF_TEXT; it.iItem = (int)i;
        std::wstring w = sys::widen(items[i].when); it.pszText = &w[0];
        ListView_InsertItem(p.list, &it);
        std::string d = items[i].text.empty() ? items[i].path.substr(items[i].path.find_last_of("\\/") + 1) : items[i].text;
        for (auto& c : d) if (c == '\r' || c == '\n') c = ' ';
        if (d.size() > 160) d.resize(160);
        std::wstring wd = sys::widen(d); ListView_SetItemText(p.list, (int)i, 1, &wd[0]);
    }
    // a freshly generated result becomes the selected one (so Play / Preview / Copy / Delete act on it); otherwise the selection stays on its file
    int want = -1;
    if (!items.empty() && items[0].path != oldNewest) want = 0;
    else if (!selPath.empty()) want = outIndexOf(p, selPath);
    if (want >= 0) { ListView_SetItemState(p.list, want, LVIS_SELECTED | LVIS_FOCUSED, LVIS_SELECTED | LVIS_FOCUSED); ListView_EnsureVisible(p.list, want, FALSE); }
}

bool isAudio(Kind k) { return k == Kind::TTS || k == Kind::Audio; }

void stopPlayback() { Player::get().release(); }
// Play / Stop on a file: a second press on the file that is playing stops it (and rewinds)
void playFile(const std::string& path) {
    Player& pl = Player::get();
    if (pl.current() == path && pl.playing()) { pl.stop(); return; }
    pl.play(path);
}

// ----- the small player under the list on the audio tabs
int outIndexOf(const Page& p, const std::string& path) {
    for (size_t i = 0; i < p.outs.size(); i++) if (p.outs[i].path == path) return (int)i;
    return -1;
}
void selectOut(Page& p, int i) {
    ListView_SetItemState(p.list, -1, 0, LVIS_SELECTED);
    if (i >= 0 && i < (int)p.outs.size()) { ListView_SetItemState(p.list, i, LVIS_SELECTED | LVIS_FOCUSED, LVIS_SELECTED | LVIS_FOCUSED); ListView_EnsureVisible(p.list, i, FALSE); }
}
void playerStep(Page& p, int dir) {                  // previous / next file of the list (newest first); starts it
    Player& pl = Player::get();
    int cur = outIndexOf(p, pl.current());
    if (cur < 0) cur = ListView_GetNextItem(p.list, -1, LVNI_SELECTED);
    if (p.outs.empty()) return;
    int n = cur < 0 ? (dir > 0 ? 0 : (int)p.outs.size() - 1) : cur + dir;
    if (n < 0 || n >= (int)p.outs.size()) return;
    selectOut(p, n);
    pl.play(p.outs[n].path);
}
void playerToggle(Page& p) {                         // Play/Pause on the file selected in the list (the newest result after a generation)
    Player& pl = Player::get();
    int sel = ListView_GetNextItem(p.list, -1, LVNI_SELECTED);
    if (sel < 0 && !p.outs.empty()) sel = 0;
    if (sel >= 0 && sel < (int)p.outs.size() && p.outs[sel].path != pl.current()) {      // another file than the loaded one: start the selected one
        selectOut(p, sel);
        pl.play(p.outs[sel].path);
        return;
    }
    if (pl.loaded()) pl.toggle();
}
void updatePlayer(Page& p) {
    if (!pagePlays(p)) return;
    Player& pl = Player::get();
    bool playing = pl.playing();
    std::string name = pl.loaded() ? pl.current().substr(pl.current().find_last_of("\\/") + 1) : "";
    std::string txt = !pl.loaded() ? "Nothing playing" : name + "    " + Player::timeText(pl.positionMs()) + " / " + Player::timeText(pl.lengthMs()) + (playing ? "" : "   (stopped)");
    if (uiText(p.lblNow) != txt) uiSetText(p.lblNow, txt);
    std::string cap = playing ? "Pause" : "Play";
    if (uiText(p.bPlay) != cap) uiSetText(p.bPlay, cap);
    if (!p.seeking) {
        long long len = pl.lengthMs();
        int pos = len > 0 ? (int)(pl.positionMs() * 1000 / len) : 0;
        if ((int)SendMessageW(p.trk, TBM_GETPOS, 0, 0) != pos) SendMessageW(p.trk, TBM_SETPOS, TRUE, pos);
    }
    EnableWindow(p.trk, pl.loaded());
    EnableWindow(p.bStop, pl.loaded());
}

void onOutputAction(Page& p, int id) {
    int sel = ListView_GetNextItem(p.list, -1, LVNI_SELECTED);
    if (sel < 0 || sel >= (int)p.outs.size()) { MessageBoxW(g_main, L"Select an item in the list first.", L"Dopplers Bakery", MB_OK | MB_ICONINFORMATION); return; }
    OutItem it = p.outs[sel];
    if (id == IDC_PREVIEW) {
        if (p.kind == Kind::LLM) showTextViewer(g_main, "LLM output - " + it.when, it.text);
        else if (isAudio(p.kind)) playFile(it.path);
        else ShellExecuteW(g_main, L"open", sys::widen(it.path).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    } else if (id == IDC_COPY) {
        if (p.kind == Kind::LLM) clipboardSetText(g_main, it.text);
        else clipboardSetFile(g_main, it.path);
    } else if (id == IDC_DELETE) {
        if (Player::get().current() == it.path) Player::get().release();      // MCI keeps the file open: let go of it before deleting
        std::wstring w = sys::widen(it.path); w.push_back(0); w.push_back(0);
        SHFILEOPSTRUCTW op = {}; op.hwnd = g_main; op.wFunc = FO_DELETE; op.pFrom = w.c_str();
        op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT;
        SHFileOperationW(&op);                       // to the Recycle Bin
        JobQueue::get().dropOutput(it.jobId, it.path);
        p.sig.clear(); refreshOutputs(p);
    }
}

void onCreateClicked(Page& p) {
    Json form = readForm(p);
    Settings::get().setForm(p.kind, form);
    std::string main = p.kind == Kind::TTS ? form.str("text") : form.str("prompt");
    if (main.find_first_not_of(" \r\n\t") == std::string::npos) {
        MessageBoxW(g_main, L"The prompt is empty.", L"Dopplers Bakery", MB_OK | MB_ICONINFORMATION); return;
    }
    Json params = makeJobParams(p.kind, form);
    JobQueue::get().add(p.kind, params, jobTitle(p.kind, params));
}

// ----------------------------------------------------------------- page window procs
LRESULT CALLBACK pageProc(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    if (m == WM_NCCREATE) SetWindowLongPtrW(h, GWLP_USERDATA, (LONG_PTR)((CREATESTRUCTW*)lp)->lpCreateParams);
    Page* p = (Page*)GetWindowLongPtrW(h, GWLP_USERDATA);
    if (p) switch (m) {
    case WM_COMMAND:
        switch (LOWORD(wp)) {
        case IDC_CREATE: onCreateClicked(*p); return 0;
        default:
            if (p->kind == Kind::Image && LOWORD(wp) >= IDC_FIELD0 && LOWORD(wp) < IDC_FIELD0 + 64 && (HIWORD(wp) == CBN_SELCHANGE || HIWORD(wp) == CBN_EDITCHANGE || HIWORD(wp) == CBN_CLOSEUP)) { applyImageLimits(*p, HIWORD(wp) == CBN_SELCHANGE); return 0; }
            if (LOWORD(wp) >= IDC_BROWSE0 && LOWORD(wp) < IDC_BROWSE0 + 64) {          // "..." next to a file field
                size_t idx = LOWORD(wp) - IDC_BROWSE0;
                if (idx < p->fc.size()) {
                    std::string path;
                    bool img = std::string(p->fc[idx].f->key) == "init_image";
                    if (pickFile(g_main, path, img ? L"Images\0*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.tif;*.webp\0All files\0*.*\0" : L"WAV audio\0*.wav\0All files\0*.*\0")) uiSetText(p->fc[idx].ctl, path);
                }
                return 0;
            }
            break;
        case IDC_SETTINGS: if (showModelSettings(g_main, p->kind)) refreshReady(*p); return 0;
        case IDC_PREVIEW: case IDC_COPY: case IDC_DELETE: onOutputAction(*p, LOWORD(wp)); return 0;
        case IDC_EXTEND:                                                // lengths beyond the five-minute trackbar, up to what the model can make
            for (auto& c : p->fc) if (!strcmp(c.f->key, "length")) {
                int cur = c.ext > 0 ? c.ext : (int)SendMessageW(c.ctl, TBM_GETPOS, 0, 0), v = 0;
                if (askNumber(g_main, "Extend the length",
                              "Length in seconds. The trackbar goes up to " + std::to_string(kAudioSliderMax) + " s (5 minutes); the audio model can make up to " +
                              std::to_string(kAudioModelMax) + " s (" + std::to_string(kAudioModelMax / 60) + " min " + std::to_string(kAudioModelMax % 60) + " s). Enter the length you want:",
                              cur, 1, kAudioModelMax, v)) {
                    c.ext = v > kAudioSliderMax ? v : 0;
                    SendMessageW(c.ctl, TBM_SETPOS, TRUE, std::min(v, kAudioSliderMax));
                    sliderLabel(c);
                }
                RedrawWindow(p->form, nullptr, nullptr, RDW_INVALIDATE | RDW_ERASE | RDW_ALLCHILDREN | RDW_UPDATENOW);       // the dialog sat on top of the form
            }
            return 0;
        case IDC_PL_PREV: playerStep(*p, -1); updatePlayer(*p); return 0;
        case IDC_PL_NEXT: playerStep(*p, +1); updatePlayer(*p); return 0;
        case IDC_PL_PLAY: playerToggle(*p); updatePlayer(*p); return 0;
        case IDC_PL_STOP: Player::get().stop(); updatePlayer(*p); return 0;
        }
        break;
    case WM_HSCROLL:                                                    // the position bar of the audio tabs
        if ((HWND)lp == p->trk) {
            int code = LOWORD(wp);
            p->seeking = code != TB_ENDTRACK;
            Player::get().seek((double)SendMessageW(p->trk, TBM_GETPOS, 0, 0) / 1000.0);
            if (code == TB_ENDTRACK) p->seeking = false;
            updatePlayer(*p);
            return 0;
        }
        break;
    case WM_NOTIFY:
        if (((NMHDR*)lp)->idFrom == IDC_OUTLIST && ((NMHDR*)lp)->code == NM_DBLCLK) { onOutputAction(*p, IDC_PREVIEW); return 0; }
        break;
    case WM_CTLCOLORSTATIC:
        if ((HWND)lp == p->warn) { SetTextColor((HDC)wp, RGB(185, 30, 30)); SetBkMode((HDC)wp, TRANSPARENT); return (LRESULT)GetSysColorBrush(COLOR_BTNFACE); }
        break;
    }
    return DefWindowProcW(h, m, wp, lp);
}

LRESULT CALLBACK formProc(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    if (m == WM_NCCREATE) SetWindowLongPtrW(h, GWLP_USERDATA, (LONG_PTR)((CREATESTRUCTW*)lp)->lpCreateParams);
    Page* p = (Page*)GetWindowLongPtrW(h, GWLP_USERDATA);
    if (p) switch (m) {
    case WM_VSCROLL: {
        SCROLLINFO si = {sizeof si, SIF_ALL}; GetScrollInfo(h, SB_VERT, &si);
        int pos = si.nPos;
        switch (LOWORD(wp)) {
        case SB_LINEUP: pos -= 24; break; case SB_LINEDOWN: pos += 24; break;
        case SB_PAGEUP: pos -= si.nPage; break; case SB_PAGEDOWN: pos += si.nPage; break;
        case SB_THUMBTRACK: case SB_THUMBPOSITION: pos = si.nTrackPos; break;
        }
        p->scroll = pos; layoutForm(*p); return 0;
    }
    case WM_HSCROLL:       // trackbars live in the form
        for (auto& c : p->fc) if (c.ctl == (HWND)lp && c.f->type == FT::Slider) { c.ext = 0; sliderLabel(c); }      // moving the trackbar ends an extended length
        return 0;
    case WM_DRAWITEM:
        if (wp == IDC_CREATE) { drawCreateButton((DRAWITEMSTRUCT*)lp); return TRUE; }
        break;
    case WM_COMMAND: case WM_NOTIFY: return SendMessageW(GetParent(h), m, wp, lp);
    case WM_CTLCOLORSTATIC: case WM_CTLCOLOREDIT: return DefWindowProcW(h, m, wp, lp);
    }
    return DefWindowProcW(h, m, wp, lp);
}

void buildPage(Page& p, Kind kind) {
    p.kind = kind;
    p.hwnd = CreateWindowExW(0, L"BakeryPage", L"", WS_CHILD | WS_CLIPCHILDREN, 0, 0, 10, 10, g_main, nullptr, GetModuleHandleW(nullptr), &p);
    p.form = CreateWindowExW(0, L"BakeryForm", L"", WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_CLIPCHILDREN, 0, 0, 10, 10, p.hwnd, nullptr, GetModuleHandleW(nullptr), &p);
    p.warn = uiMake(p.hwnd, L"STATIC", "", SS_LEFT);
    int i = 0;
    for (auto& f : fieldsOf(kind)) {
        FieldCtl c; c.f = &f;
        int id = IDC_FIELD0 + i++;
        if (f.type != FT::Check) c.label = uiMake(p.form, L"STATIC", f.label, SS_LEFT);
        switch (f.type) {
        case FT::Multi: c.ctl = uiMake(p.form, L"EDIT", "", ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN | WS_VSCROLL | WS_TABSTOP, id, WS_EX_CLIENTEDGE); break;
        case FT::Text: case FT::Int: case FT::Float: c.ctl = uiMake(p.form, L"EDIT", "", ES_AUTOHSCROLL | WS_TABSTOP, id, WS_EX_CLIENTEDGE); break;
        case FT::File:
            c.ctl = uiMake(p.form, L"EDIT", "", ES_AUTOHSCROLL | WS_TABSTOP, id, WS_EX_CLIENTEDGE);
            c.btn = uiMake(p.form, L"BUTTON", "...", BS_PUSHBUTTON | WS_TABSTOP, IDC_BROWSE0 + (id - IDC_FIELD0));
            break;
        case FT::Combo: {
            c.ctl = uiMake(p.form, L"COMBOBOX", "", CBS_DROPDOWN | CBS_AUTOHSCROLL | WS_VSCROLL | WS_TABSTOP, id);
            std::string items = f.items; size_t s = 0;
            while (s <= items.size()) {
                size_t e = items.find('|', s); if (e == std::string::npos) e = items.size();
                SendMessageW(c.ctl, CB_ADDSTRING, 0, (LPARAM)sys::widen(items.substr(s, e - s)).c_str());
                s = e + 1;
            }
            break;
        }
        case FT::Check: c.ctl = uiMake(p.form, L"BUTTON", f.label, BS_AUTOCHECKBOX | WS_TABSTOP, id); break;
        case FT::Slider:
            c.ctl = uiMake(p.form, TRACKBAR_CLASSW, "", TBS_HORZ | TBS_NOTICKS | WS_TABSTOP, id);
            SendMessageW(c.ctl, TBM_SETRANGE, TRUE, MAKELONG((int)f.mn, (int)f.mx));
            SendMessageW(c.ctl, TBM_SETPAGESIZE, 0, 50);
            break;
        }
        p.fc.push_back(c);
    }
    if (kind == Kind::Audio) p.bExtend = uiMake(p.form, L"BUTTON", "Extend...", BS_PUSHBUTTON | WS_TABSTOP, IDC_EXTEND);
    p.bCreate = uiMake(p.form, L"BUTTON", kind == Kind::Audio ? "Create  (F2)" : "Create", BS_OWNERDRAW | WS_TABSTOP, IDC_CREATE);
    SetWindowSubclass(p.bCreate, hoverProc, 1, 0);
    p.bSettings = uiMake(p.form, L"BUTTON", "Model settings...", BS_PUSHBUTTON | WS_TABSTOP, IDC_SETTINGS);
    SendMessageW(p.bSettings, WM_SETFONT, (WPARAM)btnFont(false), TRUE);
    p.list = uiMake(p.hwnd, WC_LISTVIEWW, "", LVS_REPORT | LVS_SINGLESEL | LVS_SHOWSELALWAYS | WS_TABSTOP, IDC_OUTLIST, WS_EX_CLIENTEDGE);
    ListView_SetExtendedListViewStyle(p.list, LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES);
    const wchar_t* cols[] = {L"Result of", L"Output (this session)"}; int cw[] = {90, 300};
    for (int k = 0; k < 2; k++) { LVCOLUMNW c = {}; c.mask = LVCF_TEXT | LVCF_WIDTH; c.pszText = (LPWSTR)cols[k]; c.cx = cw[k]; ListView_InsertColumn(p.list, k, &c); }
    bool audio = isAudio(kind);
    const bool music = kind == Kind::Audio;           // the Audio (music) tab also has keyboard shortcuts: F2 / F3 / Shift+Delete
    p.bPreview = uiMake(p.hwnd, L"BUTTON", music ? "Play / Stop latest (F3)" : audio ? "Play / Stop" : "Preview", BS_PUSHBUTTON | WS_TABSTOP, IDC_PREVIEW);
    p.bCopy = uiMake(p.hwnd, L"BUTTON", kind == Kind::LLM ? "Copy text" : "Copy file", BS_PUSHBUTTON | WS_TABSTOP, IDC_COPY);
    p.bDelete = uiMake(p.hwnd, L"BUTTON", music ? "Delete (Shift+Del)" : "Delete", BS_PUSHBUTTON | WS_TABSTOP, IDC_DELETE);
    if (audio) {
        p.lblNow = uiMake(p.hwnd, L"STATIC", "Nothing playing", SS_LEFT | SS_ENDELLIPSIS);
        p.bPrev = uiMake(p.hwnd, L"BUTTON", "Prev", BS_PUSHBUTTON | WS_TABSTOP, IDC_PL_PREV);
        p.bPlay = uiMake(p.hwnd, L"BUTTON", "Play", BS_PUSHBUTTON | WS_TABSTOP, IDC_PL_PLAY);
        p.bStop = uiMake(p.hwnd, L"BUTTON", "Stop", BS_PUSHBUTTON | WS_TABSTOP, IDC_PL_STOP);
        p.bNext = uiMake(p.hwnd, L"BUTTON", "Next", BS_PUSHBUTTON | WS_TABSTOP, IDC_PL_NEXT);
        p.trk = uiMake(p.hwnd, TRACKBAR_CLASSW, "", TBS_HORZ | TBS_NOTICKS | WS_TABSTOP, IDC_PL_TRK);
        SendMessageW(p.trk, TBM_SETRANGE, TRUE, MAKELONG(0, 1000));
    }
    Json saved = Settings::get().form(kind);
    writeForm(p, saved);
    applyImageLimits(p);
}

// ----------------------------------------------------------------- queue panel
std::string fmtTime(double s) {
    char b[32];
    if (s < 60) snprintf(b, sizeof b, "%.1f s", s);
    else if (s < 3600) snprintf(b, sizeof b, "%d:%02d", (int)s / 60, (int)s % 60);
    else snprintf(b, sizeof b, "%d:%02d:%02d", (int)s / 3600, (int)s % 3600 / 60, (int)s % 60);
    return b;
}
// "37%" while running, otherwise the outcome
std::string progressText(const Job& j) {
    switch (j.state) {
    case JobState::Running: return std::to_string((int)(j.progress * 100)) + "%";
    case JobState::Paused: return std::to_string((int)(j.progress * 100)) + "% paused";
    case JobState::Queued: return "Queued";
    case JobState::Held: return "Held";
    case JobState::Done: return "Done";
    case JobState::Failed: return "Failed";
    default: return "Canceled";
    }
}
// tokens/s (LLM), image size, audio length - whatever applies to the job
std::string infoText(const Job& j) {
    if (!j.info.empty()) return j.info;
    if (j.kind == Kind::Image && j.params.has("width")) return std::to_string(j.params.integer("width", 0)) + "x" + std::to_string(j.params.integer("height", 0));
    if (j.kind == Kind::Audio && j.params.has("length")) return std::to_string(j.params.integer("length", 0)) + " s";
    return "";
}
std::string timeText(const Job& j) {
    if (j.state == JobState::Running || j.state == JobState::Paused) return j.tStart > 0 ? fmtTime(sys::now() - j.tStart) : "";
    if (j.state == JobState::Done || j.state == JobState::Failed || j.state == JobState::Canceled) return j.tStart > 0 && j.tEnd >= j.tStart ? fmtTime(j.tEnd - j.tStart) : "";
    return "";
}

int selectedJob() {
    int s = ListView_GetNextItem(g_qList, -1, LVNI_SELECTED);
    return s >= 0 && s < (int)g_qIds.size() ? g_qIds[s] : 0;
}

std::string crlf(const std::string& s) { std::string r; for (char c : s) { if (c == '\n' && (r.empty() || r.back() != '\r')) r += '\r'; r += c; } return r; }
std::string baseName(const std::string& p) { return p.substr(p.find_last_of("\\/") + 1); }

// everything about a job: prompt first, then outcome, then the settings it ran with
std::string jobDetails(const Job& j) {
    std::string s = "Job " + std::to_string(j.id) + " - " + kindTitle(j.kind) + " - " + progressText(j) + "\n";
    std::string meta;
    if (!timeText(j).empty()) meta += "Time: " + timeText(j) + "   ";
    if (!infoText(j).empty()) meta += (j.kind == Kind::LLM ? "Speed: " : j.kind == Kind::Image ? "Size: " : "Length: ") + infoText(j);
    if (!meta.empty()) s += meta + "\n";
    if (!j.status.empty() && j.state != JobState::Done) s += "Status: " + j.status + "\n";
    if (!j.error.empty()) s += "Error: " + j.error + "\n";
    std::string prompt = j.kind == Kind::TTS ? j.params.str("text") : j.params.str("prompt");
    if (j.params.has("pre_prompt") && !j.params.str("pre_prompt").empty()) prompt = j.params.str("pre_prompt") + "\n" + prompt;
    if (prompt.empty() && j.params.get("messages").t == Json::Arr && !j.params.get("messages").a.empty()) {
        s += "\nConversation (last message):\n" + j.params.get("messages").a.back().str("content") + "\n";
    } else if (!prompt.empty()) s += "\nPrompt:\n" + prompt + "\n";
    if (!j.params.str("negative").empty()) s += "\nNegative prompt:\n" + j.params.str("negative") + "\n";
    if (!j.text.empty() && j.kind == Kind::LLM) s += "\nResult:\n" + j.text + "\n";
    else if (!j.text.empty() && j.kind == Kind::Audio) s += "\nExpanded prompt used:\n" + j.text + "\n";
    if (!j.outputs.empty()) { s += "\nFiles:\n"; for (auto& o : j.outputs) s += o + "\n"; }
    std::string st;
    for (auto& kv : j.params.o) {
        const std::string& k = kv.first; const Json& v = kv.second;
        if (k == "prompt" || k == "text" || k == "negative" || k == "pre_prompt" || k == "messages" || k == "system" || k == "no_output") continue;
        if (v.t == Json::Str) { if (v.s.empty()) continue; st += "  " + k + ": " + ((k == "model" || k == "clip" || k == "vae" || k == "lora" || k == "llm" || k == "init_image" || k == "ref_audio") ? baseName(v.s) : v.s) + "\n"; }
        else if (v.t == Json::Num || v.t == Json::Bool) st += "  " + k + ": " + v.dump() + "\n";
    }
    if (!st.empty()) s += "\nSettings:\n" + st;
    return crlf(s);
}

std::vector<std::array<std::string, 3>> g_qCache;     // text of columns 2..4 per row, so the list is touched only when something changed
std::string g_qDetailText;

void refreshDetails() {
    std::string t;
    Job sel;
    if (int id = selectedJob()) { if (JobQueue::get().find(id, sel)) t = jobDetails(sel); }
    else t = "Click a job to see its prompt and details.";
    if (t == g_qDetailText) return;
    g_qDetailText = t;
    int first = (int)SendMessageW(g_qDetail, EM_GETFIRSTVISIBLELINE, 0, 0);
    uiSetText(g_qDetail, t);
    SendMessageW(g_qDetail, EM_LINESCROLL, 0, first);        // keep the reading position while a running job updates
}

// ----- display order (click a column header to sort: ascending, descending, back to queue order)
int g_qSortCol = -1; bool g_qSortAsc = true;
bool g_qDirty = true;                 // something changed outside refreshQueue (clear, move, sort): rebuild the rows
int g_qFollowId = 0;                  // running job the list was last scrolled to

double leadingNumber(const std::string& s) { return s.empty() ? 0.0 : atof(s.c_str()); }
double progressRank(const Job& j) {
    switch (j.state) {
    case JobState::Queued: return -2; case JobState::Held: return -1;
    case JobState::Running: case JobState::Paused: return j.progress * 100.0;
    case JobState::Done: return 200; case JobState::Failed: return 201; default: return 202;
    }
}
double seconds(const Job& j) {
    if (j.tStart <= 0) return 0;
    if (j.state == JobState::Running || j.state == JobState::Paused) return sys::now() - j.tStart;
    return j.tEnd >= j.tStart ? j.tEnd - j.tStart : 0;
}
std::vector<const Job*> displayOrder(const std::vector<Job>& jobs) {
    std::vector<const Job*> o;
    for (auto& j : jobs) o.push_back(&j);
    if (g_qSortCol < 0) return o;
    const int c = g_qSortCol; const bool asc = g_qSortAsc;
    std::stable_sort(o.begin(), o.end(), [&](const Job* a, const Job* b) {
        double ka = 0, kb = 0; std::string sa, sb; bool text = false;
        switch (c) {
        case 0: ka = a->id; kb = b->id; break;
        case 1: sa = kindTitle(a->kind); sb = kindTitle(b->kind); text = true; break;
        case 2: ka = progressRank(*a); kb = progressRank(*b); break;
        case 3: ka = leadingNumber(infoText(*a)); kb = leadingNumber(infoText(*b)); break;
        default: ka = seconds(*a); kb = seconds(*b); break;
        }
        if (text) return asc ? sa < sb : sa > sb;
        return asc ? ka < kb : ka > kb;
    });
    return o;
}
void showSortArrow() {
    HWND hd = ListView_GetHeader(g_qList);
    for (int i = 0; i < 5; i++) {
        HDITEMW it = {}; it.mask = HDI_FORMAT;
        Header_GetItem(hd, i, &it);
        it.fmt &= ~(HDF_SORTUP | HDF_SORTDOWN);
        if (i == g_qSortCol) it.fmt |= g_qSortAsc ? HDF_SORTUP : HDF_SORTDOWN;
        Header_SetItem(hd, i, &it);
    }
}
void onQueueColumnClick(int col) {
    if (g_qSortCol == col) { if (g_qSortAsc) g_qSortAsc = false; else g_qSortCol = -1; }
    else { g_qSortCol = col; g_qSortAsc = true; }
    showSortArrow();
    for (int i = 1; i <= 2; i++) EnableWindow(g_qBtn[i - 1], g_qSortCol < 0);                  // Move up / Move down act on the queue order
    g_qDirty = true;
}
int listRowHeight() {
    RECT r;
    if (ListView_GetItemRect(g_qList, 0, &r, LVIR_BOUNDS)) return std::max<int>(1, r.bottom - r.top);
    return 19;
}
void scrollListBy(int rows) { if (rows) ListView_Scroll(g_qList, 0, rows * listRowHeight()); }

void refreshQueue() {
    auto jobs = JobQueue::get().snapshot();
    auto order = displayOrder(jobs);

    std::string sig = std::to_string(g_qSortCol) + (g_qSortAsc ? "a" : "d") + ":";
    for (auto* j : order) { sig += std::to_string(j->id); if (g_qSortCol >= 2) sig += stateName(j->state); sig += ";"; }
    if (g_qDirty || sig != g_qSig) {                         // rows added, removed, reordered or re-sorted: rebuild, but keep the scroll position
        g_qDirty = false; g_qSig = sig;
        int keep = selectedJob(), top = ListView_GetTopIndex(g_qList);
        SendMessageW(g_qList, WM_SETREDRAW, FALSE, 0);
        ListView_DeleteAllItems(g_qList);
        g_qIds.clear(); g_qCache.clear();
        int row = 0;
        for (auto* jp : order) {
            const Job& j = *jp;
            LVITEMW it = {}; it.mask = LVIF_TEXT; it.iItem = row;
            std::wstring a = sys::widen(std::to_string(j.id)); it.pszText = &a[0];
            ListView_InsertItem(g_qList, &it);
            std::wstring b = sys::widen(kindTitle(j.kind));
            ListView_SetItemText(g_qList, row, 1, &b[0]);
            g_qIds.push_back(j.id); g_qCache.push_back({"\x01", "\x01", "\x01"});
            if (j.id == keep) ListView_SetItemState(g_qList, row, LVIS_SELECTED | LVIS_FOCUSED, LVIS_SELECTED | LVIS_FOCUSED);
            row++;
        }
        scrollListBy(std::min(top, std::max(0, row - 1)));   // back to where the user was looking
        SendMessageW(g_qList, WM_SETREDRAW, TRUE, 0);
        InvalidateRect(g_qList, nullptr, TRUE);
    }
    // the job that is running now: keep it in view, about the 4th row from the top, when it starts
    int runningId = 0, runningRow = -1;
    for (size_t r = 0; r < order.size(); r++) if (order[r]->state == JobState::Running) { runningId = order[r]->id; runningRow = (int)r; break; }
    if (runningId != g_qFollowId) {
        g_qFollowId = runningId;
        if (runningId && ListView_GetCountPerPage(g_qList) < (int)order.size()) scrollListBy(std::max(0, runningRow - 3) - ListView_GetTopIndex(g_qList));
    }
    for (size_t row = 0; row < order.size() && row < g_qCache.size(); row++) {       // live cells: progress, info, time
        std::array<std::string, 3> now = {progressText(*order[row]), infoText(*order[row]), timeText(*order[row])};
        for (int c = 0; c < 3; c++) if (now[c] != g_qCache[row][c]) {
            g_qCache[row][c] = now[c];
            std::wstring w = sys::widen(now[c]); ListView_SetItemText(g_qList, (int)row, 2 + c, &w[0]);
        }
    }
    // how many jobs: "Queue: 7 jobs (1 running, 3 waiting, 3 done)"
    {
        int run = 0, wait = 0, done = 0, bad = 0;
        for (auto& j : jobs) {
            if (j.state == JobState::Running || j.state == JobState::Paused) run++;
            else if (j.state == JobState::Queued || j.state == JobState::Held) wait++;
            else if (j.state == JobState::Done) done++; else bad++;
        }
        std::string t = "Queue: " + std::to_string(jobs.size()) + (jobs.size() == 1 ? " job" : " jobs");
        if (!jobs.empty()) {
            t += " (";
            auto add = [&](int n, const char* w) { if (n) { if (t.back() != '(') t += ", "; t += std::to_string(n) + " " + w; } };
            add(run, "running"); add(wait, "waiting"); add(done, "done"); add(bad, "failed/canceled");
            t += ")";
        }
        if (uiText(g_qTitle) != t) uiSetText(g_qTitle, t);
    }
    // status line for the active job
    std::string stat = "Idle"; int prog = 0;
    for (auto& j : jobs) if (j.state == JobState::Running || j.state == JobState::Paused) {
        stat = std::string(j.state == JobState::Paused ? "Paused - " : "") + "Job " + std::to_string(j.id) + ": " + j.status;
        prog = (int)(j.progress * 1000);
    }
    uiSetText(g_qStat, stat);
    SendMessageW(g_qProg, PBM_SETPOS, prog, 0);
    // pause button caption follows the selected job
    Job sel; std::string cap = "Pause";
    if (int id = selectedJob()) if (JobQueue::get().find(id, sel) && (sel.state == JobState::Paused || sel.state == JobState::Held)) cap = "Resume";
    if (uiText(g_qBtn[3]) != cap) uiSetText(g_qBtn[3], cap);
    refreshDetails();
    for (auto& p : g_pages) refreshOutputs(p);
}

void onQueueCommand(int id) {
    auto& q = JobQueue::get();
    int jid = selectedJob();
    if (id == IDC_QCLEAR) { q.clearFinished(); g_qDirty = true; refreshQueue(); return; }
    if (!jid) { MessageBoxW(g_main, L"Select a job in the queue first.", L"Dopplers Bakery", MB_OK | MB_ICONINFORMATION); return; }
    Job j; if (!q.find(jid, j)) return;
    switch (id) {
    case IDC_QUP: q.moveUp(jid); break;
    case IDC_QDOWN: q.moveDown(jid); break;
    case IDC_QREMOVE:
        if (!q.remove(jid)) MessageBoxW(g_main, L"The active job cannot be removed - use Cancel first.", L"Dopplers Bakery", MB_OK | MB_ICONINFORMATION);
        break;
    case IDC_QPAUSE:
        if (j.state == JobState::Paused || j.state == JobState::Held) q.resume(jid); else q.pause(jid);
        break;
    case IDC_QCANCEL: q.cancel(jid); break;
    case IDC_QDETAILS: showTextViewer(g_main, "Job " + std::to_string(jid), jobDetails(j)); break;
    }
    g_qDirty = true; refreshQueue();
    // keep the moved job selected
    for (size_t i = 0; i < g_qIds.size(); i++) if (g_qIds[i] == jid) ListView_SetItemState(g_qList, (int)i, LVIS_SELECTED | LVIS_FOCUSED, LVIS_SELECTED | LVIS_FOCUSED);
}

void buildQueue() {
    g_qTitle = uiMake(g_main, L"STATIC", "Queue", SS_LEFT);
    g_qList = uiMake(g_main, WC_LISTVIEWW, "", LVS_REPORT | LVS_SINGLESEL | LVS_SHOWSELALWAYS | WS_TABSTOP, IDC_QLIST, WS_EX_CLIENTEDGE);
    ListView_SetExtendedListViewStyle(g_qList, LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES);
    const wchar_t* cols[] = {L"#", L"Tab", L"Progress", L"Speed / size / length", L"Time"}; int cw[] = {34, 90, 90, 130, 70};
    for (int i = 0; i < 5; i++) { LVCOLUMNW c = {}; c.mask = LVCF_TEXT | LVCF_WIDTH; c.pszText = (LPWSTR)cols[i]; c.cx = cw[i]; ListView_InsertColumn(g_qList, i, &c); }
    const char* names[] = {"Move up", "Move down", "Remove", "Pause", "Cancel", "Clear finished", "Details...", ""};
    int ids[] = {IDC_QUP, IDC_QDOWN, IDC_QREMOVE, IDC_QPAUSE, IDC_QCANCEL, IDC_QCLEAR, IDC_QDETAILS, 0};
    for (int i = 0; i < 7; i++) g_qBtn[i] = uiMake(g_main, L"BUTTON", names[i], BS_PUSHBUTTON | WS_TABSTOP, ids[i]);
    g_qDetail = uiMake(g_main, L"EDIT", "Click a job to see its prompt and details.", ES_MULTILINE | ES_AUTOVSCROLL | ES_READONLY | WS_VSCROLL, IDC_QDETAIL, WS_EX_CLIENTEDGE);
    g_qStat = uiMake(g_main, L"STATIC", "Idle", SS_LEFT | SS_ENDELLIPSIS);
    g_qProg = uiMake(g_main, PROGRESS_CLASSW, "", PBS_SMOOTH, IDC_QPROG);
    SendMessageW(g_qProg, PBM_SETRANGE32, 0, 1000);
    g_qToggle = uiMake(g_main, L"BUTTON", "Hide", BS_PUSHBUTTON | WS_TABSTOP, IDC_QTOGGLE);
}

// ----- queue panel: can be hidden, dragged wider/narrower (splitter), columns resized with the header
const int kSplit = 8, kToggleW = 22, kQMin = 220;
bool g_qVis = true, g_qColsCustom = false, g_qDrag = false;
int g_qW = 0;                                   // 0 = automatic (36 % of the window)


int queueWidth(int W) {
    if (!g_qVis) return 0;
    int q = g_qW > 0 ? g_qW : std::min(std::max(W * 36 / 100, kQueueMinW), 520);
    return std::max(kQMin, std::min(q, W - 420));
}
RECT splitterRect() {
    RECT rc; GetClientRect(g_main, &rc);
    int qW = queueWidth(rc.right);
    return {rc.right - qW - kSplit, 0, rc.right - qW, rc.bottom};
}
void saveQueueUi() {
    Settings& s = Settings::get();
    s.setUiInt("queue_visible", g_qVis ? 1 : 0); s.setUiInt("queue_width", g_qW);
    if (g_qColsCustom) for (int i = 0; i < 5; i++) s.setUiInt("qc2_" + std::to_string(i), ListView_GetColumnWidth(g_qList, i));
}
void loadQueueUi() {
    Settings& s = Settings::get();
    g_qVis = s.uiInt("queue_visible", 1) != 0; g_qW = s.uiInt("queue_width", 0);
    if (s.uiInt("qc2_0", 0) > 0) {
        for (int i = 0; i < 5; i++) { int w = s.uiInt("qc2_" + std::to_string(i), 0); if (w > 0) ListView_SetColumnWidth(g_qList, i, w); }
        g_qColsCustom = true;
    }
}
void setQueueVisible(bool v) {
    g_qVis = v;
    HMENU m = GetMenu(g_main); if (m) CheckMenuItem(m, IDM_QUEUE, MF_BYCOMMAND | (v ? MF_CHECKED : MF_UNCHECKED));
    saveQueueUi();
}

void layoutMainImpl() {
    RECT rc; GetClientRect(g_main, &rc);
    int W = rc.right, H = rc.bottom;
    int qW = queueWidth(W);
    int tabW = g_qVis ? W - qW - kSplit : W - kToggleW - 2;
    for (HWND c : {g_qTitle, g_qList, g_qDetail, g_qStat, g_qProg}) ShowWindow(c, g_qVis ? SW_SHOW : SW_HIDE);
    for (int i = 0; i < 7; i++) ShowWindow(g_qBtn[i], g_qVis ? SW_SHOW : SW_HIDE);
    uiPlace(g_tab, 4, 4, tabW - 4, H - 8);
    RECT tr = {4, 4, 4 + tabW - 4, 4 + H - 8};
    TabCtrl_AdjustRect(g_tab, FALSE, &tr);
    for (auto& p : g_pages) { uiPlace(p.hwnd, tr.left, tr.top, tr.right - tr.left, tr.bottom - tr.top); }
    uiPlace(chatViewHwnd(), tr.left, tr.top, tr.right - tr.left, tr.bottom - tr.top);
    int x = W - qW, w = qW - 6;
    uiSetText(g_qToggle, g_qVis ? "Hide \xE2\x80\xBA" : "\xE2\x80\xB9");
    if (g_qVis) uiPlace(g_qToggle, x + w - 58, 3, 58, 22); else uiPlace(g_qToggle, W - kToggleW - 1, H / 2 - 40, kToggleW, 80);
    if (!g_qVis) return;
    uiPlace(g_qTitle, x, 6, w - 62, 18);
    int listH = H - 6 - 22 - 22 - 66 - 28 - 30;
    int detailH = std::min(190, std::max(90, listH / 3)), listH2 = listH - detailH - 6;
    uiPlace(g_qList, x, 26, w, listH2);
    uiPlace(g_qDetail, x, 26 + listH2 + 6, w, detailH);
    int bw = (w - 8) / 3, y = 26 + listH + 6;
    for (int i = 0; i < 7; i++) { int r = i / 3, c = i % 3; uiPlace(g_qBtn[i], x + c * (bw + 4), y + r * 30, bw, 26); }
    uiPlace(g_qStat, x, H - 46, w, 18);
    uiPlace(g_qProg, x, H - 24, w, 16);

}

void layoutMain() {
    uiBatchBegin(g_main, 64);                    // move every child of the window in one go (no flicker, no half-laid-out frames)
    layoutMainImpl();
    uiBatchEnd();
    for (auto& p : g_pages) layoutPage(p);       // the pages read their own size, so they follow once the move is done
    RedrawWindow(g_main, nullptr, nullptr, RDW_INVALIDATE | RDW_ERASE | RDW_ALLCHILDREN);
}

void showSelectedPage() {
    int s = TabCtrl_GetCurSel(g_tab);
    // hide what is not wanted first, then show and fully repaint the chosen page (it must not be painted while another page still covers it)
    SetWindowPos(g_tab, HWND_BOTTOM, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);              // the tab control covers the whole content area: keep it below the pages
    chatViewShow(s == 0);                                                                                  // tab 0 is Chat
    for (int i = 0; i < kKindCount; i++) if (i + 1 != s) ShowWindow(g_pages[i].hwnd, SW_HIDE);
    if (s >= 1 && s <= kKindCount) {
        Page& pg = g_pages[s - 1];
        ShowWindow(pg.hwnd, SW_SHOW);
        layoutPage(pg);
        RedrawWindow(pg.hwnd, nullptr, nullptr, RDW_INVALIDATE | RDW_ERASE | RDW_ALLCHILDREN | RDW_UPDATENOW);
    }
}

void refreshAllReady() { for (auto& p : g_pages) refreshReady(p); }

void firstRun() {
    Settings::get().autoSelect();
    if (!Settings::get().firstRunDone()) {
        bool missing = false;
        for (auto& e : loadCatalog()) if (e.downloadable() && !catalogInstalled(e)) missing = true;
        if (missing) showDownloadDialog(g_main, true);
        Settings::get().setFirstRunDone(true);
        Settings::get().autoSelect();
    }
    refreshAllReady();
    if (!g_recovered.empty()) {                                   // the previous run ended inside a GPU job: explain and what changed
        std::string what = g_recovered; g_recovered.clear();
        MessageBoxW(g_main, sys::widen("The previous session ended unexpectedly while it was working on:\n\n" + what +
                    "\n\nTo avoid another crash, pictures, speech and audio now use the CPU (slower but safe). To try the GPU again choose Backend = Vulkan on the Text to image tab; "
                    "if that job succeeds the app goes back to using the GPU on its own.\n\nTechnical details are in bakery.log next to the program.").c_str(),
                    L"Dopplers Bakery", MB_OK | MB_ICONWARNING);
    }
}

HMENU buildMenu() {
    HMENU bar = CreateMenu(), file = CreatePopupMenu(), models = CreatePopupMenu(), help = CreatePopupMenu();
    AppendMenuW(file, MF_STRING, IDM_OPENOUT, L"Open output folder");
    AppendMenuW(file, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(file, MF_STRING, IDM_EXIT, L"Exit");
    AppendMenuW(models, MF_STRING, IDM_DOWNLOAD, L"Download models...");
    AppendMenuW(models, MF_STRING, IDM_ADDROOT, L"Add models folder (ComfyUI style)...");
    AppendMenuW(help, MF_STRING, IDM_ABOUT, L"About");
    HMENU web = CreatePopupMenu();
    AppendMenuW(web, MF_STRING, IDM_WEB, L"Open the chat in the browser");
    AppendMenuW(web, MF_STRING, IDM_WEBLAN, L"Share on the local network...");
    HMENU view = CreatePopupMenu();
    AppendMenuW(view, MF_STRING | MF_CHECKED, IDM_QUEUE, L"Queue panel");
    AppendMenuW(bar, MF_POPUP, (UINT_PTR)file, L"File");
    AppendMenuW(bar, MF_POPUP, (UINT_PTR)view, L"View");
    AppendMenuW(bar, MF_POPUP, (UINT_PTR)web, L"Web");
    AppendMenuW(bar, MF_POPUP, (UINT_PTR)models, L"Models");
    AppendMenuW(bar, MF_POPUP, (UINT_PTR)help, L"Help");
    return bar;
}

LRESULT CALLBACK mainProc(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    switch (m) {
    case WM_CREATE: {
        g_main = h;
        INITCOMMONCONTROLSEX ic = {sizeof ic, ICC_TAB_CLASSES | ICC_LISTVIEW_CLASSES | ICC_PROGRESS_CLASS | ICC_BAR_CLASSES | ICC_STANDARD_CLASSES};
        InitCommonControlsEx(&ic);
        g_tab = uiMake(h, WC_TABCONTROLW, "", WS_CLIPSIBLINGS | WS_TABSTOP);
        { TCITEMW ti = {}; ti.mask = TCIF_TEXT; std::wstring t = L"Chat"; ti.pszText = &t[0]; TabCtrl_InsertItem(g_tab, 0, &ti); }
        for (int i = 0; i < kKindCount; i++) {
            TCITEMW ti = {}; ti.mask = TCIF_TEXT; std::wstring t = sys::widen(kindTitle((Kind)i)); ti.pszText = &t[0];
            TabCtrl_InsertItem(g_tab, i + 1, &ti);
        }
        for (int i = 0; i < kKindCount; i++) buildPage(g_pages[i], (Kind)i);
        chatViewCreate(h);
        DragAcceptFiles(h, TRUE);
        buildQueue();
        loadQueueUi();
        { std::string e; WebServer::get().start(8787, false, e); }
        if (HMENU mb = GetMenu(h)) CheckMenuItem(mb, IDM_QUEUE, MF_BYCOMMAND | (g_qVis ? MF_CHECKED : MF_UNCHECKED));
        layoutMain();
        showSelectedPage();
        refreshAllReady();
        SetTimer(h, 1, 300, nullptr);
        PostMessageW(h, WM_FIRSTRUN, 0, 0);
        return 0;
    }
    case WM_FIRSTRUN: firstRun(); return 0;
    case WM_DROPFILES: {
        HDROP hd = (HDROP)wp;
        UINT n = DragQueryFileW(hd, 0xFFFFFFFF, nullptr, 0);
        if (IsWindowVisible(chatViewHwnd())) for (UINT i = 0; i < n; i++) { wchar_t buf[MAX_PATH * 2]; if (DragQueryFileW(hd, i, buf, MAX_PATH * 2)) chatViewAddFile(sys::narrow(buf)); }
        DragFinish(hd);
        return 0;
    }
    case WM_SIZE: if (wp != SIZE_MINIMIZED && g_tab) layoutMain(); return 0;
    case WM_GETMINMAXINFO: { auto* mm = (MINMAXINFO*)lp; mm->ptMinTrackSize.x = 980; mm->ptMinTrackSize.y = 640; return 0; }
    case WM_TIMER: refreshQueue(); chatViewTick(); for (auto& pg : g_pages) updatePlayer(pg); return 0;
    case WM_SETCURSOR: {
        POINT pt; GetCursorPos(&pt); ScreenToClient(h, &pt);
        RECT sp = splitterRect();
        if (g_qVis && LOWORD(lp) == HTCLIENT && pt.x >= sp.left && pt.x < sp.right) { SetCursor(LoadCursorW(nullptr, IDC_SIZEWE)); return TRUE; }
        break;
    }
    case WM_LBUTTONDOWN: {
        RECT sp = splitterRect(); int x = (short)LOWORD(lp);
        if (g_qVis && x >= sp.left && x < sp.right) { g_qDrag = true; SetCapture(h); }
        return 0;
    }
    case WM_MOUSEMOVE:
        if (g_qDrag) {
            RECT rc; GetClientRect(h, &rc);
            g_qW = std::max<int>(kQMin, std::min<int>(rc.right - (int)(short)LOWORD(lp) - kSplit / 2, rc.right - 420));
            layoutMain();
        }
        return 0;
    case WM_LBUTTONUP:
        if (g_qDrag) { g_qDrag = false; ReleaseCapture(); saveQueueUi(); }
        return 0;
    case WM_LBUTTONDBLCLK: {
        RECT sp = splitterRect(); int x = (short)LOWORD(lp);
        if (g_qVis && x >= sp.left && x < sp.right) { g_qW = 0; saveQueueUi(); layoutMain(); }        // back to the automatic width
        return 0;
    }
    case WM_NOTIFY: {
        NMHDR* n = (NMHDR*)lp;
        if (n->hwndFrom == ListView_GetHeader(g_qList) && (n->code == HDN_ENDTRACKW || n->code == HDN_ENDTRACKA)) { g_qColsCustom = true; saveQueueUi(); }
        if (n->hwndFrom == g_tab && n->code == TCN_SELCHANGE) showSelectedPage();
        if (n->hwndFrom == g_qList && n->code == NM_DBLCLK) onQueueCommand(IDC_QDETAILS);
        if (n->hwndFrom == g_qList && n->code == LVN_COLUMNCLICK) onQueueColumnClick(((NMLISTVIEW*)lp)->iSubItem);
        return 0;
    }
    case WM_COMMAND: {
        int id = LOWORD(wp);
        if (id == IDM_WEB) {
            WebServer& ws = WebServer::get();
            if (ws.running()) ShellExecuteW(h, L"open", sys::widen(ws.url()).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
            else MessageBoxW(h, L"The web server is not running (no free port).", L"Dopplers Bakery", MB_OK | MB_ICONWARNING);
        }
        else if (id == IDM_WEBLAN) {
            WebServer& ws = WebServer::get();
            bool lan = !ws.lan();
            if (lan && MessageBoxW(h, L"Other devices on your network will be able to use this chat (and make the PC generate images, audio and speech) if they open the address with the access key.\n\nThe key is part of the link; only share it with people you trust. Turn on?", L"Share on the local network", MB_YESNO | MB_ICONWARNING) != IDYES) return 0;
            ws.stop(); std::string e; Sleep(100); ws.start(8787, lan, e);
            CheckMenuItem(GetMenu(h), IDM_WEBLAN, MF_BYCOMMAND | (ws.lan() ? MF_CHECKED : MF_UNCHECKED));
            if (ws.lan()) {
                std::string u = ws.url(); u.replace(u.find("127.0.0.1"), 9, "<this-PC-address>");
                MessageBoxW(h, sys::widen("Open this on the other device (replace <this-PC-address> with this PC's IP address):\n\n" + u).c_str(), L"Dopplers Bakery", MB_OK | MB_ICONINFORMATION);
            }
        }
        else if (id == IDM_QUEUE || id == IDC_QTOGGLE) { setQueueVisible(!g_qVis); layoutMain(); }
        else if (id >= IDC_QUP && id <= IDC_QDETAILS) onQueueCommand(id);
        else if (id == IDC_QCLEAR) onQueueCommand(id);
        else if (id == IDM_EXIT) PostMessageW(h, WM_CLOSE, 0, 0);
        else if (id == IDM_OPENOUT) { sys::makeDirs(sys::outputDir()); ShellExecuteW(h, L"open", sys::widen(sys::outputDir()).c_str(), nullptr, nullptr, SW_SHOWNORMAL); }
        else if (id == IDM_DOWNLOAD) { showDownloadDialog(h, false); refreshAllReady(); }
        else if (id == IDM_ADDROOT) {
            std::string p;
            if (pickFolder(h, p, L"Select a ComfyUI-style models folder (contains checkpoints, vae, text_encoders ...)")) {
                auto r = Settings::get().extraRoots(); r.push_back(p); Settings::get().setExtraRoots(r);
                Settings::get().autoSelect(); refreshAllReady();
            }
        }
        else if (id == IDM_ABOUT)
            MessageBoxW(h, L"Dopplers Bakery\nCompact offline baker for text, speech, images and audio.\nModels: " L"models\\   Output: output\\", L"About", MB_OK | MB_ICONINFORMATION);
        return 0;
    }
    case WM_CLOSE:
        if (JobQueue::get().busy() && MessageBoxW(h, L"A job is still running. Cancel it and quit?", L"Dopplers Bakery", MB_YESNO | MB_ICONQUESTION) != IDYES) return 0;
        for (auto& j : JobQueue::get().snapshot()) JobQueue::get().cancel(j.id);
        stopPlayback();
        DestroyWindow(h);
        return 0;
    case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(h, m, wp, lp);
}

// Mouse wheel over a tab's settings form scrolls the form, whichever child has focus.
// Keyboard shortcuts of the Audio (music) tab: F2 generate, F3 play/stop the latest generated file, Shift+Delete delete the selected file
bool routeKey(MSG& msg) {
    if (msg.message != WM_KEYDOWN) return false;
    if (msg.wParam != VK_F2 && msg.wParam != VK_F3 && msg.wParam != VK_DELETE) return false;
    if (TabCtrl_GetCurSel(g_tab) != 1 + (int)Kind::Audio) return false;
    if (msg.hwnd != g_main && !IsChild(g_main, msg.hwnd)) return false;
    Page& p = g_pages[(int)Kind::Audio];
    if (msg.wParam == VK_F2) { if (p.ready) onCreateClicked(p); return true; }
    if (msg.wParam == VK_F3) {
        if (p.outs.empty()) return true;
        Player& pl = Player::get();
        const std::string latest = p.outs[0].path;                       // the list is newest first
        if (pl.current() == latest && pl.playing()) pl.stop();
        else { selectOut(p, 0); pl.play(latest); }
        updatePlayer(p);
        return true;
    }
    // Shift+Delete: not while typing (there it means Cut)
    if (!(GetKeyState(VK_SHIFT) & 0x8000)) return false;
    wchar_t cls[32] = {}; GetClassNameW(GetFocus(), cls, 32);
    if (!wcscmp(cls, L"Edit") || !wcscmp(cls, L"ComboBox")) return false;
    onOutputAction(p, IDC_DELETE);
    return true;
}

bool routeWheel(MSG& msg) {
    POINT pt; GetCursorPos(&pt);
    HWND w = WindowFromPoint(pt);
    for (HWND a = w; a && a != g_main; a = GetParent(a)) {
        for (auto& p : g_pages) if (a == p.form) {
            if (w != a && (GetWindowLongW(w, GWL_STYLE) & WS_VSCROLL) == 0 && false) return false;
            int delta = GET_WHEEL_DELTA_WPARAM(msg.wParam);
            SendMessageW(p.form, WM_VSCROLL, delta > 0 ? SB_LINEUP : SB_LINEDOWN, 0);
            if (std::abs(delta) > 120) SendMessageW(p.form, WM_VSCROLL, delta > 0 ? SB_LINEUP : SB_LINEDOWN, 0);
            return true;
        }
    }
    return false;
}
}  // namespace

int runGui(HINSTANCE inst) {
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    Settings::get().load();
    if (crashGuardRecover(g_recovered)) Settings::get().setUiInt("safe_cpu", 1);          // the last run died inside a GPU job
    Compute::setSafeMode(Settings::get().uiInt("safe_cpu", 0) != 0);
    sys::makeDirs(sys::modelsDir()); sys::makeDirs(sys::outputDir());
    JobQueue::get().start();

    WNDCLASSW wc = {};
    wc.hInstance = inst; wc.hCursor = LoadCursorW(nullptr, IDC_ARROW); wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    wc.style = CS_DBLCLKS | CS_HREDRAW | CS_VREDRAW; wc.lpfnWndProc = mainProc; wc.lpszClassName = L"BakeryMain"; wc.hIcon = LoadIconW(inst, MAKEINTRESOURCEW(1));
    RegisterClassW(&wc);
    wc.lpfnWndProc = pageProc; wc.lpszClassName = L"BakeryPage"; wc.hIcon = nullptr; RegisterClassW(&wc);
    wc.lpfnWndProc = formProc; wc.lpszClassName = L"BakeryForm"; RegisterClassW(&wc);

    HWND w = CreateWindowExW(0, L"BakeryMain", L"Dopplers Bakery", WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
                             CW_USEDEFAULT, CW_USEDEFAULT, 1280, 800, nullptr, buildMenu(), inst, nullptr);
    HICON small = (HICON)LoadImageW(inst, MAKEINTRESOURCEW(1), IMAGE_ICON, GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), 0);
    if (small) SendMessageW(w, WM_SETICON, ICON_SMALL, (LPARAM)small);
    ShowWindow(w, SW_SHOW);
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        if (msg.message == WM_MOUSEWHEEL && routeWheel(msg)) continue;
        if (msg.message == WM_KEYDOWN && routeKey(msg)) continue;
        if (IsDialogMessageW(w, &msg)) continue;
        TranslateMessage(&msg); DispatchMessageW(&msg);
    }
    JobQueue::get().shutdown();
    return 0;
}
