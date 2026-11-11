// Main window: operation tabs on the left, queue manager on the right.
#include "ui/ui.h"
#include <objbase.h>
#include <commctrl.h>
#include <cstring>
#include <shellapi.h>
#include <mmsystem.h>
#include <algorithm>
#include <cmath>
#include "ui/chat_view.h"
#include "web/server.h"
#include "core/download.h"
#include "core/forms.h"
#include "core/queue.h"
#include "core/settings.h"
#include "ui/dialogs.h"
#include "ui/modal.h"
#include "util/image_io.h"
#include "util/sys.h"

namespace {

enum {
    IDC_CREATE = 1001, IDC_SETTINGS, IDC_OUTLIST, IDC_PREVIEW, IDC_COPY, IDC_DELETE,
    IDC_FIELD0 = 2000, IDC_BROWSE0 = 2500,
    IDC_QLIST = 3001, IDC_QUP, IDC_QDOWN, IDC_QREMOVE, IDC_QPAUSE, IDC_QCANCEL, IDC_QCLEAR, IDC_QDETAILS, IDC_QPROG, IDC_QSTAT, IDC_QTOGGLE,

    IDM_OPENOUT = 4001, IDM_EXIT, IDM_DOWNLOAD, IDM_ADDROOT, IDM_ABOUT, IDM_WEB, IDM_WEBLAN, IDM_QUEUE,
};
const UINT WM_FIRSTRUN = WM_APP + 1;
const int kQueueMinW = 340;

struct FieldCtl {
    const Field* f; HWND label = nullptr, ctl = nullptr, btn = nullptr;
    int x = 0, ly = 0, cy = 0, w = 0, ch = 0;      // logical layout (before scrolling)
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
};

HWND g_main, g_tab, g_qTitle, g_qList, g_qStat, g_qProg, g_qToggle = nullptr;
HWND g_qBtn[8];
Page g_pages[kKindCount];
std::vector<int> g_qIds;          // job id per queue row
std::string g_qSig;
std::string g_playing;

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
        case FT::Slider: j[f.key] = (double)SendMessageW(c.ctl, TBM_GETPOS, 0, 0); break;
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
    int v = (int)SendMessageW(c.ctl, TBM_GETPOS, 0, 0);
    uiSetText(c.label, std::string(c.f->label) + ": " + std::to_string(v));
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
        case FT::Slider: SendMessageW(c.ctl, TBM_SETPOS, TRUE, (LPARAM)(v.isNull() ? f.def : v.num())); sliderLabel(c); break;
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
    }
}

void layoutPage(Page& p) {
    RECT rc; GetClientRect(p.hwnd, &rc);
    int W = rc.right, H = rc.bottom;
    int warnH = p.ready ? 0 : 36, actH = 0, outH = 120, outBtnH = 32;
    ShowWindow(p.warn, p.ready ? SW_HIDE : SW_SHOW);
    uiPlace(p.warn, 6, 4, W - 12, warnH);
    int top = warnH ? warnH + 6 : 0;
    int formH = std::max(60, H - top - actH - outH - outBtnH - 6);
    uiPlace(p.form, 0, top, W, formH);
    int y = top + formH + 4;
    y += actH;
    uiPlace(p.list, 8, y, W - 16, outH);
    ListView_SetColumnWidth(p.list, 1, std::max(120, W - 16 - 130 - 22));
    y += outH + 4;
    int bw = 96;
    uiPlace(p.bPreview, 8, y, bw, 26);
    uiPlace(p.bCopy, 8 + bw + 6, y, bw, 26);
    uiPlace(p.bDelete, 8 + 2 * (bw + 6), y, bw, 26);
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
    p.sig = sig; p.outs = items;
    int sel = ListView_GetNextItem(p.list, -1, LVNI_SELECTED);
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
    if (sel >= 0 && sel < (int)items.size()) ListView_SetItemState(p.list, sel, LVIS_SELECTED, LVIS_SELECTED);
}

bool isAudio(Kind k) { return k == Kind::TTS || k == Kind::Audio; }

void stopPlayback() {
    mciSendStringW(L"close bakery", nullptr, 0, nullptr);
    g_playing.clear();
}
void playFile(const std::string& path) {
    bool same = g_playing == path;
    stopPlayback();
    if (same) return;                                 // second click on the same item stops it
    std::string lp = path; for (auto& c : lp) c = (char)tolower((unsigned char)c);
    bool wav = lp.size() > 4 && lp.compare(lp.size() - 4, 4, ".wav") == 0;
    std::wstring cmd = L"open \"" + sys::widen(path) + L"\" type " + (wav ? L"waveaudio" : L"mpegvideo") + L" alias bakery";
    if (mciSendStringW(cmd.c_str(), nullptr, 0, nullptr) == 0) {
        mciSendStringW(L"play bakery", nullptr, 0, nullptr);
        g_playing = path;
    }
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
        if (g_playing == it.path) stopPlayback();
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
        for (auto& c : p->fc) if (c.ctl == (HWND)lp && c.f->type == FT::Slider) sliderLabel(c);
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
    p.bCreate = uiMake(p.form, L"BUTTON", "Create", BS_OWNERDRAW | WS_TABSTOP, IDC_CREATE);
    SetWindowSubclass(p.bCreate, hoverProc, 1, 0);
    p.bSettings = uiMake(p.form, L"BUTTON", "Model settings...", BS_PUSHBUTTON | WS_TABSTOP, IDC_SETTINGS);
    SendMessageW(p.bSettings, WM_SETFONT, (WPARAM)btnFont(false), TRUE);
    p.list = uiMake(p.hwnd, WC_LISTVIEWW, "", LVS_REPORT | LVS_SINGLESEL | LVS_SHOWSELALWAYS | WS_TABSTOP, IDC_OUTLIST, WS_EX_CLIENTEDGE);
    ListView_SetExtendedListViewStyle(p.list, LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES);
    const wchar_t* cols[] = {L"Result of", L"Output (this session)"}; int cw[] = {90, 300};
    for (int k = 0; k < 2; k++) { LVCOLUMNW c = {}; c.mask = LVCF_TEXT | LVCF_WIDTH; c.pszText = (LPWSTR)cols[k]; c.cx = cw[k]; ListView_InsertColumn(p.list, k, &c); }
    bool audio = isAudio(kind);
    p.bPreview = uiMake(p.hwnd, L"BUTTON", audio ? "Play / Stop" : "Preview", BS_PUSHBUTTON | WS_TABSTOP, IDC_PREVIEW);
    p.bCopy = uiMake(p.hwnd, L"BUTTON", kind == Kind::LLM ? "Copy text" : "Copy file", BS_PUSHBUTTON | WS_TABSTOP, IDC_COPY);
    p.bDelete = uiMake(p.hwnd, L"BUTTON", "Delete", BS_PUSHBUTTON | WS_TABSTOP, IDC_DELETE);
    Json saved = Settings::get().form(kind);
    writeForm(p, saved);
}

// ----------------------------------------------------------------- queue panel
std::string pct(const Job& j) { return j.state == JobState::Running || j.state == JobState::Paused ? std::to_string((int)(j.progress * 100)) + "%" : ""; }

int selectedJob() {
    int s = ListView_GetNextItem(g_qList, -1, LVNI_SELECTED);
    return s >= 0 && s < (int)g_qIds.size() ? g_qIds[s] : 0;
}

void refreshQueue() {
    auto jobs = JobQueue::get().snapshot();
    std::string sig;
    for (auto& j : jobs) sig += std::to_string(j.id) + stateName(j.state) + pct(j) + ";";
    if (sig != g_qSig) {
        g_qSig = sig;
        int keep = selectedJob();
        ListView_DeleteAllItems(g_qList);
        g_qIds.clear();
        int row = 0;
        for (auto& j : jobs) {
            LVITEMW it = {}; it.mask = LVIF_TEXT; it.iItem = row;
            std::wstring a = sys::widen(std::to_string(j.id)); it.pszText = &a[0];
            ListView_InsertItem(g_qList, &it);
            std::wstring b = sys::widen(kindTitle(j.kind)), c = sys::widen(j.title), d = sys::widen(std::string(stateName(j.state)) + (j.state == JobState::Failed ? "!" : "")), e = sys::widen(pct(j));
            ListView_SetItemText(g_qList, row, 1, &b[0]); ListView_SetItemText(g_qList, row, 2, &c[0]);
            ListView_SetItemText(g_qList, row, 3, &d[0]); ListView_SetItemText(g_qList, row, 4, &e[0]);
            g_qIds.push_back(j.id);
            if (j.id == keep) ListView_SetItemState(g_qList, row, LVIS_SELECTED | LVIS_FOCUSED, LVIS_SELECTED | LVIS_FOCUSED);
            row++;
        }
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
    for (auto& p : g_pages) refreshOutputs(p);
}

std::string jobDetails(const Job& j) {
    std::string s = "Job " + std::to_string(j.id) + " - " + kindTitle(j.kind) + "\r\nState: " + stateName(j.state) + "\r\n";
    if (!j.status.empty()) s += "Status: " + j.status + "\r\n";
    if (!j.error.empty()) s += "Error: " + j.error + "\r\n";
    for (auto& o : j.outputs) s += "Output: " + o + "\r\n";
    s += "\r\nParameters:\r\n" + j.params.dump(2);
    std::string r; for (char c : s) { if (c == '\n' && (r.empty() || r.back() != '\r')) r += '\r'; r += c; }
    return r;
}

void onQueueCommand(int id) {
    auto& q = JobQueue::get();
    int jid = selectedJob();
    if (id == IDC_QCLEAR) { q.clearFinished(); g_qSig.clear(); refreshQueue(); return; }
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
    g_qSig.clear(); refreshQueue();
    // keep the moved job selected
    for (size_t i = 0; i < g_qIds.size(); i++) if (g_qIds[i] == jid) ListView_SetItemState(g_qList, (int)i, LVIS_SELECTED | LVIS_FOCUSED, LVIS_SELECTED | LVIS_FOCUSED);
}

void buildQueue() {
    g_qTitle = uiMake(g_main, L"STATIC", "Queue", SS_LEFT);
    g_qList = uiMake(g_main, WC_LISTVIEWW, "", LVS_REPORT | LVS_SINGLESEL | LVS_SHOWSELALWAYS | WS_TABSTOP, IDC_QLIST, WS_EX_CLIENTEDGE);
    ListView_SetExtendedListViewStyle(g_qList, LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES);
    const wchar_t* cols[] = {L"#", L"Tab", L"Prompt", L"State", L"%"}; int cw[] = {30, 70, 150, 70, 40};
    for (int i = 0; i < 5; i++) { LVCOLUMNW c = {}; c.mask = LVCF_TEXT | LVCF_WIDTH; c.pszText = (LPWSTR)cols[i]; c.cx = cw[i]; ListView_InsertColumn(g_qList, i, &c); }
    const char* names[] = {"Move up", "Move down", "Remove", "Pause", "Cancel", "Clear finished", "Details...", ""};
    int ids[] = {IDC_QUP, IDC_QDOWN, IDC_QREMOVE, IDC_QPAUSE, IDC_QCANCEL, IDC_QCLEAR, IDC_QDETAILS, 0};
    for (int i = 0; i < 7; i++) g_qBtn[i] = uiMake(g_main, L"BUTTON", names[i], BS_PUSHBUTTON | WS_TABSTOP, ids[i]);
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
    if (g_qColsCustom) for (int i = 0; i < 5; i++) s.setUiInt("qcol" + std::to_string(i), ListView_GetColumnWidth(g_qList, i));
}
void loadQueueUi() {
    Settings& s = Settings::get();
    g_qVis = s.uiInt("queue_visible", 1) != 0; g_qW = s.uiInt("queue_width", 0);
    if (s.uiInt("qcol0", 0) > 0) {
        for (int i = 0; i < 5; i++) { int w = s.uiInt("qcol" + std::to_string(i), 0); if (w > 0) ListView_SetColumnWidth(g_qList, i, w); }
        g_qColsCustom = true;
    }
}
void setQueueVisible(bool v) {
    g_qVis = v;
    HMENU m = GetMenu(g_main); if (m) CheckMenuItem(m, IDM_QUEUE, MF_BYCOMMAND | (v ? MF_CHECKED : MF_UNCHECKED));
    saveQueueUi();
}

void layoutMain() {
    RECT rc; GetClientRect(g_main, &rc);
    int W = rc.right, H = rc.bottom;
    int qW = queueWidth(W);
    int tabW = g_qVis ? W - qW - kSplit : W - kToggleW - 2;
    for (HWND c : {g_qTitle, g_qList, g_qStat, g_qProg}) ShowWindow(c, g_qVis ? SW_SHOW : SW_HIDE);
    for (int i = 0; i < 7; i++) ShowWindow(g_qBtn[i], g_qVis ? SW_SHOW : SW_HIDE);
    uiPlace(g_tab, 4, 4, tabW - 4, H - 8);
    RECT tr = {4, 4, 4 + tabW - 4, 4 + H - 8};
    TabCtrl_AdjustRect(g_tab, FALSE, &tr);
    for (auto& p : g_pages) { uiPlace(p.hwnd, tr.left, tr.top, tr.right - tr.left, tr.bottom - tr.top); layoutPage(p); }
    uiPlace(chatViewHwnd(), tr.left, tr.top, tr.right - tr.left, tr.bottom - tr.top);
    int x = W - qW, w = qW - 6;
    uiSetText(g_qToggle, g_qVis ? "Hide \xE2\x80\xBA" : "\xE2\x80\xB9");
    if (g_qVis) uiPlace(g_qToggle, x + w - 58, 3, 58, 22); else uiPlace(g_qToggle, W - kToggleW - 1, H / 2 - 40, kToggleW, 80);
    if (!g_qVis) return;
    uiPlace(g_qTitle, x, 6, w - 62, 18);
    int listH = H - 6 - 22 - 22 - 66 - 28 - 30;
    uiPlace(g_qList, x, 26, w, listH);
    int bw = (w - 8) / 3, y = 26 + listH + 6;
    for (int i = 0; i < 7; i++) { int r = i / 3, c = i % 3; uiPlace(g_qBtn[i], x + c * (bw + 4), y + r * 30, bw, 26); }
    uiPlace(g_qStat, x, H - 46, w, 18);
    uiPlace(g_qProg, x, H - 24, w, 16);
    if (!g_qColsCustom) ListView_SetColumnWidth(g_qList, 2, std::max(80, w - 30 - 70 - 70 - 40 - 24));
}

void showSelectedPage() {
    int s = TabCtrl_GetCurSel(g_tab);
    for (int i = 0; i < kKindCount; i++) ShowWindow(g_pages[i].hwnd, i + 1 == s ? SW_SHOW : SW_HIDE);      // tab 0 is Chat
    chatViewShow(s == 0);
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
    case WM_TIMER: refreshQueue(); chatViewTick(); return 0;
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
    sys::makeDirs(sys::modelsDir()); sys::makeDirs(sys::outputDir());
    JobQueue::get().start();

    WNDCLASSW wc = {};
    wc.hInstance = inst; wc.hCursor = LoadCursorW(nullptr, IDC_ARROW); wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    wc.style = CS_DBLCLKS; wc.lpfnWndProc = mainProc; wc.lpszClassName = L"BakeryMain"; wc.hIcon = LoadIconW(inst, MAKEINTRESOURCEW(1));
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
        if (IsDialogMessageW(w, &msg)) continue;
        TranslateMessage(&msg); DispatchMessageW(&msg);
    }
    JobQueue::get().shutdown();
    return 0;
}
