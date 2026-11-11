#include "ui/chat_view.h"
#include <commctrl.h>
#include <mmsystem.h>
#include <shellapi.h>
#include <algorithm>
#include <map>
#include <vector>
#include "chat/agent.h"
#include "ui/modal.h"
#include "util/image_io.h"
#include "util/sys.h"
#include "web/server.h"

namespace {
enum ItemKind { K_TEXT, K_IMAGE, K_AUDIO, K_NOTE, K_TYPING, K_EMPTY };
enum HitId { H_NONE = 0, H_IMG, H_AUDIO, H_CHIPX, H_PLUS, H_VOICE, H_SEND };
struct Line { int s, e; };
struct Item { int kind = K_TEXT; bool user = false; int x = 0, y = 0, w = 0, h = 0; std::wstring text; std::string path; std::vector<Line> lines; int lineH = 0; };
struct Hit { RECT r; int id; int idx; std::string path; };
struct Thumb { HBITMAP bmp = nullptr; int w = 0, h = 0; };

const COLORREF cBg = RGB(255, 255, 255), cText = RGB(31, 35, 40), cMute = RGB(107, 114, 128), cUser = RGB(235, 240, 250),
               cPanel = RGB(244, 244, 245), cLine = RGB(221, 221, 226), cAcc = RGB(37, 99, 235), cErr = RGB(180, 35, 24), cHot = RGB(228, 228, 233);
const wchar_t* kClass = L"BakeryChatView";
const int kPad = 8, kBtn = 36, kChip = 60, kLine = 22;

HWND g_view = nullptr, g_edit = nullptr;
HFONT g_f = nullptr, g_fs = nullptr, g_fh = nullptr;
std::vector<Item> g_items;
std::vector<Hit> g_hits;                           // composer hits (client coordinates)
std::vector<std::string> g_attached;
std::map<std::string, Thumb> g_thumbs;
std::string g_playing, g_sig;
std::vector<ChatMsg> g_tr;
std::string g_live, g_status;
bool g_busy = false;
int g_contentH = 0, g_scroll = 0, g_phase = 0, g_hot = 0;
RECT g_composer = {}, g_logRect = {}, g_statusRect = {};
RECT g_rPlus = {}, g_rVoice = {}, g_rSend = {};
bool g_tracking = false;
// text selection: (item index, character index) anchor and focus
int g_selAi = -1, g_selAc = 0, g_selBi = -1, g_selBc = 0;
bool g_selecting = false;
POINT g_downPt = {};

std::wstring W(const std::string& s) { return sys::widen(s); }
std::string fileOf(const std::string& p) { return p.substr(p.find_last_of("\\/") + 1); }
bool ends(const std::string& s, const char* e) { std::string l = s; for (auto& c : l) c = (char)tolower((unsigned char)c); size_t n = strlen(e); return l.size() >= n && l.compare(l.size() - n, n, e) == 0; }
bool isImg(const std::string& p) { for (const char* e : {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}) if (ends(p, e)) return true; return false; }
bool isAud(const std::string& p) { return ends(p, ".mp3") || ends(p, ".wav"); }
bool inRect(const RECT& r, int x, int y) { return x >= r.left && x < r.right && y >= r.top && y < r.bottom; }

// ------------------------------------------------------------------ sound
void stopSound() { mciSendStringW(L"close bkchat", nullptr, 0, nullptr); g_playing.clear(); }
void playSound(const std::string& path) {
    bool same = g_playing == path;
    stopSound();
    if (same) return;
    std::wstring cmd = L"open \"" + W(path) + L"\" type " + (ends(path, ".wav") ? L"waveaudio" : L"mpegvideo") + L" alias bkchat";
    if (mciSendStringW(cmd.c_str(), nullptr, 0, nullptr) == 0) { mciSendStringW(L"play bkchat", nullptr, 0, nullptr); g_playing = path; }
}
void checkSoundEnded() {
    if (g_playing.empty()) return;
    wchar_t mode[32] = {};
    mciSendStringW(L"status bkchat mode", mode, 32, nullptr);
    if (wcscmp(mode, L"playing") != 0) { stopSound(); InvalidateRect(g_view, nullptr, FALSE); }
}

// ------------------------------------------------------------------ pictures
const Thumb& thumbOf(const std::string& path, int maxSide) {
    auto it = g_thumbs.find(path);
    if (it != g_thumbs.end()) return it->second;
    Thumb t; t.bmp = (HBITMAP)loadThumbnail(path, maxSide, t.w, t.h);
    return g_thumbs[path] = t;
}

void drawThumb(HDC dc, const Thumb& t, int x, int y, int w, int h, int sx, int sy, int sw, int sh, int radius) {
    if (!t.bmp) {
        HBRUSH b = CreateSolidBrush(cPanel); RECT r = {x, y, x + w, y + h}; FillRect(dc, &r, b); DeleteObject(b);
        return;
    }
    int saved = SaveDC(dc);
    HRGN rg = CreateRoundRectRgn(x, y, x + w + 1, y + h + 1, radius * 2, radius * 2);
    ExtSelectClipRgn(dc, rg, RGN_AND); DeleteObject(rg);
    HDC m = CreateCompatibleDC(dc);
    HGDIOBJ old = SelectObject(m, t.bmp);
    BLENDFUNCTION bf = {AC_SRC_OVER, 0, 255, AC_SRC_ALPHA};
    AlphaBlend(dc, x, y, w, h, m, sx, sy, sw, sh, bf);
    SelectObject(m, old); DeleteDC(m);
    RestoreDC(dc, saved);
}

void fillRound(HDC dc, const RECT& r, COLORREF fill, COLORREF line, int rad) {
    HBRUSH b = CreateSolidBrush(fill); HPEN p = CreatePen(PS_SOLID, 1, line);
    HGDIOBJ ob = SelectObject(dc, b), op = SelectObject(dc, p);
    RoundRect(dc, r.left, r.top, r.right, r.bottom, rad * 2, rad * 2);
    SelectObject(dc, ob); SelectObject(dc, op); DeleteObject(b); DeleteObject(p);
}
void fillCircle(HDC dc, const RECT& r, COLORREF fill) {
    HBRUSH b = CreateSolidBrush(fill); HPEN p = CreatePen(PS_SOLID, 1, fill);
    HGDIOBJ ob = SelectObject(dc, b), op = SelectObject(dc, p);
    Ellipse(dc, r.left, r.top, r.right, r.bottom);
    SelectObject(dc, ob); SelectObject(dc, op); DeleteObject(b); DeleteObject(p);
}

// ------------------------------------------------------------------ text layout
SIZE measure(HDC dc, const std::wstring& t, int maxW, HFONT f) {
    HGDIOBJ o = SelectObject(dc, f);
    RECT r = {0, 0, std::max(40, maxW), 0};
    DrawTextW(dc, t.c_str(), -1, &r, DT_CALCRECT | DT_WORDBREAK | DT_NOPREFIX);
    SelectObject(dc, o);
    return {r.right, r.bottom};
}

// word-wrap `t` into lines of at most maxW pixels (own wrapping so drawing and hit-testing agree)
void wrapLines(HDC dc, HFONT f, const std::wstring& t, int maxW, std::vector<Line>& out, int& wMax, int& lineH) {
    HGDIOBJ o = SelectObject(dc, f);
    TEXTMETRICW tm; GetTextMetricsW(dc, &tm);
    lineH = tm.tmHeight + tm.tmExternalLeading + 2;
    out.clear(); wMax = 0;
    const int n = (int)t.size();
    int i = 0;
    while (i <= n) {
        int pe = i; while (pe < n && t[pe] != L'\n') pe++;
        int ps = i, pend = pe; if (pend > ps && t[pend - 1] == L'\r') pend--;
        if (pend == ps) out.push_back({ps, ps});
        else {
            int s = ps;
            while (s < pend) {
                int fit = 0; SIZE sz;
                GetTextExtentExPointW(dc, &t[s], pend - s, std::max(20, maxW), &fit, nullptr, &sz);
                if (fit <= 0) fit = 1;
                int e = s + fit;
                if (e < pend) { int b = e; while (b > s && t[b - 1] != L' ') b--; if (b > s) e = b; }
                int e2 = e; while (e2 > s && t[e2 - 1] == L' ') e2--;
                SIZE w; GetTextExtentPoint32W(dc, &t[s], e2 - s, &w);
                wMax = std::max<int>(wMax, w.cx);
                out.push_back({s, e}); s = e;
            }
        }
        i = pe + 1;
    }
    SelectObject(dc, o);
}

void selRange(int& ai, int& ac, int& bi, int& bc) {            // normalized: a <= b
    ai = g_selAi; ac = g_selAc; bi = g_selBi; bc = g_selBc;
    if (ai > bi || (ai == bi && ac > bc)) { std::swap(ai, bi); std::swap(ac, bc); }
}
bool hasSelection() { return g_selAi >= 0 && g_selBi >= 0 && !(g_selAi == g_selBi && g_selAc == g_selBc); }
std::string selectedText() {
    if (!hasSelection()) return "";
    int ai, ac, bi, bc; selRange(ai, ac, bi, bc);
    std::wstring out;
    for (int i = ai; i <= bi && i < (int)g_items.size(); i++) {
        const Item& it = g_items[i];
        if (it.kind != K_TEXT) continue;
        int s = i == ai ? ac : 0, e = i == bi ? bc : (int)it.text.size();
        s = std::min<int>(s, (int)it.text.size()); e = std::min<int>(e, (int)it.text.size());
        if (e > s) { if (!out.empty()) out += L"\r\n\r\n"; out += it.text.substr(s, e - s); }
    }
    return sys::narrow(out);
}

void buildItems(HDC dc, int W_, int colX, int colW) {
    (void)W_;
    g_items.clear();
    int y = 18;
    auto add = [&](Item it, bool user) {
        it.user = user; it.x = user ? colX + colW - it.w : colX; it.y = y; y += it.h; g_items.push_back(it);
    };
    for (auto& m : g_tr) {
        if (m.role == "note") {
            Item it; it.kind = K_NOTE; it.text = W(m.text);
            SIZE s = measure(dc, it.text, colW, g_fs); it.w = s.cx; it.h = s.cy + 4;
            y += 6; add(it, false); continue;
        }
        bool user = m.role == "user";
        y += 14;
        if (!m.text.empty()) {
            Item it; it.kind = K_TEXT; it.text = W(m.text);
            int maxW = user ? colW * 78 / 100 : colW, padX = user ? 16 : 2, padY = user ? 10 : 2;
            int wm = 0; wrapLines(dc, g_f, it.text, maxW - 2 * padX, it.lines, wm, it.lineH);
            it.w = wm + 2 * padX + 2; it.h = (int)it.lines.size() * it.lineH + 2 * padY;
            add(it, user);
        }
        for (auto& f : m.files) {
            if (isImg(f)) {
                const Thumb& t = thumbOf(f, 360);
                Item it; it.kind = K_IMAGE; it.path = f;
                int tw = t.w ? t.w : 240, th = t.h ? t.h : 240;
                double sc = std::min(1.0, 300.0 / std::max(tw, th));
                it.w = std::max(40, (int)(tw * sc)); it.h = std::max(40, (int)(th * sc));
                y += 8; add(it, user);
            } else if (isAud(f)) {
                Item it; it.kind = K_AUDIO; it.path = f; it.w = std::min(300, colW); it.h = 46;
                y += 8; add(it, user);
            }
        }
    }
    if (g_busy) {
        y += 14;
        if (!g_live.empty()) { Item it; it.kind = K_TEXT; it.text = W(g_live); int wm = 0; wrapLines(dc, g_f, it.text, colW - 4, it.lines, wm, it.lineH); it.w = wm + 6; it.h = (int)it.lines.size() * it.lineH + 4; add(it, false); }
        else { Item it; it.kind = K_TYPING; it.w = 52; it.h = 28; add(it, false); }
    }
    if (g_tr.empty() && !g_busy) { g_items.clear(); Item it; it.kind = K_EMPTY; it.x = colX; it.w = colW; it.h = 90; it.y = 0; g_items.push_back(it); y = 0; }
    g_contentH = y + 24;
}

// ------------------------------------------------------------------ geometry
int editLines() { int n = (int)SendMessageW(g_edit, EM_GETLINECOUNT, 0, 0); return std::min(8, std::max(1, n)); }

void layoutAll(bool stick) {
    if (!g_view) return;
    RECT cr; GetClientRect(g_view, &cr);
    int Wd = cr.right, Ht = cr.bottom;
    if (Wd < 50 || Ht < 50) return;
    int colW = std::min(Wd - 40, 760), colX = (Wd - colW) / 2;
    int chipsH = g_attached.empty() ? 0 : kChip + 10;
    int editH = editLines() * kLine + 6, rowH = std::max(kBtn, editH + 4);
    int compH = kPad + chipsH + rowH + kPad;
    int compBottom = Ht - 14, compTop = compBottom - compH;
    g_composer = {colX, compTop, colX + colW, compBottom};
    int rowTop = compTop + kPad + chipsH, btnTop = rowTop + rowH - kBtn;
    g_rPlus = {colX + kPad, btnTop, colX + kPad + kBtn, btnTop + kBtn};
    g_rSend = {colX + colW - kPad - kBtn, btnTop, colX + colW - kPad, btnTop + kBtn};
    g_rVoice = {g_rSend.left - 4 - kBtn, btnTop, g_rSend.left - 4, btnTop + kBtn};
    int ex = g_rPlus.right + 6, ew = g_rVoice.left - 6 - ex;
    SetWindowPos(g_edit, nullptr, ex, rowTop + (rowH - editH) / 2, ew, editH, SWP_NOZORDER | SWP_NOACTIVATE);
    g_hits.clear();
    g_hits.push_back({g_rPlus, H_PLUS, 0, ""});
    g_hits.push_back({g_rVoice, H_VOICE, 0, ""});
    g_hits.push_back({g_rSend, H_SEND, 0, ""});
    for (size_t i = 0; i < g_attached.size(); i++) {
        int x = colX + 14 + (int)i * (kChip + 8), y = compTop + kPad;
        RECT xr = {x + kChip - 20, y + 2, x + kChip - 2, y + 20};
        g_hits.push_back({xr, H_CHIPX, (int)i, ""});
    }
    int statH = g_busy && !g_status.empty() ? 20 : 0;
    g_statusRect = {colX, compTop - statH - 4, colX + colW, compTop - 4};
    g_logRect = {0, 0, Wd, compTop - statH - 6};

    HDC dc = GetDC(g_view);
    buildItems(dc, Wd, colX, colW);
    ReleaseDC(g_view, dc);
    int viewH = g_logRect.bottom;
    int maxScroll = std::max(0, g_contentH - viewH);
    if (stick) g_scroll = maxScroll;
    g_scroll = std::min(std::max(0, g_scroll), maxScroll);
    SCROLLINFO si = {sizeof si, SIF_RANGE | SIF_PAGE | SIF_POS, 0, std::max(0, g_contentH - 1), (UINT)viewH, g_scroll, 0};
    SetScrollInfo(g_view, SB_VERT, &si, TRUE);
    InvalidateRect(g_view, nullptr, FALSE);
}

// ------------------------------------------------------------------ painting
void drawIconPlus(HDC dc, const RECT& r) {
    int cx = (r.left + r.right) / 2, cy = (r.top + r.bottom) / 2;
    HPEN p = CreatePen(PS_SOLID, 2, cText); HGDIOBJ o = SelectObject(dc, p);
    MoveToEx(dc, cx - 8, cy, nullptr); LineTo(dc, cx + 9, cy); MoveToEx(dc, cx, cy - 8, nullptr); LineTo(dc, cx, cy + 9);
    SelectObject(dc, o); DeleteObject(p);
}
void drawIconVoice(HDC dc, const RECT& r, bool on) {
    int cx = (r.left + r.right) / 2 - 2, cy = (r.top + r.bottom) / 2;
    COLORREF c = on ? cAcc : cMute;
    HPEN p = CreatePen(PS_SOLID, 2, c); HBRUSH b = CreateSolidBrush(c);
    HGDIOBJ op = SelectObject(dc, p), ob = SelectObject(dc, b);
    POINT pts[6] = {{cx - 8, cy - 3}, {cx - 4, cy - 3}, {cx + 1, cy - 8}, {cx + 1, cy + 8}, {cx - 4, cy + 3}, {cx - 8, cy + 3}};
    Polygon(dc, pts, 6);
    SelectObject(dc, GetStockObject(NULL_BRUSH));
    Arc(dc, cx + 1 - 5, cy - 5, cx + 1 + 5, cy + 5, cx + 1, cy + 5, cx + 1, cy - 5);
    Arc(dc, cx + 1 - 9, cy - 9, cx + 1 + 9, cy + 9, cx + 1, cy + 9, cx + 1, cy - 9);
    SelectObject(dc, op); SelectObject(dc, ob); DeleteObject(p); DeleteObject(b);
}
void drawIconSend(HDC dc, const RECT& r, bool stop, bool enabled) {
    fillCircle(dc, r, enabled ? cText : RGB(205, 205, 210));
    int cx = (r.left + r.right) / 2, cy = (r.top + r.bottom) / 2;
    HBRUSH b = CreateSolidBrush(RGB(255, 255, 255)); HPEN p = CreatePen(PS_SOLID, 1, RGB(255, 255, 255));
    HGDIOBJ ob = SelectObject(dc, b), op = SelectObject(dc, p);
    if (stop) { Rectangle(dc, cx - 6, cy - 6, cx + 7, cy + 7); }
    else { POINT pts[7] = {{cx, cy - 8}, {cx - 7, cy - 1}, {cx - 3, cy - 1}, {cx - 3, cy + 8}, {cx + 4, cy + 8}, {cx + 4, cy - 1}, {cx + 8, cy - 1}}; Polygon(dc, pts, 7); }
    SelectObject(dc, ob); SelectObject(dc, op); DeleteObject(b); DeleteObject(p);
}

void paintItem(HDC dc, const Item& it, int yy) {
    RECT r = {it.x, yy, it.x + it.w, yy + it.h};
    switch (it.kind) {
    case K_TEXT: {
        if (it.user) fillRound(dc, r, cUser, cUser, 18);
        int padX = it.user ? 16 : 2, padY = it.user ? 10 : 2;
        SelectObject(dc, g_f);
        int idx = (int)(&it - &g_items[0]);
        int ai = 0, ac = 0, bi = 0, bc = 0; bool sel = hasSelection();
        if (sel) selRange(ai, ac, bi, bc);
        for (size_t li = 0; li < it.lines.size(); li++) {
            const Line& L = it.lines[li];
            int ly = yy + padY + (int)li * it.lineH;
            if (sel && idx >= ai && idx <= bi) {
                int s = idx == ai ? ac : 0, e = idx == bi ? bc : (int)it.text.size();
                int ss = std::max(s, L.s), ee = std::min(e, L.e);
                if (ee > ss) {
                    SIZE a = {0, 0}, b2 = {0, 0};
                    if (ss > L.s) GetTextExtentPoint32W(dc, &it.text[L.s], ss - L.s, &a);
                    GetTextExtentPoint32W(dc, &it.text[L.s], ee - L.s, &b2);
                    HBRUSH hb = CreateSolidBrush(RGB(176, 204, 250)); RECT sr = {it.x + padX + a.cx, ly, it.x + padX + b2.cx, ly + it.lineH}; FillRect(dc, &sr, hb); DeleteObject(hb);
                }
            }
            SetTextColor(dc, cText);
            if (L.e > L.s) TextOutW(dc, it.x + padX, ly + 1, &it.text[L.s], L.e - L.s);
        }
        break;
    }
    case K_NOTE:
        SetTextColor(dc, cErr); SelectObject(dc, g_fs);
        DrawTextW(dc, it.text.c_str(), -1, &r, DT_WORDBREAK | DT_NOPREFIX);
        break;
    case K_IMAGE: {
        const Thumb& t = thumbOf(it.path, 360);
        drawThumb(dc, t, it.x, yy, it.w, it.h, 0, 0, t.w ? t.w : 1, t.h ? t.h : 1, 14);
        break;
    }
    case K_AUDIO: {
        bool playing = g_playing == it.path;
        fillRound(dc, r, cPanel, cLine, 23);
        RECT cb = {r.left + 7, r.top + 7, r.left + 39, r.top + 39};
        fillCircle(dc, cb, cText);
        HBRUSH wb = CreateSolidBrush(RGB(255, 255, 255)); HPEN wp = CreatePen(PS_SOLID, 1, RGB(255, 255, 255));
        HGDIOBJ ob = SelectObject(dc, wb), op = SelectObject(dc, wp);
        int cx = (cb.left + cb.right) / 2, cy = (cb.top + cb.bottom) / 2;
        if (playing) { Rectangle(dc, cx - 5, cy - 6, cx - 1, cy + 7); Rectangle(dc, cx + 2, cy - 6, cx + 6, cy + 7); }
        else { POINT tri[3] = {{cx - 4, cy - 7}, {cx - 4, cy + 7}, {cx + 7, cy}}; Polygon(dc, tri, 3); }
        SelectObject(dc, ob); SelectObject(dc, op); DeleteObject(wb); DeleteObject(wp);
        RECT tr = {r.left + 50, r.top, r.right - 12, r.bottom};
        std::wstring label = ends(it.path, ".wav") ? L"Voice / sound  •  " : L"Audio  •  ";
        label += W(fileOf(it.path));
        SetTextColor(dc, cText); SelectObject(dc, g_fs);
        DrawTextW(dc, label.c_str(), -1, &tr, DT_SINGLELINE | DT_VCENTER | DT_END_ELLIPSIS | DT_NOPREFIX);
        break;
    }
    case K_TYPING:
        for (int i = 0; i < 3; i++) {
            int a = (g_phase + i) % 3;
            COLORREF c = a == 0 ? cText : a == 1 ? RGB(130, 135, 145) : RGB(190, 193, 200);
            RECT d = {r.left + 4 + i * 16, r.top + 9, r.left + 14 + i * 16, r.top + 19};
            fillCircle(dc, d, c);
        }
        break;
    case K_EMPTY: {
        RECT t1 = {r.left, r.top, r.right, r.top + 40};
        SetTextColor(dc, cText); SelectObject(dc, g_fh);
        DrawTextW(dc, L"What shall we make today?", -1, &t1, DT_CENTER | DT_SINGLELINE | DT_NOPREFIX);
        RECT t2 = {r.left, r.top + 44, r.right, r.top + 100};
        SetTextColor(dc, cMute); SelectObject(dc, g_f);
        DrawTextW(dc, L"Ask for a picture, a song or a sound effect.\nAttach a picture with + (or drop it here) and say how to change it.", -1, &t2, DT_CENTER | DT_NOPREFIX);
        break;
    }
    }
}

void paint(HDC wdc, const RECT& cr) {
    int Wd = cr.right, Ht = cr.bottom;
    HDC dc = CreateCompatibleDC(wdc);
    HBITMAP bmp = CreateCompatibleBitmap(wdc, Wd, Ht);
    HGDIOBJ oldBmp = SelectObject(dc, bmp);
    HBRUSH bg = CreateSolidBrush(cBg); FillRect(dc, &cr, bg); DeleteObject(bg);
    SetBkMode(dc, TRANSPARENT);
    ChatAgent& ag = ChatAgent::get();

    // messages
    int saved = SaveDC(dc);
    IntersectClipRect(dc, g_logRect.left, g_logRect.top, g_logRect.right, g_logRect.bottom);
    for (auto& it : g_items) {
        int yy = it.y - g_scroll;
        if (yy + it.h < 0 || yy > g_logRect.bottom) continue;
        paintItem(dc, it, yy);
    }
    RestoreDC(dc, saved);

    // status line
    if (g_statusRect.bottom - g_statusRect.top > 4) {
        SetTextColor(dc, cMute); SelectObject(dc, g_fs);
        DrawTextW(dc, W(g_status).c_str(), -1, &g_statusRect, DT_SINGLELINE | DT_VCENTER | DT_END_ELLIPSIS | DT_NOPREFIX);
    }

    // composer
    fillRound(dc, g_composer, cPanel, cLine, 26);
    for (size_t i = 0; i < g_attached.size(); i++) {
        int x = g_composer.left + 14 + (int)i * (kChip + 8), y = g_composer.top + kPad;
        const Thumb& t = thumbOf(g_attached[i], 360);
        int sw = t.w && t.h ? std::min(t.w, t.h) : 1;
        drawThumb(dc, t, x, y, kChip, kChip, t.w ? (t.w - sw) / 2 : 0, t.h ? (t.h - sw) / 2 : 0, sw, sw, 10);
        RECT xr = {x + kChip - 20, y + 2, x + kChip - 2, y + 20};
        fillCircle(dc, xr, RGB(40, 42, 48));
        HPEN p = CreatePen(PS_SOLID, 2, RGB(255, 255, 255)); HGDIOBJ o = SelectObject(dc, p);
        MoveToEx(dc, xr.left + 5, xr.top + 5, nullptr); LineTo(dc, xr.right - 5, xr.bottom - 5);
        MoveToEx(dc, xr.right - 6, xr.top + 5, nullptr); LineTo(dc, xr.left + 4, xr.bottom - 5);
        SelectObject(dc, o); DeleteObject(p);
    }
    if (g_hot == H_PLUS) fillCircle(dc, g_rPlus, cHot);
    if (g_hot == H_VOICE) fillCircle(dc, g_rVoice, cHot);
    drawIconPlus(dc, g_rPlus);
    drawIconVoice(dc, g_rVoice, ag.settings.voice);
    bool hasText = GetWindowTextLengthW(g_edit) > 0;
    drawIconSend(dc, g_rSend, g_busy, g_busy || hasText || !g_attached.empty());

    BitBlt(wdc, 0, 0, Wd, Ht, dc, 0, 0, SRCCOPY);
    SelectObject(dc, oldBmp); DeleteObject(bmp); DeleteDC(dc);
}

// ------------------------------------------------------------------ actions
std::string chatDir() { std::string d = sys::join(sys::outputDir(), "chat"); sys::makeDirs(d); return d; }

void doSend() {
    ChatAgent& ag = ChatAgent::get();
    if (g_busy) { ag.cancel(); return; }
    int n = GetWindowTextLengthW(g_edit);
    std::wstring w(n + 1, 0); GetWindowTextW(g_edit, &w[0], n + 1); w.resize(n);
    std::string text = sys::narrow(w), err;
    if (text.find_first_not_of(" \r\n\t") == std::string::npos && g_attached.empty()) return;
    if (!ag.send(text, g_attached, err)) { MessageBoxW(g_view, W(err).c_str(), L"Chat", MB_OK | MB_ICONINFORMATION); return; }
    SetWindowTextW(g_edit, L"");
    g_attached.clear();
    g_busy = true; g_sig.clear();
    chatViewTick();
    layoutAll(true);
}

void pickPicture() {
    std::string p;
    if (pickFile(g_view, p, L"Pictures\0*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.tif;*.webp\0All files\0*.*\0")) chatViewAddFile(p);
}
void pasteClipboardPicture(bool& handled) {
    std::string path, err;
    handled = clipboardImageToFile(g_view, chatDir(), path, err);
    if (handled) chatViewAddFile(path);
}

void plusMenu() {
    ChatAgent& ag = ChatAgent::get();
    HMENU m = CreatePopupMenu(), spk = CreatePopupMenu();
    static const char* speakers[] = {"Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee"};
    for (int i = 0; i < 9; i++) AppendMenuW(spk, MF_STRING | (ag.settings.speaker == speakers[i] ? MF_CHECKED : 0), 100 + i, W(speakers[i]).c_str());
    AppendMenuW(m, MF_STRING, 1, L"Upload a picture...");
    AppendMenuW(m, MF_STRING, 2, L"Paste picture from clipboard");
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(m, MF_STRING | (ag.settings.voice ? MF_CHECKED : 0), 3, L"Voice replies");
    AppendMenuW(m, MF_POPUP, (UINT_PTR)spk, L"Voice");
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(m, MF_STRING, 4, L"New chat");
    AppendMenuW(m, MF_STRING, 5, L"Open this chat in the browser");
    POINT pt = {g_rPlus.left, g_rPlus.top - 4}; ClientToScreen(g_view, &pt);
    int cmd = TrackPopupMenu(m, TPM_RETURNCMD | TPM_BOTTOMALIGN | TPM_LEFTALIGN, pt.x, pt.y, 0, g_view, nullptr);
    DestroyMenu(m);
    if (cmd == 1) pickPicture();
    else if (cmd == 2) { bool h = false; pasteClipboardPicture(h); if (!h) MessageBoxW(g_view, L"There is no picture on the clipboard.", L"Chat", MB_OK | MB_ICONINFORMATION); }
    else if (cmd == 3) ag.settings.voice = !ag.settings.voice;
    else if (cmd >= 100 && cmd < 109) { ag.settings.speaker = speakers[cmd - 100]; ag.settings.voice = true; }
    else if (cmd == 4) { stopSound(); ag.reset(); g_attached.clear(); g_sig.clear(); }
    else if (cmd == 5 && WebServer::get().running()) ShellExecuteW(g_view, L"open", W(WebServer::get().url()).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    InvalidateRect(g_view, nullptr, FALSE);
    if (cmd) { g_sig.clear(); chatViewTick(); }
}

Hit* hitAt(int x, int y, Hit& out) {
    for (auto& h : g_hits) if (inRect(h.r, x, y)) { out = h; return &out; }
    if (y < g_logRect.bottom) {
        for (size_t i = 0; i < g_items.size(); i++) {
            auto& it = g_items[i];
            if ((it.kind == K_IMAGE || it.kind == K_AUDIO) && x >= it.x && x < it.x + it.w && y + g_scroll >= it.y && y + g_scroll < it.y + it.h) {
                out = {{it.x, it.y, it.x + it.w, it.y + it.h}, it.kind == K_IMAGE ? H_IMG : H_AUDIO, (int)i, it.path};
                return &out;
            }
        }
    }
    return nullptr;
}

// ----- text selection with the mouse
int textItemAt(int x, int y) {                                   // y in client coordinates; -1 if none
    if (y >= g_logRect.bottom) return -1;
    for (size_t i = 0; i < g_items.size(); i++) {
        const Item& it = g_items[i];
        if (it.kind == K_TEXT && x >= it.x && x < it.x + it.w && y + g_scroll >= it.y && y + g_scroll < it.y + it.h) return (int)i;
    }
    return -1;
}
int nearestTextItem(int y) {
    int best = -1, bd = 1 << 30;
    for (size_t i = 0; i < g_items.size(); i++) {
        const Item& it = g_items[i];
        if (it.kind != K_TEXT) continue;
        int top = it.y - g_scroll, bot = top + it.h;
        int d = y < top ? top - y : y > bot ? y - bot : 0;
        if (d < bd) { bd = d; best = (int)i; }
    }
    return best;
}
int charAt(int idx, int x, int y) {
    const Item& it = g_items[idx];
    if (it.lines.empty()) return 0;
    HDC dc = GetDC(g_view); SelectObject(dc, g_f);
    int padX = it.user ? 16 : 2, padY = it.user ? 10 : 2;
    int li = std::min<int>((int)it.lines.size() - 1, std::max(0, (y + g_scroll - it.y - padY) / std::max(1, it.lineH)));
    if (y + g_scroll < it.y) li = 0;
    const Line L = it.lines[li];
    int rel = x - it.x - padX, res;
    int len = L.e - L.s;
    if (rel <= 0 || len <= 0) res = L.s;
    else {
        int fit = 0; SIZE sz;
        GetTextExtentExPointW(dc, &it.text[L.s], len, rel, &fit, nullptr, &sz);
        if (fit >= len) res = L.e;
        else {
            SIZE a = {0, 0}, b = {0, 0};
            if (fit > 0) GetTextExtentPoint32W(dc, &it.text[L.s], fit, &a);
            GetTextExtentPoint32W(dc, &it.text[L.s], fit + 1, &b);
            res = L.s + fit + ((rel - a.cx) > (b.cx - rel) ? 1 : 0);
        }
    }
    ReleaseDC(g_view, dc);
    return res;
}
void clearSelection() { g_selAi = g_selBi = -1; g_selAc = g_selBc = 0; g_selecting = false; }
void copySelection() { std::string t = selectedText(); if (!t.empty()) clipboardSetText(g_view, t); }
void selectAll() {
    int first = -1, last = -1;
    for (size_t i = 0; i < g_items.size(); i++) if (g_items[i].kind == K_TEXT) { if (first < 0) first = (int)i; last = (int)i; }
    if (first < 0) return;
    g_selAi = first; g_selAc = 0; g_selBi = last; g_selBc = (int)g_items[last].text.size();
    InvalidateRect(g_view, nullptr, FALSE);
}
void contextMenu(int x, int y, POINT screen) {
    int over = textItemAt(x, y);
    HMENU m = CreatePopupMenu();
    AppendMenuW(m, MF_STRING | (hasSelection() ? 0 : MF_GRAYED), 1, L"Copy");
    AppendMenuW(m, MF_STRING | (over >= 0 ? 0 : MF_GRAYED), 2, L"Copy whole message");
    AppendMenuW(m, MF_STRING, 3, L"Select all");
    int cmd = TrackPopupMenu(m, TPM_RETURNCMD | TPM_LEFTALIGN | TPM_TOPALIGN, screen.x, screen.y, 0, g_view, nullptr);
    DestroyMenu(m);
    if (cmd == 1) copySelection();
    else if (cmd == 2 && over >= 0) clipboardSetText(g_view, sys::narrow(g_items[over].text));
    else if (cmd == 3) selectAll();
}

LRESULT CALLBACK editProc(HWND h, UINT m, WPARAM wp, LPARAM lp, UINT_PTR, DWORD_PTR) {
    if (m == WM_PASTE) { bool handled = false; pasteClipboardPicture(handled); if (handled) return 0; }
    if (m == WM_KEYDOWN && (wp == 'C' || wp == 'A') && (GetKeyState(VK_CONTROL) & 0x8000) && hasSelection()) {
        DWORD s = 0, e = 0; SendMessageW(h, EM_GETSEL, (WPARAM)&s, (LPARAM)&e);
        if (s == e && wp == 'C') { copySelection(); return 0; }                          // nothing selected in the input: copy the chat selection
    }
    if (m == WM_KEYDOWN && wp == VK_RETURN && !(GetKeyState(VK_SHIFT) & 0x8000)) { doSend(); return 0; }
    if (m == WM_CHAR && wp == VK_RETURN && !(GetKeyState(VK_SHIFT) & 0x8000)) return 0;
    return DefSubclassProc(h, m, wp, lp);
}

LRESULT CALLBACK viewProc(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    switch (m) {
    case WM_ERASEBKGND: return 1;
    case WM_PAINT: {
        PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps);
        RECT cr; GetClientRect(h, &cr);
        paint(dc, cr);
        EndPaint(h, &ps);
        return 0;
    }
    case WM_SIZE: layoutAll(false); return 0;
    case WM_COMMAND:
        if ((HWND)lp == g_edit && HIWORD(wp) == EN_CHANGE) { layoutAll(false); }
        return 0;
    case WM_CTLCOLOREDIT: { HDC dc = (HDC)wp; SetBkColor(dc, cPanel); SetTextColor(dc, cText); static HBRUSH b = CreateSolidBrush(cPanel); return (LRESULT)b; }
    case WM_VSCROLL: {
        SCROLLINFO si = {sizeof si, SIF_ALL}; GetScrollInfo(h, SB_VERT, &si);
        int pos = si.nPos;
        switch (LOWORD(wp)) {
        case SB_LINEUP: pos -= 40; break; case SB_LINEDOWN: pos += 40; break;
        case SB_PAGEUP: pos -= si.nPage; break; case SB_PAGEDOWN: pos += si.nPage; break;
        case SB_THUMBTRACK: case SB_THUMBPOSITION: pos = si.nTrackPos; break;
        }
        g_scroll = std::max(0, std::min(pos, (int)(g_contentH - g_logRect.bottom)));
        si.fMask = SIF_POS; si.nPos = g_scroll; SetScrollInfo(h, SB_VERT, &si, TRUE);
        InvalidateRect(h, nullptr, FALSE);
        return 0;
    }
    case WM_MOUSEWHEEL: {
        int d = GET_WHEEL_DELTA_WPARAM(wp);
        g_scroll = std::max(0, std::min(g_scroll - d / 2, std::max(0, g_contentH - (int)g_logRect.bottom)));
        SCROLLINFO si = {sizeof si, SIF_POS, 0, 0, 0, g_scroll, 0}; SetScrollInfo(h, SB_VERT, &si, TRUE);
        InvalidateRect(h, nullptr, FALSE);
        return 0;
    }
    case WM_MOUSEMOVE: {
        int x = (short)LOWORD(lp), y = (short)HIWORD(lp);
        if (g_selecting) {
            int ti = nearestTextItem(y);
            if (ti >= 0) { g_selBi = ti; g_selBc = charAt(ti, x, y); InvalidateRect(h, nullptr, FALSE); }
            return 0;
        }
        Hit hh; Hit* p = hitAt(x, y, hh);
        int hot = p && (p->id == H_PLUS || p->id == H_VOICE) ? p->id : 0;
        if (hot != g_hot) { g_hot = hot; InvalidateRect(h, nullptr, FALSE); }
        if (!g_tracking) { TRACKMOUSEEVENT t = {sizeof t, TME_LEAVE, h, 0}; TrackMouseEvent(&t); g_tracking = true; }
        return 0;
    }
    case WM_MOUSELEAVE: g_tracking = false; if (g_hot) { g_hot = 0; InvalidateRect(h, nullptr, FALSE); } return 0;
    case WM_SETCURSOR: {
        POINT pt; GetCursorPos(&pt); ScreenToClient(h, &pt);
        Hit hh;
        if (LOWORD(lp) == HTCLIENT && hitAt(pt.x, pt.y, hh)) { SetCursor(LoadCursorW(nullptr, IDC_HAND)); return TRUE; }
        if (LOWORD(lp) == HTCLIENT && (g_selecting || textItemAt(pt.x, pt.y) >= 0)) { SetCursor(LoadCursorW(nullptr, IDC_IBEAM)); return TRUE; }
        break;
    }
    case WM_LBUTTONDOWN: {
        int x = (short)LOWORD(lp), y = (short)HIWORD(lp);
        int ti = textItemAt(x, y);
        if (ti >= 0) {
            g_selecting = true; SetCapture(h);
            g_selAi = g_selBi = ti; g_selAc = g_selBc = charAt(ti, x, y);
            g_downPt = {x, y};
        } else if (hasSelection()) clearSelection();
        InvalidateRect(h, nullptr, FALSE);
        SetFocus(g_edit);
        return 0;
    }
    case WM_RBUTTONUP: {
        POINT pt = {(short)LOWORD(lp), (short)HIWORD(lp)}, sc = pt; ClientToScreen(h, &sc);
        contextMenu(pt.x, pt.y, sc);
        return 0;
    }
    case WM_LBUTTONUP: {
        int x = (short)LOWORD(lp), y = (short)HIWORD(lp);
        if (g_selecting) {
            g_selecting = false; ReleaseCapture();
            if (abs(x - g_downPt.x) < 3 && abs(y - g_downPt.y) < 3) clearSelection();
            InvalidateRect(h, nullptr, FALSE);
            return 0;
        }
        Hit hh; Hit* p = hitAt(x, y, hh);
        if (!p) return 0;
        switch (p->id) {
        case H_PLUS: plusMenu(); break;
        case H_VOICE: { auto& s = ChatAgent::get().settings; s.voice = !s.voice; InvalidateRect(h, nullptr, FALSE); break; }
        case H_SEND: doSend(); break;
        case H_CHIPX: if (p->idx < (int)g_attached.size()) { g_attached.erase(g_attached.begin() + p->idx); layoutAll(false); } break;
        case H_IMG: ShellExecuteW(h, L"open", W(p->path).c_str(), nullptr, nullptr, SW_SHOWNORMAL); break;
        case H_AUDIO: playSound(p->path); InvalidateRect(h, nullptr, FALSE); break;
        }
        SetFocus(g_edit);
        return 0;
    }
    }
    return DefWindowProcW(h, m, wp, lp);
}

HFONT mkFont(int h, int weight) { return CreateFontW(h, 0, 0, 0, weight, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI"); }
}  // namespace

// ------------------------------------------------------------------ public API
HWND chatViewHwnd() { return g_view; }

HWND chatViewCreate(HWND parent) {
    WNDCLASSW wc = {};
    wc.lpfnWndProc = viewProc; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = kClass;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW); wc.hbrBackground = nullptr;
    RegisterClassW(&wc);
    g_f = mkFont(-17, FW_NORMAL); g_fs = mkFont(-14, FW_NORMAL); g_fh = mkFont(-30, FW_SEMIBOLD);
    g_view = CreateWindowExW(0, kClass, L"", WS_CHILD | WS_VSCROLL | WS_CLIPCHILDREN, 0, 0, 10, 10, parent, nullptr, wc.hInstance, nullptr);
    g_edit = CreateWindowExW(0, L"EDIT", L"", WS_CHILD | WS_VISIBLE | ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN, 0, 0, 100, 28, g_view, nullptr, wc.hInstance, nullptr);
    SendMessageW(g_edit, WM_SETFONT, (WPARAM)g_f, TRUE);
    SendMessageW(g_edit, EM_SETCUEBANNER, TRUE, (LPARAM)L"Message Baker...");
    SetWindowSubclass(g_edit, editProc, 1, 0);
    return g_view;
}

void chatViewShow(bool on) {
    ShowWindow(g_view, on ? SW_SHOW : SW_HIDE);
    if (on) { layoutAll(true); SetFocus(g_edit); }
}

void chatViewStopSound() { stopSound(); }

void chatViewAddFile(const std::string& path) {
    if (!isImg(path)) { MessageBoxW(g_view, L"Only picture files can be attached (png, jpg, bmp, gif, tif, webp).", L"Chat", MB_OK | MB_ICONINFORMATION); return; }
    std::string root = sys::outputDir(), p = path;
    std::string use = path;
    if (_strnicmp(p.c_str(), root.c_str(), root.size()) != 0) {              // keep a copy inside output\ so the history keeps working
        std::string name = "att-" + sys::timestamp() + "-" + std::to_string(g_attached.size()) + "-" + fileOf(path);
        use = sys::join(chatDir(), name);
        if (!CopyFileW(W(path).c_str(), W(use).c_str(), FALSE)) use = path;
    }
    g_attached.push_back(use);
    layoutAll(false);
    SetFocus(g_edit);
}

void chatViewTick() {
    if (!g_view) return;
    ChatAgent& ag = ChatAgent::get();
    ag.poll();
    std::string snd = ag.takeSpeechToPlay();
    if (!snd.empty() && ag.settings.voice && !WebServer::get().clientActive()) playSound(snd);      // the browser plays it when one is open
    checkSoundEnded();
    g_busy = ag.busy(); g_status = g_busy ? ag.status() : ""; g_live = ag.live();
    auto tr = ag.transcript();
    std::string sig = std::to_string(tr.size()) + "|" + std::to_string(g_live.size()) + "|" + (g_busy ? "b" : "i") + g_status;
    if (!tr.empty()) { auto& l = tr.back(); sig += "|" + std::to_string(l.files.size()) + std::to_string(l.text.size()); }
    if (g_busy) { g_phase = (g_phase + 1) % 3; if (g_live.empty()) InvalidateRect(g_view, nullptr, FALSE); }
    if (sig != g_sig) {
        if (tr.size() != g_tr.size() && !g_selecting) clearSelection();      // item numbers change when messages are added
        g_sig = sig; g_tr = tr;
        bool atBottom = g_scroll >= g_contentH - (int)g_logRect.bottom - 30;
        layoutAll(atBottom || g_busy);
    }
}
