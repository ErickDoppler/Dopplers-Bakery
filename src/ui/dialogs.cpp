#include "ui/dialogs.h"
#include <commctrl.h>
#include "core/cuda_libs.h"
#include "core/download.h"
#include "core/settings.h"
#include "ui/modal.h"
#include "util/sys.h"

// =============================================================== model settings
namespace {
// start a fresh copy of the app a moment after this one has closed, then close this one (the main window is the dialog's owner)
void restartApp(HWND dlg) {
    wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring cmd = std::wstring(L"cmd.exe /c ping -n 3 127.0.0.1 >nul & start \"\" \"") + exe + L"\"";
    STARTUPINFOW si = {sizeof si}; PROCESS_INFORMATION pi = {};
    if (CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) { CloseHandle(pi.hProcess); CloseHandle(pi.hThread); }
    HWND owner = GetWindow(dlg, GW_OWNER);
    PostMessageW(owner ? owner : dlg, WM_CLOSE, 0, 0);
}

struct RoleRow { const Role* role; HWND combo; std::vector<std::string> values; std::vector<ModelFile> extras; };

struct SettingsDlg : Modal {
    Kind kind; bool changed = false;
    std::vector<RoleRow> rows;
    HWND status = nullptr, cudaStatus = nullptr, cudaBtn = nullptr;
    bool cudaAsked = false;

    void fill(RoleRow& r, const std::string& current) {
        SendMessageW(r.combo, CB_RESETCONTENT, 0, 0);
        r.values.clear();
        SendMessageW(r.combo, CB_ADDSTRING, 0, (LPARAM)L"(none)"); r.values.push_back("");
        int sel = 0;
        auto items = Settings::get().scan(kind, *r.role);
        for (auto& x : r.extras) {          // candidates from "Search in folder"
            bool dup = false; for (auto& it : items) if (it.value == x.value) dup = true;
            if (!dup) items.push_back(x);
        }
        bool found = current.empty();
        for (auto& it : items) {
            SendMessageW(r.combo, CB_ADDSTRING, 0, (LPARAM)sys::widen(it.label).c_str());
            r.values.push_back(it.value);
            if (it.value == current) { sel = (int)r.values.size() - 1; found = true; }
        }
        if (!found) {
            SendMessageW(r.combo, CB_ADDSTRING, 0, (LPARAM)sys::widen("[custom] " + current).c_str());
            r.values.push_back(current); sel = (int)r.values.size() - 1;
        }
        SendMessageW(r.combo, CB_SETCURSEL, sel, 0);
    }
    void refreshCuda() {
        CudaInstaller& ci = CudaInstaller::get();
        bool inst = cudaLibsInstalled();
        EnableWindow(cudaBtn, !inst && !ci.running());
        uiSetText(cudaBtn, inst ? "CUDA libraries installed" : "Download CUDA libraries");
        uiSetText(cudaStatus, inst && !ci.done() ? "Installed. Restart the app after installing for the first time." : ci.status());
    }
    void onTimer() override {
        if (!cudaStatus) return;
        CudaInstaller& ci = CudaInstaller::get();
        refreshCuda();
        if (ci.done() && !cudaAsked) {
            cudaAsked = true;
            if (MessageBoxW(hwnd, L"The CUDA libraries are installed.\n\nRestart the app now to start using them?", L"Dopplers Bakery", MB_YESNO | MB_ICONQUESTION) == IDYES) restartApp(hwnd);
        }
    }
    void onCreate() override {
        const auto& roles = rolesOf(kind);
        int y = 14, W = 640;
        uiPlace(uiMake(hwnd, L"STATIC", "Files are looked up in the models folder next to the program:\n" + sys::modelsDir() +
                       "\n(plus any ComfyUI-style folder added through Models > Add models folder).", SS_LEFT), 14, y, W - 28, 48);
        y += 58;
        for (size_t i = 0; i < roles.size(); i++) {
            RoleRow r; r.role = &roles[i];
            uiPlace(uiMake(hwnd, L"STATIC", std::string(roles[i].label) + (roles[i].required ? " *" : ""), SS_LEFT), 14, y, W - 28, 18);
            r.combo = uiMake(hwnd, L"COMBOBOX", "", CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP, 100 + (int)i);
            uiPlace(r.combo, 14, y + 20, W - 130, 300);
            uiPlace(uiMake(hwnd, L"BUTTON", "Browse...", BS_PUSHBUTTON | WS_TABSTOP, 200 + (int)i), W - 106, y + 19, 92, 26);
            fill(r, Settings::get().selection(kind, roles[i].key));
            rows.push_back(r);
            y += 56;
        }
        // optional GPU runtime: Vulkan is built in; CUDA allows images above 1264 px
        uiPlace(uiMake(hwnd, L"STATIC", "GPU: Vulkan is built in (fast, images up to 1264 px). The optional CUDA libraries (about 0.45 GB, downloaded from github.com/ggml-org/llama.cpp) allow larger images.", SS_LEFT), 14, y + 4, W - 28, 34);
        cudaBtn = uiMake(hwnd, L"BUTTON", "Download CUDA libraries", BS_PUSHBUTTON | WS_TABSTOP, 301);
        uiPlace(cudaBtn, 14, y + 42, 190, 28);
        cudaStatus = uiMake(hwnd, L"STATIC", "", SS_LEFT | SS_ENDELLIPSIS);
        uiPlace(cudaStatus, 212, y + 48, W - 226, 18);
        refreshCuda();
        y += 84;
        SetTimer(hwnd, 1, 300, nullptr);
        uiPlace(uiMake(hwnd, L"BUTTON", "Search in folder...", BS_PUSHBUTTON | WS_TABSTOP, 300), 14, y + 6, 150, 28);
        status = uiMake(hwnd, L"STATIC", "", SS_LEFT | SS_ENDELLIPSIS);
        uiPlace(status, 172, y + 12, W - 172 - 220, 18);
        uiPlace(uiMake(hwnd, L"BUTTON", "OK", BS_DEFPUSHBUTTON | WS_TABSTOP, IDOK), W - 200, y + 6, 90, 28);
        uiPlace(uiMake(hwnd, L"BUTTON", "Cancel", BS_PUSHBUTTON | WS_TABSTOP, IDCANCEL), W - 104, y + 6, 90, 28);
        SetWindowPos(hwnd, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER);
        // grow the window to fit the rows
        RECT wr; GetWindowRect(hwnd, &wr);
        RECT want = {0, 0, W, y + 48}; AdjustWindowRectEx(&want, WS_CAPTION | WS_SYSMENU, FALSE, WS_EX_DLGMODALFRAME);
        SetWindowPos(hwnd, nullptr, 0, 0, want.right - want.left, want.bottom - want.top, SWP_NOMOVE | SWP_NOZORDER);
    }
    void onCommand(int id, int) override {
        if (id >= 200 && id < 200 + (int)rows.size()) {
            RoleRow& r = rows[id - 200];
            std::string p; bool ok;
            if (std::string(r.role->exts) == "dir") ok = pickFolder(hwnd, p, L"Select the model folder");
            else ok = pickFile(hwnd, p, L"Model files\0*.safetensors;*.gguf;*.pth;*.bin;*.ckpt;*.pt\0All files\0*.*\0");
            if (ok) fill(r, p);
        } else if (id == 301) {
            if (MessageBoxW(hwnd, L"Download the CUDA libraries (about 450 MB) from github.com/ggml-org/llama.cpp and unpack them next to the program?\n\nThey need an NVIDIA GPU with a recent driver. The app has to be restarted afterwards.", L"Download CUDA libraries", MB_YESNO | MB_ICONQUESTION) == IDYES) { CudaInstaller::get().start(); refreshCuda(); }
        } else if (id == 300) {
            std::string folder;
            if (!pickFolder(hwnd, folder, L"Select a folder to search (subfolders are included)")) return;
            uiSetText(status, "Searching...");
            UpdateWindow(hwnd);
            std::vector<CatEntry> mine;
            for (auto& e : loadCatalog()) if (e.kind == kind) mine.push_back(e);
            int hits = scanFolderForCatalog(folder, mine);          // known models are selected right away
            int extra = 0;
            for (auto& r : rows) {
                r.extras = Settings::get().scanFolderForRole(folder, kind, *r.role);
                extra += (int)r.extras.size();
                fill(r, Settings::get().selection(kind, r.role->key));
            }
            changed = true;
            uiSetText(status, std::to_string(hits) + " known model(s) selected, " + std::to_string(extra) + " candidate file(s) added to the lists");
        } else if (id == IDOK) {
            for (auto& r : rows) {
                int s = (int)SendMessageW(r.combo, CB_GETCURSEL, 0, 0);
                std::string v = s >= 0 && s < (int)r.values.size() ? r.values[s] : "";
                if (v != Settings::get().selection(kind, r.role->key)) { Settings::get().setSelection(kind, r.role->key, v); changed = true; }
            }
            close();
        } else if (id == IDCANCEL) close();
    }
};
}  // namespace

bool showModelSettings(HWND parent, Kind k) {
    SettingsDlg d; d.kind = k;
    runModal(d, parent, std::string(kindTitle(k)) + " - model settings", 640, 300);
    return d.changed;
}

// =============================================================== downloads
namespace {
struct DownloadDlg : Modal {
    bool first = false, any = false, running = false;
    std::vector<CatEntry> cat;
    HWND list = nullptr, bar = nullptr, status = nullptr, go = nullptr, skip = nullptr, search = nullptr;
    Downloader dl;

    void refreshRows() {
        ListView_DeleteAllItems(list);
        for (size_t i = 0; i < cat.size(); i++) {
            LVITEMW it = {}; it.mask = LVIF_TEXT; it.iItem = (int)i;
            std::wstring w = sys::widen(cat[i].label); it.pszText = &w[0];
            ListView_InsertItem(list, &it);
            std::wstring k = sys::widen(kindTitle(cat[i].kind)); ListView_SetItemText(list, (int)i, 1, &k[0]);
            std::wstring s = sys::widen(sys::humanSize(cat[i].approxBytes)); ListView_SetItemText(list, (int)i, 2, &s[0]);
            std::string loc = catalogLocation(cat[i]);
            bool inst = !loc.empty();
            bool inModels = inst && loc.size() >= sys::modelsDir().size() && _strnicmp(loc.c_str(), sys::modelsDir().c_str(), sys::modelsDir().size()) == 0;
            std::wstring st = inst ? (inModels ? L"installed" : L"found") : (cat[i].downloadable() ? L"missing" : L"manual");
            ListView_SetItemText(list, (int)i, 3, &st[0]);
            ListView_SetCheckState(list, (int)i, !inst && cat[i].downloadable());
        }
    }
    void onCreate() override {
        cat = loadCatalog();
        int W = 700;
        uiPlace(uiMake(hwnd, L"STATIC",
            first ? "Welcome. Pick the models to download into the models folder next to the program. You can skip this and select your own files in each tab's [Settings]."
                  : "Models available for download. They are saved to the models folder next to the program.", SS_LEFT), 14, 12, W - 28, 36);
        list = uiMake(hwnd, WC_LISTVIEWW, "", LVS_REPORT | LVS_SINGLESEL | LVS_SHOWSELALWAYS | WS_TABSTOP, 10, WS_EX_CLIENTEDGE);
        ListView_SetExtendedListViewStyle(list, LVS_EX_CHECKBOXES | LVS_EX_FULLROWSELECT);
        const wchar_t* cols[] = {L"Model", L"Tab", L"Size", L"State"}; int cw[] = {330, 110, 100, 110};
        for (int i = 0; i < 4; i++) { LVCOLUMNW c = {}; c.mask = LVCF_TEXT | LVCF_WIDTH; c.pszText = (LPWSTR)cols[i]; c.cx = cw[i]; ListView_InsertColumn(list, i, &c); }
        uiPlace(list, 14, 54, W - 28, 230);
        refreshRows();
        status = uiMake(hwnd, L"STATIC", "", SS_LEFT | SS_ENDELLIPSIS);
        uiPlace(status, 14, 294, W - 28, 18);
        bar = uiMake(hwnd, PROGRESS_CLASSW, "", PBS_SMOOTH);
        SendMessageW(bar, PBM_SETRANGE32, 0, 1000);
        uiPlace(bar, 14, 316, W - 28, 20);
        go = uiMake(hwnd, L"BUTTON", "Download selected", BS_DEFPUSHBUTTON | WS_TABSTOP, 1);
        skip = uiMake(hwnd, L"BUTTON", first ? "Skip" : "Close", BS_PUSHBUTTON | WS_TABSTOP, 2);
        uiPlace(go, W - 270, 348, 140, 28); uiPlace(skip, W - 120, 348, 106, 28);
        search = uiMake(hwnd, L"BUTTON", "SEARCH IN FOLDER", BS_PUSHBUTTON | WS_TABSTOP, 3);
        uiPlace(search, 14, 348, 170, 28);
        SetTimer(hwnd, 1, 200, nullptr);
    }
    void onCommand(int id, int) override {
        if (id == 1 && !running) {
            std::vector<CatEntry> pick;
            for (size_t i = 0; i < cat.size(); i++) if (ListView_GetCheckState(list, (int)i) && cat[i].downloadable()) pick.push_back(cat[i]);
            if (pick.empty()) { uiSetText(status, "Nothing downloadable is checked."); return; }
            running = true; any = true;
            uiSetText(go, "Working..."); EnableWindow(go, FALSE); EnableWindow(search, FALSE); uiSetText(skip, "Cancel");
            dl.start(pick);
        } else if (id == 3 && !running) {
            std::string folder;
            if (!pickFolder(hwnd, folder, L"Select a folder to search for models (subfolders are included)")) return;
            uiSetText(status, "Searching " + folder + " ...");
            UpdateWindow(hwnd);
            int hits = scanFolderForCatalog(folder, cat);
            any = any || hits > 0;
            refreshRows();
            uiSetText(status, "Found " + std::to_string(hits) + " of " + std::to_string(cat.size()) + " models; they are selected for their tabs.");
        } else if (id == 2) {
            if (running) { dl.st.cancel = true; uiSetText(status, "Cancelling..."); }
            else close();
        }
    }
    void onTimer() override {
        if (!running) return;
        auto& st = dl.st;
        unsigned long long d = st.doneBytes, t = st.totalBytes;
        SendMessageW(bar, PBM_SETPOS, t ? (WPARAM)(d * 1000 / t) : 0, 0);
        std::string cur; int idx, cnt;
        { std::lock_guard<std::mutex> g(st.m); cur = st.current; idx = st.fileIndex; cnt = st.fileCount; }
        if (!st.finished) {
            uiSetText(status, "(" + std::to_string(idx) + "/" + std::to_string(cnt) + ") " + cur + "   " + sys::humanSize(d) + " / " + sys::humanSize(t));
        } else {
            dl.join(); running = false;
            std::string err; { std::lock_guard<std::mutex> g(st.m); err = st.error; }
            if (st.ok) { uiSetText(status, "Done."); SendMessageW(bar, PBM_SETPOS, 1000, 0); }
            else uiSetText(status, err.empty() ? "Cancelled (partial files are kept and resume next time)." : "Failed: " + err);
            uiSetText(go, "Download selected"); EnableWindow(go, TRUE); EnableWindow(search, TRUE); uiSetText(skip, "Close");
            refreshRows();
            Settings::get().autoSelect();
        }
    }
    bool onClose() override {
        if (running) { dl.st.cancel = true; uiSetText(status, "Cancelling... close again when it stops."); return false; }
        return true;
    }
};
}  // namespace

bool showDownloadDialog(HWND parent, bool firstLaunch) {
    DownloadDlg d; d.first = firstLaunch;
    runModal(d, parent, firstLaunch ? "Welcome - download models" : "Download models", 700, 392);
    return d.any;
}
