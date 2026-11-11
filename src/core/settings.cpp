#include "core/settings.h"
#include <algorithm>
#include <cctype>
#include "util/sys.h"

static std::string lower(std::string s) { for (auto& c : s) c = (char)tolower((unsigned char)c); return s; }
static std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> r; std::string cur;
    for (char c : s) { if (c == d) { if (!cur.empty()) r.push_back(cur); cur.clear(); } else cur += c; }
    if (!cur.empty()) r.push_back(cur);
    return r;
}
static bool endsWith(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

Settings& Settings::get() { static Settings s; return s; }

void Settings::load() {
    std::lock_guard<std::mutex> g(m_);
    path_ = sys::join(sys::exeDir(), "settings.json");
    std::string txt;
    if (sys::readFile(path_, txt)) { bool ok; Json j = Json::parse(txt, &ok); if (ok && j.t == Json::Obj) j_ = j; }
}
void Settings::save() {
    std::lock_guard<std::mutex> g(m_);
    sys::writeFile(path_, j_.dump(2));
}

int Settings::uiInt(const std::string& key, int def) { std::lock_guard<std::mutex> g(m_); return j_.get("ui").integer(key, def); }
void Settings::setUiInt(const std::string& key, int v) { { std::lock_guard<std::mutex> g(m_); j_["ui"][key] = v; } save(); }

bool Settings::firstRunDone() { std::lock_guard<std::mutex> g(m_); return j_.boolean("first_run_done", false); }
void Settings::setFirstRunDone(bool v) { { std::lock_guard<std::mutex> g(m_); j_["first_run_done"] = v; } save(); }

std::string Settings::selection(Kind k, const std::string& role) {
    std::lock_guard<std::mutex> g(m_);
    return j_.get("sel").get(kindKey(k)).str(role);
}
void Settings::setSelection(Kind k, const std::string& role, const std::string& value) {
    { std::lock_guard<std::mutex> g(m_); j_["sel"][kindKey(k)][role] = value; }
    save();
}
std::string Settings::resolve(const std::string& stored) {
    if (stored.empty()) return "";
    std::string p = stored;
    bool abs = p.size() > 1 && (p[1] == ':' || p[0] == '\\' || p[0] == '/');
    if (!abs) p = sys::join(sys::modelsDir(), stored);
    for (auto& c : p) if (c == '/') c = '\\';
    return sys::exists(p) ? p : "";
}
std::string Settings::resolvedSelection(Kind k, const std::string& role) { return resolve(selection(k, role)); }

bool Settings::tabReady(Kind k, std::string* why) {
    for (auto& r : rolesOf(k)) {
        if (!r.required) continue;
        if (resolvedSelection(k, r.key).empty()) {
            if (why) *why = std::string("No ") + r.label + " selected.";
            return false;
        }
    }
    return true;
}

Json Settings::form(Kind k) { std::lock_guard<std::mutex> g(m_); return j_.get("form").get(kindKey(k)); }
void Settings::setForm(Kind k, const Json& v) { { std::lock_guard<std::mutex> g(m_); j_["form"][kindKey(k)] = v; } save(); }

std::string Settings::found(const std::string& id) { std::lock_guard<std::mutex> g(m_); return j_.get("found").str(id); }
void Settings::setFound(const std::string& id, const std::string& path) {
    { std::lock_guard<std::mutex> g(m_); j_["found"][id] = path; }
    save();
}

std::vector<std::string> Settings::extraRoots() {
    std::lock_guard<std::mutex> g(m_);
    std::vector<std::string> r;
    for (auto& x : j_.get("extra_roots").a) r.push_back(x.asStr());
    return r;
}
void Settings::setExtraRoots(const std::vector<std::string>& r) {
    { std::lock_guard<std::mutex> g(m_); Json a = Json::array(); for (auto& s : r) a.push(s); j_["extra_roots"] = a; }
    save();
}

std::vector<ModelFile> Settings::scan(Kind k, const Role& role) {
    std::vector<ModelFile> out;
    std::vector<std::string> kws = split(role.folders, '|');
    std::vector<std::string> exts = split(role.exts, '|');
    bool wantDir = exts.size() == 1 && exts[0] == "dir";

    std::vector<std::pair<std::string, bool>> roots;   // path, isPrimary
    roots.push_back({sys::modelsDir(), true});
    for (auto& r : extraRoots()) roots.push_back({r, false});

    for (auto& rt : roots) {
        const std::string& root = rt.first;
        if (!sys::isDir(root)) continue;
        auto fits = [&](const std::string& rel) {
            if (kws.empty()) return true;
            std::string l = lower(rel);
            for (auto& kw : kws) if (l.find(lower(kw)) != std::string::npos) return true;
            return false;
        };
        auto add = [&](const std::string& rel) {
            std::string full = sys::join(root, rel);
            for (auto& c : full) if (c == '/') c = '\\';
            ModelFile mf;
            mf.value = rt.second ? rel : full;
            mf.label = rt.second ? rel : (std::string("[ext] ") + rel);
            out.push_back(mf);
        };
        if (wantDir) {
            std::vector<std::string> dirs; sys::listDirs(root, dirs, 3);
            for (auto& d : dirs) {
                std::string full = sys::join(root, d);
                if (!fits(d)) continue;
                if (sys::exists(sys::join(full, "config.json")) &&
                    (sys::exists(sys::join(full, "model.safetensors")) || sys::exists(sys::join(full, "model.safetensors.index.json"))))
                    add(d);
            }
        } else {
            std::vector<std::string> files; sys::listFiles(root, files, 4);
            for (auto& f : files) {
                std::string l = lower(f);
                bool ok = false;
                for (auto& e : exts) if (endsWith(l, e)) ok = true;
                if (ok && fits(f)) add(f);
            }
        }
    }
    std::sort(out.begin(), out.end(), [](const ModelFile& a, const ModelFile& b) { return a.label < b.label; });
    return out;
}

void Settings::autoSelect() {
    for (int i = 0; i < kKindCount; i++) {
        Kind k = (Kind)i;
        for (auto& r : rolesOf(k)) {
            if (!r.required || !resolvedSelection(k, r.key).empty()) continue;
            auto c = scan(k, r);
            if (!c.empty()) { std::lock_guard<std::mutex> g(m_); j_["sel"][kindKey(k)][r.key] = c[0].value; }
        }
    }
    save();
}

std::vector<ModelFile> Settings::scanFolderForRole(const std::string& folder, Kind, const Role& role) {
    std::vector<ModelFile> out;
    std::vector<std::string> exts = split(role.exts, '|');
    bool wantDir = exts.size() == 1 && exts[0] == "dir";
    auto add = [&](const std::string& rel) {
        std::string full = sys::join(folder, rel);
        for (auto& c : full) if (c == '/') c = '\\';
        out.push_back({"[folder] " + rel, full});
    };
    if (wantDir) {
        std::vector<std::string> dirs; sys::listDirs(folder, dirs, 12);
        for (auto& d : dirs) {
            std::string full = sys::join(folder, d);
            if (sys::exists(sys::join(full, "config.json")) && (sys::exists(sys::join(full, "model.safetensors")) || sys::exists(sys::join(full, "model.safetensors.index.json")))) add(d);
        }
    } else {
        std::vector<std::string> files; sys::listFiles(folder, files, 12);
        for (auto& f : files) { std::string l = lower(f); for (auto& e : exts) if (endsWith(l, e)) { add(f); break; } }
    }
    std::sort(out.begin(), out.end(), [](const ModelFile& a, const ModelFile& b) { return a.label < b.label; });
    return out;
}
