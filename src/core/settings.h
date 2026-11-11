// Persistent settings (settings.json next to the exe) and model discovery.
#pragma once
#include <mutex>
#include <string>
#include <vector>
#include "core/types.h"
#include "util/json.h"

struct ModelFile {
    std::string label;   // shown in the combo box
    std::string value;   // stored (relative to models\ when inside it, else absolute)
};

class Settings {
public:
    static Settings& get();
    void load();
    void save();

    int uiInt(const std::string& key, int def);                 // small remembered window settings (queue width, column widths ...)
    void setUiInt(const std::string& key, int v);
    bool firstRunDone();
    void setFirstRunDone(bool v);

    std::string selection(Kind k, const std::string& role);
    void setSelection(Kind k, const std::string& role, const std::string& value);
    // Resolve a stored value to an absolute path ("" if empty or missing on disk).
    std::string resolve(const std::string& stored);
    std::string resolvedSelection(Kind k, const std::string& role);
    // True when all required roles of the tab point at existing files.
    bool tabReady(Kind k, std::string* why = nullptr);

    Json form(Kind k);                 // remembered form values
    void setForm(Kind k, const Json& v);

    // Locations found by "Search in folder" for catalog entries (entry id -> absolute path)
    std::string found(const std::string& id);
    void setFound(const std::string& id, const std::string& path);

    std::vector<std::string> extraRoots();
    void setExtraRoots(const std::vector<std::string>& r);

    // Files/folders under models\ (and extra roots) that fit the role.
    std::vector<ModelFile> scan(Kind k, const Role& role);

    // Every file (or model folder) under `folder`, subfolders included, that fits the role by type (folder names are not used).
    std::vector<ModelFile> scanFolderForRole(const std::string& folder, Kind k, const Role& role);

    // Pre-select the first matching file for every unset role (called after downloads/launch).
    void autoSelect();

private:
    std::mutex m_;
    Json j_ = Json::object();
    std::string path_;
};
