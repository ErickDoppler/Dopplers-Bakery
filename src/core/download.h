// Model catalog + resumable HTTPS downloader (WinHTTP, no third-party code).
#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "core/types.h"

struct CatFile { std::string url; std::string rel; unsigned long long size = 0; };

struct CatEntry {
    std::string id;
    std::string label;
    Kind kind = Kind::LLM;
    std::string note;
    std::string hfRepo;           // if set: every file of the repo is fetched into models\<destDir>
    std::string destDir;
    std::string role;             // which [Settings] slot of the tab this model fills ("model", "clip", "vae", ...)
    std::vector<CatFile> files;   // explicit files
    unsigned long long approxBytes = 0;
    bool downloadable() const {
        if (!hfRepo.empty()) return true;
        for (auto& f : files) if (!f.url.empty()) return true;
        return false;
    }
};

// Where the model is available right now (models\, an added models folder, or a folder found by "Search in folder");
// empty when missing. For repo-style entries this is the model folder.
std::string catalogLocation(const CatEntry& e);
// Recursively looks through `folder` for the catalog's models (by file name / folder name), remembers where they are
// and selects them for their tabs. Returns how many entries were found.
int scanFolderForCatalog(const std::string& folder, const std::vector<CatEntry>& cat);

// Built-in entries plus <exe>\catalog.json (same shape) if present.
std::vector<CatEntry> loadCatalog();
// True when the entry's files already exist under models\ (or an extra root).
bool catalogInstalled(const CatEntry& e);

bool httpGetString(const std::string& url, std::string& body, std::string& err);

struct DlState {
    std::atomic<unsigned long long> doneBytes{0}, totalBytes{0};
    std::atomic<bool> cancel{false}, finished{false}, ok{false};
    std::mutex m;
    std::string current, error;
    int fileIndex = 0, fileCount = 0;
};

class Downloader {
public:
    ~Downloader() { if (th_.joinable()) { st.cancel = true; th_.join(); } }
    void start(std::vector<CatEntry> entries);     // runs on a background thread
    void join() { if (th_.joinable()) th_.join(); }
    DlState st;
private:
    std::thread th_;
};
