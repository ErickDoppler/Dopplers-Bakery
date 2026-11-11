// Command line mode:  DopplersBakery <llm|tts|image|audio|models> [--flag value ...]
#include "cli/cli.h"
#include <cstdio>
#include <map>
#include "backends/engines.h"
#include <windows.h>
#include "chat/agent.h"
#include "core/cuda_libs.h"
#include "web/server.h"
#include "core/queue.h"
#include "core/download.h"
#include "core/forms.h"
#include "core/settings.h"
#include "util/sys.h"

namespace {
struct ConsoleCtl : JobCtl {
    std::string lastOut, text; int lastPct = -1;
    void progress(float f, const std::string& s) override {
        int pct = (int)(f * 100);
        if (pct != lastPct) { fprintf(stderr, "\r[%3d%%] %-60s", pct, s.c_str()); fflush(stderr); lastPct = pct; }
    }
    void addOutput(const std::string& p) override { lastOut = p; }
    void setText(const std::string& t) override { text = t; }
    bool checkpoint() override { return true; }
};

void usage() {
    fprintf(stdout,
        "Dopplers Bakery - command line\n\n"
        "  DopplersBakery                       start the window UI\n"
        "  DopplersBakery llm   --prompt \"...\" [--max_tokens N --temperature T --seed S ...]\n"
        "  DopplersBakery tts   --text \"...\"   [--speaker Ryan --language Auto ...]\n"
        "  DopplersBakery image --prompt \"...\" [--width 720 --height 720 --steps 10 --cfg 1 ...]\n"
        "  DopplersBakery audio --prompt \"...\" [--length 60 --steps 8 ...]\n"
        "  DopplersBakery models [list | download <id> | scan]\n\n"
        "Every setting of a tab can be given as --<name>. Model overrides: --model --clip --vae --lora --llm <path>.\n"
        "Without overrides the files chosen in the window's [Settings] are used. Results land in output\\.\n"
        "Run `DopplersBakery <tab> --help` to list the settings and defaults of a tab.\n");
}

void tabHelp(Kind k) {
    fprintf(stdout, "Settings for %s:\n", kindTitle(k));
    for (auto& f : fieldsOf(k)) {
        std::string d = f.type == FT::Check ? (f.def ? "true" : "false") : (f.type == FT::Text || f.type == FT::File || f.type == FT::Multi || f.type == FT::Combo) ? f.sdef : std::to_string(f.def);
        fprintf(stdout, "  --%-16s %s  [default: %s]\n", f.key, f.label, d.c_str());
    }
    fprintf(stdout, "  --out <file>         copy the result to this path\n");
    for (auto& r : rolesOf(k)) fprintf(stdout, "  --%-16s %s\n", r.key, r.label);
}

int cmdModels(const std::vector<std::string>& a) {
    std::string sub = a.size() > 1 ? a[1] : "list";
    auto cat = loadCatalog();
    if (sub == "list") {
        for (auto& e : cat) fprintf(stdout, "%-20s %-14s %-10s %s\n", e.id.c_str(), kindTitle(e.kind), sys::humanSize(e.approxBytes).c_str(), catalogInstalled(e) ? "installed" : "missing");
        return 0;
    }
    if (sub == "find" && a.size() > 2) {
        int n = scanFolderForCatalog(a[2], cat);
        fprintf(stdout, "found %d of %zu catalog models in %s\n", n, cat.size(), a[2].c_str());
        for (auto& e : cat) { std::string l = catalogLocation(e); if (!l.empty()) fprintf(stdout, "  %-20s %s\n", e.id.c_str(), l.c_str()); }
        return 0;
    }
    if (sub == "scan") {
        for (int i = 0; i < kKindCount; i++) for (auto& r : rolesOf((Kind)i)) {
            fprintf(stdout, "[%s] %s\n", kindKey((Kind)i), r.label);
            for (auto& m : Settings::get().scan((Kind)i, r)) fprintf(stdout, "    %s\n", m.value.c_str());
        }
        return 0;
    }
    if (sub == "download" && a.size() > 2) {
        std::vector<CatEntry> pick;
        for (size_t i = 2; i < a.size(); i++) for (auto& e : cat) if (e.id == a[i] || a[i] == "all") pick.push_back(e);
        if (pick.empty()) { fprintf(stderr, "unknown model id (see `models list`)\n"); return 2; }
        Downloader d; d.start(pick);
        while (!d.st.finished) {
            unsigned long long done = d.st.doneBytes, tot = d.st.totalBytes;
            std::string cur; { std::lock_guard<std::mutex> g(d.st.m); cur = d.st.current; }
            fprintf(stderr, "\r%5.1f%%  %s / %s  %-40.40s", tot ? done * 100.0 / tot : 0.0, sys::humanSize(done).c_str(), sys::humanSize(tot).c_str(), cur.c_str());
            Sleep(500);
        }
        d.join(); fprintf(stderr, "\n");
        if (!d.st.ok) { fprintf(stderr, "download failed: %s\n", d.st.error.c_str()); return 1; }
        Settings::get().autoSelect();
        return 0;
    }
    usage(); return 2;
}
}  // namespace

int runSelfTest(const std::vector<std::string>& args);
int runBackendsList();
int runTextEncoderTest(const std::vector<std::string>& a);
int runDiTTest(const std::vector<std::string>& a);
int runDiTBench(const std::vector<std::string>& a);
int runVaeTest(const std::vector<std::string>& a);
int runTtsDecoderTest(const std::vector<std::string>& a);
int runTalkerTest(const std::vector<std::string>& a);
int runSaTextTest(const std::vector<std::string>& a);
int runSpeakerTest(const std::vector<std::string>& a);
int runCloneSim(const std::vector<std::string>& a);
int runCodecTest(const std::vector<std::string>& a);
int runSaDiTTest(const std::vector<std::string>& a);
int runSaVaeTest(const std::vector<std::string>& a);

// chat --say "message" [--say "next message" ...] [--voice] [--speaker Ryan]
int runChatCli(const std::vector<std::string>& a) {
    std::vector<std::string> says, attach; ChatAgent& ag = ChatAgent::get();
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] == "--say" && i + 1 < a.size()) says.push_back(a[++i]);
        else if (a[i] == "--image" && i + 1 < a.size()) attach.push_back(a[++i]);          // attached to the first message
        else if (a[i] == "--voice") ag.settings.voice = true;
        else if (a[i] == "--speaker" && i + 1 < a.size()) ag.settings.speaker = a[++i];
    }
    if (says.empty()) { fprintf(stderr, "usage: chat --say \"message\" [--say ...] [--voice] [--speaker Ryan]\n"); return 2; }
    JobQueue::get().start();
    size_t shown = 0;
    for (auto& s : says) {
        fprintf(stdout, "\n> %s\n", s.c_str());
        std::string err;
        bool sent = ag.send(s, attach, err); attach.clear();
        if (!sent) { fprintf(stderr, "%s\n", err.c_str()); JobQueue::get().shutdown(); return 1; }
        std::string lastStatus;
        while (ag.poll() || ag.busy()) {
            std::string st = ag.status();
            if (st != lastStatus) { fprintf(stderr, "[%s]\n", st.c_str()); lastStatus = st; }
            Sleep(200);
            auto tr = ag.transcript();
            for (; shown < tr.size(); shown++) {
                if (tr[shown].role == "user") continue;
                fprintf(stdout, "%s: %s\n", tr[shown].role.c_str(), tr[shown].text.c_str());
                for (auto& f : tr[shown].files) fprintf(stdout, "   file: %s\n", f.c_str());
            }
        }
        auto tr = ag.transcript();
        for (; shown < tr.size(); shown++) {
            if (tr[shown].role == "user") continue;
            fprintf(stdout, "%s: %s\n", tr[shown].role.c_str(), tr[shown].text.c_str());
            for (auto& f : tr[shown].files) fprintf(stdout, "   file: %s\n", f.c_str());
        }
        fflush(stdout);
    }
    JobQueue::get().shutdown();
    return 0;
}

// serve [--port 8787] [--lan]   chat in a browser without the window
int runServeCli(const std::vector<std::string>& a) {
    int port = 8787; bool lan = false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] == "--port" && i + 1 < a.size()) port = atoi(a[++i].c_str());
        else if (a[i] == "--lan") lan = true;
    }
    JobQueue::get().start();
    std::string err;
    if (!WebServer::get().start(port, lan, err)) { fprintf(stderr, "cannot start the web server: %s\n", err.c_str()); return 1; }
    fprintf(stdout, "Chat is running at %s\n%sPress Ctrl+C to stop.\n", WebServer::get().url().c_str(),
            lan ? "Other devices: use this PC's IP address instead of 127.0.0.1 (keep the ?key=... part).\n" : "");
    fflush(stdout);
    for (;;) Sleep(1000);
}

// cuda-install: download the optional CUDA libraries next to the exe (the window offers the same in Model settings)
int runCudaInstallCli() {
    if (cudaLibsInstalled()) { fprintf(stdout, "CUDA libraries are already installed.\n"); return 0; }
    CudaInstaller& ci = CudaInstaller::get();
    ci.start();
    std::string last;
    while (ci.running()) { std::string s = ci.status(); if (s != last) { fprintf(stderr, "\r%-70s", s.c_str()); last = s; } Sleep(500); }
    fprintf(stderr, "\n%s\n", ci.status().c_str());
    return ci.done() ? 0 : 1;
}

// switch-test: two small image jobs on different backends in one process (checks the runtime CUDA/Vulkan switch)
int runSwitchTestCli() {
    JobQueue& q = JobQueue::get();
    q.start();
    std::vector<int> ids;
    for (const char* be : {"Vulkan", "CUDA", "Vulkan"}) {
        Json form = Json::object();
        form["prompt"] = "a red fox in snow"; form["width"] = 512; form["height"] = 512; form["steps"] = 2; form["seed"] = 3; form["backend"] = be;
        ids.push_back(q.add(Kind::Image, makeJobParams(Kind::Image, form), std::string("switch ") + be));
    }
    for (int id : ids) {
        Job j;
        for (;;) { if (q.find(id, j) && (j.state == JobState::Done || j.state == JobState::Failed || j.state == JobState::Canceled)) break; Sleep(300); }
        fprintf(stdout, "job %d (%s): %s | %s%s\n", id, j.title.c_str(), stateName(j.state), j.status.c_str(), j.error.empty() ? "" : (" ERROR " + j.error).c_str());
    }
    q.shutdown();
    return 0;
}

int runCli(const std::vector<std::string>& a) {
    if (a.empty() || a[0] == "help" || a[0] == "--help" || a[0] == "-h") { usage(); return 0; }
    Settings::get().load();
    Settings::get().autoSelect();      // repairs empty or wrong model choices (checked by file content)
    sys::makeDirs(sys::modelsDir()); sys::makeDirs(sys::outputDir());
    if (a[0] == "models") return cmdModels(a);
    if (a[0] == "selftest") return runSelfTest(std::vector<std::string>(a.begin() + 1, a.end()));
    if (a[0] == "backends") return runBackendsList();
    if (a[0] == "selftest-te") return runTextEncoderTest(std::vector<std::string>(a.begin() + 1, a.end()));
    if (a[0] == "selftest-dit") return runDiTTest(std::vector<std::string>(a.begin() + 1, a.end()));
    if (a[0] == "bench-dit") return runDiTBench(std::vector<std::string>(a.begin() + 1, a.end()));
    if (a[0] == "selftest-vae") return runVaeTest(std::vector<std::string>(a.begin() + 1, a.end()));
    if (a[0] == "selftest-tts-dec") return runTtsDecoderTest(std::vector<std::string>(a.begin() + 1, a.end()));
    if (a[0] == "selftest-talker") return runTalkerTest(std::vector<std::string>(a.begin() + 1, a.end()));
    if (a[0] == "selftest-sa-te") return runSaTextTest(std::vector<std::string>(a.begin() + 1, a.end()));
    if (a[0] == "selftest-base-spk") return runSpeakerTest(std::vector<std::string>(a.begin() + 1, a.end()));
    if (a[0] == "clone-sim") return runCloneSim(std::vector<std::string>(a.begin() + 1, a.end()));
    if (a[0] == "selftest-codec") return runCodecTest(std::vector<std::string>(a.begin() + 1, a.end()));
    if (a[0] == "selftest-sa-dit") return runSaDiTTest(std::vector<std::string>(a.begin() + 1, a.end()));
    if (a[0] == "selftest-sa-vae") return runSaVaeTest(std::vector<std::string>(a.begin() + 1, a.end()));
    if (a[0] == "switch-test") return runSwitchTestCli();
    if (a[0] == "cuda-install") return runCudaInstallCli();
    if (a[0] == "serve") return runServeCli(std::vector<std::string>(a.begin() + 1, a.end()));
    if (a[0] == "chat") return runChatCli(std::vector<std::string>(a.begin() + 1, a.end()));

    Kind k;
    bool found = false;
    for (int i = 0; i < kKindCount; i++) if (a[0] == kindKey((Kind)i)) { k = (Kind)i; found = true; }
    if (!found) { fprintf(stderr, "unknown command '%s'\n", a[0].c_str()); usage(); return 2; }

    std::map<std::string, std::string> opt;
    for (size_t i = 1; i < a.size(); i++) {
        if (a[i] == "--help" || a[i] == "-h") { tabHelp(k); return 0; }
        if (a[i].rfind("--", 0) != 0) { fprintf(stderr, "unexpected argument '%s'\n", a[i].c_str()); return 2; }
        std::string key = a[i].substr(2);
        opt[key] = (i + 1 < a.size() && a[i + 1].rfind("--", 0) != 0) ? a[++i] : "true";
    }

    Json form = Json::object();
    for (auto& f : fieldsOf(k)) {
        auto it = opt.find(f.key);
        if (it == opt.end()) continue;
        switch (f.type) {
        case FT::Check: form[f.key] = it->second != "false" && it->second != "0"; break;
        case FT::Int: case FT::Float: case FT::Slider: form[f.key] = atof(it->second.c_str()); break;
        default: form[f.key] = it->second;
        }
        opt.erase(it);
    }
    Json params = makeJobParams(k, form);
    for (auto& r : rolesOf(k)) {
        auto it = opt.find(r.key);
        if (it != opt.end()) { params[r.key] = it->second; opt.erase(it); }
    }
    std::string out = opt.count("out") ? opt["out"] : "";
    opt.erase("out");
    for (auto& kv : opt) fprintf(stderr, "warning: ignoring unknown option --%s\n", kv.first.c_str());

    std::string why;
    std::string mainText = k == Kind::TTS ? params.str("text") : params.str("prompt");
    if (mainText.empty()) { fprintf(stderr, "error: %s is required\n", k == Kind::TTS ? "--text" : "--prompt"); return 2; }

    Job job; job.id = 1; job.kind = k; job.params = params;
    ConsoleCtl ctl;
    std::string err;
    bool ok = false;
    try { ok = backendFor(k)(job, ctl, err); } catch (const std::exception& e) { err = e.what(); }
    fprintf(stderr, "\r%-72s\r", "");
    if (!ok) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    if (k == Kind::LLM) fprintf(stdout, "%s\n", ctl.text.c_str());
    else if (k == Kind::Audio && !ctl.text.empty()) fprintf(stderr, "prompt used: %s\n", ctl.text.c_str());
    if (!ctl.lastOut.empty()) {
        if (!out.empty()) { std::string d; sys::readFile(ctl.lastOut, d); sys::writeFile(out, d); ctl.lastOut = out; }
        fprintf(k == Kind::LLM ? stderr : stdout, "%s\n", ctl.lastOut.c_str());
    }
    return 0;
}
