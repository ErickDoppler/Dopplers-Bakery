// Job queue: one worker thread, pause/cancel/reorder/remove, session output history.
#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "core/types.h"
#include "util/json.h"

enum class JobState { Queued, Held, Running, Paused, Done, Failed, Canceled };
const char* stateName(JobState s);

struct Job {
    int id = 0;
    Kind kind = Kind::LLM;
    JobState state = JobState::Queued;
    Json params;                       // everything the backend needs (prompt, sizes, model paths ...)
    std::string title;                 // short label for the queue list
    float progress = 0;                // 0..1
    std::string status;                // "step 3/10", "loading model" ...
    std::vector<std::string> outputs;  // files written
    std::string text;                  // text result (LLM)
    std::string error;
    double tStart = 0, tEnd = 0;
};

// Handed to a backend while it runs. All calls are thread-safe.
class JobCtl {
public:
    virtual ~JobCtl() {}
    virtual void progress(float frac, const std::string& status) = 0;
    virtual void addOutput(const std::string& path) = 0;
    virtual void setText(const std::string& t) = 0;
    // Call between steps: blocks while paused; returns false when cancelled (stop at once).
    virtual bool checkpoint() = 0;
};

// A backend returns true on success; on failure fills err (also used for "cancelled").
using BackendFn = std::function<bool(const Job& job, JobCtl& ctl, std::string& err)>;
BackendFn backendFor(Kind k);           // defined in backends/backend.cpp

class JobQueue {
public:
    static JobQueue& get();
    void start();
    void shutdown();

    int add(Kind k, const Json& params, const std::string& title);
    std::vector<Job> snapshot();
    bool find(int id, Job& out);
    bool moveUp(int id);
    bool moveDown(int id);
    bool remove(int id);                // queued/finished only; running must be cancelled first
    void clearFinished();
    void pause(int id);                 // running -> paused, queued -> held
    void resume(int id);
    void cancel(int id);
    bool busy();
    // Remove one output of a finished job (file is deleted by the caller).
    void dropOutput(int id, const std::string& path);
    void dropJobText(int id);

private:
    friend class CtlImpl;
    void loop();
    std::mutex m_;
    std::condition_variable cv_;
    std::vector<Job> jobs_;
    int nextId_ = 1;
    int running_ = 0;
    std::atomic<bool> stop_{false};
    std::atomic<bool> cancelFlag_{false};
    std::thread th_;
};
