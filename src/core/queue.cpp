#include "core/queue.h"
#include <algorithm>
#include "util/sys.h"

const char* stateName(JobState s) {
    switch (s) {
    case JobState::Queued: return "Queued"; case JobState::Held: return "Held"; case JobState::Running: return "Running";
    case JobState::Paused: return "Paused"; case JobState::Done: return "Done"; case JobState::Failed: return "Failed";
    default: return "Canceled";
    }
}

static bool finished(JobState s) { return s == JobState::Done || s == JobState::Failed || s == JobState::Canceled; }

JobQueue& JobQueue::get() { static JobQueue q; return q; }

void JobQueue::start() { if (!th_.joinable()) th_ = std::thread([this] { loop(); }); }
void JobQueue::shutdown() {
    { std::lock_guard<std::mutex> g(m_); stop_ = true; cancelFlag_ = true; }
    cv_.notify_all();
    if (th_.joinable()) th_.join();
}

int JobQueue::add(Kind k, const Json& params, const std::string& title) {
    std::lock_guard<std::mutex> g(m_);
    Job j; j.id = nextId_++; j.kind = k; j.params = params; j.title = title;
    jobs_.push_back(j);
    cv_.notify_all();
    return j.id;
}
std::vector<Job> JobQueue::snapshot() { std::lock_guard<std::mutex> g(m_); return jobs_; }
bool JobQueue::find(int id, Job& out) {
    std::lock_guard<std::mutex> g(m_);
    for (auto& j : jobs_) if (j.id == id) { out = j; return true; }
    return false;
}
static int indexOf(std::vector<Job>& v, int id) { for (size_t i = 0; i < v.size(); i++) if (v[i].id == id) return (int)i; return -1; }

bool JobQueue::moveUp(int id) {
    std::lock_guard<std::mutex> g(m_);
    int i = indexOf(jobs_, id);
    if (i <= 0 || finished(jobs_[i].state) || jobs_[i].state == JobState::Running) return false;
    // do not jump over running/finished entries
    if (jobs_[i - 1].state == JobState::Running || finished(jobs_[i - 1].state)) return false;
    std::swap(jobs_[i], jobs_[i - 1]); return true;
}
bool JobQueue::moveDown(int id) {
    std::lock_guard<std::mutex> g(m_);
    int i = indexOf(jobs_, id);
    if (i < 0 || i + 1 >= (int)jobs_.size() || jobs_[i].state == JobState::Running || finished(jobs_[i].state)) return false;
    std::swap(jobs_[i], jobs_[i + 1]); return true;
}
bool JobQueue::remove(int id) {
    std::lock_guard<std::mutex> g(m_);
    int i = indexOf(jobs_, id);
    if (i < 0 || jobs_[i].state == JobState::Running || jobs_[i].state == JobState::Paused) return false;
    jobs_.erase(jobs_.begin() + i); return true;
}
void JobQueue::clearFinished() {
    std::lock_guard<std::mutex> g(m_);
    jobs_.erase(std::remove_if(jobs_.begin(), jobs_.end(), [](const Job& j) { return finished(j.state); }), jobs_.end());
}
void JobQueue::pause(int id) {
    std::lock_guard<std::mutex> g(m_);
    int i = indexOf(jobs_, id); if (i < 0) return;
    if (jobs_[i].state == JobState::Running) jobs_[i].state = JobState::Paused;
    else if (jobs_[i].state == JobState::Queued) jobs_[i].state = JobState::Held;
}
void JobQueue::resume(int id) {
    std::lock_guard<std::mutex> g(m_);
    int i = indexOf(jobs_, id); if (i < 0) return;
    if (jobs_[i].state == JobState::Paused) jobs_[i].state = JobState::Running;
    else if (jobs_[i].state == JobState::Held) jobs_[i].state = JobState::Queued;
    cv_.notify_all();
}
void JobQueue::cancel(int id) {
    std::lock_guard<std::mutex> g(m_);
    int i = indexOf(jobs_, id); if (i < 0) return;
    JobState s = jobs_[i].state;
    if (s == JobState::Running || s == JobState::Paused) { cancelFlag_ = true; if (s == JobState::Paused) jobs_[i].state = JobState::Running; }
    else if (s == JobState::Queued || s == JobState::Held) jobs_[i].state = JobState::Canceled;
    cv_.notify_all();
}
bool JobQueue::busy() {
    std::lock_guard<std::mutex> g(m_);
    for (auto& j : jobs_) if (j.state == JobState::Running || j.state == JobState::Paused) return true;
    return false;
}
void JobQueue::dropOutput(int id, const std::string& path) {
    std::lock_guard<std::mutex> g(m_);
    int i = indexOf(jobs_, id); if (i < 0) return;
    auto& o = jobs_[i].outputs;
    o.erase(std::remove(o.begin(), o.end(), path), o.end());
}
void JobQueue::dropJobText(int id) {
    std::lock_guard<std::mutex> g(m_);
    int i = indexOf(jobs_, id); if (i >= 0) jobs_[i].text.clear();
}

class CtlImpl : public JobCtl {
public:
    CtlImpl(JobQueue& q, int id) : q_(q), id_(id) {}
    void progress(float f, const std::string& s) override {
        std::lock_guard<std::mutex> g(q_.m_);
        int i = indexOf(q_.jobs_, id_); if (i < 0) return;
        q_.jobs_[i].progress = f; q_.jobs_[i].status = s;
    }
    void addOutput(const std::string& p) override {
        std::lock_guard<std::mutex> g(q_.m_);
        int i = indexOf(q_.jobs_, id_); if (i >= 0) q_.jobs_[i].outputs.push_back(p);
    }
    void setText(const std::string& t) override {
        std::lock_guard<std::mutex> g(q_.m_);
        int i = indexOf(q_.jobs_, id_); if (i >= 0) q_.jobs_[i].text = t;
    }
    bool checkpoint() override {
        std::unique_lock<std::mutex> lk(q_.m_);
        for (;;) {
            if (q_.cancelFlag_) return false;
            int i = indexOf(q_.jobs_, id_);
            if (i < 0 || q_.jobs_[i].state != JobState::Paused) return true;
            q_.cv_.wait_for(lk, std::chrono::milliseconds(200));
        }
    }
private:
    JobQueue& q_; int id_;
};

void JobQueue::loop() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(m_);
            int pick = -1;
            cv_.wait(lk, [&] {
                if (stop_) return true;
                for (size_t i = 0; i < jobs_.size(); i++) if (jobs_[i].state == JobState::Queued) { pick = (int)i; return true; }
                return false;
            });
            if (stop_) return;
            cancelFlag_ = false;
            jobs_[pick].state = JobState::Running;
            jobs_[pick].tStart = sys::now();
            jobs_[pick].status = "starting";
            job = jobs_[pick];
        }
        CtlImpl ctl(*this, job.id);
        std::string err;
        bool ok = false;
        try {
            BackendFn fn = backendFor(job.kind);
            ok = fn(job, ctl, err);
        } catch (const std::exception& e) { err = std::string("exception: ") + e.what(); }
        catch (...) { err = "unknown exception"; }
        {
            std::lock_guard<std::mutex> g(m_);
            int i = indexOf(jobs_, job.id);
            if (i >= 0) {
                Job& j = jobs_[i];
                j.tEnd = sys::now();
                if (cancelFlag_) { j.state = JobState::Canceled; j.status = "cancelled"; }
                else if (ok) { j.state = JobState::Done; j.progress = 1; j.status = "done"; }
                else { j.state = JobState::Failed; j.error = err; j.status = "failed"; }
            }
            cancelFlag_ = false;
        }
    }
}
