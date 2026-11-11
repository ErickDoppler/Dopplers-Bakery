#include "chat/agent.h"
#include <algorithm>
#include <cmath>
#include "core/forms.h"
#include "core/queue.h"
#include "core/settings.h"
#include "gpu/compute.h"

namespace {
const int kMaxToolsPerTurn = 3, kMaxRounds = 2, kKeepHistory = 24;

std::string trim(const std::string& s) { size_t a = s.find_first_not_of(" \r\n\t"), b = s.find_last_not_of(" \r\n\t"); return a == std::string::npos ? "" : s.substr(a, b - a + 1); }

// text with every <tool_call>...</tool_call> block (and a half-written one at the end) removed
std::string hideToolCalls(const std::string& s) {
    std::string out; size_t pos = 0;
    for (;;) {
        size_t a = s.find("<tool_call>", pos);
        if (a == std::string::npos) { out += s.substr(pos); break; }
        out += s.substr(pos, a - pos);
        size_t b = s.find("</tool_call>", a);
        if (b == std::string::npos) break;
        pos = b + 12;
    }
    size_t lt = out.rfind('<');                                   // a started tag like "<tool_c" at the very end
    if (lt != std::string::npos && out.size() - lt < 12 && std::string("<tool_call>").compare(0, out.size() - lt, out, lt, std::string::npos) == 0) out.resize(lt);
    return trim(out);
}

int clampInt(double v, int lo, int hi) { return (int)std::min<double>(hi, std::max<double>(lo, std::llround(v))); }
// the chat always uses CUDA when its libraries are installed (any size), otherwise Vulkan (capped)
std::string chatBackend() { return Compute::hasCuda() ? "CUDA" : (Compute::hasVulkan() ? "Vulkan" : ""); }
int chatImageLimit() { return std::min(2048, Compute::imageSideLimit(chatBackend())); }

std::string speakable(const std::string& s) {                     // plain sentences only, short enough to read aloud
    std::string t;
    for (char c : s) { if (c == '*' || c == '#' || c == '`' || c == '_' || c == '~' || c == '[' || c == ']' || c == '\n' || c == '\r') t += c == '\n' ? ' ' : (c == '\r' ? ' ' : '\0'); else t += c; }
    std::string clean; for (char c : t) if (c) clean += c;
    clean = trim(clean);
    if (clean.size() > 420) {
        size_t cut = clean.find_last_of(".!?", 420);
        clean = cut != std::string::npos && cut > 80 ? clean.substr(0, cut + 1) : clean.substr(0, 420);
    }
    return clean;
}
}  // namespace

ChatAgent& ChatAgent::get() { static ChatAgent a; return a; }

std::string ChatAgent::systemPrompt() {
    std::string s =
        "You are Baker, the assistant inside Dopplers Bakery, an offline studio that makes images, music, sound effects and speech on this computer. "
        "Chat naturally and keep replies short (1-3 sentences). Reply in the language the user writes in, but always write tool prompts in English.\n"
        "You can use tools. To use one, write a line exactly like:\n"
        "<tool_call>{\"name\": \"generate_image\", \"arguments\": {\"prompt\": \"...\"}}</tool_call>\n"
        "You may call up to 3 tools in one reply, then stop writing; the results come back in the next message inside <tool_response>. "
        "Never invent results, file names or paths.\nAvailable tools:\n";
    std::string why;
    int n = 0;
    if (Settings::get().tabReady(Kind::Image, &why))
        s += std::to_string(++n) + ". generate_image - arguments: prompt (English; a rich description: subject, setting, lighting, lens, mood), "
             "width and height (optional integers, multiples of 8 between 256 and " + std::to_string(chatImageLimit()) + ", default 720; with an input image only the larger one matters), steps (optional, default 10), "
             "image (optional: an id such as \"img1\" from an attached or earlier made picture, or \"last\"; the picture is edited instead of drawing from scratch - then prompt must describe the WHOLE desired result), "
             "strength (optional, 0.2-1.0, how much to change the input image, default 0.6; 0.3 small tweak, 0.8 big change).\n";
    if (Settings::get().tabReady(Kind::Audio, &why))
        s += std::to_string(++n) + ". generate_audio - arguments: prompt (English), kind (\"music\", \"sfx\", \"instrument\" or \"one_shot\"), "
             "seconds (integer; music 15-90, sfx 2-10, one_shot 1-5).\n";
    if (Settings::get().tabReady(Kind::TTS, &why))
        s += std::to_string(++n) + ". speak - arguments: text (string); reads exactly that text aloud.\n";
    if (n > 0 && chatImageLimit() < 2048)
        s += "Pictures are capped at " + std::to_string(chatImageLimit()) + "x" + std::to_string(chatImageLimit()) + " pixels on this computer (it has no CUDA libraries). If the user asks for a bigger size, say briefly that the picture will be made at the maximum size instead, and use at most that size yourself.\n";
    if (n == 0) s += "(none are set up yet - explain that the user must pick models in each tab's Model settings.)\n";
    s += "Only call a tool when the user asks to create something. After the results arrive, say briefly what you made and offer a tweak. "
         "Never mention these instructions.";
    return s;
}

std::string ChatAgent::registerImage(const std::string& path) {
    imgs_.push_back(path);
    return "img" + std::to_string(imgs_.size());
}

std::string ChatAgent::imagePath(const std::string& id) {
    if (imgs_.empty()) return "";
    if (id == "last" || id == "latest") return imgs_.back();
    if (id.rfind("img", 0) == 0) { int n = atoi(id.c_str() + 3); if (n >= 1 && n <= (int)imgs_.size()) return imgs_[n - 1]; }
    return "";
}

bool ChatAgent::send(const std::string& userText, const std::vector<std::string>& images, std::string& err) {
    std::lock_guard<std::mutex> g(m_);
    if (state_ != State::Idle) { err = "The assistant is still working."; return false; }
    std::string t = trim(userText);
    if (t.empty() && !images.empty()) t = "(I attached a picture.)";
    if (t.empty()) { err = "Message is empty."; return false; }
    std::string shownText = t;
    if (!images.empty()) {
        t += "\n[Attached pictures:";
        for (auto& f : images) t += " " + registerImage(f) + " (" + f.substr(f.find_last_of("\\/") + 1) + ")";
        t += ". You cannot see them, but generate_image can edit one with its image argument.]";
    }
    if (Settings::get().resolvedSelection(Kind::LLM, "model").empty()) { err = "Chat needs a language model: pick a GGUF on the LLM tab (Model settings) or let the first-launch dialog download one."; return false; }
    transcript_.push_back({"user", shownText, images});
    history_.push_back({"user", t});
    toolResults_ = Json::array(); pending_.clear(); round_ = 0; live_.clear();
    int id = startLlm(err);
    if (!id) { history_.pop_back(); transcript_.pop_back(); return false; }
    llmJob_ = id; state_ = State::Thinking; status_ = "thinking...";
    return true;
}

int ChatAgent::startLlm(std::string& err) {
    (void)err;
    Json msgs = Json::array();
    { Json m = Json::object(); m["role"] = "system"; m["content"] = systemPrompt(); msgs.push(m); }
    size_t from = history_.size() > (size_t)kKeepHistory ? history_.size() - kKeepHistory : 0;
    for (size_t i = from; i < history_.size(); i++) { Json m = Json::object(); m["role"] = history_[i].first; m["content"] = history_[i].second; msgs.push(m); }
    Json p = defaultsOf(Kind::LLM);
    p["model"] = Settings::get().resolvedSelection(Kind::LLM, "model");
    p["lora"] = Settings::get().resolvedSelection(Kind::LLM, "lora");
    p["messages"] = msgs; p["prompt"] = ""; p["system"] = "";
    p["max_tokens"] = 600; p["ctx"] = 8192; p["temperature"] = 0.6; p["top_k"] = 40; p["top_p"] = 0.9; p["min_p"] = 0.05; p["repeat_penalty"] = 1.05;
    p["seed"] = -1; p["no_output"] = true;
    return JobQueue::get().add(Kind::LLM, p, "chat");
}

bool ChatAgent::launchCall(const Call& c, std::vector<Pending>& pending, Json& results) {
    auto fail = [&](const std::string& why) {
        Json r = Json::object(); r["tool"] = c.name; r["status"] = "error"; r["message"] = why; results.push(r); return false;
    };
    std::string why;
    if (c.name == "generate_image") {
        if (!Settings::get().tabReady(Kind::Image, &why)) return fail("image generation is not set up: " + why);
        std::string prompt = trim(c.args.str("prompt"));
        if (prompt.empty()) return fail("missing prompt");
        Json form = Json::object();
        form["prompt"] = prompt; form["negative"] = c.args.str("negative");
        form["backend"] = chatBackend();
        const int lim = chatImageLimit();
        double rw = std::max(64.0, c.args.num("width", 720)), rh = std::max(64.0, c.args.num("height", 720));
        std::string sizeNote;
        if (std::max(rw, rh) > lim) {                                   // keep the aspect ratio, fit the longest side to the limit
            double sc = lim / std::max(rw, rh);
            int nw = std::max(256, (int)(rw * sc) / 8 * 8), nh = std::max(256, (int)(rh * sc) / 8 * 8);
            sizeNote = "The picture was made at " + std::to_string(nw) + "x" + std::to_string(nh) + " instead of the requested " + std::to_string((int)rw) + "x" + std::to_string((int)rh) + ": " +
                       (Compute::hasCuda() ? "the chat limit is " + std::to_string(lim) + " px." : "pictures are capped at " + std::to_string(lim) + "x" + std::to_string(lim) + " here (install the CUDA libraries in Model settings for larger ones).");
            rw = nw; rh = nh;
        }
        form["width"] = clampInt((int)rw / 8 * 8, 256, lim); form["height"] = clampInt((int)rh / 8 * 8, 256, lim);
        form["steps"] = clampInt(c.args.num("steps", 10), 1, 40); form["seed"] = -1;
        std::string imgId = trim(c.args.str("image"));
        if (!imgId.empty()) {
            std::string src = imagePath(imgId);
            if (src.empty()) return fail("unknown image '" + imgId + "'");
            form["init_image"] = src;
            form["denoise"] = std::min(1.0, std::max(0.15, c.args.num("strength", 0.6)));
        }
        int id = JobQueue::get().add(Kind::Image, makeJobParams(Kind::Image, form), "chat: " + jobTitle(Kind::Image, form));
        pending.push_back({id, c.name, sizeNote});
        if (!sizeNote.empty()) transcript_.push_back({"info", sizeNote, {}});
    } else if (c.name == "generate_audio") {
        if (!Settings::get().tabReady(Kind::Audio, &why)) return fail("audio generation is not set up: " + why);
        std::string prompt = trim(c.args.str("prompt"));
        if (prompt.empty()) return fail("missing prompt");
        std::string kind = c.args.str("kind", "music"), mode = "Music"; int def = 30, lo = 15, hi = 90;
        if (kind == "sfx") { mode = "SFX"; def = 5; lo = 1; hi = 15; }
        else if (kind == "instrument") { mode = "Instrument"; def = 12; lo = 4; hi = 30; }
        else if (kind == "one_shot" || kind == "one-shot") { mode = "One-shot"; def = 2; lo = 1; hi = 6; }
        Json form = Json::object();
        form["prompt"] = prompt; form["mode"] = mode; form["length"] = clampInt(c.args.num("seconds", def), lo, hi);
        form["enhance"] = true; form["format"] = "mp3"; form["seed"] = -1; form["backend"] = chatBackend();
        int id = JobQueue::get().add(Kind::Audio, makeJobParams(Kind::Audio, form), "chat: " + jobTitle(Kind::Audio, form));
        pending.push_back({id, c.name});
    } else if (c.name == "speak") {
        if (!Settings::get().tabReady(Kind::TTS, &why)) return fail("speech is not set up: " + why);
        std::string text = trim(c.args.str("text"));
        if (text.empty()) return fail("missing text");
        Json form = Json::object();
        form["text"] = text; form["speaker"] = settings.speaker; form["language"] = settings.language; form["format"] = "wav"; form["seed"] = -1; form["backend"] = chatBackend();
        int id = JobQueue::get().add(Kind::TTS, makeJobParams(Kind::TTS, form), "chat: " + jobTitle(Kind::TTS, form));
        pending.push_back({id, c.name});
    } else return fail("unknown tool '" + c.name + "'");
    return true;
}

void ChatAgent::finishLlmTurn(const std::string& raw, bool allowTools) {
    // parse tool calls
    std::vector<Call> calls;
    size_t pos = 0;
    while (true) {
        size_t a = raw.find("<tool_call>", pos);
        if (a == std::string::npos) break;
        size_t b = raw.find("</tool_call>", a);
        std::string body = raw.substr(a + 11, (b == std::string::npos ? raw.size() : b) - (a + 11));
        bool ok; Json j = Json::parse(trim(body), &ok);
        if (ok && j.t == Json::Obj) {
            Call c; c.name = j.str("name"); c.args = j.get("arguments").t == Json::Obj ? j.get("arguments") : Json::object();
            if (!c.name.empty()) calls.push_back(c);
        }
        if (b == std::string::npos) break;
        pos = b + 12;
    }
    std::string visible = hideToolCalls(raw);
    history_.push_back({"assistant", trim(raw)});
    if (!visible.empty()) transcript_.push_back({"assistant", visible, {}});

    if (!calls.empty() && allowTools) {
        pending_.clear(); toolResults_ = Json::array();
        int n = 0;
        for (auto& c : calls) { if (n++ >= kMaxToolsPerTurn) break; launchCall(c, pending_, toolResults_); }
        state_ = State::Working; status_ = "working...";
        return;
    }
    if (visible.empty()) { transcript_.push_back({"note", "(the assistant had nothing to say)", {}}); state_ = State::Idle; status_.clear(); return; }
    if (settings.voice) startSpeech(visible); else { state_ = State::Idle; status_.clear(); }
}

void ChatAgent::startSpeech(const std::string& reply) {
    std::string say = speakable(reply), why;
    if (say.empty()) { state_ = State::Idle; status_.clear(); return; }
    if (!Settings::get().tabReady(Kind::TTS, &why)) { transcript_.push_back({"note", "Voice reply skipped: " + why, {}}); state_ = State::Idle; status_.clear(); return; }
    Json form = Json::object();
    form["text"] = say; form["speaker"] = settings.speaker; form["language"] = settings.language; form["format"] = "wav"; form["seed"] = -1; form["backend"] = chatBackend();
    speechJob_ = JobQueue::get().add(Kind::TTS, makeJobParams(Kind::TTS, form), "chat voice");
    state_ = State::Speaking; status_ = "speaking...";
}

bool ChatAgent::poll() {
    std::lock_guard<std::mutex> g(m_);
    auto& q = JobQueue::get();
    auto active = [](JobState s) { return s == JobState::Queued || s == JobState::Held || s == JobState::Running || s == JobState::Paused; };
    switch (state_) {
    case State::Idle: return false;
    case State::Thinking: case State::Replying: {
        Job j;
        if (!q.find(llmJob_, j)) { state_ = State::Idle; live_.clear(); return false; }
        if (active(j.state)) { live_ = hideToolCalls(j.text); status_ = j.state == JobState::Running ? "thinking..." : "waiting in the queue..."; return true; }
        live_.clear();
        if (j.state == JobState::Done) { finishLlmTurn(j.text, state_ == State::Thinking || round_ < kMaxRounds); return state_ != State::Idle; }
        transcript_.push_back({"note", j.state == JobState::Failed ? "Language model error: " + j.error : "Stopped.", {}});
        state_ = State::Idle; status_.clear();
        return false;
    }
    case State::Working: {
        bool all = true;
        for (auto& pnd : pending_) {
            Job j; if (!q.find(pnd.jobId, j)) continue;
            if (active(j.state)) { all = false; status_ = std::string(pnd.tool == "generate_image" ? "making an image" : pnd.tool == "generate_audio" ? "making audio" : "speaking") + " (job " + std::to_string(pnd.jobId) + "): " + j.status; }
        }
        if (!all) return true;
        std::vector<std::string> files;
        for (auto& pnd : pending_) {
            Job j; Json r = Json::object(); r["tool"] = pnd.tool;
            if (!q.find(pnd.jobId, j) || j.state != JobState::Done) { r["status"] = "error"; r["message"] = (j.state == JobState::Failed ? j.error : std::string("cancelled")); }
            else {
                r["status"] = "ok";
                Json fl = Json::array();
                for (auto& f : j.outputs) {
                    files.push_back(f);
                    std::string id = pnd.tool == "generate_image" ? registerImage(f) : "";
                    Json e = Json::object(); e["file"] = f.substr(f.find_last_of("\\/") + 1); if (!id.empty()) e["image_id"] = id;
                    fl.push(e);
                }
                r["files"] = fl;
                if (pnd.tool == "generate_audio" && !j.text.empty()) r["expanded_prompt"] = j.text;
                if (pnd.tool == "generate_image") { if (!j.info.empty()) r["size"] = j.info; if (!pnd.note.empty()) r["note"] = pnd.note; }
            }
            toolResults_.push(r);
        }
        if (!files.empty()) transcript_.push_back({"result", "", files});
        for (size_t i = 0; i < toolResults_.a.size(); i++)
            if (toolResults_.a[i].str("status") == "error") transcript_.push_back({"note", toolResults_.a[i].str("tool") + " failed: " + toolResults_.a[i].str("message"), {}});
        history_.push_back({"user", "<tool_response>\n" + toolResults_.dump() + "\n</tool_response>"});
        pending_.clear(); toolResults_ = Json::array();
        round_++;
        std::string err;
        int id = startLlm(err);
        if (!id) { state_ = State::Idle; return false; }
        llmJob_ = id; state_ = State::Replying; status_ = "thinking...";
        return true;
    }
    case State::Speaking: {
        Job j;
        if (!q.find(speechJob_, j)) { state_ = State::Idle; return false; }
        if (active(j.state)) { status_ = "speaking..."; return true; }
        if (j.state == JobState::Done && !j.outputs.empty()) {
            play_ = j.outputs[0];
            for (size_t i = transcript_.size(); i-- > 0;) if (transcript_[i].role == "assistant") { transcript_[i].files.push_back(j.outputs[0]); break; }
        } else if (j.state == JobState::Failed) transcript_.push_back({"note", "Voice reply failed: " + j.error, {}});
        state_ = State::Idle; status_.clear();
        return false;
    }
    }
    return false;
}

bool ChatAgent::busy() { std::lock_guard<std::mutex> g(m_); return state_ != State::Idle; }

void ChatAgent::cancel() {
    std::lock_guard<std::mutex> g(m_);
    auto& q = JobQueue::get();
    if (llmJob_) q.cancel(llmJob_);
    for (auto& p : pending_) q.cancel(p.jobId);
    if (speechJob_) q.cancel(speechJob_);
    pending_.clear(); live_.clear(); status_.clear(); state_ = State::Idle;
    transcript_.push_back({"note", "Stopped.", {}});
}

void ChatAgent::reset() {
    cancel();
    std::lock_guard<std::mutex> g(m_);
    transcript_.clear(); history_.clear(); toolResults_ = Json::array(); play_.clear(); imgs_.clear();
}

std::vector<ChatMsg> ChatAgent::transcript() { std::lock_guard<std::mutex> g(m_); return transcript_; }
std::string ChatAgent::live() { std::lock_guard<std::mutex> g(m_); return live_; }
std::string ChatAgent::status() { std::lock_guard<std::mutex> g(m_); return status_; }
std::string ChatAgent::takeSpeechToPlay() { std::lock_guard<std::mutex> g(m_); std::string r; r.swap(play_); return r; }
