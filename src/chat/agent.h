// Chat assistant: talks to the LLM, turns requests into image / audio / speech jobs on the normal queue,
// reads the results back to the model and (optionally) speaks its reply with the TTS engine.
// Driven by poll() from a timer (window) or a loop (command line); every heavy step is an ordinary queue job.
#pragma once
#include <mutex>
#include <string>
#include <vector>
#include "util/json.h"

struct ChatMsg {
    std::string role;                    // "user", "assistant" or "note" (status lines such as errors)
    std::string text;
    std::vector<std::string> files;      // images / audio produced for this message
};

struct ChatSettings {
    bool voice = false;                  // speak the assistant's replies
    std::string speaker = "Ryan", language = "English";
};

class ChatAgent {
public:
    static ChatAgent& get();
    ChatSettings settings;

    // images: picture files the user attached; the model cannot see them but can edit them (generate_image with image = "imgN")
    bool send(const std::string& userText, const std::vector<std::string>& images, std::string& err);
    bool send(const std::string& userText, std::string& err) { return send(userText, {}, err); }
    bool poll();                         // advance the state machine; true while the agent is busy
    bool busy();
    void cancel();                       // cancel the running steps and go idle
    void reset();                        // forget the conversation

    std::vector<ChatMsg> transcript();
    std::string live();                  // assistant text streamed so far (tool calls hidden)
    std::string status();                // "thinking...", "making an image (job 7)", ...
    // file produced by the speech job that should be played now ("" if none); cleared when read
    std::string takeSpeechToPlay();

private:
    enum class State { Idle, Thinking, Working, Replying, Speaking };
    struct Call { std::string name; Json args; };
    struct Pending { int jobId; std::string tool; std::string note; };

    std::string registerImage(const std::string& path);           // -> "img3"
    std::string imagePath(const std::string& id);                  // "" if unknown; "last" = newest picture
    std::vector<std::string> imgs_;                                // attached and generated pictures, id = "img" + (index+1)
    int startLlm(std::string& err);
    std::string systemPrompt();
    bool launchCall(const Call& c, std::vector<Pending>& pending, Json& results);
    void finishLlmTurn(const std::string& raw, bool allowTools);
    void startSpeech(const std::string& reply);

    std::mutex m_;
    State state_ = State::Idle;
    std::vector<ChatMsg> transcript_;
    std::vector<std::pair<std::string, std::string>> history_;    // role, content as the model sees it
    int llmJob_ = 0, round_ = 0, speechJob_ = 0;
    std::vector<Pending> pending_;
    Json toolResults_ = Json::array();
    std::string live_, status_, play_;
};
