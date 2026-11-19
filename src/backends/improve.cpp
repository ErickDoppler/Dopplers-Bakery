// The "Improve" button of the Video, Audio, Text to image and TTS tabs: the chat model rewrites what is in the prompt box (the result is put
// back into the box, nothing is generated). Only the request is built here; the UI runs it as an internal LLM job.
#include <cmath>
#include "backends/engines.h"
#include "core/settings.h"

namespace {
const char* kImageSystemPrompt =
    "You write prompts for a text-to-image model. Rewrite the user idea as ONE detailed paragraph of 3 to 6 sentences, plain text. Describe the subject "
    "and its pose or action, the setting, the composition and camera (shot size, angle, lens, depth of field), the lighting and colors, materials and "
    "textures, the mood and the visual style. Keep everything the user asked for and do not contradict it. No titles, lists, comments or quotation "
    "marks around the answer: output only the prompt.";
const char* kSpeechSystemPrompt =
    "You prepare text for a speech synthesizer. Rewrite the text so it sounds natural when read aloud: keep the meaning, the language and the order, "
    "fix punctuation, write numbers, dates, symbols and abbreviations as spoken words, split overlong sentences, and use commas, dashes and ellipses "
    "for natural pauses. Do not add new facts, comments or stage directions. Output only the rewritten text.";
}

bool buildImproveJob(Kind k, const Json& form, Json& q, std::string& err) {
    const std::string model = Settings::get().resolvedSelection(Kind::LLM, "model");
    if (model.empty()) { err = "No chat model is selected: pick a GGUF model on the LLM tab or in Model settings."; return false; }
    std::string idea = k == Kind::TTS ? form.str("text") : form.str("prompt");
    const size_t a = idea.find_first_not_of(" \r\n\t");
    if (a == std::string::npos) { err = "The prompt is empty: write an idea first."; return false; }
    idea = idea.substr(a, idea.find_last_not_of(" \r\n\t") - a + 1);
    std::string request; int maxTokens = 400;
    switch (k) {
    case Kind::Video:
        request = std::string(videoImproveSystemPrompt()) + "\n\nIdea: " + idea + "\nVideo length: " + std::to_string((int)std::lround(form.num("seconds", 4))) + " seconds.\nPrompt:"; break;
    case Kind::Audio:
        request = audioImproveSystemPrompt(form.str("mode", "Music")) + "\n\nInput: " + idea + "\nTarget audio length: " + std::to_string((int)std::lround(form.num("length", 60))) + " seconds.\nOutput:"; maxTokens = 256; break;
    case Kind::Image:
        request = std::string(kImageSystemPrompt) + "\n\nIdea: " + idea + "\nPrompt:"; break;
    case Kind::TTS:
        request = std::string(kSpeechSystemPrompt) + "\n\nText: " + idea + "\nRewritten text:"; maxTokens = std::min(2048, std::max(256, (int)idea.size())); break;
    default: err = "This tab has no Improve button."; return false;
    }
    q = Json::object();
    q["model"] = model; q["prompt"] = request;
    q["max_tokens"] = maxTokens; q["ctx"] = 6144; q["temperature"] = k == Kind::TTS ? 0.3 : 0.7; q["top_k"] = 64; q["top_p"] = 0.95; q["min_p"] = 0.05; q["repeat_penalty"] = 1.05;
    q["seed"] = -1; q["no_output"] = true;
    return true;
}
