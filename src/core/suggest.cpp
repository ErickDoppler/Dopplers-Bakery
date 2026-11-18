#include "core/suggest.h"
#include <cstring>

namespace {
struct Mood { const char* text; const char* instruct; };

const Mood kTts[] = {
    {"The rain had not stopped for three days, and the harbour lights swam in the puddles like small, patient fires. Nobody spoke of the ship that had not returned.",
     "A low, weary narrator, slow and quiet, with a hint of sadness and long pauses between the sentences."},
    {"And then, just when we had given up hope, the door swung open - and there she was, laughing, with snow in her hair and a cake in her hands!",
     "Bright, delighted and fast, a storyteller bursting with joy, rising pitch on the surprise."},
    {"Do not move. Do not breathe. Whatever is in the corridor has stopped outside this door, and it is listening for us.",
     "A tense whisper, close to the microphone, fearful and urgent, with sudden stillness before the last sentence."},
    {"Welcome back, traveller. The kettle is warm, the fire is lit, and the road will keep until the morning. Tell me everything.",
     "Warm, gentle and unhurried, like a kind old innkeeper smiling as they speak."},
    {"In the year the sky turned copper, the last of the engineers climbed the tower, set down her tools and, for the first time in her life, looked up.",
     "Calm, reverent and cinematic, a documentary voice with awe building slowly."},
    {"I told you a hundred times. A hundred! And still you went in there, and still you came out like this. What do you want from me?",
     "Angry and exasperated, then breaking into tired disbelief at the end; sharp consonants, rising volume."},
    {"Listen closely. This is the secret nobody told you: the river does not run to the sea - the sea runs back to the river, every night, in the dark.",
     "Mysterious and playful, a conspiratorial murmur with a smile in the voice."},
};

const char* kImage[] = {
    "a lone lighthouse on black basalt cliffs in a storm, enormous waves, a shaft of golden light cutting through the clouds, cinematic, highly detailed",
    "an old street market at dusk in a rain-washed neon city, steam from food stalls, reflections on wet asphalt, a crowd of silhouettes under umbrellas",
    "a portrait of a weathered desert nomad with a turquoise scarf, soft late-afternoon light, shallow depth of field, fine skin detail, 85mm photograph",
    "a cosy cabin in a snowy pine forest at night, lit windows, smoke from the chimney, aurora borealis in the sky, painterly, serene",
    "a giant clockwork whale drifting through clouds above a medieval town, brass plates, steam, warm sunrise, fantasy illustration",
};
const char* kPre[] = {
    "masterpiece, best quality, sharp focus, rich colour",
    "cinematic lighting, volumetric light, ultra detailed, 8k",
    "film photograph, natural light, fine grain, realistic textures",
};
const char* kNegative[] = {
    "blurry, low quality, deformed hands, extra fingers, watermark, text, oversaturated",
    "lowres, jpeg artifacts, bad anatomy, cropped, duplicate, signature",
};
const char* kAudio[] = {
    "slow ambient pads with soft piano, distant wind chimes, warm and spacious, 70 bpm, calm and dreamy",
    "tense cinematic percussion with low brass swells, building suspense, 110 bpm, dark and driving",
    "rain on a tin roof with distant thunder, soft and steady, no music",
    "a sci-fi door opening with a hydraulic hiss and a rising electronic chirp",
    "upbeat lo-fi hip hop beat, dusty drums, mellow electric piano, relaxed, 85 bpm",
};
const char* kVideo[] = {
    "A slow dolly shot along a rain-soaked neon street at night; a cyclist passes through puddles, reflections ripple, distant sirens and soft synth music.",
    "A drone shot rising over a misty mountain valley at sunrise, golden light spilling across the clouds, birds in the distance, calm orchestral swell.",
    "Close-up of a chef folding dumplings in a steamy kitchen, hands quick and precise, warm light, sizzling sounds and quiet chatter.",
};
const char* kLlm[] = {
    "Write a short, atmospheric opening paragraph for a noir story set in a flooded city.",
    "Explain in simple words how a repulsor craft could stay in the air, then list three problems an engineer would worry about.",
    "Give me five names for a rusty tavern on the edge of a radioactive desert, each with a one-line description.",
};
const char* kSystem[] = {
    "You are a concise assistant. Answer in plain language and keep it under 150 words.",
    "You are a witty game writer who answers in vivid, short sentences.",
};

template <size_t N> const char* pick(const char* (&a)[N], unsigned s) { return a[s % N]; }
}  // namespace

std::string suggestionFor(Kind k, const char* key, unsigned seed) {
    const size_t nTts = sizeof(kTts) / sizeof(kTts[0]);
    switch (k) {
    case Kind::TTS:
        if (!strcmp(key, "text")) return kTts[seed % nTts].text;
        if (!strcmp(key, "instruct")) return kTts[seed % nTts].instruct;
        break;
    case Kind::Image:
        if (!strcmp(key, "prompt")) return pick(kImage, seed);
        if (!strcmp(key, "pre_prompt")) return pick(kPre, seed / 3);
        if (!strcmp(key, "negative")) return pick(kNegative, seed / 5);
        break;
    case Kind::Audio:
        if (!strcmp(key, "prompt")) return pick(kAudio, seed);
        break;
    case Kind::Video:
        if (!strcmp(key, "prompt")) return pick(kVideo, seed);
        break;
    case Kind::LLM:
        if (!strcmp(key, "prompt")) return pick(kLlm, seed);
        if (!strcmp(key, "system")) return pick(kSystem, seed);
        break;
    }
    return "";
}
