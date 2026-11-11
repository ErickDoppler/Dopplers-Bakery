// Qwen3-TTS talker (28-layer LM) + code predictor (5 layers) on ggml with KV caches.
// Produces the 16 codec codes per 80 ms frame that the speech decoder turns into audio.
#pragma once
#include <functional>
#include <map>
#include <string>
#include <vector>

struct ggml_context; struct ggml_backend_buffer; struct ggml_tensor; struct ggml_cgraph;
class StFile;
class QwenTokenizer;

class QwenTalker {
public:
    struct Params {
        std::string text, speaker = "ryan", language = "auto", instruct;
        bool sample = true;                    // false = greedy (used for validation)
        float temperature = 0.9f, subTemperature = 0.9f, repetition = 1.05f;
        int topK = 50, subTopK = 50;
        int maxFrames = 1500;
        // voice cloning (Base model): speaker embedding from the speaker encoder; with refCodes + refText the
        // in-context mode is used (reference audio codes continue into the new text)
        const std::vector<float>* speakerEmbed = nullptr;
        const std::vector<int>* refCodes = nullptr;      // frames x 16
        std::string refText;
        const std::vector<int>* forced = nullptr;     // teacher forcing for validation: frame-major codes to follow
        unsigned long long seed = 1;
    };
    QwenTalker();
    ~QwenTalker();
    // modelDir contains config.json, model.safetensors, vocab.json, merges.txt, tokenizer_config.json
    bool load(const std::string& modelDir, int threads, std::string& err, const std::function<void(float)>& progress = nullptr);
    void unload();
    bool loaded() const { return buf_ != nullptr; }
    // codes: frames x 16, frame-major. onFrame(frame) returns false to cancel.
    bool generate(const Params& p, std::vector<int>& codes, int& frames, std::string& err, const std::function<bool(int)>& onFrame = nullptr);
    std::vector<std::string> speakers() const;

    // debugging aids for the self test
    std::vector<float> dbgPrefill, dbgTrailing;
    int forcedAgree0 = 0, forcedAgreeSub = 0, forcedTotal0 = 0, forcedTotalSub = 0;

private:
    struct LW { ggml_tensor *in, *qn, *kn, *q, *k, *v, *o, *post, *gate, *up, *down; };
    struct Stack {
        int D = 0, heads = 0, kv = 0, maxT = 0;
        std::vector<LW> w;
        ggml_tensor* norm = nullptr;
        std::vector<ggml_tensor*> kc, vc;
        ggml_context* cctx = nullptr;
        ggml_backend_buffer* cbuf = nullptr;
    };
    ggml_tensor* runStack(ggml_context* c, ggml_cgraph* g, Stack& s, ggml_tensor* x, ggml_tensor* pos, ggml_tensor* mask, int past);
    bool runTalker(const float* embeds, int n, int past, std::vector<float>& hidden, std::vector<float>& logits, std::string& err);
    bool runPredictor(const float* embeds, int n, int past, int step, std::vector<float>& logits, std::string& err);
    bool textProject(const std::vector<int>& ids, std::vector<float>& out, std::string& err);   // ids -> n x 2048
    void embedRow(const char* table, int row, float* out);                                      // host lookups
    bool makeStack(Stack& s, const std::string& prefix, int D, int layers, int maxT, std::string& err, void* jobs);

    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer* buf_ = nullptr;
    StFile* st_ = nullptr;
    QwenTokenizer* tok_ = nullptr;
    Stack talker_, pred_;
    ggml_tensor *codecHead_ = nullptr, *tp1w_ = nullptr, *tp1b_ = nullptr, *tp2w_ = nullptr, *tp2b_ = nullptr, *smallW_ = nullptr, *smallB_ = nullptr;
    std::vector<ggml_tensor*> lmHead_;
    std::string dir_;
    // config
    int bosId_ = 2149, eosId_ = 2150, padId_ = 2148, thinkId_ = 2154, nothinkId_ = 2155, thinkBos_ = 2156, thinkEos_ = 2157;
    int ttsBos_ = 151672, ttsEos_ = 151673, ttsPad_ = 151671;
    std::map<std::string, int> spk_, lang_;
    std::map<std::string, std::string> spkDialect_;
};
