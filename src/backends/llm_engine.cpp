// LLM text generation on the embedded llama.cpp/ggml runtime (CPU).
#include <algorithm>
#include <random>
#include <thread>
#include "backends/engines.h"
#include "llama.h"
#include "util/sys.h"

namespace {
struct ModelCache {
    std::string path;
    llama_model* model = nullptr;
    std::string loraPath;
    llama_adapter_lora* lora = nullptr;
} g_cache;

void quietLog(ggml_log_level, const char*, void*) {}

bool ensureModel(const std::string& path, const std::string& lora, JobCtl& ctl, std::string& err) {
    static bool inited = false;
    if (!inited) { llama_log_set(quietLog, nullptr); llama_backend_init(); inited = true; }
    if (g_cache.model && g_cache.path == path && g_cache.loraPath == lora) return true;
    if (g_cache.model) { llama_model_free(g_cache.model); g_cache = ModelCache(); }
    ctl.progress(0, "loading model");
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;      // GPU backend DLL if present, otherwise falls back to CPU
    g_cache.model = llama_model_load_from_file(path.c_str(), mp);
    if (!g_cache.model) { err = "cannot load model: " + path; return false; }
    g_cache.path = path;
    if (!lora.empty()) {
        g_cache.lora = llama_adapter_lora_init(g_cache.model, lora.c_str());
        if (!g_cache.lora) { err = "cannot load LoRA: " + lora; return false; }
        g_cache.loraPath = lora;
    }
    return true;
}

std::string stripThink(const std::string& s) {
    size_t a = s.find("<think>"), b = s.find("</think>");
    if (a == std::string::npos && b != std::string::npos) return s.substr(b + 8);
    if (a != std::string::npos && b != std::string::npos && b > a) return s.substr(0, a) + s.substr(b + 8);
    return s;
}
}  // namespace

bool runLLM(const Job& job, JobCtl& ctl, std::string& err) {
    const Json& p = job.params;
    std::string path = p.str("model");
    if (path.empty()) { err = "No model selected (open [Settings] on this tab)."; return false; }
    std::string prompt = p.str("prompt");
    const bool chat = p.get("messages").t == Json::Arr && !p.get("messages").a.empty();      // whole conversation instead of one prompt
    if (prompt.empty() && !chat) { err = "Prompt is empty."; return false; }
    if (!ensureModel(path, p.str("lora"), ctl, err)) return false;
    llama_model* model = g_cache.model;
    const llama_vocab* vocab = llama_model_get_vocab(model);

    int nCtx = std::max(256, p.integer("ctx", 4096));
    int maxNew = std::max(1, p.integer("max_tokens", 512));
    int threads = p.integer("threads", 0);
    if (threads <= 0) threads = std::max(1u, std::thread::hardware_concurrency() > 2 ? std::thread::hardware_concurrency() - 1 : 1u);

    // ---- prompt via the model's own chat template
    std::string sys = p.str("system");
    std::vector<llama_chat_message> msgs;
    std::vector<std::string> keep;                                 // owns the strings the message structs point at
    size_t totalChars = prompt.size() + sys.size();
    if (chat) {
        for (auto& m : p.get("messages").a) { keep.push_back(m.str("role")); keep.push_back(m.str("content")); totalChars += keep.back().size(); }
        for (size_t i = 0; i + 1 < keep.size(); i += 2) msgs.push_back({keep[i].c_str(), keep[i + 1].c_str()});
        if (prompt.empty()) prompt = keep.back();
    } else {
        if (!sys.empty()) msgs.push_back({"system", sys.c_str()});
        msgs.push_back({"user", prompt.c_str()});
    }
    const char* tmpl = llama_model_chat_template(model, nullptr);
    std::string text = prompt;
    bool templated = false;
    if (tmpl) {
        std::vector<char> buf(totalChars * 2 + 1024);
        int n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, buf.data(), (int32_t)buf.size());
        if (n > (int)buf.size()) { buf.resize(n); n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), true, buf.data(), (int32_t)buf.size()); }
        if (n > 0) { text.assign(buf.data(), n); templated = true; }
    }
    std::vector<llama_token> toks(text.size() + 16);
    int nt = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), toks.data(), (int32_t)toks.size(), !templated, true);
    if (nt < 0) { toks.resize(-nt); nt = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), toks.data(), (int32_t)toks.size(), !templated, true); }
    if (nt <= 0) { err = "tokenization failed"; return false; }
    toks.resize(nt);
    if (nt + 1 >= nCtx) { err = "Prompt is " + std::to_string(nt) + " tokens, larger than the context (" + std::to_string(nCtx) + "). Raise Context in settings."; return false; }
    maxNew = std::min(maxNew, nCtx - nt);

    // ---- context
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = nCtx; cp.n_batch = 512; cp.n_threads = threads; cp.n_threads_batch = threads;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { err = "cannot create context (out of memory? try a smaller Context)"; return false; }
    struct Free { llama_context* c; llama_sampler* s = nullptr; ~Free() { if (s) llama_sampler_free(s); llama_free(c); } } guard{ctx};
    if (g_cache.lora) { llama_adapter_lora* ad = g_cache.lora; float sc = (float)p.num("lora_scale", 1.0); llama_set_adapters_lora(ctx, &ad, 1, &sc); }

    // ---- sampler
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    guard.s = smpl;
    float temp = (float)p.num("temperature", 0.7);
    int seed = p.integer("seed", -1);
    if (seed < 0) seed = (int)(std::random_device{}() & 0x7fffffff);
    float rp = (float)p.num("repeat_penalty", 1.05);
    if (rp != 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), 64, rp, 0.0f, 0.0f));
    if (temp <= 0.0f) llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    else {
        int topk = p.integer("top_k", 40); float topp = (float)p.num("top_p", 0.95), minp = (float)p.num("min_p", 0.05);
        if (topk > 0) llama_sampler_chain_add(smpl, llama_sampler_init_top_k(topk));
        if (topp < 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_top_p(topp, 1));
        if (minp > 0.0f) llama_sampler_chain_add(smpl, llama_sampler_init_min_p(minp, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist((uint32_t)seed));
    }

    // ---- prompt ingestion
    ctl.progress(0, "reading prompt (" + std::to_string(nt) + " tokens)");
    for (int i = 0; i < nt; i += cp.n_batch) {
        if (!ctl.checkpoint()) { err = "cancelled"; return false; }
        int n = std::min<int>(cp.n_batch, nt - i);
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, n)) != 0) { err = "decode failed while reading the prompt"; return false; }
    }

    // ---- generation
    std::string out;
    int gen = 0;
    llama_token tok = 0;
    for (; gen < maxNew; gen++) {
        if (!ctl.checkpoint()) { err = "cancelled"; return false; }
        tok = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        char piece[256];
        int n = llama_token_to_piece(vocab, tok, piece, sizeof piece, 0, false);
        if (n > 0) out.append(piece, n);
        if (gen % 6 == 0) { ctl.setText(stripThink(out)); ctl.progress((float)gen / maxNew, std::to_string(gen) + " / " + std::to_string(maxNew) + " tokens"); }
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0) { err = "decode failed during generation"; return false; }
    }
    out = stripThink(out);
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
    size_t lead = out.find_first_not_of("\n ");
    if (lead != std::string::npos) out = out.substr(lead);
    ctl.setText(out);
    if (p.boolean("no_output", false)) return true;                 // used internally (audio prompt expansion): text only

    std::string file = outputPath("llm", job.id, ".txt");
    if (!sys::writeFile(file, out)) { err = "cannot write " + file; return false; }
    ctl.addOutput(file);
    return true;
}
