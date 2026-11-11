#include "tts/talker.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <numeric>
#include <thread>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gpu/compute.h"
#include "krea2/bpe.h"
#include "krea2/safetensors.h"
#include "krea2/sampling.h"
#include "util/json.h"
#include "util/sys.h"

using TT = ggml_tensor*;
namespace {
struct Job { TT t; std::string name; int mode; };      // 0: matrix -> F16, 1: float vector
std::string lowerS(std::string s) { for (auto& c : s) c = (char)tolower((unsigned char)c); return s; }
const int kHead = 128;
}  // namespace

QwenTalker::QwenTalker() {}
QwenTalker::~QwenTalker() { unload(); }

void QwenTalker::unload() {
    for (Stack* s : {&talker_, &pred_}) {
        if (s->cbuf) ggml_backend_buffer_free(s->cbuf);
        if (s->cctx) ggml_free(s->cctx);
        *s = Stack();
    }
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
    buf_ = nullptr; ctx_ = nullptr;
    delete st_; delete tok_; st_ = nullptr; tok_ = nullptr;
    lmHead_.clear();
}

std::vector<std::string> QwenTalker::speakers() const {
    std::vector<std::string> r; for (auto& kv : spk_) r.push_back(kv.first); return r;
}

bool QwenTalker::makeStack(Stack& s, const std::string& pre, int D, int layers, int maxT, std::string& err, void* jobsP) {
    auto& jobs = *(std::vector<Job>*)jobsP;
    s.D = D; s.maxT = maxT;
    bool ok = true;
    auto mat = [&](const std::string& n) -> TT {
        const StTensor* t = st_->find(n);
        if (!t || t->shape.size() != 2) { if (ok) err = "missing tensor " + n; ok = false; return nullptr; }
        TT g = ggml_new_tensor_2d(ctx_, GGML_TYPE_F16, t->shape[1], t->shape[0]);
        ggml_set_name(g, n.c_str()); jobs.push_back({g, n, 0}); return g;
    };
    auto vec = [&](const std::string& n) -> TT {
        const StTensor* t = st_->find(n);
        if (!t) { if (ok) err = "missing tensor " + n; ok = false; return nullptr; }
        TT g = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, t->shape[0]);
        ggml_set_name(g, n.c_str()); jobs.push_back({g, n, 1}); return g;
    };
    for (int i = 0; i < layers; i++) {
        std::string p = pre + "layers." + std::to_string(i) + ".";
        LW L;
        L.in = vec(p + "input_layernorm.weight"); L.post = vec(p + "post_attention_layernorm.weight");
        L.qn = vec(p + "self_attn.q_norm.weight"); L.kn = vec(p + "self_attn.k_norm.weight");
        L.q = mat(p + "self_attn.q_proj.weight"); L.k = mat(p + "self_attn.k_proj.weight");
        L.v = mat(p + "self_attn.v_proj.weight"); L.o = mat(p + "self_attn.o_proj.weight");
        L.gate = mat(p + "mlp.gate_proj.weight"); L.up = mat(p + "mlp.up_proj.weight"); L.down = mat(p + "mlp.down_proj.weight");
        s.w.push_back(L);
    }
    s.norm = vec(pre + "norm.weight");
    if (!ok) return false;
    s.heads = (int)(s.w[0].q->ne[1] / kHead);
    s.kv = (int)(s.w[0].k->ne[1] / kHead);
    // KV caches, F16, layout [128, maxT, kvHeads] per layer
    ggml_init_params ip = {ggml_tensor_overhead() * (size_t)(layers * 2 + 4), nullptr, true};
    s.cctx = ggml_init(ip);
    for (int i = 0; i < layers; i++) {
        s.kc.push_back(ggml_new_tensor_3d(s.cctx, GGML_TYPE_F16, kHead, maxT, s.kv));
        s.vc.push_back(ggml_new_tensor_3d(s.cctx, GGML_TYPE_F16, kHead, maxT, s.kv));
    }
    s.cbuf = Compute::get().allocDeviceWeights(s.cctx);
    if (!s.cbuf) { err = "cannot allocate KV cache"; return false; }
    return true;
}

bool QwenTalker::load(const std::string& dir, int threads, std::string& err, const std::function<void(float)>& progress) {
    unload();
    dir_ = dir;
    std::string cfgText;
    if (!sys::readFile(sys::join(dir, "config.json"), cfgText)) { err = "config.json not found in " + dir; return false; }
    bool ok; Json cfg = Json::parse(cfgText, &ok);
    if (!ok) { err = "bad config.json"; return false; }
    const Json& tc = cfg.get("talker_config");
    bosId_ = tc.integer("codec_bos_id", bosId_); eosId_ = tc.integer("codec_eos_token_id", eosId_); padId_ = tc.integer("codec_pad_id", padId_);
    thinkId_ = tc.integer("codec_think_id", thinkId_); nothinkId_ = tc.integer("codec_nothink_id", nothinkId_);
    thinkBos_ = tc.integer("codec_think_bos_id", thinkBos_); thinkEos_ = tc.integer("codec_think_eos_id", thinkEos_);
    ttsBos_ = cfg.integer("tts_bos_token_id", ttsBos_); ttsEos_ = cfg.integer("tts_eos_token_id", ttsEos_); ttsPad_ = cfg.integer("tts_pad_token_id", ttsPad_);
    spk_.clear(); lang_.clear(); spkDialect_.clear();
    for (auto& kv : tc.get("spk_id").o) spk_[lowerS(kv.first)] = kv.second.integer();
    for (auto& kv : tc.get("codec_language_id").o) lang_[lowerS(kv.first)] = kv.second.integer();
    for (auto& kv : tc.get("spk_is_dialect").o) spkDialect_[lowerS(kv.first)] = kv.second.t == Json::Str ? kv.second.s : "";

    st_ = new StFile();
    if (!st_->open(sys::join(dir, "model.safetensors"), err)) { unload(); return false; }
    tok_ = new QwenTokenizer();
    if (!tok_->load(dir, err)) { unload(); return false; }
    std::string e2;
    if (!Compute::get().init(threads, e2)) { err = e2; unload(); return false; }

    ggml_init_params ip = {ggml_tensor_overhead() * 600, nullptr, true};
    ctx_ = ggml_init(ip);
    std::vector<Job> jobs;
    const int kMaxT = 4096;
    if (!makeStack(talker_, "talker.model.", 2048, 28, kMaxT, err, &jobs)) { unload(); return false; }
    if (!makeStack(pred_, "talker.code_predictor.model.", 1024, 5, 24, err, &jobs)) { unload(); return false; }
    auto mat = [&](const std::string& n) -> TT {
        const StTensor* t = st_->find(n);
        if (!t) { err = "missing tensor " + n; return nullptr; }
        TT g = ggml_new_tensor_2d(ctx_, GGML_TYPE_F16, t->shape[1], t->shape[0]); ggml_set_name(g, n.c_str()); jobs.push_back({g, n, 0}); return g;
    };
    auto vec = [&](const std::string& n) -> TT {
        const StTensor* t = st_->find(n);
        if (!t) { err = "missing tensor " + n; return nullptr; }
        TT g = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, t->shape[0]); ggml_set_name(g, n.c_str()); jobs.push_back({g, n, 1}); return g;
    };
    codecHead_ = mat("talker.codec_head.weight");
    tp1w_ = mat("talker.text_projection.linear_fc1.weight"); tp1b_ = vec("talker.text_projection.linear_fc1.bias");
    tp2w_ = mat("talker.text_projection.linear_fc2.weight"); tp2b_ = vec("talker.text_projection.linear_fc2.bias");
    smallW_ = mat("talker.code_predictor.small_to_mtp_projection.weight"); smallB_ = vec("talker.code_predictor.small_to_mtp_projection.bias");
    for (int i = 0; i < 15; i++) lmHead_.push_back(mat("talker.code_predictor.lm_head." + std::to_string(i) + ".weight"));
    if (!codecHead_ || !tp1w_ || !tp1b_ || !tp2w_ || !tp2b_ || !smallW_ || !smallB_) { unload(); return false; }
    for (auto* h : lmHead_) if (!h) { unload(); return false; }

    buf_ = Compute::get().allocDeviceWeights(ctx_);
    if (!buf_) { err = "cannot allocate talker weights"; unload(); return false; }

    std::atomic<size_t> next{0}, done{0};
    auto worker = [&]() {
        std::vector<float> f; std::vector<ggml_fp16_t> h;
        for (;;) {
            size_t i = next++;
            if (i >= jobs.size()) return;
            Job& j = jobs[i];
            const StTensor* t = st_->find(j.name);
            st_->toFloat(*t, f);
            if (j.mode == 0) { h.resize(f.size()); ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t)f.size()); ggml_backend_tensor_set(j.t, h.data(), 0, h.size() * 2); }
            else ggml_backend_tensor_set(j.t, f.data(), 0, f.size() * 4);
            size_t d = ++done;
            if (progress && d % 24 == 0) progress((float)d / jobs.size());
        }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < std::max(2, std::min(threads, 8)); i++) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    if (progress) progress(1.0f);
    return true;
}

void QwenTalker::embedRow(const char* table, int row, float* out) {
    const StTensor* t = st_->find(table);
    const int D = (int)t->shape[1];
    const uint16_t* r = (const uint16_t*)t->data + (size_t)row * D;
    for (int i = 0; i < D; i++) out[i] = bf16ToF32(r[i]);
}

// ---------------------------------------------------------------- graphs
TT QwenTalker::runStack(ggml_context* c, ggml_cgraph* g, Stack& s, TT x, TT pos, TT mask, int past) {
    const int n = (int)x->ne[1];
    for (size_t l = 0; l < s.w.size(); l++) {
        LW& L = s.w[l];
        TT h = ggml_mul(c, ggml_rms_norm(c, x, 1e-6f), L.in);
        TT q = ggml_reshape_3d(c, ggml_mul_mat(c, L.q, h), kHead, s.heads, n);
        TT k = ggml_reshape_3d(c, ggml_mul_mat(c, L.k, h), kHead, s.kv, n);
        TT v = ggml_reshape_3d(c, ggml_mul_mat(c, L.v, h), kHead, s.kv, n);
        q = ggml_mul(c, ggml_rms_norm(c, q, 1e-6f), L.qn);
        k = ggml_mul(c, ggml_rms_norm(c, k, 1e-6f), L.kn);
        q = ggml_rope_ext(c, q, pos, nullptr, kHead, GGML_ROPE_TYPE_NEOX, 0, 1000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(c, k, pos, nullptr, kHead, GGML_ROPE_TYPE_NEOX, 0, 1000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        TT kc = s.kc[l], vc = s.vc[l];
        TT kdst = ggml_view_3d(c, kc, kHead, n, s.kv, kc->nb[1], kc->nb[2], (size_t)past * kc->nb[1]);
        TT vdst = ggml_view_3d(c, vc, kHead, n, s.kv, vc->nb[1], vc->nb[2], (size_t)past * vc->nb[1]);
        ggml_build_forward_expand(g, ggml_cpy(c, ggml_permute(c, k, 0, 2, 1, 3), kdst));
        ggml_build_forward_expand(g, ggml_cpy(c, ggml_permute(c, v, 0, 2, 1, 3), vdst));
        TT kall = ggml_view_3d(c, kc, kHead, past + n, s.kv, kc->nb[1], kc->nb[2], 0);
        TT vall = ggml_view_3d(c, vc, kHead, past + n, s.kv, vc->nb[1], vc->nb[2], 0);
        TT a = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), kall, vall, mask, 1.0f / std::sqrt((float)kHead), 0.0f, 0.0f);
        a = ggml_reshape_2d(c, a, kHead * s.heads, n);
        x = ggml_add(c, x, ggml_mul_mat(c, L.o, a));
        TT h2 = ggml_mul(c, ggml_rms_norm(c, x, 1e-6f), L.post);
        TT m = ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, L.gate, h2)), ggml_mul_mat(c, L.up, h2));
        x = ggml_add(c, x, ggml_mul_mat(c, L.down, m));
    }
    return ggml_mul(c, ggml_rms_norm(c, x, 1e-6f), s.norm);
}

namespace {
struct GraphCtx {
    std::vector<uint8_t> meta; ggml_context* c; ggml_cgraph* g;
    explicit GraphCtx(size_t nodes = 4096) {
        meta.resize(ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false) + (1 << 20));
        ggml_init_params ip = {meta.size(), meta.data(), true};
        c = ggml_init(ip);
        g = ggml_new_graph_custom(c, nodes, false);
    }
    ~GraphCtx() { ggml_free(c); }
};
void causalMask(std::vector<ggml_fp16_t>& mk, int n, int past) {
    const int nkv = past + n, pad = (n + 63) / 64 * 64;
    mk.assign((size_t)nkv * pad, ggml_fp32_to_fp16(-INFINITY));
    ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    for (int i = 0; i < n; i++) for (int j = 0; j <= past + i; j++) mk[(size_t)i * nkv + j] = zero;
}
}  // namespace

bool QwenTalker::runTalker(const float* embeds, int n, int past, std::vector<float>& hidden, std::vector<float>& logits, std::string& err) {
    if (past + n > talker_.maxT) { err = "audio too long for the talker context"; return false; }
    Compute& cm = Compute::get();
    GraphCtx gc; ggml_context* c = gc.c;
    TT emb = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2048, n); ggml_set_input(emb);
    TT pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n); ggml_set_input(pos);
    const int nkv = past + n, padN = (n + 63) / 64 * 64;
    TT mask = n > 1 ? ggml_new_tensor_2d(c, GGML_TYPE_F16, nkv, padN) : nullptr;
    if (mask) ggml_set_input(mask);
    TT x = runStack(c, gc.g, talker_, emb, pos, mask, past);
    TT last = ggml_cont(c, ggml_view_2d(c, x, 2048, 1, x->nb[1], (size_t)(n - 1) * x->nb[1]));
    TT lg = ggml_mul_mat(c, codecHead_, last);
    ggml_set_output(last); ggml_set_output(lg);
    ggml_build_forward_expand(gc.g, last); ggml_build_forward_expand(gc.g, lg);
    ggml_backend_sched_reset(cm.sched);
    if (!ggml_backend_sched_alloc_graph(cm.sched, gc.g)) { err = "cannot allocate talker graph"; return false; }
    ggml_backend_tensor_set(emb, embeds, 0, (size_t)2048 * n * 4);
    std::vector<int32_t> p(n); std::iota(p.begin(), p.end(), past);
    ggml_backend_tensor_set(pos, p.data(), 0, n * 4);
    if (mask) { std::vector<ggml_fp16_t> mk; causalMask(mk, n, past); ggml_backend_tensor_set(mask, mk.data(), 0, mk.size() * 2); }
    if (ggml_backend_sched_graph_compute(cm.sched, gc.g) != GGML_STATUS_SUCCESS) { err = "talker compute failed"; return false; }
    hidden.resize(2048); logits.resize(lg->ne[0]);
    ggml_backend_tensor_get(last, hidden.data(), 0, 2048 * 4);
    ggml_backend_tensor_get(lg, logits.data(), 0, logits.size() * 4);
    ggml_backend_sched_reset(cm.sched);
    return true;
}

bool QwenTalker::runPredictor(const float* embeds, int n, int past, int step, std::vector<float>& logits, std::string& err) {
    Compute& cm = Compute::get();
    GraphCtx gc(2048); ggml_context* c = gc.c;
    TT emb = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2048, n); ggml_set_input(emb);
    TT pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n); ggml_set_input(pos);
    const int nkv = past + n, padN = (n + 63) / 64 * 64;
    TT mask = n > 1 ? ggml_new_tensor_2d(c, GGML_TYPE_F16, nkv, padN) : nullptr;
    if (mask) ggml_set_input(mask);
    TT h = ggml_add(c, ggml_mul_mat(c, smallW_, emb), smallB_);                // [1024, n]
    TT x = runStack(c, gc.g, pred_, h, pos, mask, past);
    TT last = ggml_cont(c, ggml_view_2d(c, x, 1024, 1, x->nb[1], (size_t)(n - 1) * x->nb[1]));
    TT lg = ggml_mul_mat(c, lmHead_[step], last);
    ggml_set_output(lg); ggml_build_forward_expand(gc.g, lg);
    ggml_backend_sched_reset(cm.sched);
    if (!ggml_backend_sched_alloc_graph(cm.sched, gc.g)) { err = "cannot allocate predictor graph"; return false; }
    ggml_backend_tensor_set(emb, embeds, 0, (size_t)2048 * n * 4);
    std::vector<int32_t> p(n); std::iota(p.begin(), p.end(), past);
    ggml_backend_tensor_set(pos, p.data(), 0, n * 4);
    if (mask) { std::vector<ggml_fp16_t> mk; causalMask(mk, n, past); ggml_backend_tensor_set(mask, mk.data(), 0, mk.size() * 2); }
    if (ggml_backend_sched_graph_compute(cm.sched, gc.g) != GGML_STATUS_SUCCESS) { err = "predictor compute failed"; return false; }
    logits.resize(lg->ne[0]);
    ggml_backend_tensor_get(lg, logits.data(), 0, logits.size() * 4);
    ggml_backend_sched_reset(cm.sched);
    return true;
}

bool QwenTalker::textProject(const std::vector<int>& ids, std::vector<float>& out, std::string& err) {
    Compute& cm = Compute::get();
    const int n = (int)ids.size();
    std::vector<float> e((size_t)n * 2048);
    for (int i = 0; i < n; i++) embedRow("talker.model.text_embedding.weight", ids[i], &e[(size_t)i * 2048]);
    GraphCtx gc(256); ggml_context* c = gc.c;
    TT in = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2048, n); ggml_set_input(in);
    TT h = ggml_add(c, ggml_mul_mat(c, tp1w_, in), tp1b_);
    h = ggml_add(c, ggml_mul_mat(c, tp2w_, ggml_silu(c, h)), tp2b_);
    ggml_set_output(h); ggml_build_forward_expand(gc.g, h);
    ggml_backend_sched_reset(cm.sched);
    if (!ggml_backend_sched_alloc_graph(cm.sched, gc.g)) { err = "cannot allocate text projection graph"; return false; }
    ggml_backend_tensor_set(in, e.data(), 0, e.size() * 4);
    if (ggml_backend_sched_graph_compute(cm.sched, gc.g) != GGML_STATUS_SUCCESS) { err = "text projection failed"; return false; }
    out.resize((size_t)n * 2048);
    ggml_backend_tensor_get(h, out.data(), 0, out.size() * 4);
    ggml_backend_sched_reset(cm.sched);
    return true;
}

// ---------------------------------------------------------------- sampling
namespace {
int pickToken(std::vector<float>& lg, bool sample, float temp, int topK, k2::Rng& rng) {
    const int n = (int)lg.size();
    if (!sample || temp <= 0.0f) return (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    int k = std::min(std::max(topK, 1), n);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    float mx = lg[idx[0]];
    std::vector<double> pr(k); double sum = 0;
    for (int i = 0; i < k; i++) { pr[i] = std::exp((double)(lg[idx[i]] - mx) / temp); sum += pr[i]; }
    std::vector<float> u(1); rng.normal(u);                          // reuse the normal generator: map to uniform via erf
    double r = 0.5 * (1.0 + std::erf(u[0] / std::sqrt(2.0))) * sum;
    for (int i = 0; i < k; i++) { r -= pr[i]; if (r <= 0) return idx[i]; }
    return idx[k - 1];
}
}  // namespace

// ---------------------------------------------------------------- generation
bool QwenTalker::generate(const Params& p, std::vector<int>& codes, int& frames, std::string& err, const std::function<bool(int)>& onFrame) {
    if (!loaded()) { err = "talker not loaded"; return false; }
    codes.clear(); frames = 0;
    const int D = 2048;

    // ---- tokens
    std::vector<int> ids = tok_->encode("<|im_start|>assistant\n" + p.text + "<|im_end|>\n<|im_start|>assistant\n");
    const int n = (int)ids.size();
    if (n < 10) { err = "text is too short"; return false; }
    std::vector<int> ins;
    if (!p.instruct.empty()) ins = tok_->encode("<|im_start|>user\n" + p.instruct + "<|im_end|>\n");

    // ---- speaker, language
    std::string spk = lowerS(p.speaker), lang = lowerS(p.language);
    int spkId = -1;
    if (p.speakerEmbed) {
        if ((int)p.speakerEmbed->size() != D) { err = "speaker embedding has the wrong size"; return false; }
        spk.clear();                                           // cloned voice: no preset speaker
    } else if (!spk.empty()) {
        auto it = spk_.find(spk);
        if (it == spk_.end()) { err = "unknown speaker '" + p.speaker + "'"; return false; }
        spkId = it->second;
    }
    int langId = -1;
    if (lang != "auto" && !lang.empty()) {
        auto it = lang_.find(lang);
        if (it == lang_.end()) { err = "unknown language '" + p.language + "'"; return false; }
        langId = it->second;
    }
    if ((lang == "chinese" || lang == "auto" || lang.empty()) && !spk.empty()) {
        auto d = spkDialect_.find(spk);
        if (d != spkDialect_.end() && !d->second.empty() && lang_.count(d->second)) langId = lang_[d->second];
    }

    // ---- text side: everything that goes through text_projection in one batch
    // layout: [bos, eos, pad, role x3, first text token, trailing text tokens..., instruct tokens...]
    std::vector<int> pj = {ttsBos_, ttsEos_, ttsPad_, ids[0], ids[1], ids[2], ids[3]};
    const int trailN = n - 9;                                  // ids[4 .. n-5)
    for (int i = 4; i < n - 5; i++) pj.push_back(ids[i]);
    const int insOff = (int)pj.size();
    for (int t : ins) pj.push_back(t);
    std::vector<float> P;
    if (!textProject(pj, P, err)) return false;
    auto prow = [&](int i) { return &P[(size_t)i * D]; };
    const float *bosE = prow(0), *eosE = prow(1), *padE = prow(2);

    // ---- codec side
    std::vector<int> cprefill;
    if (langId < 0) cprefill = {nothinkId_, thinkBos_, thinkEos_};
    else cprefill = {thinkId_, thinkBos_, langId, thinkEos_};
    std::vector<std::vector<float>> cs;
    auto crow = [&](int id) { std::vector<float> r(D); embedRow("talker.model.codec_embedding.weight", id, r.data()); return r; };
    for (int id : cprefill) cs.push_back(crow(id));
    if (p.speakerEmbed) cs.push_back(*p.speakerEmbed);
    else if (spkId >= 0) cs.push_back(crow(spkId));
    cs.push_back(crow(padId_)); cs.push_back(crow(bosId_));
    const int L = (int)cs.size();

    // ---- prefill sequence
    std::vector<float> seq;
    auto push = [&](const float* a, const float* b = nullptr) { for (int i = 0; i < D; i++) seq.push_back(a[i] + (b ? b[i] : 0.0f)); };
    for (size_t i = 0; i < ins.size(); i++) push(prow(insOff + (int)i));
    for (int i = 3; i < 6; i++) push(prow(i));                                  // role
    for (int i = 0; i < L - 1; i++) push(i < L - 2 ? padE : bosE, cs[i].data());
    std::vector<float> trailing;
    int trailRows = 0;
    const bool icl = p.refCodes && !p.refCodes->empty() && !p.refText.empty();
    if (icl) {
        // in-context prompt: reference text + target text against the reference codec frames
        const int refFrames = (int)(p.refCodes->size() / 16);
        std::vector<int> rid = tok_->encode("<|im_start|>assistant\n" + p.refText + "<|im_end|>\n");
        std::vector<int> tids;
        if (rid.size() > 5) tids.assign(rid.begin() + 3, rid.end() - 2);        // ref_id[:, 3:-2]
        for (int i = 3; i < n - 5; i++) tids.push_back(ids[i]);                 // + target text input_id[:, 3:-5]
        std::vector<float> T;
        if (!textProject(tids, T, err)) return false;
        const int textLens = (int)tids.size() + 1;                              // + tts_eos
        const int codecLens = refFrames + 1;                                    // + codec bos row first
        std::vector<float> textRows(T);
        textRows.insert(textRows.end(), eosE, eosE + D);
        std::vector<float> codecRows((size_t)codecLens * D, 0.0f);
        { std::vector<float> e = crow(bosId_); memcpy(codecRows.data(), e.data(), D * 4); }
        std::vector<float> tmp(D);
        for (int f = 0; f < refFrames; f++) {
            float* dst = &codecRows[(size_t)(f + 1) * D];
            embedRow("talker.model.codec_embedding.weight", (*p.refCodes)[(size_t)f * 16], dst);
            for (int g = 1; g < 16; g++) {
                embedRow(("talker.code_predictor.model.codec_embedding." + std::to_string(g - 1) + ".weight").c_str(), (*p.refCodes)[(size_t)f * 16 + g], tmp.data());
                for (int i = 0; i < D; i++) dst[i] += tmp[i];
            }
        }
        if (textLens > codecLens) {
            for (int r = 0; r < codecLens; r++) push(&textRows[(size_t)r * D], &codecRows[(size_t)r * D]);
            trailing.assign(textRows.begin() + (size_t)codecLens * D, textRows.end());
            trailRows = textLens - codecLens;
        } else {
            for (int r = 0; r < codecLens; r++) push(r < textLens ? &textRows[(size_t)r * D] : padE, &codecRows[(size_t)r * D]);
            trailing.assign(padE, padE + D);
            trailRows = 1;
        }
    } else {
        push(prow(6), cs[L - 1].data());                                        // first text token + codec bos
        for (int i = 0; i < trailN; i++) trailing.insert(trailing.end(), prow(7 + i), prow(7 + i) + D);
        trailing.insert(trailing.end(), eosE, eosE + D);
        trailRows = trailN + 1;
    }
    const int prefillN = (int)(seq.size() / D);
    dbgPrefill = seq;
    dbgTrailing = trailing;
    std::vector<float> padRow(padE, padE + D);

    const int maxFrames = std::min(p.maxFrames, talker_.maxT - prefillN - 4);
    k2::Rng rng(p.seed);

    // ---- prefill
    std::vector<float> hidden, logits, plog;
    if (!runTalker(seq.data(), prefillN, 0, hidden, logits, err)) return false;
    int past = prefillN;
    std::vector<int> history;

    for (int f = 0; f < maxFrames; f++) {
        if (onFrame && !onFrame(f)) { err = "cancelled"; return false; }
        // code 0
        for (int id : history) logits[id] = logits[id] < 0 ? logits[id] * p.repetition : logits[id] / p.repetition;
        for (int i = 2048; i < (int)logits.size(); i++) if (i != eosId_) logits[i] = -INFINITY;
        if (f < 2) logits[eosId_] = -INFINITY;
        int c0 = pickToken(logits, p.sample, p.temperature, p.topK, rng);
        const int nForced = p.forced ? (int)(p.forced->size() / 16) : 0;
        if (p.forced) { if (f >= nForced) break; forcedTotal0++; if (c0 == (*p.forced)[f * 16]) forcedAgree0++; c0 = (*p.forced)[f * 16]; }
        if (c0 == eosId_) break;
        history.push_back(c0);
        int fr[16]; fr[0] = c0;
        // codes 1..15 from the code predictor
        std::vector<float> e0 = crow(c0), pin(2 * (size_t)D);
        memcpy(pin.data(), hidden.data(), D * 4); memcpy(pin.data() + D, e0.data(), D * 4);
        if (!runPredictor(pin.data(), 2, 0, 0, plog, err)) return false;
        fr[1] = pickToken(plog, p.sample, p.subTemperature, p.subTopK, rng);
        if (p.forced) { forcedTotalSub++; if (fr[1] == (*p.forced)[f * 16 + 1]) forcedAgreeSub++; fr[1] = (*p.forced)[f * 16 + 1]; }
        std::vector<float> sum = e0, e(D);
        for (int k = 1; k <= 14; k++) {
            embedRow(("talker.code_predictor.model.codec_embedding." + std::to_string(k - 1) + ".weight").c_str(), fr[k], e.data());
            for (int i = 0; i < D; i++) sum[i] += e[i];
            if (!runPredictor(e.data(), 1, k + 1, k, plog, err)) return false;
            fr[k + 1] = pickToken(plog, p.sample, p.subTemperature, p.subTopK, rng);
            if (p.forced) { forcedTotalSub++; if (fr[k + 1] == (*p.forced)[f * 16 + k + 1]) forcedAgreeSub++; fr[k + 1] = (*p.forced)[f * 16 + k + 1]; }
        }
        embedRow("talker.code_predictor.model.codec_embedding.14.weight", fr[15], e.data());
        for (int i = 0; i < D; i++) sum[i] += e[i];
        for (int i = 0; i < 16; i++) codes.push_back(fr[i]);
        frames++;
        // next talker input: summed codec embeddings + (trailing text | pad)
        const float* add = f < trailRows ? &trailing[(size_t)f * D] : padRow.data();
        for (int i = 0; i < D; i++) sum[i] += add[i];
        if (!runTalker(sum.data(), 1, past, hidden, logits, err)) return false;
        past++;
    }
    return true;
}
