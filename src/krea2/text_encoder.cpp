#include "krea2/text_encoder.h"
#include <atomic>
#include <cmath>
#include <cstring>
#include <thread>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gpu/compute.h"
#include "krea2/safetensors.h"

static const int kTapList[12] = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35};
const int* Krea2TextEncoder::tapLayers() { return kTapList; }

static const char* kPre = "model.layers.";

void Krea2TextEncoder::unload() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
    delete (StFile*)st_;
    buf_ = nullptr; ctx_ = nullptr; backend_ = nullptr; st_ = nullptr; layers_.clear();
}

bool Krea2TextEncoder::load(const std::string& path, int threads, std::string& err, const std::function<void(float)>& progress) {
    unload();
    threads_ = threads > 0 ? threads : 4;
    StFile* st = new StFile();
    st_ = st;
    if (!st->open(path, err)) { unload(); return false; }

    if (!Compute::get().init(threads_, err)) { unload(); return false; }
    backend_ = Compute::get().cpu;
    ggml_init_params ip = {ggml_tensor_overhead() * (kLayers * 11 + 16), nullptr, true};
    ctx_ = ggml_init(ip);

    struct Job { ggml_tensor* t; std::string name; bool fp8; };
    std::vector<Job> jobs;
    auto shapeOf = [&](const std::string& n, int64_t& ne0, int64_t& ne1) -> bool {
        const StTensor* t = st->find(n);
        if (!t) { err = "missing tensor " + n; return false; }
        ne0 = t->shape.back(); ne1 = t->shape.size() > 1 ? t->shape[0] : 1; return true;
    };
    auto lin = [&](const std::string& n) -> ggml_tensor* {
        int64_t a, b; if (!shapeOf(n, a, b)) return nullptr;
        ggml_tensor* t = ggml_new_tensor_2d(ctx_, GGML_TYPE_Q8_0, a, b);
        ggml_set_name(t, n.c_str()); jobs.push_back({t, n, true}); return t;
    };
    auto vec = [&](const std::string& n) -> ggml_tensor* {
        int64_t a, b; if (!shapeOf(n, a, b)) return nullptr;
        ggml_tensor* t = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, a);
        ggml_set_name(t, n.c_str()); jobs.push_back({t, n, false}); return t;
    };
    layers_.resize(kLayers);
    for (int i = 0; i < kLayers; i++) {
        std::string p = kPre + std::to_string(i) + ".";
        Layer& L = layers_[i];
        L.qn = vec(p + "self_attn.q_norm.weight"); L.kn = vec(p + "self_attn.k_norm.weight");
        L.q = lin(p + "self_attn.q_proj.weight"); L.k = lin(p + "self_attn.k_proj.weight");
        L.v = lin(p + "self_attn.v_proj.weight"); L.o = lin(p + "self_attn.o_proj.weight");
        L.inLn = vec(p + "input_layernorm.weight"); L.postLn = vec(p + "post_attention_layernorm.weight");
        L.gate = lin(p + "mlp.gate_proj.weight"); L.up = lin(p + "mlp.up_proj.weight"); L.down = lin(p + "mlp.down_proj.weight");
        if (!L.qn || !L.kn || !L.q || !L.k || !L.v || !L.o || !L.inLn || !L.postLn || !L.gate || !L.up || !L.down) { unload(); return false; }
    }
    if (!st->find("model.embed_tokens.weight")) { err = "missing embed_tokens"; unload(); return false; }
    buf_ = Compute::get().allocWeights(ctx_);
    if (!buf_) { err = "out of memory allocating text encoder weights"; unload(); return false; }

    // convert in parallel
    std::atomic<size_t> next{0}, done{0};
    std::atomic<bool> bad{false};
    auto worker = [&]() {
        std::vector<float> f; std::vector<uint8_t> q;
        for (;;) {
            size_t i = next++;
            if (i >= jobs.size() || bad) return;
            Job& j = jobs[i];
            const StTensor* t = st->find(j.name);
            float scale = 1.0f;
            if (j.fp8) st->scalar(j.name.substr(0, j.name.size() - 6) + "weight_scale", scale);
            if (!st->toFloat(*t, f, scale)) { bad = true; return; }
            if (j.fp8) {
                int64_t cols = j.t->ne[0], rows = j.t->ne[1];
                q.resize(ggml_nbytes(j.t));
                ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, rows, cols, nullptr);
                ggml_backend_tensor_set(j.t, q.data(), 0, q.size());
            } else ggml_backend_tensor_set(j.t, f.data(), 0, f.size() * sizeof(float));
            size_t d = ++done;
            if (progress && d % 16 == 0) progress((float)d / jobs.size());
        }
    };
    ggml_quantize_init(GGML_TYPE_Q8_0);
    std::vector<std::thread> pool;
    for (int i = 0; i < threads_; i++) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    if (bad) { err = "unsupported dtype in text encoder file"; unload(); return false; }
    if (progress) progress(1.0f);
    return true;
}

bool Krea2TextEncoder::encode(const std::vector<int>& ids, std::vector<float>& out, int& seqOut, std::string& err) {
    if (!loaded()) { err = "text encoder not loaded"; return false; }
    StFile* st = (StFile*)st_;
    const int n = (int)ids.size();
    if (n < 4) { err = "prompt too short"; return false; }
    const float eps = 1e-6f;

    const size_t nodes = 8192;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false) + (1 << 20));
    ggml_init_params ip = {meta.size(), meta.data(), true};
    ggml_context* c = ggml_init(ip);

    ggml_tensor* emb = ggml_new_tensor_2d(c, GGML_TYPE_F32, kHidden, n); ggml_set_input(emb);
    ggml_tensor* pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n); ggml_set_input(pos);
    const int padN = (n + 63) / 64 * 64;
    ggml_tensor* mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n, padN); ggml_set_input(mask);

    std::vector<ggml_tensor*> taps(kTaps, nullptr);
    int tapIdx = 0;
    ggml_tensor* x = emb;
    for (int i = 0; i < kLayers; i++) {
        if (tapIdx < kTaps && i == kTapList[tapIdx]) { ggml_set_output(x); taps[tapIdx++] = x; }
        Layer& L = layers_[i];
        ggml_tensor* h = ggml_mul(c, ggml_rms_norm(c, x, eps), L.inLn);
        ggml_tensor* q = ggml_reshape_3d(c, ggml_mul_mat(c, L.q, h), kHeadDim, kHeads, n);
        ggml_tensor* k = ggml_reshape_3d(c, ggml_mul_mat(c, L.k, h), kHeadDim, kKvHeads, n);
        ggml_tensor* v = ggml_reshape_3d(c, ggml_mul_mat(c, L.v, h), kHeadDim, kKvHeads, n);
        q = ggml_mul(c, ggml_rms_norm(c, q, eps), L.qn);
        k = ggml_mul(c, ggml_rms_norm(c, k, eps), L.kn);
        q = ggml_rope_ext(c, q, pos, nullptr, kHeadDim, GGML_ROPE_TYPE_NEOX, 0, 5000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(c, k, pos, nullptr, kHeadDim, GGML_ROPE_TYPE_NEOX, 0, 5000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        ggml_tensor* qp = ggml_permute(c, q, 0, 2, 1, 3);
        ggml_tensor* kp = ggml_cast(c, ggml_permute(c, k, 0, 2, 1, 3), GGML_TYPE_F16);
        ggml_tensor* vp = ggml_cast(c, ggml_permute(c, v, 0, 2, 1, 3), GGML_TYPE_F16);
        ggml_tensor* a = ggml_flash_attn_ext(c, qp, kp, vp, mask, 1.0f / std::sqrt((float)kHeadDim), 0.0f, 0.0f);
        a = ggml_reshape_2d(c, a, kHeads * kHeadDim, n);
        x = ggml_add(c, x, ggml_mul_mat(c, L.o, a));
        ggml_tensor* h2 = ggml_mul(c, ggml_rms_norm(c, x, eps), L.postLn);
        ggml_tensor* m = ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, L.gate, h2)), ggml_mul_mat(c, L.up, h2));
        x = ggml_add(c, x, ggml_mul_mat(c, L.down, m));
    }
    ggml_set_output(x);       // keep the final state alive so the graph has a defined end
    ggml_cgraph* g = ggml_new_graph_custom(c, nodes, false);
    ggml_build_forward_expand(g, x);
    for (auto* t : taps) ggml_build_forward_expand(g, t);

    Compute& cm = Compute::get();
    ggml_backend_sched_reset(cm.sched);
    if (!ggml_backend_sched_alloc_graph(cm.sched, g)) { err = "cannot allocate text encoder graph"; ggml_free(c); return false; }

    // inputs: embedding rows (bf16 -> f32), positions, causal mask
    const StTensor* et = st->find("model.embed_tokens.weight");
    std::vector<float> e((size_t)n * kHidden);
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= et->shape[0]) { err = "token id out of range"; ggml_free(c); return false; }
        const uint16_t* row = (const uint16_t*)et->data + (size_t)ids[i] * kHidden;
        for (int j = 0; j < kHidden; j++) e[(size_t)i * kHidden + j] = bf16ToF32(row[j]);
    }
    ggml_backend_tensor_set(emb, e.data(), 0, e.size() * sizeof(float));
    std::vector<int32_t> p(n); for (int i = 0; i < n; i++) p[i] = i;
    ggml_backend_tensor_set(pos, p.data(), 0, n * sizeof(int32_t));
    std::vector<ggml_fp16_t> mk((size_t)n * padN, ggml_fp32_to_fp16(0.0f));
    ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int r = 0; r < padN; r++) for (int cc = 0; cc < n; cc++) if (r >= n || cc > r) mk[(size_t)r * n + cc] = ninf;
    ggml_backend_tensor_set(mask, mk.data(), 0, mk.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(cm.sched, g) != GGML_STATUS_SUCCESS) { err = "text encoder compute failed"; ggml_free(c); return false; }

    // strip the system + "user\n" prefix like Krea2TEModel.encode_token_weights
    int templateEnd = -1, count = 0;
    for (int i = 0; i < n; i++) if (ids[i] == 151644 && count < 2) { templateEnd = i; count++; }
    if (templateEnd < 0) templateEnd = 0;
    if (n > templateEnd + 3 && ids[templateEnd + 1] == 872 && ids[templateEnd + 2] == 198) templateEnd += 3;
    seqOut = n - templateEnd;
    out.assign((size_t)seqOut * kTaps * kHidden, 0.0f);
    std::vector<float> tmp((size_t)n * kHidden);
    for (int t = 0; t < kTaps; t++) {
        ggml_backend_tensor_get(taps[t], tmp.data(), 0, tmp.size() * sizeof(float));
        for (int i = 0; i < seqOut; i++)
            memcpy(&out[((size_t)i * kTaps + t) * kHidden], &tmp[(size_t)(templateEnd + i) * kHidden], kHidden * sizeof(float));
    }
        ggml_free(c);
    return true;
}
