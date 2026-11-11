#include "audio/t5gemma.h"
#include <cmath>
#include <cstring>
#include <numeric>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gpu/compute.h"
#include "krea2/safetensors.h"

using TT = ggml_tensor*;

void T5GemmaEncoder::unload() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
    delete st_;
    buf_ = nullptr; ctx_ = nullptr; st_ = nullptr; layers_.clear();
}

bool T5GemmaEncoder::load(const std::string& path, std::string& err) {
    unload();
    st_ = new StFile();
    if (!st_->open(path, err)) { unload(); return false; }
    const StTensor* sp = st_->find("spiece_model");
    if (!sp) { err = "spiece_model not found in " + path; unload(); return false; }
    if (!sp_.load(std::string((const char*)sp->data, (size_t)sp->bytes), err)) { unload(); return false; }
    std::string e2;
    if (!Compute::get().init(0, e2)) { err = e2; unload(); return false; }

    ggml_init_params ip = {ggml_tensor_overhead() * 200, nullptr, true};
    ctx_ = ggml_init(ip);
    struct Job { TT t; std::string name; int mode; };      // 0 matrix F16, 1 vector F32, 2 vector + 1
    std::vector<Job> jobs;
    bool ok = true;
    auto mat = [&](const std::string& n) -> TT {
        const StTensor* t = st_->find(n); if (!t) { if (ok) err = "missing tensor " + n; ok = false; return nullptr; }
        TT g = ggml_new_tensor_2d(ctx_, GGML_TYPE_F16, t->shape[1], t->shape[0]); jobs.push_back({g, n, 0}); return g;
    };
    auto vec1 = [&](const std::string& n) -> TT {
        const StTensor* t = st_->find(n); if (!t) { if (ok) err = "missing tensor " + n; ok = false; return nullptr; }
        TT g = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, t->shape[0]); jobs.push_back({g, n, 2}); return g;
    };
    for (int i = 0; i < 12; i++) {
        std::string p = "model.encoder.layers." + std::to_string(i) + ".";
        Layer L;
        L.n1 = vec1(p + "pre_self_attn_layernorm.weight"); L.n2 = vec1(p + "post_self_attn_layernorm.weight");
        L.n3 = vec1(p + "pre_feedforward_layernorm.weight"); L.n4 = vec1(p + "post_feedforward_layernorm.weight");
        L.q = mat(p + "self_attn.q_proj.weight"); L.k = mat(p + "self_attn.k_proj.weight"); L.v = mat(p + "self_attn.v_proj.weight"); L.o = mat(p + "self_attn.o_proj.weight");
        L.gate = mat(p + "mlp.gate_proj.weight"); L.up = mat(p + "mlp.up_proj.weight"); L.down = mat(p + "mlp.down_proj.weight");
        layers_.push_back(L);
    }
    norm_ = vec1("model.encoder.norm.weight");
    if (!st_->find("model.encoder.embed_tokens.weight")) { err = "missing embed_tokens"; ok = false; }
    if (!ok) { unload(); return false; }
    buf_ = Compute::get().allocDeviceWeights(ctx_);
    if (!buf_) { err = "cannot allocate T5Gemma weights"; unload(); return false; }
    std::vector<float> f; std::vector<ggml_fp16_t> h;
    for (auto& j : jobs) {
        st_->toFloat(*st_->find(j.name), f);
        if (j.mode == 0) { h.resize(f.size()); ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t)f.size()); ggml_backend_tensor_set(j.t, h.data(), 0, h.size() * 2); }
        else { if (j.mode == 2) for (auto& v : f) v += 1.0f; ggml_backend_tensor_set(j.t, f.data(), 0, f.size() * 4); }
    }
    return true;
}

bool T5GemmaEncoder::encode(const std::vector<int>& ids, std::vector<float>& out, std::string& err) {
    if (!loaded()) { err = "T5Gemma not loaded"; return false; }
    const int n = (int)ids.size();
    if (n < 1) { err = "empty prompt"; return false; }
    Compute& cm = Compute::get();
    const size_t nodes = 2048;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false) + (1 << 20));
    ggml_init_params ip = {meta.size(), meta.data(), true};
    ggml_context* c = ggml_init(ip);
    TT emb = ggml_new_tensor_2d(c, GGML_TYPE_F32, 768, n); ggml_set_input(emb);
    TT pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n); ggml_set_input(pos);
    TT x = emb;
    for (auto& L : layers_) {
        TT h = ggml_mul(c, ggml_rms_norm(c, x, 1e-6f), L.n1);
        TT q = ggml_reshape_3d(c, ggml_mul_mat(c, L.q, h), 64, 12, n);
        TT k = ggml_reshape_3d(c, ggml_mul_mat(c, L.k, h), 64, 12, n);
        TT v = ggml_reshape_3d(c, ggml_mul_mat(c, L.v, h), 64, 12, n);
        q = ggml_rope_ext(c, q, pos, nullptr, 64, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(c, k, pos, nullptr, 64, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        // attention with tanh logit soft-capping, written out (short sequences; the fused CUDA kernel crashes for this shape)
        TT qp = ggml_permute(c, q, 0, 2, 1, 3), kp = ggml_permute(c, k, 0, 2, 1, 3);          // [64, n, 12]
        TT sc = ggml_mul_mat(c, kp, qp);                                                       // [keys, queries, 12]
        sc = ggml_scale(c, ggml_tanh(c, ggml_scale(c, sc, 0.125f / 50.0f)), 50.0f);
        sc = ggml_soft_max(c, sc);
        TT vp = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));                                  // [keys, 64, 12]
        TT a = ggml_cont(c, ggml_permute(c, ggml_mul_mat(c, vp, sc), 0, 2, 1, 3));             // [64, 12, n]
        a = ggml_mul_mat(c, L.o, ggml_reshape_2d(c, a, 768, n));
        x = ggml_add(c, x, ggml_mul(c, ggml_rms_norm(c, a, 1e-6f), L.n2));
        TT f = ggml_mul(c, ggml_rms_norm(c, x, 1e-6f), L.n3);
        TT m = ggml_mul_mat(c, L.down, ggml_mul(c, ggml_gelu(c, ggml_mul_mat(c, L.gate, f)), ggml_mul_mat(c, L.up, f)));
        x = ggml_add(c, x, ggml_mul(c, ggml_rms_norm(c, m, 1e-6f), L.n4));
    }
    x = ggml_mul(c, ggml_rms_norm(c, x, 1e-6f), norm_);
    ggml_set_output(x);
    ggml_cgraph* g = ggml_new_graph_custom(c, nodes, false);
    ggml_build_forward_expand(g, x);
    ggml_backend_sched_reset(cm.sched);
    if (!ggml_backend_sched_alloc_graph(cm.sched, g)) { err = "cannot allocate T5Gemma graph"; ggml_free(c); return false; }

    const StTensor* et = st_->find("model.encoder.embed_tokens.weight");
    std::vector<float> e((size_t)n * 768);
    const float scale = std::sqrt(768.0f);
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= et->shape[0]) { err = "token id out of range"; ggml_free(c); return false; }
        const uint16_t* row = (const uint16_t*)et->data + (size_t)ids[i] * 768;
        for (int j = 0; j < 768; j++) e[(size_t)i * 768 + j] = bf16ToF32(row[j]) * scale;
    }
    ggml_backend_tensor_set(emb, e.data(), 0, e.size() * 4);
    std::vector<int32_t> p(n); std::iota(p.begin(), p.end(), 0);
    ggml_backend_tensor_set(pos, p.data(), 0, n * 4);
    if (ggml_backend_sched_graph_compute(cm.sched, g) != GGML_STATUS_SUCCESS) { err = "T5Gemma compute failed"; ggml_free(c); return false; }
    out.resize((size_t)n * 768);
    ggml_backend_tensor_get(x, out.data(), 0, out.size() * 4);
    ggml_backend_sched_reset(cm.sched);
    ggml_free(c);
    return true;
}
