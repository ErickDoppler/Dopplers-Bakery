#include "ltx/ltx_text.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gpu/compute.h"
#include "util/sys.h"

using TT = ggml_tensor*;
namespace {
constexpr int kHid = 3840, kFF = 15360, kHeads = 16, kKv = 8, kHd = 256, kLayers = 48;

// weights of one block / one layer: uploaded as F16 (matrices) or F32 (vectors), freed together
struct WeightSet {
    ggml_context* ctx = nullptr; ggml_backend_buffer_t buf = nullptr;
    struct Item { TT t; std::vector<float> data; bool f16; };
    std::vector<Item> items;
    WeightSet() { ggml_init_params ip = {ggml_tensor_overhead() * 256, nullptr, true}; ctx = ggml_init(ip); }
    ~WeightSet() { if (buf) ggml_backend_buffer_free(buf); if (ctx) ggml_free(ctx); }
    // from the file: matrix [rows][cols] -> ggml [cols, rows]
    TT mat(const LtxWeights& w, const std::string& n, std::string& err) {
        std::vector<int64_t> sh; std::vector<float> d;
        if (!w.shapeOf(n, sh) || !w.loadF32(n, d, err)) { if (err.empty()) err = "missing tensor " + n; return nullptr; }
        TT t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, sh[1], sh[0]);
        items.push_back({t, std::move(d), true}); return t;
    }
    TT vec(const LtxWeights& w, const std::string& n, std::string& err, float add = 0.f) {
        std::vector<float> d;
        if (!w.has(n) || !w.loadF32(n, d, err)) { if (err.empty()) err = "missing tensor " + n; return nullptr; }
        if (add != 0.f) for (auto& v : d) v += add;
        TT t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)d.size());
        items.push_back({t, std::move(d), false}); return t;
    }
    bool upload(std::string& err) {
        buf = Compute::get().allocDeviceWeights(ctx);
        if (!buf) { err = "cannot allocate model weights on the device"; return false; }
        std::vector<ggml_fp16_t> h;
        for (auto& it : items) {
            if (it.f16) { h.resize(it.data.size()); ggml_fp32_to_fp16_row(it.data.data(), h.data(), (int64_t)it.data.size()); ggml_backend_tensor_set(it.t, h.data(), 0, h.size() * 2); }
            else ggml_backend_tensor_set(it.t, it.data.data(), 0, it.data.size() * 4);
            std::vector<float>().swap(it.data);
        }
        return true;
    }
};

struct Graph {
    std::vector<uint8_t> meta; ggml_context* c = nullptr; ggml_cgraph* g = nullptr;
    explicit Graph(size_t nodes) {
        meta.resize(ggml_tensor_overhead() * nodes * 2 + ggml_graph_overhead_custom(nodes, false) + (1 << 20));
        ggml_init_params ip = {meta.size(), meta.data(), true}; c = ggml_init(ip); g = ggml_new_graph_custom(c, nodes, false);
    }
    ~Graph() { ggml_free(c); }
};

bool runGraph(Graph& G, std::string& err) {
    Compute& cm = Compute::get();
    ggml_backend_sched_reset(cm.sched);
    if (!ggml_backend_sched_alloc_graph(cm.sched, G.g)) { err = "cannot allocate graph (not enough memory)"; return false; }
    return true;
}
bool computeGraph(Graph& G, std::string& err) {
    if (ggml_backend_sched_graph_compute(Compute::get().sched, G.g) != GGML_STATUS_SUCCESS) { err = "compute failed"; return false; }
    return true;
}

void rmsNormRow(const float* x, float* o, int n, const float* w, float add, float eps) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float r = 1.0f / std::sqrt((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * r * (w ? w[i] + add : 1.0f);
}
}  // namespace

bool LtxTextEncoder::open(const std::string& gemmaPath, const std::string& ckptPath, std::string& err) {
    if (!gemma.open(gemmaPath, err) || !ckpt.open(ckptPath, err)) return false;
    const StTensor* sp_ = gemma.file.find("spiece_model");
    if (!sp_) { err = "spiece_model not found in the Gemma file"; return false; }
    return sp.load(std::string((const char*)sp_->data, (size_t)sp_->bytes), err);
}

std::vector<int> LtxTextEncoder::tokenize(const std::string& text) const {
    std::vector<int> ids = {2};                                  // <bos>
    for (int t : sp.encode(text)) ids.push_back(t);
    return ids;
}

bool LtxTextEncoder::gemmaHidden(const std::vector<int>& ids, std::vector<std::vector<float>>& hidden, const std::function<void(int)>& progress, std::string& err) {
    const int n = (int)ids.size();
    if (n < 1) { err = "empty prompt"; return false; }
    if (!Compute::get().init(0, err)) return false;
    hidden.assign(kLayers + 1, std::vector<float>());
    const StTensor* et = gemma.file.find("model.embed_tokens.weight");
    if (!et) { err = "missing embed_tokens"; return false; }
    std::vector<float> x((size_t)n * kHid);
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= et->shape[0]) { err = "token id out of range"; return false; }
        const uint16_t* row = (const uint16_t*)et->data + (size_t)ids[i] * kHid;
        for (int j = 0; j < kHid; j++) x[(size_t)i * kHid + j] = bf16ToF32(row[j]) * std::sqrt((float)kHid);
    }
    hidden[0] = x;
    // masks: causal, plus the 1024-token sliding window of the local layers
    auto makeMask = [&](bool local) {
        const int pad = (n + 63) / 64 * 64;
        std::vector<ggml_fp16_t> m((size_t)n * pad, ggml_fp32_to_fp16(0.f));
        const ggml_fp16_t neg = ggml_fp32_to_fp16(-65000.f);
        for (int q = 0; q < n; q++) for (int k = 0; k < n; k++) if (k > q || (local && q - k >= 1024)) m[(size_t)q * n + k] = neg;
        return m;
    };
    const auto maskGlobal = makeMask(false), maskLocal = makeMask(true);
    std::vector<int32_t> pos(n); std::iota(pos.begin(), pos.end(), 0);
    for (int L = 0; L < kLayers; L++) {
        if (progress) progress(L);
        const std::string p = "model.layers." + std::to_string(L) + ".";
        WeightSet W;
        TT q = W.mat(gemma, p + "self_attn.q_proj.weight", err), k = W.mat(gemma, p + "self_attn.k_proj.weight", err), v = W.mat(gemma, p + "self_attn.v_proj.weight", err);
        TT o = W.mat(gemma, p + "self_attn.o_proj.weight", err), gt = W.mat(gemma, p + "mlp.gate_proj.weight", err), up = W.mat(gemma, p + "mlp.up_proj.weight", err), dn = W.mat(gemma, p + "mlp.down_proj.weight", err);
        TT n1 = W.vec(gemma, p + "input_layernorm.weight", err, 1.f), n2 = W.vec(gemma, p + "post_attention_layernorm.weight", err, 1.f);
        TT n3 = W.vec(gemma, p + "pre_feedforward_layernorm.weight", err, 1.f), n4 = W.vec(gemma, p + "post_feedforward_layernorm.weight", err, 1.f);
        TT qn = W.vec(gemma, p + "self_attn.q_norm.weight", err, 1.f), kn = W.vec(gemma, p + "self_attn.k_norm.weight", err, 1.f);
        if (!err.empty() || !W.upload(err)) return false;
        const bool global = L % 6 == 5;
        Graph G(512); ggml_context* c = G.c;
        TT in = ggml_new_tensor_2d(c, GGML_TYPE_F32, kHid, n); ggml_set_input(in);
        TT pt = ggml_new_tensor_1d(c, GGML_TYPE_I32, n); ggml_set_input(pt);
        TT mk = ggml_new_tensor_2d(c, GGML_TYPE_F16, n, (n + 63) / 64 * 64); ggml_set_input(mk);
        TT h = ggml_mul(c, ggml_rms_norm(c, in, 1e-6f), n1);
        TT Q = ggml_reshape_3d(c, ggml_mul_mat(c, q, h), kHd, kHeads, n), K = ggml_reshape_3d(c, ggml_mul_mat(c, k, h), kHd, kKv, n), V = ggml_reshape_3d(c, ggml_mul_mat(c, v, h), kHd, kKv, n);
        Q = ggml_mul(c, ggml_rms_norm(c, Q, 1e-6f), qn); K = ggml_mul(c, ggml_rms_norm(c, K, 1e-6f), kn);
        const float base = global ? 1000000.f : 10000.f, fscale = global ? 1.f / 8.f : 1.f;
        Q = ggml_rope_ext(c, Q, pt, nullptr, kHd, GGML_ROPE_TYPE_NEOX, 0, base, fscale, 0.f, 1.f, 32.f, 1.f);
        K = ggml_rope_ext(c, K, pt, nullptr, kHd, GGML_ROPE_TYPE_NEOX, 0, base, fscale, 0.f, 1.f, 32.f, 1.f);
        TT Qp = ggml_permute(c, Q, 0, 2, 1, 3), Kp = ggml_permute(c, K, 0, 2, 1, 3);                  // [256, n, heads]
        TT sc = ggml_soft_max_ext(c, ggml_mul_mat(c, Kp, Qp), mk, 1.0f / 16.0f, 0.f);                  // [keys, queries, 16], GQA by broadcast
        TT Vp = ggml_cont(c, ggml_permute(c, V, 1, 2, 0, 3));                                          // [keys, 256, 8]
        TT a = ggml_cont(c, ggml_permute(c, ggml_mul_mat(c, Vp, sc), 0, 2, 1, 3));                     // [256, 16, n]
        a = ggml_mul_mat(c, o, ggml_reshape_2d(c, a, kHeads * kHd, n));
        TT r = ggml_add(c, in, ggml_mul(c, ggml_rms_norm(c, a, 1e-6f), n2));
        TT f = ggml_mul(c, ggml_rms_norm(c, r, 1e-6f), n3);
        TT m = ggml_mul_mat(c, dn, ggml_mul(c, ggml_gelu(c, ggml_mul_mat(c, gt, f)), ggml_mul_mat(c, up, f)));
        TT out = ggml_add(c, r, ggml_mul(c, ggml_rms_norm(c, m, 1e-6f), n4));
        ggml_set_output(out); ggml_build_forward_expand(G.g, out);
        if (!runGraph(G, err)) return false;
        ggml_backend_tensor_set(in, x.data(), 0, x.size() * 4);
        ggml_backend_tensor_set(pt, pos.data(), 0, n * 4);
        const auto& mm = global ? maskGlobal : maskLocal;
        ggml_backend_tensor_set(mk, mm.data(), 0, mm.size() * 2);
        if (!computeGraph(G, err)) return false;
        ggml_backend_tensor_get(out, x.data(), 0, x.size() * 4);
        ggml_backend_sched_reset(Compute::get().sched);
        if (L + 1 < kLayers) hidden[L + 1] = x;
    }
    const std::vector<float> nw = [&] { std::vector<float> w; gemma.loadF32("model.norm.weight", w, err); return w; }();
    if (nw.empty()) return false;
    hidden[kLayers].resize(x.size());
    for (int i = 0; i < n; i++) rmsNormRow(&x[(size_t)i * kHid], &hidden[kLayers][(size_t)i * kHid], kHid, nw.data(), 1.f, 1e-6f);
    return true;
}

bool LtxTextEncoder::project(const std::vector<std::vector<float>>& hidden, int n, std::vector<float>& video, std::vector<float>& audio, std::string& err) {
    const int NL = (int)hidden.size();                              // 49
    const int in = kHid * NL;
    std::vector<float> feat((size_t)n * in);
    for (int t = 0; t < n; t++) for (int l = 0; l < NL; l++) {
        const float* hs = &hidden[l][(size_t)t * kHid];
        double ss = 0; for (int d = 0; d < kHid; d++) ss += (double)hs[d] * hs[d];
        const float r = 1.0f / std::sqrt((float)(ss / kHid) + 1e-6f);
        for (int d = 0; d < kHid; d++) feat[(size_t)t * in + (size_t)d * NL + l] = hs[d] * r;       // flatten order: channel major, layer minor
    }
    for (int which = 0; which < 2; which++) {
        const int outDim = which == 0 ? 4096 : 2048;
        const std::string base = std::string("text_embedding_projection.") + (which == 0 ? "video" : "audio") + "_aggregate_embed.";
        WeightSet W; TT w = W.mat(ckpt, base + "weight", err), b = W.vec(ckpt, base + "bias", err);
        if (!err.empty() || !W.upload(err)) return false;
        Graph G(32); ggml_context* c = G.c;
        TT x = ggml_new_tensor_2d(c, GGML_TYPE_F32, in, n); ggml_set_input(x);
        TT y = ggml_add(c, ggml_mul_mat(c, w, ggml_scale(c, x, std::sqrt((float)outDim / kHid))), b);
        ggml_set_output(y); ggml_build_forward_expand(G.g, y);
        if (!runGraph(G, err)) return false;
        ggml_backend_tensor_set(x, feat.data(), 0, feat.size() * 4);
        if (!computeGraph(G, err)) return false;
        std::vector<float>& dst = which == 0 ? video : audio; dst.resize((size_t)n * outDim);
        ggml_backend_tensor_get(y, dst.data(), 0, dst.size() * 4);
        ggml_backend_sched_reset(Compute::get().sched);
    }
    return true;
}

bool LtxTextEncoder::connector(bool video, const std::vector<float>& in, int n, std::vector<float>& out, int& L, std::string& err) {
    const int dim = video ? 4096 : 2048, heads = 32, hd = dim / heads, half = hd / 2;
    const std::string P = std::string("model.diffusion_model.") + (video ? "video" : "audio") + "_embeddings_connector.";
    const int tile = (std::max(1024, n) + 127) / 128;
    L = tile * 128;
    std::vector<float> reg; if (!ckpt.loadF32(P + "learnable_registers", reg, err)) return false;
    std::vector<float> x((size_t)L * dim);
    memcpy(x.data(), in.data(), (size_t)n * dim * 4);
    for (int t = n; t < L; t++) memcpy(&x[(size_t)t * dim], &reg[(size_t)(t % 128) * dim], (size_t)dim * 4);
    // split-rope tables [half, heads, L]: head h uses frequencies h*half .. (h+1)*half of the dim/2 grid
    std::vector<float> cs((size_t)half * heads * L), sn(cs.size());
    const int nf = dim / 2;
    for (int t = 0; t < L; t++) {
        const float frac = (float)t / 4096.0f * 2.0f - 1.0f;
        for (int h = 0; h < heads; h++) for (int j = 0; j < half; j++) {
            const int k = h * half + j;
            const float fk = (float)(std::pow(10000.0, (double)k / (double)(nf - 1)) * 3.14159265358979323846 / 2.0);
            const float a = fk * frac;
            cs[((size_t)t * heads + h) * half + j] = std::cos(a); sn[((size_t)t * heads + h) * half + j] = std::sin(a);
        }
    }
    for (int blk = 0; blk < 8; blk++) {
        const std::string b = P + "transformer_1d_blocks." + std::to_string(blk) + ".";
        WeightSet W;
        TT wq = W.mat(ckpt, b + "attn1.to_q.weight", err), bq = W.vec(ckpt, b + "attn1.to_q.bias", err), wk = W.mat(ckpt, b + "attn1.to_k.weight", err), bk = W.vec(ckpt, b + "attn1.to_k.bias", err);
        TT wv = W.mat(ckpt, b + "attn1.to_v.weight", err), bv = W.vec(ckpt, b + "attn1.to_v.bias", err), wo = W.mat(ckpt, b + "attn1.to_out.0.weight", err), bo = W.vec(ckpt, b + "attn1.to_out.0.bias", err);
        TT qn = W.vec(ckpt, b + "attn1.q_norm.weight", err), kn = W.vec(ckpt, b + "attn1.k_norm.weight", err), wg = W.mat(ckpt, b + "attn1.to_gate_logits.weight", err), bg = W.vec(ckpt, b + "attn1.to_gate_logits.bias", err);
        TT w1 = W.mat(ckpt, b + "ff.net.0.proj.weight", err), b1 = W.vec(ckpt, b + "ff.net.0.proj.bias", err), w2 = W.mat(ckpt, b + "ff.net.2.weight", err), b2 = W.vec(ckpt, b + "ff.net.2.bias", err);
        if (!err.empty() || !W.upload(err)) return false;
        Graph G(256); ggml_context* c = G.c;
        TT xin = ggml_new_tensor_2d(c, GGML_TYPE_F32, dim, L); ggml_set_input(xin);
        TT ct = ggml_new_tensor_3d(c, GGML_TYPE_F32, half, heads, L); ggml_set_input(ct);
        TT st = ggml_new_tensor_3d(c, GGML_TYPE_F32, half, heads, L); ggml_set_input(st);
        auto rope = [&](TT t) {                                                                       // t: [hd, heads, L]
            TT x1 = ggml_view_3d(c, t, half, heads, L, t->nb[1], t->nb[2], 0), x2 = ggml_view_3d(c, t, half, heads, L, t->nb[1], t->nb[2], (size_t)half * 4);
            TT o1 = ggml_sub(c, ggml_mul(c, x1, ct), ggml_mul(c, x2, st)), o2 = ggml_add(c, ggml_mul(c, x2, ct), ggml_mul(c, x1, st));
            return ggml_concat(c, o1, o2, 0);
        };
        TT h = ggml_rms_norm(c, xin, 1e-6f);
        TT q = ggml_mul(c, ggml_rms_norm(c, ggml_add(c, ggml_mul_mat(c, wq, h), bq), 1e-5f), qn), k = ggml_mul(c, ggml_rms_norm(c, ggml_add(c, ggml_mul_mat(c, wk, h), bk), 1e-5f), kn);
        TT v = ggml_add(c, ggml_mul_mat(c, wv, h), bv);
        q = rope(ggml_reshape_3d(c, q, hd, heads, L)); k = rope(ggml_reshape_3d(c, k, hd, heads, L)); v = ggml_reshape_3d(c, v, hd, heads, L);
        TT qp = ggml_permute(c, q, 0, 2, 1, 3), kp = ggml_permute(c, k, 0, 2, 1, 3);                   // [hd, L, heads]
        TT sc = ggml_soft_max_ext(c, ggml_mul_mat(c, kp, qp), nullptr, 1.0f / std::sqrt((float)hd), 0.f);
        TT vp = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));                                          // [keys, hd, heads]
        TT a = ggml_cont(c, ggml_permute(c, ggml_mul_mat(c, vp, sc), 0, 2, 1, 3));                     // [hd, heads, L]
        TT gate = ggml_scale(c, ggml_sigmoid(c, ggml_add(c, ggml_mul_mat(c, wg, h), bg)), 2.0f);        // [heads, L]
        a = ggml_mul(c, a, ggml_reshape_3d(c, gate, 1, heads, L));
        TT att = ggml_add(c, ggml_mul_mat(c, wo, ggml_reshape_2d(c, a, dim, L)), bo);
        TT r = ggml_add(c, xin, att);
        TT f = ggml_add(c, ggml_mul_mat(c, w2, ggml_gelu(c, ggml_add(c, ggml_mul_mat(c, w1, ggml_rms_norm(c, r, 1e-6f)), b1))), b2);
        TT y = ggml_add(c, r, f);
        if (blk == 7) y = ggml_rms_norm(c, y, 1e-6f);
        ggml_set_output(y); ggml_build_forward_expand(G.g, y);
        if (!runGraph(G, err)) return false;
        ggml_backend_tensor_set(xin, x.data(), 0, x.size() * 4);
        ggml_backend_tensor_set(ct, cs.data(), 0, cs.size() * 4); ggml_backend_tensor_set(st, sn.data(), 0, sn.size() * 4);
        if (!computeGraph(G, err)) return false;
        ggml_backend_tensor_get(y, x.data(), 0, x.size() * 4);
        ggml_backend_sched_reset(Compute::get().sched);
    }
    out = std::move(x);
    return true;
}

// ---- self test against tools/ltx_ref_text.py output
static bool readBin(const std::string& p, std::vector<float>& v) {
    std::string s; if (!sys::readFile(p, s)) return false;
    v.resize(s.size() / 4); memcpy(v.data(), s.data(), v.size() * 4); return true;
}
static void compare(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) { printf("%-12s size mismatch: %zu vs reference %zu\n", what, a.size(), b.size()); return; }
    double md = 0, dot = 0, na = 0, nb = 0, mr = 0;
    for (size_t i = 0; i < a.size(); i++) { md = std::max(md, (double)std::fabs(a[i] - b[i])); dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; mr = std::max(mr, (double)std::fabs(b[i])); }
    printf("%-12s n=%zu max|diff|=%.5g (max|ref|=%.4g) cosine=%.6f\n", what, a.size(), md, mr, dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30));
}
int runLtxTextTest(const std::vector<std::string>& a) {
    if (a.size() < 4) { fprintf(stderr, "usage: ltx-text <gemma> <ltx checkpoint> <reference dir> \"<prompt>\" [stage: tokens|gemma|all]\n"); return 2; }
    LtxTextEncoder te; std::string err;
    if (!te.open(a[0], a[1], err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::vector<int> ids = te.tokenize(a[3]);
    std::vector<float> ref; readBin(a[2] + "\\ids.bin", ref);
    std::string s; sys::readFile(a[2] + "\\ids.bin", s);
    std::vector<int> rid(s.size() / 4); memcpy(rid.data(), s.data(), s.size());
    printf("tokens: %zu (reference %zu) %s\n", ids.size(), rid.size(), ids == rid ? "IDENTICAL" : "DIFFERENT");
    if (ids != rid) { for (size_t i = 0; i < std::max(ids.size(), rid.size()); i++) printf("  %zu: %d | %d\n", i, i < ids.size() ? ids[i] : -1, i < rid.size() ? rid[i] : -1); return 1; }
    const std::string stage = a.size() > 4 ? a[4] : "all";
    if (stage == "tokens") return 0;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<float>> hid;
    if (!te.gemmaHidden(ids, hid, [](int l) { if (l % 8 == 0) fprintf(stderr, "gemma layer %d/48\n", l); }, err)) { fprintf(stderr, "gemma: %s\n", err.c_str()); return 1; }
    printf("gemma hidden states: %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    const int n = (int)ids.size();
    std::vector<float> vid, aud;
    if (!te.project(hid, n, vid, aud, err)) { fprintf(stderr, "project: %s\n", err.c_str()); return 1; }
    std::vector<float> proj; readBin(a[2] + "\\proj.bin", proj);
    std::vector<float> mine((size_t)n * 6144);
    for (int t = 0; t < n; t++) { memcpy(&mine[(size_t)t * 6144], &vid[(size_t)t * 4096], 4096 * 4); memcpy(&mine[(size_t)t * 6144 + 4096], &aud[(size_t)t * 2048], 2048 * 4); }
    compare("projection", mine, proj);
    if (stage == "gemma") return 0;
    for (int which = 0; which < 2; which++) {
        std::vector<float> out, rv; int L = 0;
        if (!te.connector(which == 0, which == 0 ? vid : aud, n, out, L, err)) { fprintf(stderr, "connector: %s\n", err.c_str()); return 1; }
        readBin(a[2] + (which == 0 ? "\\video_ctx.bin" : "\\audio_ctx.bin"), rv);
        compare(which == 0 ? "video ctx" : "audio ctx", out, rv);
    }
    return 0;
}
