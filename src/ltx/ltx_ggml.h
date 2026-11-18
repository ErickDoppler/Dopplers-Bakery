// Small ggml helpers shared by the LTX-2.3 modules: weights uploaded per block, one-shot graphs.
#pragma once
#include <cmath>
#include <vector>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gpu/compute.h"
#include "ltx/ltx_weights.h"

using TT = ggml_tensor*;
namespace ltxg {
// weights of one block / one layer: uploaded as F16 (matrices) or F32 (vectors), freed together
struct WeightSet {
    ggml_context* ctx = nullptr; ggml_backend_buffer_t buf = nullptr;
    struct Item { TT t; std::vector<float> data; bool f16; const std::vector<uint16_t>* h16 = nullptr; };
    std::vector<Item> items;
    WeightSet() { ggml_init_params ip = {ggml_tensor_overhead() * 256, nullptr, true}; ctx = ggml_init(ip); }
    ~WeightSet() { if (buf) ggml_backend_buffer_free(buf); if (ctx) ggml_free(ctx); }
    // from the file: matrix [rows][cols] -> ggml [cols, rows]
    TT mat(const LtxWeights& w, const std::string& n, std::string& err) {
        std::vector<int64_t> sh; std::vector<float> d;
        if (!w.shapeOf(n, sh)) { err = "missing tensor " + n; return nullptr; }
        if (w.cacheF16) {
            auto& slot = w.f16cache[n];
            if (slot.empty()) {
                if (!w.loadF32(n, d, err)) { if (err.empty()) err = "missing tensor " + n; w.f16cache.erase(n); return nullptr; }
                slot.resize(d.size()); ggml_fp32_to_fp16_row(d.data(), (ggml_fp16_t*)slot.data(), (int64_t)d.size());
            }
            TT t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, sh[1], sh[0]);
            Item it{t, {}, true}; it.h16 = &slot; items.push_back(std::move(it)); return t;
        }
        if (!w.loadF32(n, d, err)) { if (err.empty()) err = "missing tensor " + n; return nullptr; }
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
            if (it.h16) { ggml_backend_tensor_set(it.t, it.h16->data(), 0, it.h16->size() * 2); continue; }
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

inline bool runGraph(Graph& G, std::string& err) {
    Compute& cm = Compute::get();
    ggml_backend_sched_reset(cm.sched);
    if (!ggml_backend_sched_alloc_graph(cm.sched, G.g)) { err = "cannot allocate graph (not enough memory)"; return false; }
    return true;
}
inline bool computeGraph(Graph& G, std::string& err) {
    if (ggml_backend_sched_graph_compute(Compute::get().sched, G.g) != GGML_STATUS_SUCCESS) { err = "compute failed"; return false; }
    return true;
}

inline void rmsNormRow(const float* x, float* o, int n, const float* w, float add, float eps) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float r = 1.0f / std::sqrt((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * r * (w ? w[i] + add : 1.0f);
}
struct ConvW { TT w[3] = {}; TT b = nullptr; int ic = 0, oc = 0; };

inline bool loadConv(WeightSet& W, const LtxWeights& ck, const std::string& p, ConvW& cw, std::string& err) {
    std::vector<int64_t> sh; std::vector<float> d;
    if (!ck.shapeOf(p + "weight", sh) || !ck.loadF32(p + "weight", d, err)) { if (err.empty()) err = "missing " + p + "weight"; return false; }
    const int oc = (int)sh[0], ic = (int)sh[1];
    cw.oc = oc; cw.ic = ic;
    for (int dt = 0; dt < 3; dt++) {
        std::vector<float> tap((size_t)oc * ic * 9);
        for (int o = 0; o < oc; o++) for (int i = 0; i < ic; i++) for (int k = 0; k < 9; k++) tap[((size_t)o * ic + i) * 9 + k] = d[(((size_t)o * ic + i) * 3 + dt) * 9 + k];
        cw.w[dt] = ggml_new_tensor_4d(W.ctx, GGML_TYPE_F16, 3, 3, ic, oc);
        W.items.push_back({cw.w[dt], std::move(tap), true});
    }
    cw.b = W.vec(ck, p + "bias", err);
    return cw.b != nullptr;
}

// x [W,H,C,T]; the time axis is padded by replicating the first / last frame (padF / padB frames), then three 2D convs are summed
inline TT conv3(ggml_context* c, const ConvW& cw, TT x, int padF, int padB, bool zeroPad = false) {
    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], T = x->ne[3];
    TT xp = x;
    if (padF) { TT first = ggml_view_4d(c, x, W, H, C, 1, x->nb[1], x->nb[2], x->nb[3], 0); if (zeroPad) first = ggml_scale(c, first, 0.0f); for (int i = 0; i < padF; i++) xp = ggml_concat(c, first, xp, 3); }
    if (padB) { TT last = ggml_view_4d(c, x, W, H, C, 1, x->nb[1], x->nb[2], x->nb[3], (size_t)(T - 1) * x->nb[3]); if (zeroPad) last = ggml_scale(c, last, 0.0f); for (int i = 0; i < padB; i++) xp = ggml_concat(c, xp, last, 3); }
    const int64_t To = xp->ne[3] - 2;
    TT acc = nullptr;
    for (int dt = 0; dt < 3; dt++) {
        TT s = ggml_cont(c, ggml_view_4d(c, xp, W, H, C, To, xp->nb[1], xp->nb[2], xp->nb[3], (size_t)dt * xp->nb[3]));
        TT y = ggml_conv_2d(c, cw.w[dt], s, 1, 1, 1, 1, 1, 1);
        acc = acc ? ggml_add(c, acc, y) : y;
    }
    return ggml_add(c, acc, ggml_reshape_4d(c, cw.b, 1, 1, cw.oc, 1));
}

inline TT pixelNorm(ggml_context* c, TT x) {          // over the channel axis (ne[2]) with eps 1e-8
    TT p = ggml_cont(c, ggml_permute(c, x, 1, 2, 0, 3));
    p = ggml_rms_norm(c, p, 1e-8f);
    return ggml_cont(c, ggml_permute(c, p, 2, 0, 1, 3));
}

}  // namespace ltxg

