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
}  // namespace ltxg
