// `DopplersBakery krea-op-bench [tokens]`: times the operations that make up a Krea2 DiT step at one token count (set BAKERY_BACKEND=cuda / vulkan),
// to see which one is slow. Weights are random; only the speed matters.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "ggml.h"
#include "ggml-backend.h"
#include "gpu/compute.h"
#include "ltx/ltx_ggml.h"

namespace {
double best3(const std::function<void()>& f) {
    double b = 1e9;
    for (int i = 0; i < 3; i++) { auto t = std::chrono::steady_clock::now(); f(); b = std::min(b, std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count()); }
    return b;
}
// builds one op over fresh random data and times its compute (the first run is a warm-up)
template <class Build> void timeOp(const char* name, double tflop, Build build) {
    Compute& cm = Compute::get(); std::string err;
    ggml_init_params wp = {ggml_tensor_overhead() * 16, nullptr, true}; ggml_context* wc = ggml_init(wp);
    ltxg::Graph G(256);
    std::vector<ggml_tensor*> weights, inputs;
    ggml_tensor* out = build(G.c, wc, weights, inputs);
    ggml_set_output(out); ggml_build_forward_expand(G.g, out);
    ggml_backend_buffer_t wb = weights.empty() ? nullptr : cm.allocDeviceWeights(wc);
    if (!weights.empty() && !wb) { printf("%-34s  cannot allocate weights\n", name); ggml_free(wc); return; }
    unsigned s = 1u;
    auto fill = [&](ggml_tensor* t) {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        if (t->type == GGML_TYPE_F32) { std::vector<float> f(ggml_nelements(t)); for (auto& v : f) { s = s * 1664525u + 1013904223u; v = ((float)(s >> 8) / 16777216.f - 0.5f); } memcpy(raw.data(), f.data(), raw.size()); }
        else for (auto& b : raw) { s = s * 1664525u + 1013904223u; b = (uint8_t)(s >> 24); }
        if (t->type == GGML_TYPE_Q8_0) for (size_t o = 0; o + 34 <= raw.size(); o += 34) { raw[o] = 0x00; raw[o + 1] = 0x1c; }      // block scale 0.0039: sane magnitudes
        if (t->type == GGML_TYPE_F16) { ggml_fp16_t* h = (ggml_fp16_t*)raw.data(); for (size_t i = 0; i < raw.size() / 2; i++) { s = s * 1664525u + 1013904223u; h[i] = ggml_fp32_to_fp16(((float)(s >> 8) / 16777216.f - 0.5f)); } }
        ggml_backend_tensor_set(t, raw.data(), 0, raw.size());
    };
    for (auto* w : weights) fill(w);
    if (!ltxg::runGraph(G, err)) { printf("%-34s  %s\n", name, err.c_str()); if (wb) ggml_backend_buffer_free(wb); ggml_free(wc); return; }
    for (auto* i : inputs) fill(i);
    ltxg::computeGraph(G, err);
    const double t = best3([&] { ltxg::computeGraph(G, err); });
    printf("%-34s %7.3f s   %6.1f TFLOPS\n", name, t, tflop / t);
    ggml_backend_sched_reset(cm.sched); if (wb) ggml_backend_buffer_free(wb); ggml_free(wc);
}
}

int runKreaOpBench(const std::vector<std::string>& a) {
    std::string err; if (!Compute::get().init(0, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const int64_t N = a.empty() ? 16384 : atoi(a[0].c_str());
    printf("backend: %s, %lld tokens\n", Compute::get().description().c_str(), (long long)N);
    auto mm = [&](const char* nm, ggml_type ty, int64_t in, int64_t out) {
        timeOp(nm, 2.0 * in * out * N / 1e12, [&](ggml_context* c, ggml_context* wc, std::vector<ggml_tensor*>& w, std::vector<ggml_tensor*>& x) {
            ggml_tensor* W = ggml_new_tensor_2d(wc, ty, in, out); w.push_back(W);
            ggml_tensor* X = ggml_new_tensor_2d(c, GGML_TYPE_F32, in, N); ggml_set_input(X); x.push_back(X);
            return ggml_mul_mat(c, W, X);
        });
    };
    mm("Q8_0 6144 -> 6144 (wq/wo/gate)", GGML_TYPE_Q8_0, 6144, 6144);
    mm("F16  6144 -> 6144 (cuBLAS)", GGML_TYPE_F16, 6144, 6144);
    mm("Q8_0 6144 -> 16384 (mlp up)", GGML_TYPE_Q8_0, 6144, 16384);
    mm("F16  6144 -> 16384", GGML_TYPE_F16, 6144, 16384);
    mm("Q8_0 16384 -> 6144 (mlp down)", GGML_TYPE_Q8_0, 16384, 6144);
    mm("F16  16384 -> 6144", GGML_TYPE_F16, 16384, 6144);
    // attention as the DiT builds it: q [128, N, 48] f32, k / v [128, N, 12] f16, grouped query
    timeOp("flash_attn_ext 48 heads / 12 kv", 4.0 * (double)N * N * 128 * 48 / 1e12, [&](ggml_context* c, ggml_context*, std::vector<ggml_tensor*>&, std::vector<ggml_tensor*>& x) {
        ggml_tensor* q = ggml_new_tensor_4d(c, GGML_TYPE_F32, 128, N, 48, 1); ggml_set_input(q); x.push_back(q);
        ggml_tensor* k = ggml_new_tensor_4d(c, GGML_TYPE_F16, 128, N, 12, 1); ggml_set_input(k); x.push_back(k);
        ggml_tensor* v = ggml_new_tensor_4d(c, GGML_TYPE_F16, 128, N, 12, 1); ggml_set_input(v); x.push_back(v);
        return ggml_flash_attn_ext(c, q, k, v, nullptr, 1.0f / std::sqrt(128.0f), 0.0f, 0.0f);
    });
    return 0;
}
