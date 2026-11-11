#include "tts/speech_decoder.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gpu/compute.h"
#include "krea2/safetensors.h"

using TT = ggml_tensor*;

void SpeechDecoder::unload() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
    buf_ = nullptr; ctx_ = nullptr; layers_.clear();
    for (auto& c : cb_) { c.clear(); c.shrink_to_fit(); }
}

bool SpeechDecoder::load(const std::string& path, std::string& err) {
    unload();
    StFile st;
    if (!st.open(path, err)) return false;
    std::string e2;
    if (!Compute::get().init(0, e2)) { err = e2; return false; }
    ggml_init_params ip = {ggml_tensor_overhead() * 600, nullptr, true};
    ctx_ = ggml_init(ip);

    struct Job { TT t; std::function<void(std::vector<float>&)> xf; std::string name; bool f16; };
    std::vector<Job> jobs;
    bool ok = true;
    auto get = [&](const std::string& n) -> const StTensor* { const StTensor* t = st.find(n); if (!t && ok) { err = "missing tensor " + n; ok = false; } return t; };
    // F32 matrix/vector with ggml dims = reversed safetensors dims (trailing 1s dropped)
    auto f32 = [&](const std::string& n, std::function<void(std::vector<float>&)> xf = nullptr) -> TT {
        const StTensor* t = get(n); if (!t) return nullptr;
        std::vector<int64_t> d;
        for (auto it = t->shape.rbegin(); it != t->shape.rend(); ++it) if (*it != 1) d.push_back(*it);
        if (d.empty()) d.push_back(1);
        TT g = d.size() == 1 ? ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, d[0]) : ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, d[0], d[1]);
        ggml_set_name(g, n.c_str()); jobs.push_back({g, xf, n, false}); return g;
    };
    // conv kernels: 3D safetensors [a,b,c] -> ggml [c,b,a]; F16 (regular conv) or F32 (transposed conv)
    auto k3 = [&](const std::string& n, bool f16) -> TT {
        const StTensor* t = get(n); if (!t) return nullptr;
        TT g = ggml_new_tensor_3d(ctx_, f16 ? GGML_TYPE_F16 : GGML_TYPE_F32, t->shape[2], t->shape[1], t->shape[0]);
        ggml_set_name(g, n.c_str()); jobs.push_back({g, nullptr, n, f16}); return g;
    };
    // per-channel vector as [1, C] (for broadcasting along time)
    auto row = [&](const std::string& n, std::function<void(std::vector<float>&)> xf) -> TT {
        const StTensor* t = get(n); if (!t) return nullptr;
        TT g = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, 1, t->shape[0]);
        ggml_set_name(g, n.c_str()); jobs.push_back({g, xf, n, false}); return g;
    };
    auto expv = [](std::vector<float>& v) { for (auto& x : v) x = std::exp(x); };
    auto invExp = [](std::vector<float>& v) { for (auto& x : v) x = 1.0f / (std::exp(x) + 1e-9f); };

    const std::string D = "decoder.";
    opFirst_ = f32(D + "quantizer.rvq_first.output_proj.weight");
    opRest_ = f32(D + "quantizer.rvq_rest.output_proj.weight");
    preW_ = k3(D + "pre_conv.conv.weight", true); preB_ = f32(D + "pre_conv.conv.bias");
    const std::string T = D + "pre_transformer.";
    inW_ = f32(T + "input_proj.weight"); inB_ = f32(T + "input_proj.bias");
    outW_ = f32(T + "output_proj.weight"); outB_ = f32(T + "output_proj.bias"); normW_ = f32(T + "norm.weight");
    for (int i = 0; i < 8; i++) {
        std::string p = T + "layers." + std::to_string(i) + ".";
        Layer L;
        L.ln1 = f32(p + "input_layernorm.weight"); L.ln2 = f32(p + "post_attention_layernorm.weight");
        L.q = f32(p + "self_attn.q_proj.weight"); L.k = f32(p + "self_attn.k_proj.weight"); L.v = f32(p + "self_attn.v_proj.weight"); L.o = f32(p + "self_attn.o_proj.weight");
        L.gate = f32(p + "mlp.gate_proj.weight"); L.up = f32(p + "mlp.up_proj.weight"); L.down = f32(p + "mlp.down_proj.weight");
        L.s1 = f32(p + "self_attn_layer_scale.scale"); L.s2 = f32(p + "mlp_layer_scale.scale");
        layers_.push_back(L);
    }
    for (int i = 0; i < 2; i++) {
        std::string p = D + "upsample." + std::to_string(i) + ".";
        Up& u = up_[i];
        u.tw = k3(p + "0.conv.weight", false); u.tb = f32(p + "0.conv.bias");
        u.dw = f32(p + "1.dwconv.conv.weight"); u.db = f32(p + "1.dwconv.conv.bias");        // [1024,1,7] -> ggml [7,1024]
        u.nw = f32(p + "1.norm.weight"); u.nb = f32(p + "1.norm.bias");
        u.p1w = f32(p + "1.pwconv1.weight"); u.p1b = f32(p + "1.pwconv1.bias");
        u.p2w = f32(p + "1.pwconv2.weight"); u.p2b = f32(p + "1.pwconv2.bias"); u.gamma = f32(p + "1.gamma");
    }
    const std::string DD = D + "decoder.";
    d0w_ = k3(DD + "0.conv.weight", true); d0b_ = f32(DD + "0.conv.bias");
    const int strides[4] = {8, 5, 4, 3}, dils[3] = {1, 3, 9};
    for (int i = 0; i < 4; i++) {
        std::string p = DD + std::to_string(i + 1) + ".block.";
        Block& b = blocks_[i]; b.stride = strides[i];
        b.sa = row(p + "0.alpha", expv); b.sb = row(p + "0.beta", invExp);
        b.tw = k3(p + "1.conv.weight", false); b.tb = f32(p + "1.conv.bias");
        for (int u = 0; u < 3; u++) {
            std::string q = p + std::to_string(u + 2) + ".";
            Unit& x = b.u[u]; x.dil = dils[u];
            x.a1 = row(q + "act1.alpha", expv); x.b1 = row(q + "act1.beta", invExp);
            x.c1w = k3(q + "conv1.conv.weight", true); x.c1b = f32(q + "conv1.conv.bias");
            x.a2 = row(q + "act2.alpha", expv); x.b2 = row(q + "act2.beta", invExp);
            x.c2w = k3(q + "conv2.conv.weight", true); x.c2b = f32(q + "conv2.conv.bias");
        }
    }
    fa_ = row(DD + "5.alpha", expv); fb_ = row(DD + "5.beta", invExp);
    flw_ = k3(DD + "6.conv.weight", true); flb_ = f32(DD + "6.conv.bias");
    if (!ok) { unload(); return false; }

    buf_ = Compute::get().allocDeviceWeights(ctx_);
    if (!buf_) { err = "cannot allocate speech decoder weights"; unload(); return false; }
    std::vector<float> f;
    for (auto& j : jobs) {
        const StTensor* t = st.find(j.name);
        st.toFloat(*t, f);
        if (j.xf) j.xf(f);
        if (j.f16) { std::vector<ggml_fp16_t> h(f.size()); ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t)f.size()); ggml_backend_tensor_set(j.t, h.data(), 0, h.size() * 2); }
        else ggml_backend_tensor_set(j.t, f.data(), 0, f.size() * 4);
    }
    // residual VQ codebooks: embedding_sum / max(cluster_usage, 1e-5)
    for (int i = 0; i < 16; i++) {
        std::string p = D + "quantizer." + (i == 0 ? "rvq_first" : "rvq_rest") + ".vq.layers." + std::to_string(i == 0 ? 0 : i - 1) + "._codebook.";
        const StTensor* es = get(p + "embedding_sum"); const StTensor* cu = get(p + "cluster_usage");
        if (!es || !cu) { err = "missing codebook " + p; unload(); return false; }
        std::vector<float> e, u; st.toFloat(*es, e); st.toFloat(*cu, u);
        for (int r = 0; r < 2048; r++) { float d = std::max(u[r], 1e-5f); for (int c = 0; c < 256; c++) e[(size_t)r * 256 + c] /= d; }
        cb_[i] = std::move(e);
    }
    return true;
}

// ---------------------------------------------------------------- graph helpers, tensors are [T, C] unless noted
namespace {
TT toCT(ggml_context* c, TT x) { return ggml_cont(c, ggml_permute(c, x, 1, 0, 2, 3)); }      // [T,C] <-> [C,T]
TT biasT(ggml_context* c, TT y, TT b) { return ggml_add(c, y, ggml_reshape_2d(c, b, 1, b->ne[0])); }   // y [T,C] + b [C]

TT causalConv(ggml_context* c, TT x, TT w, TT b, int dil) {
    int pad = (int)(w->ne[0] - 1) * dil;
    TT xp = pad ? ggml_pad_ext(c, x, pad, 0, 0, 0, 0, 0, 0, 0) : x;
    TT y = ggml_conv_1d(c, w, xp, 1, 0, dil);
    return biasT(c, y, b);
}
TT transConv(ggml_context* c, TT x, TT w, TT b, int stride) {
    int K = (int)w->ne[0], pad = K - stride;
    TT y = ggml_conv_transpose_1d(c, w, x, stride, 0, 1);                  // [(T-1)*s+K, Cout]
    y = biasT(c, y, b);
    int64_t T = x->ne[0] * stride;
    return ggml_cont(c, ggml_view_2d(c, y, T, y->ne[1], y->nb[1], (size_t)pad * sizeof(float)));
}
TT snake(ggml_context* c, TT x, TT ea, TT ib) {
    TT s = ggml_sqr(c, ggml_sin(c, ggml_mul(c, x, ea)));
    return ggml_add(c, x, ggml_mul(c, s, ib));
}
// depthwise causal conv k=7: y[t,c] = sum_k w[k,c] * x[t-6+k, c]   (w is ggml [K, C])
TT dwConv(ggml_context* c, TT x, TT w, TT b) {
    int K = (int)w->ne[0];
    TT xp = ggml_pad_ext(c, x, K - 1, 0, 0, 0, 0, 0, 0, 0);
    TT wt = ggml_cont(c, ggml_permute(c, w, 1, 0, 2, 3));                  // [C, K]
    TT acc = nullptr;
    for (int k = 0; k < K; k++) {
        TT sl = ggml_view_2d(c, xp, x->ne[0], x->ne[1], xp->nb[1], (size_t)k * sizeof(float));
        TT wk = ggml_reshape_2d(c, ggml_view_1d(c, wt, w->ne[1], (size_t)k * w->ne[1] * sizeof(float)), 1, w->ne[1]);
        TT t = ggml_mul(c, sl, wk);
        acc = acc ? ggml_add(c, acc, t) : t;
    }
    return biasT(c, acc, b);
}
}  // namespace

bool SpeechDecoder::decodeChunk(const int* codes, int T, std::vector<float>& wav, std::string& err) {
    Compute& cm = Compute::get();
    const size_t nodes = 8192;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false) + (1 << 20));
    ggml_init_params ip = {meta.size(), meta.data(), true};
    ggml_context* c = ggml_init(ip);

    TT q1 = ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, T); ggml_set_input(q1);
    TT q2 = ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, T); ggml_set_input(q2);
    TT pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, T); ggml_set_input(pos);
    const int padT = (T + 63) / 64 * 64;
    TT mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, T, padT); ggml_set_input(mask);

    // RVQ output projections (1x1 convs), sum of the semantic and acoustic parts
    TT x = ggml_add(c, ggml_mul_mat(c, opFirst_, q1), ggml_mul_mat(c, opRest_, q2));      // [512, T]
    x = causalConv(c, toCT(c, x), preW_, preB_, 1);                                       // [T, 1024]

    // sliding-window transformer on [C, T]
    TT h = ggml_add(c, ggml_mul_mat(c, inW_, toCT(c, x)), inB_);                          // [512, T]
    for (auto& L : layers_) {
        TT n = ggml_mul(c, ggml_rms_norm(c, h, 1e-5f), L.ln1);
        TT q = ggml_reshape_3d(c, ggml_mul_mat(c, L.q, n), 64, 16, T);
        TT k = ggml_reshape_3d(c, ggml_mul_mat(c, L.k, n), 64, 16, T);
        TT v = ggml_reshape_3d(c, ggml_mul_mat(c, L.v, n), 64, 16, T);
        q = ggml_rope_ext(c, q, pos, nullptr, 64, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(c, k, pos, nullptr, 64, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        TT a = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), ggml_cast(c, ggml_permute(c, k, 0, 2, 1, 3), GGML_TYPE_F16),
                                   ggml_cast(c, ggml_permute(c, v, 0, 2, 1, 3), GGML_TYPE_F16), mask, 0.125f, 0.0f, 0.0f);
        a = ggml_mul_mat(c, L.o, ggml_reshape_2d(c, a, 1024, T));
        h = ggml_add(c, h, ggml_mul(c, a, L.s1));
        n = ggml_mul(c, ggml_rms_norm(c, h, 1e-5f), L.ln2);
        TT m = ggml_mul_mat(c, L.down, ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, L.gate, n)), ggml_mul_mat(c, L.up, n)));
        h = ggml_add(c, h, ggml_mul(c, m, L.s2));
    }
    h = ggml_mul(c, ggml_rms_norm(c, h, 1e-5f), normW_);
    h = ggml_add(c, ggml_mul_mat(c, outW_, h), outB_);                                    // [1024, T]
    x = toCT(c, h);                                                                       // [T, 1024]

    // 2x (ConvTranspose x2 + ConvNeXt)
    for (auto& u : up_) {
        x = transConv(c, x, u.tw, u.tb, 2);
        TT in = x;
        TT d = dwConv(c, x, u.dw, u.db);
        TT n = toCT(c, d);                                                                // [C, T]
        n = ggml_add(c, ggml_mul(c, ggml_norm(c, n, 1e-6f), u.nw), u.nb);
        n = ggml_add(c, ggml_mul_mat(c, u.p1w, n), u.p1b);
        n = ggml_gelu_erf(c, n);
        n = ggml_add(c, ggml_mul_mat(c, u.p2w, n), u.p2b);
        n = ggml_mul(c, n, u.gamma);
        x = ggml_add(c, in, toCT(c, n));
    }

    // waveform decoder
    x = causalConv(c, x, d0w_, d0b_, 1);
    for (auto& b : blocks_) {
        x = snake(c, x, b.sa, b.sb);
        x = transConv(c, x, b.tw, b.tb, b.stride);
        for (auto& u : b.u) {
            TT r = x;
            TT y = snake(c, x, u.a1, u.b1);
            y = causalConv(c, y, u.c1w, u.c1b, u.dil);
            y = snake(c, y, u.a2, u.b2);
            y = causalConv(c, y, u.c2w, u.c2b, 1);
            x = ggml_add(c, y, r);
        }
    }
    x = snake(c, x, fa_, fb_);
    x = causalConv(c, x, flw_, flb_, 1);
    x = ggml_clamp(c, x, -1.0f, 1.0f);
    ggml_set_output(x);

    ggml_cgraph* g = ggml_new_graph_custom(c, nodes, false);
    ggml_build_forward_expand(g, x);
    ggml_backend_sched_reset(cm.sched);
    if (!ggml_backend_sched_alloc_graph(cm.sched, g)) { err = "cannot allocate speech decoder graph"; ggml_free(c); return false; }

    std::vector<float> a1((size_t)256 * T, 0.0f), a2((size_t)256 * T, 0.0f);
    for (int t = 0; t < T; t++) {
        memcpy(&a1[(size_t)t * 256], &cb_[0][(size_t)codes[t * 16] * 256], 256 * 4);
        for (int g2 = 1; g2 < 16; g2++) {
            const float* e = &cb_[g2][(size_t)codes[t * 16 + g2] * 256];
            for (int k = 0; k < 256; k++) a2[(size_t)t * 256 + k] += e[k];
        }
    }
    ggml_backend_tensor_set(q1, a1.data(), 0, a1.size() * 4);
    ggml_backend_tensor_set(q2, a2.data(), 0, a2.size() * 4);
    std::vector<int32_t> p(T); for (int i = 0; i < T; i++) p[i] = i;
    ggml_backend_tensor_set(pos, p.data(), 0, T * 4);
    std::vector<ggml_fp16_t> mk((size_t)T * padT, ggml_fp32_to_fp16(-INFINITY));
    ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) if (j <= i && j > i - 72) mk[(size_t)i * T + j] = zero;
    ggml_backend_tensor_set(mask, mk.data(), 0, mk.size() * 2);

    if (ggml_backend_sched_graph_compute(cm.sched, g) != GGML_STATUS_SUCCESS) { err = "speech decoder compute failed"; ggml_free(c); return false; }
    wav.resize((size_t)ggml_nelements(x));
    ggml_backend_tensor_get(x, wav.data(), 0, wav.size() * 4);
    ggml_backend_sched_reset(cm.sched);
    ggml_free(c);
    return true;
}

bool SpeechDecoder::decode(const std::vector<int>& codes, int frames, std::vector<float>& wav, std::string& err) {
    if (!loaded()) { err = "speech decoder not loaded"; return false; }
    const int chunk = 300, left = 25;
    wav.clear();
    for (int start = 0; start < frames;) {
        int end = std::min(start + chunk, frames);
        int ctxN = (start - left > 0) ? left : start;
        std::vector<float> part;
        if (!decodeChunk(&codes[(size_t)(start - ctxN) * 16], end - (start - ctxN), part, err)) return false;
        wav.insert(wav.end(), part.begin() + (size_t)ctxN * kSamplesPerFrame, part.end());
        start = end;
    }
    return true;
}
