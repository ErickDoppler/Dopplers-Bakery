#include "tts/codec_encoder.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <thread>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gpu/compute.h"
#include "krea2/safetensors.h"

using TT = ggml_tensor*;

void CodecEncoder::unload() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
    buf_ = nullptr; ctx_ = nullptr; layers_.clear();
    for (auto& v : inProj_) { v.clear(); v.shrink_to_fit(); }
    for (auto& v : cb_) { v.clear(); v.shrink_to_fit(); }
}

bool CodecEncoder::load(const std::string& path, std::string& err) {
    unload();
    StFile st;
    if (!st.open(path, err)) return false;
    std::string e2;
    if (!Compute::get().init(0, e2)) { err = e2; return false; }
    ggml_init_params ip = {ggml_tensor_overhead() * 300, nullptr, true};
    ctx_ = ggml_init(ip);
    struct Job { TT t; std::string name; int mode; };      // 0 conv kernel F16, 1 F32 matrix/vector, 2 matrix F16
    std::vector<Job> jobs;
    bool ok = true;
    auto get = [&](const std::string& n) -> const StTensor* { const StTensor* t = st.find(n); if (!t && ok) { err = "missing tensor " + n; ok = false; } return t; };
    auto kern = [&](const std::string& n) -> TT {
        const StTensor* t = get(n); if (!t) return nullptr;
        TT g = ggml_new_tensor_3d(ctx_, GGML_TYPE_F16, t->shape[2], t->shape[1], t->shape[0]); jobs.push_back({g, n, 0}); return g;
    };
    auto vec = [&](const std::string& n) -> TT {
        const StTensor* t = get(n); if (!t) return nullptr;
        TT g = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, t->shape[0]); jobs.push_back({g, n, 1}); return g;
    };
    auto mat = [&](const std::string& n) -> TT {
        const StTensor* t = get(n); if (!t) return nullptr;
        TT g = ggml_new_tensor_2d(ctx_, GGML_TYPE_F16, t->shape[1], t->shape[0]); jobs.push_back({g, n, 2}); return g;
    };
    auto conv = [&](const std::string& p) { Conv c; c.w = kern(p + ".weight"); c.b = vec(p + ".bias"); return c; };
    const std::string E = "encoder.encoder.layers.";
    c0_ = conv(E + "0.conv");
    for (int i = 0; i < 4; i++) {
        int base = 1 + 3 * i;
        res_[i].c1 = conv(E + std::to_string(base) + ".block.1.conv");
        res_[i].c2 = conv(E + std::to_string(base) + ".block.3.conv");
        down_[i] = conv(E + std::to_string(base + 2) + ".conv");
    }
    last_ = conv(E + "14.conv");
    downsample_ = kern("encoder.downsample.conv.weight");
    for (int i = 0; i < 8; i++) {
        std::string p = "encoder.encoder_transformer.layers." + std::to_string(i) + ".";
        Layer L;
        L.n1w = vec(p + "input_layernorm.weight"); L.n1b = vec(p + "input_layernorm.bias");
        L.n2w = vec(p + "post_attention_layernorm.weight"); L.n2b = vec(p + "post_attention_layernorm.bias");
        L.q = mat(p + "self_attn.q_proj.weight"); L.k = mat(p + "self_attn.k_proj.weight"); L.v = mat(p + "self_attn.v_proj.weight"); L.o = mat(p + "self_attn.o_proj.weight");
        L.fc1 = mat(p + "mlp.fc1.weight"); L.fc2 = mat(p + "mlp.fc2.weight");
        L.s1 = vec(p + "self_attn_layer_scale.scale"); L.s2 = vec(p + "mlp_layer_scale.scale");
        layers_.push_back(L);
    }
    if (!ok) { unload(); return false; }
    buf_ = Compute::get().allocHostWeights(ctx_);
    if (!buf_) { err = "cannot allocate codec encoder weights"; unload(); return false; }
    std::vector<float> f;
    for (auto& j : jobs) {
        st.toFloat(*st.find(j.name), f);
        if (j.mode == 1) ggml_backend_tensor_set(j.t, f.data(), 0, f.size() * 4);
        else { std::vector<ggml_fp16_t> h(f.size()); ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t)f.size()); ggml_backend_tensor_set(j.t, h.data(), 0, h.size() * 2); }
    }
    // host side: quantizer projections and codebooks
    const std::string Q = "encoder.quantizer.";
    const char* names[2] = {"semantic_residual_vector_quantizer.", "acoustic_residual_vector_quantizer."};
    for (int s = 0; s < 2; s++) {
        const StTensor* w = get(Q + names[s] + "input_proj.weight");
        if (!w) { unload(); return false; }
        st.toFloat(*w, inProj_[s]);                                   // [256, 512, 1] row-major = 256 x 512
    }
    for (int i = 0; i < 16; i++) {
        std::string p = Q + names[i == 0 ? 0 : 1] + "layers." + std::to_string(i == 0 ? 0 : i - 1) + ".codebook.";
        const StTensor* es = get(p + "embed_sum"); const StTensor* cu = get(p + "cluster_usage");
        if (!es || !cu) { unload(); return false; }
        std::vector<float> e, u; st.toFloat(*es, e); st.toFloat(*cu, u);
        for (int r = 0; r < 2048; r++) { float d = std::max(u[r], 1e-5f); for (int c = 0; c < 256; c++) e[(size_t)r * 256 + c] /= d; }
        cb_[i] = std::move(e);
    }
    return true;
}

namespace {
TT biasT(ggml_context* c, TT y, TT b) { return ggml_add(c, y, ggml_reshape_2d(c, b, 1, b->ne[0])); }
TT toCT(ggml_context* c, TT x) { return ggml_cont(c, ggml_permute(c, x, 1, 0, 2, 3)); }

// causal conv on [T, C]: left padding (k-1)*dil... generalised: padding_total = k_eff - stride, extra on the right so the last frame is complete
TT causalConv(ggml_context* c, TT x, TT w, TT b, int stride, int dil) {
    const int k = (int)w->ne[0], keff = (k - 1) * dil + 1, total = keff - stride;
    const int L = (int)x->ne[0];
    const int extra = ((L + stride - 1) / stride) * stride - L;                    // ceil(L/stride)*stride - L
    TT xp = (total > 0 || extra > 0) ? ggml_pad_ext(c, x, total, extra, 0, 0, 0, 0, 0, 0) : x;
    TT y = ggml_conv_1d(c, w, xp, stride, 0, dil);
    return b ? biasT(c, y, b) : y;
}
}  // namespace

bool CodecEncoder::encode(const std::vector<float>& wav, std::vector<int>& codes, int& frames, std::string& err) {
    if (!loaded()) { err = "codec encoder not loaded"; return false; }
    const int L = (int)wav.size();
    if (L < 1920) { err = "reference audio too short"; return false; }
    Compute& cm = Compute::get();
    const size_t nodes = 4096;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false) + (1 << 20));
    ggml_init_params ip = {meta.size(), meta.data(), true};
    ggml_context* c = ggml_init(ip);

    TT in = ggml_new_tensor_2d(c, GGML_TYPE_F32, L, 1); ggml_set_input(in);
    TT x = causalConv(c, in, c0_.w, c0_.b, 1, 1);
    const int ratios[4] = {4, 5, 6, 8};
    for (int i = 0; i < 4; i++) {
        TT h = ggml_elu(c, x);
        h = causalConv(c, h, res_[i].c1.w, res_[i].c1.b, 1, 1);
        h = ggml_elu(c, h);
        h = causalConv(c, h, res_[i].c2.w, res_[i].c2.b, 1, 1);
        x = ggml_add(c, x, h);
        x = ggml_elu(c, x);
        x = causalConv(c, x, down_[i].w, down_[i].b, ratios[i], 1);
    }
    x = ggml_elu(c, x);
    x = causalConv(c, x, last_.w, last_.b, 1, 1);                              // [T25, 512]
    const int T = (int)x->ne[0];

    // 8-layer transformer, sliding window 250, rope theta 10000
    TT pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, T); ggml_set_input(pos);
    const int padT = (T + 63) / 64 * 64;
    TT mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, T, padT); ggml_set_input(mask);
    TT h = toCT(c, x);                                                         // [512, T]
    for (auto& Ly : layers_) {
        TT n = ggml_add(c, ggml_mul(c, ggml_norm(c, h, 1e-5f), Ly.n1w), Ly.n1b);
        TT q = ggml_reshape_3d(c, ggml_mul_mat(c, Ly.q, n), 64, 8, T);
        TT k = ggml_reshape_3d(c, ggml_mul_mat(c, Ly.k, n), 64, 8, T);
        TT v = ggml_reshape_3d(c, ggml_mul_mat(c, Ly.v, n), 64, 8, T);
        q = ggml_rope_ext(c, q, pos, nullptr, 64, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(c, k, pos, nullptr, 64, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        TT a = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), ggml_cast(c, ggml_permute(c, k, 0, 2, 1, 3), GGML_TYPE_F16),
                                   ggml_cast(c, ggml_permute(c, v, 0, 2, 1, 3), GGML_TYPE_F16), mask, 0.125f, 0.0f, 0.0f);
        a = ggml_mul_mat(c, Ly.o, ggml_reshape_2d(c, a, 512, T));
        h = ggml_add(c, h, ggml_mul(c, a, Ly.s1));
        n = ggml_add(c, ggml_mul(c, ggml_norm(c, h, 1e-5f), Ly.n2w), Ly.n2b);
        TT m = ggml_mul_mat(c, Ly.fc2, ggml_gelu_erf(c, ggml_mul_mat(c, Ly.fc1, n)));
        h = ggml_add(c, h, ggml_mul(c, m, Ly.s2));
    }
    x = toCT(c, h);                                                            // [T, 512]

    // downsample 25 Hz -> 12.5 Hz: k4 stride 2, replicate padding (2 left, extra right)
    {
        const int extra = ((T + 1) / 2) * 2 - T;
        TT first = ggml_view_2d(c, x, 1, 512, x->nb[1], 0);
        TT lastc = ggml_view_2d(c, x, 1, 512, x->nb[1], (size_t)(T - 1) * sizeof(float));
        TT left = ggml_repeat(c, ggml_cont(c, first), ggml_new_tensor_2d(c, GGML_TYPE_F32, 2, 512));
        TT xp = ggml_concat(c, left, x, 0);
        if (extra > 0) xp = ggml_concat(c, xp, ggml_repeat(c, ggml_cont(c, lastc), ggml_new_tensor_2d(c, GGML_TYPE_F32, extra, 512)), 0);
        x = ggml_conv_1d(c, downsample_, xp, 2, 0, 1);                         // [T/2, 512]
    }
    ggml_set_output(x);

    ggml_cgraph* g = ggml_new_graph_custom(c, nodes, false);
    ggml_build_forward_expand(g, x);
    ggml_backend_sched_reset(cm.schedCpu);
    if (!ggml_backend_sched_alloc_graph(cm.schedCpu, g)) { err = "cannot allocate codec encoder graph"; ggml_free(c); return false; }
    ggml_backend_tensor_set(in, wav.data(), 0, wav.size() * 4);
    std::vector<int32_t> p(T); std::iota(p.begin(), p.end(), 0);
    ggml_backend_tensor_set(pos, p.data(), 0, T * 4);
    std::vector<ggml_fp16_t> mk((size_t)T * padT, ggml_fp32_to_fp16(-INFINITY));
    ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) if (j <= i && j > i - 250) mk[(size_t)i * T + j] = zero;
    ggml_backend_tensor_set(mask, mk.data(), 0, mk.size() * 2);
    if (ggml_backend_sched_graph_compute(cm.schedCpu, g) != GGML_STATUS_SUCCESS) { err = "codec encoder compute failed"; ggml_free(c); return false; }

    const int T2 = (int)x->ne[0];
    std::vector<float> emb((size_t)T2 * 512);                                  // [T2, 512] time-fastest -> emb[ch*T2 + t]
    ggml_backend_tensor_get(x, emb.data(), 0, emb.size() * 4);
    dbgEmb = emb;
    ggml_backend_sched_reset(cm.sched);
    ggml_free(c);

    // residual vector quantization on the host: semantic chain (1 layer) and acoustic chain (15 layers), same input embeddings
    frames = std::min(T2, (L + 1919) / 1920);
    codes.assign((size_t)frames * 16, 0);
    unsigned nth = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    auto work = [&](int t0, int t1) {
        std::vector<float> e(512), r(256);
        for (int t = t0; t < t1; t++) {
            for (int ch = 0; ch < 512; ch++) e[ch] = emb[(size_t)ch * T2 + t];
            for (int s = 0; s < 2; s++) {
                const std::vector<float>& W = inProj_[s];
                for (int o = 0; o < 256; o++) { double acc = 0; const float* w = &W[(size_t)o * 512]; for (int i = 0; i < 512; i++) acc += (double)w[i] * e[i]; r[o] = (float)acc; }
                int nl = s == 0 ? 1 : 15;
                for (int l = 0; l < nl; l++) {
                    const std::vector<float>& cb = cb_[s == 0 ? 0 : 1 + l];
                    int best = 0; float bd = 1e30f;
                    for (int k = 0; k < 2048; k++) {
                        const float* ev = &cb[(size_t)k * 256]; float d = 0;
                        for (int i = 0; i < 256; i++) { float df = r[i] - ev[i]; d += df * df; }
                        if (d < bd) { bd = d; best = k; }
                    }
                    codes[(size_t)t * 16 + (s == 0 ? 0 : 1 + l)] = best;
                    const float* ev = &cb[(size_t)best * 256];
                    for (int i = 0; i < 256; i++) r[i] -= ev[i];
                }
            }
        }
    };
    std::vector<std::thread> pool;
    int per = (frames + (int)nth - 1) / (int)nth;
    for (unsigned i = 0; i < nth; i++) { int a = (int)i * per, b = std::min(frames, a + per); if (a < b) pool.emplace_back(work, a, b); }
    for (auto& th : pool) th.join();
    return true;
}
