#include "krea2/dit.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gpu/compute.h"
#include "krea2/safetensors.h"

static std::string P = "model.diffusion_model.";      // tensor-name prefix; some repacks (e.g. krea2_turbo_fp8) have none

void Krea2DiT::unload() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
    delete (StFile*)st_;
    buf_ = nullptr; ctx_ = nullptr; backend_ = nullptr; st_ = nullptr;
    blocks_.clear(); layerwise_.clear(); refiner_.clear();
}

// W[out][in] += s * B[out][r] * A[r][in]   (LoRA keys: diffusion_model.<path>.lora_A.weight / lora_B.weight)
static std::atomic<int> g_loraCount{0};
static void applyLora(const StFile& lora, const std::string& weightName, std::vector<float>& W, int64_t in, int64_t out, float s, int& patched) {
    (void)patched;
    std::string stem = weightName.substr(6, weightName.size() - 6 - strlen(".weight"));      // drop "model." and ".weight"
    const StTensor* ta = lora.find(stem + ".lora_A.weight");
    const StTensor* tb = lora.find(stem + ".lora_B.weight");
    if (!ta || !tb || ta->shape.size() != 2 || tb->shape.size() != 2) return;
    int64_t r = ta->shape[0];
    if (ta->shape[1] != in || tb->shape[0] != out || tb->shape[1] != r) return;
    float alpha = 0; float scale = s;
    if (lora.scalar(stem + ".alpha", alpha)) scale = s * alpha / (float)r;
    std::vector<float> A, B;
    lora.toFloat(*ta, A); lora.toFloat(*tb, B);
    for (int64_t o = 0; o < out; o++) {
        float* w = W.data() + (size_t)o * in;
        for (int64_t k = 0; k < r; k++) {
            float c = scale * B[(size_t)o * r + k];
            const float* a = A.data() + (size_t)k * in;
            for (int64_t i = 0; i < in; i++) w[i] += c * a[i];
        }
    }
    g_loraCount++;
}

namespace {
struct LoadJob { ggml_tensor* t; std::string name; int mode; };   // 0 fp8/bf16 matrix, 1 float vector, 2 float vector + 1
}

bool Krea2DiT::load(const std::string& path, int threads, std::string& err, const std::function<void(float)>& progress,
                     const std::string& loraPath, float loraStrength) {
    unload();
    threads_ = threads > 0 ? threads : 8;
    StFile* st = new StFile(); st_ = st;
    if (!st->open(path, err)) { unload(); return false; }
    for (const char* pre : {"model.diffusion_model.", "diffusion_model.", ""}) if (st->find(std::string(pre) + "blocks.0.mod.lin")) { P = pre; break; }
    StFile lora; bool useLora = false;
    loraPatched = 0;
    if (!loraPath.empty() && loraStrength != 0.0f) {
        if (!lora.open(loraPath, err)) { unload(); return false; }
        useLora = true;
    }
    if (!Compute::get().init(threads_, err)) { unload(); return false; }
    backend_ = Compute::get().cpu;
    ggml_init_params ip = {ggml_tensor_overhead() * 1400, nullptr, true};
    ctx_ = ggml_init(ip);

    std::vector<LoadJob> jobs;
    bool okAll = true;
    auto fail = [&](const std::string& n) { if (okAll) err = "missing or unsupported tensor " + n; okAll = false; return (ggml_tensor*)nullptr; };
    auto mat = [&](const std::string& rel) -> ggml_tensor* {
        std::string n = P + rel; const StTensor* t = st->find(n);
        if (!t || t->shape.size() != 2) return fail(n);
        ggml_type ty = t->dtype == "F8_E4M3" ? (getenv("BAKERY_DIT_F16") ? GGML_TYPE_F16 : GGML_TYPE_Q8_0) : (t->dtype == "BF16" ? GGML_TYPE_BF16 : GGML_TYPE_COUNT);
        if (ty == GGML_TYPE_COUNT) return fail(n);
        ggml_tensor* g = ggml_new_tensor_2d(ctx_, ty, t->shape[1], t->shape[0]);
        ggml_set_name(g, n.c_str()); jobs.push_back({g, n, 0}); return g;
    };
    auto vec = [&](const std::string& rel, bool plusOne) -> ggml_tensor* {
        std::string n = P + rel; const StTensor* t = st->find(n);
        if (!t) return fail(n);
        ggml_tensor* g = t->shape.size() == 1 ? ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, t->shape[0])
                                              : ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, t->shape[1], t->shape[0]);
        ggml_set_name(g, n.c_str()); jobs.push_back({g, n, plusOne ? 2 : 1}); return g;
    };
    auto block = [&](const std::string& p, bool withMod) {
        Blk b;
        if (withMod) b.mod = vec(p + "mod.lin", false);
        b.pre = vec(p + "prenorm.scale", true); b.post = vec(p + "postnorm.scale", true);
        b.qn = vec(p + "attn.qknorm.qnorm.scale", true); b.kn = vec(p + "attn.qknorm.knorm.scale", true);
        b.wq = mat(p + "attn.wq.weight"); b.wk = mat(p + "attn.wk.weight"); b.wv = mat(p + "attn.wv.weight");
        b.gate = mat(p + "attn.gate.weight"); b.wo = mat(p + "attn.wo.weight");
        b.mg = mat(p + "mlp.gate.weight"); b.mu = mat(p + "mlp.up.weight"); b.md = mat(p + "mlp.down.weight");
        return b;
    };
    for (int i = 0; i < kBlocks; i++) blocks_.push_back(block("blocks." + std::to_string(i) + ".", true));
    for (int i = 0; i < 2; i++) layerwise_.push_back(block("txtfusion.layerwise_blocks." + std::to_string(i) + ".", false));
    for (int i = 0; i < 2; i++) refiner_.push_back(block("txtfusion.refiner_blocks." + std::to_string(i) + ".", false));
    firstW = mat("first.weight"); firstB = vec("first.bias", false);
    tm0W = mat("tmlp.0.weight"); tm0B = vec("tmlp.0.bias", false); tm2W = mat("tmlp.2.weight"); tm2B = vec("tmlp.2.bias", false);
    tpW = mat("tproj.1.weight"); tpB = vec("tproj.1.bias", false);
    projector = vec("txtfusion.projector.weight", false);
    txtNorm = vec("txtmlp.0.scale", true); txt1W = mat("txtmlp.1.weight"); txt1B = vec("txtmlp.1.bias", false);
    txt3W = mat("txtmlp.3.weight"); txt3B = vec("txtmlp.3.bias", false);
    lastNorm = vec("last.norm.scale", true); lastLin = vec("last.modulation.lin", false);
    lastW = mat("last.linear.weight"); lastB = vec("last.linear.bias", false);
    if (!okAll) { unload(); return false; }

    buf_ = Compute::get().allocWeights(ctx_);
    if (!buf_) { err = "out of memory allocating DiT weights (needs ~14 GB)"; unload(); return false; }

    std::atomic<size_t> next{0}, done{0};
    std::atomic<bool> bad{false};
    auto worker = [&]() {
        std::vector<float> f; std::vector<uint8_t> q;
        for (;;) {
            size_t i = next++;
            if (i >= jobs.size() || bad) return;
            LoadJob& j = jobs[i];
            const StTensor* t = st->find(j.name);
            if (j.mode == 0) {
                if (t->dtype == "F8_E4M3") {
                    float scale = 1.0f; st->scalar(j.name.substr(0, j.name.size() - 6) + "weight_scale", scale);
                    st->toFloat(*t, f, scale);
                    if (useLora) applyLora(lora, "model.diffusion_model." + j.name.substr(P.size()), f, j.t->ne[0], j.t->ne[1], loraStrength, loraPatched);
                    q.resize(ggml_nbytes(j.t));
                    if (j.t->type == GGML_TYPE_F16) ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t*)q.data(), f.size());
                    else ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, j.t->ne[1], j.t->ne[0], nullptr);
                    ggml_backend_tensor_set(j.t, q.data(), 0, q.size());
                } else ggml_backend_tensor_set(j.t, t->data, 0, ggml_nbytes(j.t));
            } else {
                if (!st->toFloat(*t, f)) { bad = true; return; }
                if (j.mode == 2) for (auto& v : f) v += 1.0f;
                ggml_backend_tensor_set(j.t, f.data(), 0, f.size() * sizeof(float));
            }
            size_t d = ++done;
            if (progress && d % 32 == 0) progress((float)d / jobs.size());
        }
    };
    ggml_quantize_init(GGML_TYPE_Q8_0);
    std::vector<std::thread> pool;
    for (int i = 0; i < threads_; i++) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    if (bad) { err = "unsupported dtype in DiT file"; unload(); return false; }
    loraPatched = g_loraCount.exchange(0);
    if (progress) progress(1.0f);
    return true;
}

// ---------------------------------------------------------------- graph pieces
namespace {
ggml_tensor* addBias(ggml_context* c, ggml_tensor* y, ggml_tensor* b) { return ggml_add(c, y, b); }

// rotate interleaved pairs: (x0,x1) -> (x0 c - x1 s, x0 s + x1 c); cos/sin are [1, 64, 1, N]
ggml_tensor* ropePairs(ggml_context* c, ggml_tensor* t /*[128,H,N]*/, ggml_tensor* cs, ggml_tensor* sn) {
    int64_t H = t->ne[1], N = t->ne[2];
    ggml_tensor* t4 = ggml_reshape_4d(c, t, 2, 64, H, N);
    ggml_tensor* x0 = ggml_view_4d(c, t4, 1, 64, H, N, t4->nb[1], t4->nb[2], t4->nb[3], 0);
    ggml_tensor* x1 = ggml_view_4d(c, t4, 1, 64, H, N, t4->nb[1], t4->nb[2], t4->nb[3], t4->nb[0]);
    ggml_tensor* o0 = ggml_sub(c, ggml_mul(c, x0, cs), ggml_mul(c, x1, sn));
    ggml_tensor* o1 = ggml_add(c, ggml_mul(c, x0, sn), ggml_mul(c, x1, cs));
    return ggml_reshape_3d(c, ggml_concat(c, o0, o1, 0), 128, H, N);
}

// x: [D, N, B]. Self attention with per-head QK RMSNorm, optional RoPE, GQA and sigmoid gate.
ggml_tensor* attention(ggml_context* c, const Krea2DiT::Blk& b, ggml_tensor* h, int heads, int kvheads,
                       ggml_tensor* cs, ggml_tensor* sn) {
    const int64_t N = h->ne[1], B = h->ne[2];
    ggml_tensor* q = ggml_reshape_4d(c, ggml_mul_mat(c, b.wq, h), 128, heads, N, B);
    ggml_tensor* k = ggml_reshape_4d(c, ggml_mul_mat(c, b.wk, h), 128, kvheads, N, B);
    ggml_tensor* v = ggml_reshape_4d(c, ggml_mul_mat(c, b.wv, h), 128, kvheads, N, B);
    ggml_tensor* g = ggml_sigmoid(c, ggml_mul_mat(c, b.gate, h));
    q = ggml_mul(c, ggml_rms_norm(c, q, 1e-5f), b.qn);
    k = ggml_mul(c, ggml_rms_norm(c, k, 1e-5f), b.kn);
    if (cs) {
        q = ggml_reshape_4d(c, ropePairs(c, ggml_reshape_3d(c, q, 128, heads, N), cs, sn), 128, heads, N, 1);
        k = ggml_reshape_4d(c, ropePairs(c, ggml_reshape_3d(c, k, 128, kvheads, N), cs, sn), 128, kvheads, N, 1);
    }
    ggml_tensor* qp = ggml_permute(c, q, 0, 2, 1, 3);
    ggml_tensor* kp = ggml_cast(c, ggml_permute(c, k, 0, 2, 1, 3), GGML_TYPE_F16);
    ggml_tensor* vp = ggml_cast(c, ggml_permute(c, v, 0, 2, 1, 3), GGML_TYPE_F16);
    ggml_tensor* a = ggml_flash_attn_ext(c, qp, kp, vp, nullptr, 1.0f / std::sqrt(128.0f), 0.0f, 0.0f);
    a = ggml_reshape_3d(c, a, 128 * heads, N, B);
    a = ggml_mul(c, a, g);
    return ggml_mul_mat(c, b.wo, a);
}

ggml_tensor* swiglu(ggml_context* c, const Krea2DiT::Blk& b, ggml_tensor* h) {
    ggml_tensor* m = ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, b.mg, h)), ggml_mul_mat(c, b.mu, h));
    return ggml_mul_mat(c, b.md, m);
}

// text-fusion block: plain pre-norm residual block, no modulation
ggml_tensor* fusionBlock(ggml_context* c, const Krea2DiT::Blk& b, ggml_tensor* x) {
    x = ggml_add(c, x, attention(c, b, ggml_mul(c, ggml_rms_norm(c, x, 1e-5f), b.pre), Krea2DiT::kTxtHeads, Krea2DiT::kTxtHeads, nullptr, nullptr));
    return ggml_add(c, x, swiglu(c, b, ggml_mul(c, ggml_rms_norm(c, x, 1e-5f), b.post)));
}
}  // namespace

bool Krea2DiT::forward(const std::vector<float>& latent, int H, int W, float sigma,
                       const std::vector<float>& ctx, int seq, std::vector<float>& out, std::string& err) {
    if (!loaded()) { err = "DiT not loaded"; return false; }
    if (H % 2 || W % 2) { err = "latent size must be even"; return false; }
    const int hp = H / 2, wp = W / 2, nImg = hp * wp, N = seq + nImg;

    const size_t nodes = 6144;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false) + (1 << 20));
    ggml_init_params ip = {meta.size(), meta.data(), true};
    ggml_context* c = ggml_init(ip);

    std::vector<std::pair<std::string, ggml_tensor*>> taps;
    static const bool statsOn = getenv("BAKERY_DIT_STATS") != nullptr;                // prints mean/std/NaN count of a few intermediate tensors
    auto tap = [&](const char* n, ggml_tensor* x) { if (debug || statsOn) { ggml_set_output(x); taps.push_back({n, x}); } };
    ggml_tensor* imgIn = ggml_new_tensor_2d(c, GGML_TYPE_F32, 64, nImg); ggml_set_input(imgIn);
    ggml_tensor* ctxIn = ggml_new_tensor_3d(c, GGML_TYPE_F32, kTxtDim, kTxtLayers, seq); ggml_set_input(ctxIn);
    ggml_tensor* tIn = ggml_new_tensor_1d(c, GGML_TYPE_F32, 256); ggml_set_input(tIn);
    ggml_tensor* cosIn = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, 64, 1, N); ggml_set_input(cosIn);
    ggml_tensor* sinIn = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, 64, 1, N); ggml_set_input(sinIn);

    // timestep -> t (6144) and per-block modulation vector (6*6144)
    ggml_tensor* t = addBias(c, ggml_mul_mat(c, tm0W, tIn), tm0B);
    t = addBias(c, ggml_mul_mat(c, tm2W, ggml_gelu(c, t)), tm2B);
    ggml_tensor* tvec = addBias(c, ggml_mul_mat(c, tpW, ggml_gelu(c, t)), tpB);
    tap("t", t); tap("tvec", tvec);

    // text: 12-layer fusion -> refiner -> projection to model width
    ggml_tensor* tx = ctxIn;
    for (auto& b : layerwise_) tx = fusionBlock(c, b, tx);
    tx = ggml_cont(c, ggml_permute(c, tx, 1, 0, 2, 3));                    // [12, 2560, seq]
    tx = ggml_reshape_2d(c, ggml_mul_mat(c, projector, tx), kTxtDim, seq);
    for (auto& b : refiner_) tx = fusionBlock(c, b, tx);
    tap("fusion", tx);
    tx = ggml_mul(c, ggml_rms_norm(c, tx, 1e-5f), txtNorm);
    tx = addBias(c, ggml_mul_mat(c, txt1W, tx), txt1B);
    tx = addBias(c, ggml_mul_mat(c, txt3W, ggml_gelu(c, tx)), txt3B);
    tap("txt", tx);

    ggml_tensor* img = addBias(c, ggml_mul_mat(c, firstW, imgIn), firstB);
    tap("img0", img);
    ggml_tensor* x = ggml_concat(c, tx, img, 1);                           // [6144, N]

    for (int i = 0; i < kBlocks; i++) {
        const Blk& b = blocks_[i];
        ggml_tensor* mod = ggml_add(c, tvec, b.mod);
        auto chunk = [&](int k) { return ggml_view_1d(c, mod, kDim, (size_t)k * kDim * sizeof(float)); };
        ggml_tensor *preS = chunk(0), *preH = chunk(1), *preG = chunk(2), *postS = chunk(3), *postH = chunk(4), *postG = chunk(5);
        ggml_tensor* h = ggml_mul(c, ggml_rms_norm(c, x, 1e-5f), b.pre);
        h = ggml_add(c, ggml_mul(c, h, ggml_scale_bias(c, preS, 1.0f, 1.0f)), preH);
        x = ggml_add(c, x, ggml_mul(c, attention(c, b, ggml_reshape_3d(c, h, kDim, N, 1), kHeads, kKvHeads, cosIn, sinIn), preG));
        h = ggml_mul(c, ggml_rms_norm(c, x, 1e-5f), b.post);
        h = ggml_add(c, ggml_mul(c, h, ggml_scale_bias(c, postS, 1.0f, 1.0f)), postH);
        x = ggml_add(c, x, ggml_mul(c, swiglu(c, b, h), postG));
        if ((debug || statsOn) && (i <= 1 || i % 4 == 3)) { static std::vector<std::string> names; names.push_back("blk" + std::to_string(i)); tap(names.back().c_str(), x); }
    }

    // final layer on the image tokens only
    ggml_tensor* xi = ggml_view_2d(c, x, kDim, nImg, x->nb[1], (size_t)seq * x->nb[1]);
    ggml_tensor* lScale = ggml_add(c, t, ggml_view_1d(c, lastLin, kDim, 0));
    ggml_tensor* lShift = ggml_add(c, t, ggml_view_1d(c, lastLin, kDim, (size_t)kDim * sizeof(float)));
    ggml_tensor* h = ggml_mul(c, ggml_rms_norm(c, xi, 1e-5f), lastNorm);
    h = ggml_add(c, ggml_mul(c, h, ggml_scale_bias(c, lScale, 1.0f, 1.0f)), lShift);
    ggml_tensor* res = addBias(c, ggml_mul_mat(c, lastW, h), lastB);        // [64, nImg]
    ggml_set_output(res);

    ggml_cgraph* g = ggml_new_graph_custom(c, nodes, false);
    ggml_build_forward_expand(g, res);
    for (auto& tp : taps) ggml_build_forward_expand(g, tp.second);
    Compute& cm = Compute::get();
    ggml_backend_sched_reset(cm.sched);
    if (!ggml_backend_sched_alloc_graph(cm.sched, g)) { err = "cannot allocate DiT graph"; ggml_free(c); return false; }

    // inputs
    std::vector<float> tok((size_t)nImg * 64);
    for (int r = 0; r < hp; r++) for (int cc = 0; cc < wp; cc++) for (int ch = 0; ch < 16; ch++) for (int ph = 0; ph < 2; ph++) for (int pw = 0; pw < 2; pw++)
        tok[((size_t)(r * wp + cc)) * 64 + ch * 4 + ph * 2 + pw] = latent[((size_t)ch * H + (r * 2 + ph)) * W + (cc * 2 + pw)];
    ggml_backend_tensor_set(imgIn, tok.data(), 0, tok.size() * sizeof(float));
    ggml_backend_tensor_set(ctxIn, ctx.data(), 0, ctx.size() * sizeof(float));
    std::vector<float> te(256);
    for (int k = 0; k < 128; k++) {
        double a = 1000.0 * sigma * std::exp(-std::log(10000.0) * k / 128.0);
        te[k] = (float)std::cos(a); te[128 + k] = (float)std::sin(a);
    }
    ggml_backend_tensor_set(tIn, te.data(), 0, te.size() * sizeof(float));
    // RoPE tables: axes (0: always 0), (1: row), (2: col) with dims 32/48/48, theta 1000; text tokens at 0
    std::vector<float> cs((size_t)N * 64, 1.0f), sn((size_t)N * 64, 0.0f);
    const int axisDim[3] = {32, 48, 48};
    for (int r = 0; r < hp; r++) for (int cc = 0; cc < wp; cc++) {
        size_t tokIdx = (size_t)seq + r * wp + cc;
        double pos[3] = {0.0, (double)r, (double)cc};
        int pair = 0;
        for (int ax = 0; ax < 3; ax++) for (int k = 0; k < axisDim[ax] / 2; k++, pair++) {
            double om = 1.0 / std::pow(1000.0, (double)(2 * k) / axisDim[ax]);
            cs[tokIdx * 64 + pair] = (float)std::cos(pos[ax] * om); sn[tokIdx * 64 + pair] = (float)std::sin(pos[ax] * om);
        }
    }
    ggml_backend_tensor_set(cosIn, cs.data(), 0, cs.size() * sizeof(float));
    ggml_backend_tensor_set(sinIn, sn.data(), 0, sn.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(cm.sched, g) != GGML_STATUS_SUCCESS) { err = "DiT compute failed"; ggml_free(c); return false; }

    dump.clear();
    for (auto& tp : taps) {
        std::vector<float> v(ggml_nelements(tp.second)); ggml_backend_tensor_get(tp.second, v.data(), 0, v.size() * 4);
        if (statsOn) { double s = 0, q = 0; size_t bad = 0; for (float f : v) { if (!std::isfinite(f)) { bad++; continue; } s += f; q += (double)f * f; } double m = s / v.size(); fprintf(stderr, "[dit N=%d] %-8s mean %9.4f std %9.4f nonfinite %zu\n", N, tp.first.c_str(), m, std::sqrt(std::max(0.0, q / v.size() - m * m)), bad); }
        if (debug) dump.push_back({tp.first, v});
    }
    std::vector<float> o((size_t)nImg * 64);
    ggml_backend_tensor_get(res, o.data(), 0, o.size() * sizeof(float));
    out.assign((size_t)16 * H * W, 0.0f);
    for (int r = 0; r < hp; r++) for (int cc = 0; cc < wp; cc++) for (int ch = 0; ch < 16; ch++) for (int ph = 0; ph < 2; ph++) for (int pw = 0; pw < 2; pw++)
        out[((size_t)ch * H + (r * 2 + ph)) * W + (cc * 2 + pw)] = o[((size_t)(r * wp + cc)) * 64 + ch * 4 + ph * 2 + pw];
    ggml_free(c);
    return true;
}
