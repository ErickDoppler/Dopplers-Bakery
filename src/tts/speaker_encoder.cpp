#include "tts/speaker_encoder.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gpu/compute.h"
#include "krea2/safetensors.h"

using TT = ggml_tensor*;

// ---------------------------------------------------------------- mel spectrogram (librosa slaney mel + torch.stft)
namespace {
double hzToMel(double f) {
    const double fsp = 200.0 / 3.0, minLogHz = 1000.0, minLogMel = minLogHz / fsp, logstep = std::log(6.4) / 27.0;
    return f >= minLogHz ? minLogMel + std::log(f / minLogHz) / logstep : f / fsp;
}
double melToHz(double m) {
    const double fsp = 200.0 / 3.0, minLogHz = 1000.0, minLogMel = minLogHz / fsp, logstep = std::log(6.4) / 27.0;
    return m >= minLogMel ? minLogHz * std::exp(logstep * (m - minLogMel)) : fsp * m;
}
void fft(std::vector<std::complex<double>>& a) {     // in-place radix-2
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / (double)len;
        std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1);
            for (size_t k = 0; k < len / 2; k++) {
                auto u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v; a[i + k + len / 2] = u - v; w *= wl;
            }
        }
    }
}
}  // namespace

void qwenMelSpectrogram(const std::vector<float>& y, std::vector<float>& mel, int& frames) {
    const int nfft = 1024, hop = 256, nMel = 128, sr = 24000, nBins = nfft / 2 + 1;
    const double fmin = 0.0, fmax = 12000.0;
    // mel filterbank (slaney scale, slaney norm)
    std::vector<double> melF(nMel + 2), fftF(nBins);
    double mlo = hzToMel(fmin), mhi = hzToMel(fmax);
    for (int i = 0; i < nMel + 2; i++) melF[i] = melToHz(mlo + (mhi - mlo) * i / (nMel + 1));
    for (int i = 0; i < nBins; i++) fftF[i] = (double)sr / 2.0 * i / (nBins - 1);
    std::vector<float> basis((size_t)nMel * nBins);
    for (int m = 0; m < nMel; m++) {
        double enorm = 2.0 / (melF[m + 2] - melF[m]);
        for (int k = 0; k < nBins; k++) {
            double lower = (fftF[k] - melF[m]) / (melF[m + 1] - melF[m]);
            double upper = (melF[m + 2] - fftF[k]) / (melF[m + 2] - melF[m + 1]);
            basis[(size_t)m * nBins + k] = (float)(std::max(0.0, std::min(lower, upper)) * enorm);
        }
    }
    // reflect padding of (nfft - hop) / 2 on both sides
    const int pad = (nfft - hop) / 2, L = (int)y.size();
    std::vector<float> p((size_t)L + 2 * pad);
    for (int i = 0; i < (int)p.size(); i++) {
        int j = i - pad;
        if (j < 0) j = -j;
        if (j >= L) j = 2 * (L - 1) - j;
        p[i] = y[std::max(0, std::min(L - 1, j))];
    }
    frames = 1 + ((int)p.size() - nfft) / hop;
    std::vector<double> win(nfft);
    for (int i = 0; i < nfft; i++) win[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / nfft);       // periodic hann
    mel.assign((size_t)frames * nMel, 0.0f);
    std::vector<std::complex<double>> buf(nfft);
    std::vector<float> mag(nBins);
    for (int t = 0; t < frames; t++) {
        for (int i = 0; i < nfft; i++) buf[i] = p[(size_t)t * hop + i] * win[i];
        fft(buf);
        for (int k = 0; k < nBins; k++) mag[k] = (float)std::sqrt(buf[k].real() * buf[k].real() + buf[k].imag() * buf[k].imag() + 1e-9);
        for (int m = 0; m < nMel; m++) {
            double s = 0; const float* b = &basis[(size_t)m * nBins];
            for (int k = 0; k < nBins; k++) s += (double)b[k] * mag[k];
            mel[(size_t)t * nMel + m] = (float)std::log(std::max(s, 1e-5));
        }
    }
}

// ---------------------------------------------------------------- ECAPA-TDNN
void SpeakerEncoder::unload() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
    buf_ = nullptr; ctx_ = nullptr;
}

bool SpeakerEncoder::load(const std::string& path, std::string& err) {
    unload();
    StFile st;
    if (!st.open(path, err)) return false;
    std::string e2;
    if (!Compute::get().init(0, e2)) { err = e2; return false; }
    ggml_init_params ip = {ggml_tensor_overhead() * 200, nullptr, true};
    ctx_ = ggml_init(ip);
    struct Job { TT t; std::string name; bool kernel; };
    std::vector<Job> jobs;
    bool ok = true;
    auto conv = [&](const std::string& p, int dil) -> Conv {
        Conv c; c.dil = dil;
        const StTensor* w = st.find(p + ".weight"); const StTensor* b = st.find(p + ".bias");
        if (!w || !b) { if (ok) err = "missing speaker encoder tensor " + p; ok = false; return c; }
        c.w = ggml_new_tensor_3d(ctx_, GGML_TYPE_F16, w->shape[2], w->shape[1], w->shape[0]);
        c.b = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, b->shape[0]);
        jobs.push_back({c.w, p + ".weight", true}); jobs.push_back({c.b, p + ".bias", false});
        return c;
    };
    const std::string S = "speaker_encoder.";
    b0_ = conv(S + "blocks.0.conv", 1);
    const int dils[3] = {2, 3, 4};
    for (int i = 0; i < 3; i++) {
        std::string p = S + "blocks." + std::to_string(i + 1) + ".";
        se_[i].t1 = conv(p + "tdnn1.conv", 1); se_[i].t2 = conv(p + "tdnn2.conv", 1);
        for (int k = 0; k < 7; k++) se_[i].r2[k] = conv(p + "res2net_block.blocks." + std::to_string(k) + ".conv", dils[i]);
        se_[i].se1 = conv(p + "se_block.conv1", 1); se_[i].se2 = conv(p + "se_block.conv2", 1);
    }
    mfa_ = conv(S + "mfa.conv", 1); asp_t_ = conv(S + "asp.tdnn.conv", 1); asp_c_ = conv(S + "asp.conv", 1); fc_ = conv(S + "fc", 1);
    if (!ok) { unload(); return false; }
    buf_ = Compute::get().allocDeviceWeights(ctx_);
    if (!buf_) { err = "cannot allocate speaker encoder weights"; unload(); return false; }
    std::vector<float> f;
    for (auto& j : jobs) {
        st.toFloat(*st.find(j.name), f);
        if (j.kernel) { std::vector<ggml_fp16_t> h(f.size()); ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t)f.size()); ggml_backend_tensor_set(j.t, h.data(), 0, h.size() * 2); }
        else ggml_backend_tensor_set(j.t, f.data(), 0, f.size() * 4);
    }
    return true;
}

namespace {
// 'same' padded conv with reflect padding on a [T, C] tensor
TT conv1(ggml_context* c, TT x, TT w, TT b, int dil) {
    const int k = (int)w->ne[0];
    const int total = dil * (k - 1), left = total / 2, right = total - left;
    TT xp = total ? ggml_pad_reflect_1d(c, x, left, right) : x;
    TT y = ggml_conv_1d(c, w, xp, 1, 0, dil);
    return ggml_add(c, y, ggml_reshape_2d(c, b, 1, b->ne[0]));
}
TT relu(ggml_context* c, TT x) { return ggml_relu(c, x); }
}  // namespace

bool SpeakerEncoder::embedFromMel(const std::vector<float>& mel, int T, std::vector<float>& emb, std::string& err) {
    if (!loaded()) { err = "speaker encoder not loaded"; return false; }
    if (T < 16) { err = "reference audio is too short (need at least ~0.2 s, ideally 3 s or more)"; return false; }
    Compute& cm = Compute::get();
    const size_t nodes = 2048;
    std::vector<uint8_t> meta(ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false) + (1 << 20));
    ggml_init_params ip = {meta.size(), meta.data(), true};
    ggml_context* c = ggml_init(ip);

    TT in = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, 128); ggml_set_input(in);            // [T, 128]
    TT x = relu(c, conv1(c, in, b0_.w, b0_.b, 1));
    std::vector<TT> outs;
    for (int i = 0; i < 3; i++) {
        SeRes& s = se_[i];
        TT res = x;
        TT h = relu(c, conv1(c, x, s.t1.w, s.t1.b, 1));
        // Res2Net: 8 channel groups of 64
        TT acc = nullptr, prev = nullptr;
        for (int g = 0; g < 8; g++) {
            TT part = ggml_cont(c, ggml_view_2d(c, h, T, 64, h->nb[1], (size_t)g * 64 * h->nb[1]));
            TT o;
            if (g == 0) o = part;
            else if (g == 1) o = relu(c, conv1(c, part, s.r2[0].w, s.r2[0].b, s.r2[0].dil));
            else o = relu(c, conv1(c, ggml_add(c, part, prev), s.r2[g - 1].w, s.r2[g - 1].b, s.r2[g - 1].dil));
            prev = o;
            acc = acc ? ggml_concat(c, acc, o, 1) : o;
        }
        h = relu(c, conv1(c, acc, s.t2.w, s.t2.b, 1));
        TT m = ggml_mean(c, h);                                                          // [1, 512]
        m = relu(c, conv1(c, m, s.se1.w, s.se1.b, 1));
        m = ggml_sigmoid(c, conv1(c, m, s.se2.w, s.se2.b, 1));
        x = ggml_add(c, ggml_mul(c, h, m), res);
        outs.push_back(x);
    }
    TT cat = ggml_concat(c, ggml_concat(c, outs[0], outs[1], 1), outs[2], 1);            // [T, 1536]
    TT h = relu(c, conv1(c, cat, mfa_.w, mfa_.b, 1));

    // attentive statistics pooling
    auto stats = [&](TT v, TT w, TT& mean, TT& stdv) {
        mean = w ? ggml_sum_rows(c, ggml_mul(c, v, w)) : ggml_mean(c, v);               // [1, C]
        TT d = ggml_sqr(c, ggml_sub(c, v, mean));
        TT var = w ? ggml_sum_rows(c, ggml_mul(c, d, w)) : ggml_mean(c, d);
        stdv = ggml_sqrt(c, ggml_clamp(c, var, 1e-12f, 1e30f));
    };
    TT mean, sd;
    stats(h, nullptr, mean, sd);
    TT att = ggml_concat(c, ggml_concat(c, h, ggml_repeat(c, mean, h), 1), ggml_repeat(c, sd, h), 1);     // [T, 4608]
    att = ggml_tanh(c, relu(c, conv1(c, att, asp_t_.w, asp_t_.b, 1)));
    att = ggml_soft_max(c, conv1(c, att, asp_c_.w, asp_c_.b, 1));                        // softmax over time
    TT m2, s2;
    stats(h, att, m2, s2);
    TT pooled = ggml_concat(c, m2, s2, 1);                                               // [1, 3072]
    TT e = conv1(c, pooled, fc_.w, fc_.b, 1);                                            // [1, 2048]
    ggml_set_output(e);

    ggml_cgraph* g = ggml_new_graph_custom(c, nodes, false);
    ggml_build_forward_expand(g, e);
    ggml_backend_sched_reset(cm.sched);
    if (!ggml_backend_sched_alloc_graph(cm.sched, g)) { err = "cannot allocate speaker encoder graph"; ggml_free(c); return false; }
    // mel is frame-major (T x 128); the graph wants [T, 128] with time fastest -> transpose
    std::vector<float> tr((size_t)T * 128);
    for (int t = 0; t < T; t++) for (int m = 0; m < 128; m++) tr[(size_t)m * T + t] = mel[(size_t)t * 128 + m];
    ggml_backend_tensor_set(in, tr.data(), 0, tr.size() * 4);
    if (ggml_backend_sched_graph_compute(cm.sched, g) != GGML_STATUS_SUCCESS) { err = "speaker encoder compute failed"; ggml_free(c); return false; }
    emb.resize((size_t)ggml_nelements(e));
    ggml_backend_tensor_get(e, emb.data(), 0, emb.size() * 4);
    ggml_backend_sched_reset(cm.sched);
    ggml_free(c);
    return true;
}

bool SpeakerEncoder::embed(const std::vector<float>& wav, std::vector<float>& emb, std::string& err) {
    std::vector<float> mel; int T = 0;
    qwenMelSpectrogram(wav, mel, T);
    return embedFromMel(mel, T, emb, err);
}
