// 16-bit PCM mono WAV writer.
#pragma once
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include "util/sys.h"

inline bool writeWavMono16(const std::string& path, const std::vector<float>& samples, int sampleRate) {
    const uint32_t dataBytes = (uint32_t)(samples.size() * 2);
    std::string out;
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; i++) out += (char)((v >> (8 * i)) & 255); };
    auto u16 = [&](uint16_t v) { out += (char)(v & 255); out += (char)(v >> 8); };
    out += "RIFF"; u32(36 + dataBytes); out += "WAVEfmt "; u32(16); u16(1); u16(1); u32(sampleRate); u32(sampleRate * 2); u16(2); u16(16);
    out += "data"; u32(dataBytes);
    out.reserve(out.size() + dataBytes);
    for (float f : samples) {
        float v = f < -1.0f ? -1.0f : (f > 1.0f ? 1.0f : f);
        int16_t s = (int16_t)(v * 32767.0f + (v >= 0 ? 0.5f : -0.5f));
        u16((uint16_t)s);
    }
    return sys::writeFile(path, out);
}

#include <cmath>
#include <cstring>

// Reads PCM16/PCM24/PCM32/float32 WAV, mixes down to mono. Returns false with a message on unsupported files.
inline bool readWavMono(const std::string& path, std::vector<float>& out, int& sampleRate, std::string& err) {
    std::string d;
    if (!sys::readFile(path, d)) { err = "cannot read " + path; return false; }
    if (d.size() < 44 || d.compare(0, 4, "RIFF") != 0 || d.compare(8, 4, "WAVE") != 0) { err = "not a WAV file (only .wav reference audio is supported)"; return false; }
    int fmt = 0, ch = 0, bits = 0; size_t dataOff = 0, dataLen = 0;
    sampleRate = 0;
    for (size_t p = 12; p + 8 <= d.size();) {
        uint32_t len; memcpy(&len, &d[p + 4], 4);
        if (d.compare(p, 4, "fmt ") == 0 && p + 8 + 16 <= d.size()) {
            uint16_t f, c, b; uint32_t sr;
            memcpy(&f, &d[p + 8], 2); memcpy(&c, &d[p + 10], 2); memcpy(&sr, &d[p + 12], 4); memcpy(&b, &d[p + 22], 2);
            fmt = f; ch = c; sampleRate = (int)sr; bits = b;
            if (fmt == 0xFFFE && len >= 26) { uint16_t sub; memcpy(&sub, &d[p + 8 + 24], 2); fmt = sub; }
        } else if (d.compare(p, 4, "data") == 0) { dataOff = p + 8; dataLen = std::min<size_t>(len, d.size() - dataOff); break; }
        p += 8 + len + (len & 1);
    }
    if (!dataOff || ch < 1 || sampleRate <= 0) { err = "malformed WAV header"; return false; }
    const int bps = bits / 8;
    if (!((fmt == 1 && (bits == 16 || bits == 24 || bits == 32)) || (fmt == 3 && bits == 32))) { err = "unsupported WAV encoding (use 16/24-bit PCM or float32)"; return false; }
    const size_t frames = dataLen / ((size_t)bps * ch);
    out.assign(frames, 0.0f);
    for (size_t i = 0; i < frames; i++) {
        double acc = 0;
        for (int c = 0; c < ch; c++) {
            const unsigned char* q = (const unsigned char*)&d[dataOff + (i * ch + c) * bps];
            double v;
            if (fmt == 3) { float f; memcpy(&f, q, 4); v = f; }
            else if (bits == 16) { int16_t s; memcpy(&s, q, 2); v = s / 32768.0; }
            else if (bits == 24) { int32_t s = (q[0] | (q[1] << 8) | (q[2] << 16)); if (s & 0x800000) s |= ~0xFFFFFF; v = s / 8388608.0; }
            else { int32_t s; memcpy(&s, q, 4); v = s / 2147483648.0; }
            acc += v;
        }
        out[i] = (float)(acc / ch);
    }
    return true;
}

// Windowed-sinc resampler (Kaiser-like Hann window, 24 zero crossings); good enough for reference audio.
inline std::vector<float> resampleSinc(const std::vector<float>& x, int srIn, int srOut) {
    if (srIn == srOut || x.empty()) return x;
    const double ratio = (double)srOut / srIn, cutoff = std::min(1.0, ratio) * 0.97;
    const int half = 24;
    const size_t n = (size_t)std::ceil(x.size() * ratio);
    std::vector<float> y(n);
    for (size_t i = 0; i < n; i++) {
        double t = i / ratio, acc = 0;
        long c = (long)std::floor(t);
        double scale = std::min(1.0, ratio);
        long span = (long)std::ceil(half / scale);
        for (long k = c - span + 1; k <= c + span; k++) {
            if (k < 0 || k >= (long)x.size()) continue;
            double u = (t - k) * cutoff * scale / std::min(1.0, ratio) ;
            double v = (t - k) * scale * 0.97;
            double s = v == 0 ? 1.0 : std::sin(M_PI * v) / (M_PI * v);
            double w = 0.5 + 0.5 * std::cos(M_PI * (t - k) / span);
            (void)u;
            acc += x[k] * s * w * scale * 0.97;
        }
        y[i] = (float)acc;
    }
    return y;
}

// 16-bit PCM WAV with any channel count (samples interleaved).
inline bool writeWavPcm16(const std::string& path, const std::vector<float>& interleaved, int channels, int sampleRate) {
    const uint32_t dataBytes = (uint32_t)(interleaved.size() * 2);
    std::string out;
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; i++) out += (char)((v >> (8 * i)) & 255); };
    auto u16 = [&](uint16_t v) { out += (char)(v & 255); out += (char)(v >> 8); };
    out += "RIFF"; u32(36 + dataBytes); out += "WAVEfmt "; u32(16); u16(1); u16((uint16_t)channels); u32(sampleRate);
    u32(sampleRate * channels * 2); u16((uint16_t)(channels * 2)); u16(16);
    out += "data"; u32(dataBytes);
    out.reserve(out.size() + dataBytes);
    for (float f : interleaved) {
        float v = f < -1.0f ? -1.0f : (f > 1.0f ? 1.0f : f);
        u16((uint16_t)(int16_t)(v * 32767.0f + (v >= 0 ? 0.5f : -0.5f)));
    }
    return sys::writeFile(path, out);
}
