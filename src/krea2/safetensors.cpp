#include "krea2/safetensors.h"
#include <windows.h>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include "ggml.h"
#include "util/json.h"
#include "util/sys.h"

float bf16ToF32(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }

float f16ToF32(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float f;
    if (e == 0) f = std::ldexp((float)m, -24);
    else if (e == 31) f = m ? NAN : INFINITY;
    else f = std::ldexp((float)(m | 1024), (int)e - 25);
    return s ? -f : f;
}

float fp8e4m3ToF32(uint8_t v) {
    int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
    if (e == 15 && m == 7) return NAN;
    float f = e == 0 ? std::ldexp((float)m, -9) : std::ldexp((float)(m | 8), e - 10);
    return s ? -f : f;
}

StFile::~StFile() {
    if (base_) UnmapViewOfFile(map_);
    if (file_) CloseHandle((HANDLE)file_);
}

bool StFile::open(const std::string& path, std::string& err) {
    HANDLE f = CreateFileW(sys::widen(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) { err = "cannot open " + path; return false; }
    LARGE_INTEGER sz; GetFileSizeEx(f, &sz);
    size_ = (uint64_t)sz.QuadPart;
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) { CloseHandle(f); err = "cannot map " + path; return false; }
    void* p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(m);
    if (!p) { CloseHandle(f); err = "cannot map view of " + path; return false; }
    file_ = f; map_ = p; base_ = (const uint8_t*)p;
    if (size_ < 8) { err = "file too small"; return false; }
    uint64_t hlen; memcpy(&hlen, base_, 8);
    if (hlen == 0 || 8 + hlen > size_) { err = "bad safetensors header"; return false; }
    bool ok;
    Json h = Json::parse(std::string((const char*)base_ + 8, hlen), &ok);
    if (!ok || h.t != Json::Obj) { err = "bad safetensors json"; return false; }
    const uint8_t* data = base_ + 8 + hlen;
    for (auto& kv : h.o) {
        if (kv.first == "__metadata__") continue;
        StTensor t; t.name = kv.first; t.dtype = kv.second.str("dtype");
        for (auto& d : kv.second.get("shape").a) t.shape.push_back((int64_t)d.num());
        const Json& off = kv.second.get("data_offsets");
        if (off.a.size() != 2) { err = "bad offsets for " + t.name; return false; }
        t.offset = (uint64_t)off.a[0].num(); t.bytes = (uint64_t)off.a[1].num() - t.offset;
        if (8 + hlen + t.offset + t.bytes > size_) { err = "tensor beyond end of file: " + t.name; return false; }
        t.data = data + t.offset;
        idx_[t.name] = list_.size();
        list_.push_back(t);
    }
    return true;
}

const StTensor* StFile::find(const std::string& name) const {
    auto it = idx_.find(name);
    return it == idx_.end() ? nullptr : &list_[it->second];
}

bool StFile::scalar(const std::string& name, float& out) const {
    const StTensor* t = find(name);
    if (!t || t->numel() != 1) return false;
    std::vector<float> v;
    if (!toFloat(*t, v)) return false;
    out = v[0]; return true;
}

bool StFile::toFloat(const StTensor& t, std::vector<float>& out, float scale) const {
    int64_t n = t.numel();
    out.resize((size_t)n);
    if (t.dtype == "F32") { memcpy(out.data(), t.data, (size_t)n * 4); }
    else if (t.dtype == "BF16") { const uint16_t* p = (const uint16_t*)t.data; for (int64_t i = 0; i < n; i++) out[i] = bf16ToF32(p[i]); }
    else if (t.dtype == "F16") { const uint16_t* p = (const uint16_t*)t.data; for (int64_t i = 0; i < n; i++) out[i] = f16ToF32(p[i]); }
    else if (t.dtype == "F8_E4M3") {
        float lut[256]; for (int i = 0; i < 256; i++) lut[i] = fp8e4m3ToF32((uint8_t)i);
        for (int64_t i = 0; i < n; i++) out[i] = lut[t.data[i]];
    } else return false;
    if (scale != 1.0f) for (auto& v : out) v *= scale;
    return true;
}

// One row of FP8 weights (through a 256-entry float table that already holds the tensor scale) straight into Q8_0 blocks: same arithmetic and rounding
// (half away from zero) as ggml's reference quantizer, 8 values at a time. n must be a multiple of 32.
#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2")))
static void fp8RowToQ8Avx2(const uint8_t* src, const float* lut, int64_t n, uint8_t* dst) {
    const __m256 signBit = _mm256_set1_ps(-0.0f), half = _mm256_set1_ps(0.5f), one = _mm256_set1_ps(1.0f);
    const __m256i fix = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int64_t b = 0; b < n / 32; b++) {
        __m256 v[4];
        for (int k = 0; k < 4; k++) v[k] = _mm256_i32gather_ps(lut, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(src + b * 32 + k * 8))), 4);
        __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(signBit, v[0]), _mm256_andnot_ps(signBit, v[1])), _mm256_max_ps(_mm256_andnot_ps(signBit, v[2]), _mm256_andnot_ps(signBit, v[3])));
        __m128 m4 = _mm_max_ps(_mm256_extractf128_ps(m, 1), _mm256_castps256_ps128(m));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4)); m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        const float d = _mm_cvtss_f32(m4) / 127.0f, id = d ? 1.0f / d : 0.0f;
        const ggml_fp16_t dh = ggml_fp32_to_fp16(d);
        uint8_t* o = dst + b * 34; memcpy(o, &dh, 2);
        const __m256 mul = _mm256_set1_ps(id);
        __m256i q[4];
        for (int k = 0; k < 4; k++) {
            __m256 x = _mm256_mul_ps(v[k], mul);
            __m256 t = _mm256_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            __m256 big = _mm256_cmp_ps(_mm256_andnot_ps(signBit, _mm256_sub_ps(x, t)), half, _CMP_GE_OQ);
            t = _mm256_add_ps(t, _mm256_and_ps(big, _mm256_or_ps(one, _mm256_and_ps(x, signBit))));
            q[k] = _mm256_cvtps_epi32(t);
        }
        __m256i a = _mm256_packs_epi32(q[0], q[1]), c = _mm256_packs_epi32(q[2], q[3]);
        __m256i r = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(a, c), fix);
        _mm256_storeu_si256((__m256i*)(o + 2), r);
    }
}
#endif

// A whole FP8 matrix [rows][cols] into Q8_0 (34 bytes per 32 values, rows back to back). false: no AVX2 or cols is not a multiple of 32 (the caller converts the slow way).
bool fp8MatrixToQ8(const uint8_t* src, const float* lut, int64_t cols, int64_t rows, uint8_t* dst) {
#if defined(__x86_64__) || defined(_M_X64)
    static const bool avx2 = __builtin_cpu_supports("avx2") && !getenv("BAKERY_DIT_NOAVX");
    if (!avx2 || cols % 32) return false;
    const size_t rb = (size_t)(cols / 32) * 34;
    for (int64_t r = 0; r < rows; r++) fp8RowToQ8Avx2(src + (size_t)r * cols, lut, cols, dst + (size_t)r * rb);
    return true;
#else
    (void)src; (void)lut; (void)cols; (void)rows; (void)dst; return false;
#endif
}
