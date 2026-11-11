#include "krea2/safetensors.h"
#include <windows.h>
#include <cmath>
#include <cstring>
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
