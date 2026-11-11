// Memory-mapped .safetensors reader (read-only, zero copy).
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct StTensor {
    std::string name;
    std::string dtype;               // "BF16", "F32", "F8_E4M3", "U8", ...
    std::vector<int64_t> shape;      // safetensors order (row-major, outermost first)
    uint64_t offset = 0, bytes = 0;  // relative to data start
    const uint8_t* data = nullptr;   // mapped pointer
    int64_t numel() const { int64_t n = 1; for (auto d : shape) n *= d; return n; }
};

class StFile {
public:
    ~StFile();
    bool open(const std::string& utf8Path, std::string& err);
    const StTensor* find(const std::string& name) const;
    const std::vector<StTensor>& tensors() const { return list_; }
    // Value of a scalar F32/BF16 tensor (e.g. *.weight_scale); returns false if missing.
    bool scalar(const std::string& name, float& out) const;
    // Convert any float tensor to float32 (BF16/F16/F32/F8_E4M3). `scale` multiplies afterwards.
    bool toFloat(const StTensor& t, std::vector<float>& out, float scale = 1.0f) const;
private:
    void* file_ = nullptr; void* map_ = nullptr; const uint8_t* base_ = nullptr; uint64_t size_ = 0;
    std::vector<StTensor> list_;
    std::map<std::string, size_t> idx_;
};

float bf16ToF32(uint16_t v);
float f16ToF32(uint16_t v);
float fp8e4m3ToF32(uint8_t v);       // OCP E4M3FN (no inf, 0x7f/0xff = NaN)
