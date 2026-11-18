// LTX-2.3 weight access: reads tensors of the ComfyUI-style quantized .safetensors files and returns them as float32.
//   plain       F32 / BF16 / F16
//   fp8         F8_E4M3 weight * "<layer>.weight_scale" (scalar)         (the 22B checkpoint; "<layer>.input_scale" is only for fp8 matmuls and is ignored)
//   nvfp4       U8 weight (2 x E2M1 per byte, high nibble = even element) * "weight_scale" (F8_E4M3 per 16 values, cuBLAS tiled layout) * "weight_scale_2" (scalar)
//               (the Gemma 3 file; "<layer>.comfy_quant" holds {"format": "nvfp4" | "float8_e4m3fn"})
#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "krea2/safetensors.h"

// A LoRA (lora_A [r, in], lora_B [out, r], alpha): W += strength * alpha / r * B A, merged into a weight when it is loaded.
struct LtxLora {
    StFile file;
    float strength = 1.0f;
    bool open(const std::string& path, std::string& err) { return file.open(path, err); }
    // `weightName` as in the checkpoint ("model.diffusion_model.X.weight"); w is [out][in]. Returns true when the weight has an adapter.
    bool apply(const std::string& weightName, std::vector<float>& w, int64_t out, int64_t in) const;
};

struct LtxWeights {
    StFile file;
    const LtxLora* lora = nullptr;                 // optional, applied by loadF32 to every weight it covers
    bool cacheF16 = false;                         // keep the dequantized (+LoRA) matrices as F16 in memory the first time they are used (the 22B transformer is ~37 GB)
    mutable std::unordered_map<std::string, std::vector<uint16_t>> f16cache;
    bool open(const std::string& path, std::string& err) { return file.open(path, err); }
    bool has(const std::string& name) const { return file.find(name) != nullptr; }
    // Logical shape of a tensor (an nvfp4 weight is [rows, cols], the file stores cols / 2 bytes per row).
    bool shapeOf(const std::string& name, std::vector<int64_t>& shape) const;
    // Dequantized float32 copy of `name` (a weight name such as "...to_q.weight", or any plain tensor).
    bool loadF32(const std::string& name, std::vector<float>& out, std::string& err) const;
    bool loadF32Raw(const std::string& name, std::vector<float>& out, std::string& err) const;      // without the LoRA
    // The quantization format of a weight: "none", "fp8" or "nvfp4".
    std::string formatOf(const std::string& name) const;
};

// Exposed for tests: cuBLAS block-scale layout -> row-major scales [rows][cols] (cols = columns / 16).
void ltxUnswizzleScales(const uint8_t* blocked, int64_t rows, int64_t cols, std::vector<uint8_t>& out);


