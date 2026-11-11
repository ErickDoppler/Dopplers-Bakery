// Wan2.1 / Qwen-Image VAE decoder, single frame (so every 3D conv is a 2D conv), on ggml.
#pragma once
#include <string>
#include <vector>

struct ggml_context; struct ggml_backend_buffer; struct ggml_tensor;

class Krea2Vae {
public:
    ~Krea2Vae() { unload(); }
    bool load(const std::string& path, std::string& err);
    void unload();
    bool loaded() const { return buf_ != nullptr; }

    // Model-space latent (what the sampler produces) -> VAE space: z * std + mean, in place. z is 16 x H x W.
    static void latentToVae(std::vector<float>& z, int H, int W);

    // z: 16 x H x W in VAE space. rgb: (8H) x (8W) x 3 floats in [0,1], row-major interleaved.
    // Large images are decoded in overlapping tiles. `cancel` is polled between tiles.
    bool decode(const std::vector<float>& z, int H, int W, std::vector<float>& rgb, std::string& err,
                int maxTileArea = 64 * 64);
private:
    struct Conv { ggml_tensor *w = nullptr, *b = nullptr; };
    struct Res { ggml_tensor *g1 = nullptr, *g2 = nullptr; Conv c1, c2, sc; };
    bool decodeTile(const std::vector<float>& z, int H, int W, std::vector<float>& rgb, std::string& err);
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer* buf_ = nullptr;
    Conv conv2_, conv1_, headConv_;
    ggml_tensor* headG_ = nullptr;
    Res mid0_, mid2_;
    ggml_tensor* attnG_ = nullptr; Conv qkv_, proj_;
    std::vector<Res> ups_;            // 12 residual blocks in order
    Conv rs_[3];                      // resample convs after stages 0, 1, 2
};

// The encoder half of the same VAE (used for image-to-image), single frame.
class Krea2VaeEncoder {
public:
    ~Krea2VaeEncoder() { unload(); }
    bool load(const std::string& path, std::string& err);
    void unload();
    bool loaded() const { return buf_ != nullptr; }
    // rgb: H x W x 3 interleaved floats in [0,1], H and W multiples of 8.
    // lat: 16 x (H/8) x (W/8) in model space (already normalised, what the sampler works on).
    bool encode(const std::vector<float>& rgb, int H, int W, std::vector<float>& lat, std::string& err);
private:
    struct Conv { ggml_tensor *w = nullptr, *b = nullptr; };
    struct Res { ggml_tensor *g1 = nullptr, *g2 = nullptr; Conv c1, c2, sc; };
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer* buf_ = nullptr;
    Conv conv1_, quant_, headConv_, qkv_, proj_, down_[3];
    ggml_tensor *headG_ = nullptr, *attnG_ = nullptr;
    Res mid0_, mid2_;
    std::vector<Res> res_;            // 8 residual blocks
};
