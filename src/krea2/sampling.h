// Flow-matching schedule (ComfyUI ModelSamplingFlux + beta scheduler) and the ER-SDE sampler.
#pragma once
#include <functional>
#include <vector>

namespace k2 {

double fluxTimeShift(double mu, double t);                       // exp(mu) / (exp(mu) + (1/t - 1))
std::vector<float> betaSigmas(int steps, double shift = 1.15, double alpha = 0.6, double beta = 0.6);
std::vector<float> simpleSigmas(int steps, double shift = 1.15);

// denoise(x, sigma, out): out = x0 prediction (for flow models x - v*sigma). Return false to abort.
using DenoiseFn = std::function<bool(const std::vector<float>& x, float sigma, std::vector<float>& out)>;
// fills n with N(0,1) noise, called once per step that injects noise
using NoiseFn = std::function<void(std::vector<float>& n)>;
using StepFn = std::function<void(int step, int total)>;

// Port of comfy.k_diffusion.sampling.sample_er_sde for CONST (flow) model sampling.
// `x` is the starting latent (sigmas[0] * noise); result is written back to x.
bool erSde(std::vector<float>& x, const std::vector<float>& sigmas, double shift, const DenoiseFn& denoise,
           const NoiseFn& noise, const StepFn& onStep = nullptr, double sNoise = 1.0, int maxStage = 3);

// Seeded standard-normal generator (mt19937 + Box-Muller). Not bit-identical to torch.
class Rng {
public:
    explicit Rng(unsigned long long seed);
    void normal(std::vector<float>& out);
private:
    unsigned long long s_[4];
    unsigned long long next();
    bool have_ = false; double spare_ = 0;
};

}  // namespace k2
