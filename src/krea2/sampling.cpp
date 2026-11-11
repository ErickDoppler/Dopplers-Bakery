#include "krea2/sampling.h"
#include <algorithm>
#include <cmath>

namespace k2 {

double fluxTimeShift(double mu, double t) { return std::exp(mu) / (std::exp(mu) + (1.0 / t - 1.0)); }

static std::vector<float> sigmaTable(double shift) {
    std::vector<float> s(10000);
    for (int i = 0; i < 10000; i++) s[i] = (float)fluxTimeShift(shift, (double)(i + 1) / 10000.0);
    return s;
}

// ---- regularized incomplete beta I_x(a, b) and its inverse (what scipy.stats.beta.ppf does)
static double betacf(double a, double b, double x) {
    const int MAXIT = 300; const double EPS = 3e-16, FPMIN = 1e-300;
    double qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    if (std::fabs(d) < FPMIN) d = FPMIN;
    d = 1 / d; double h = d;
    for (int m = 1; m <= MAXIT; m++) {
        int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1 + aa * d; if (std::fabs(d) < FPMIN) d = FPMIN;
        c = 1 + aa / c; if (std::fabs(c) < FPMIN) c = FPMIN;
        d = 1 / d; h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1 + aa * d; if (std::fabs(d) < FPMIN) d = FPMIN;
        c = 1 + aa / c; if (std::fabs(c) < FPMIN) c = FPMIN;
        d = 1 / d; double del = d * c; h *= del;
        if (std::fabs(del - 1) < EPS) break;
    }
    return h;
}
static double incBeta(double a, double b, double x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) + b * std::log1p(-x));
    if (x < (a + 1) / (a + b + 2)) return bt * betacf(a, b, x) / a;
    return 1 - bt * betacf(b, a, 1 - x) / b;
}
static double betaPpf(double p, double a, double b) {
    if (p <= 0) return 0;
    if (p >= 1) return 1;
    double lo = 0, hi = 1;
    for (int i = 0; i < 100; i++) {
        double mid = 0.5 * (lo + hi);
        if (incBeta(a, b, mid) < p) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

std::vector<float> betaSigmas(int steps, double shift, double alpha, double beta) {
    std::vector<float> table = sigmaTable(shift);
    int total = (int)table.size() - 1;
    std::vector<float> sigs;
    double last = -1;
    for (int i = 0; i < steps; i++) {
        double ts = 1.0 - (double)i / steps;
        double t = std::nearbyint(betaPpf(ts, alpha, beta) * total);   // numpy rint = round half to even
        if (t != last) sigs.push_back(table[(int)t]);
        last = t;
    }
    sigs.push_back(0.0f);
    return sigs;
}

std::vector<float> simpleSigmas(int steps, double shift) {
    std::vector<float> table = sigmaTable(shift);
    std::vector<float> sigs;
    double ss = (double)table.size() / steps;
    for (int x = 0; x < steps; x++) sigs.push_back(table[table.size() - 1 - (int)(x * ss)]);
    sigs.push_back(0.0f);
    return sigs;
}

static double noiseScaler(double x) { return x * (std::exp(std::pow(x, 0.3)) + 10.0); }

bool erSde(std::vector<float>& x, const std::vector<float>& sigmasIn, double shift, const DenoiseFn& denoise,
           const NoiseFn& noise, const StepFn& onStep, double sNoise, int maxStage) {
    const int n = (int)sigmasIn.size() - 1;
    std::vector<double> sig(sigmasIn.begin(), sigmasIn.end());
    if (n > 1 && sig[0] >= 1.0) sig[0] = (float)fluxTimeShift(shift, 1.0 - 1e-4);      // offset_first_sigma_for_snr
    std::vector<double> lam(sig.size());
    for (size_t i = 0; i < sig.size(); i++) lam[i] = sig[i] <= 0 ? 0 : sig[i] / (1.0 - sig[i]);   // exp(-half_log_snr) for CONST

    const double POINTS = 200.0;
    size_t sz = x.size();
    std::vector<float> den(sz), oldDen, denD(sz), oldDenD, nz;
    for (int i = 0; i < n; i++) {
        if (!denoise(x, (float)sig[i], den)) return false;
        if (onStep) onStep(i, n);
        int stage = std::min(maxStage, i + 1);
        if (sig[i + 1] == 0) { x = den; }
        else {
            double ls = lam[i], lt = lam[i + 1];
            double as = sig[i] / ls, at = sig[i + 1] / lt;
            double rAlpha = at / as;
            double r = noiseScaler(lt) / noiseScaler(ls);
            for (size_t k = 0; k < sz; k++) x[k] = (float)(rAlpha * r * x[k] + at * (1 - r) * den[k]);

            if (stage >= 2) {
                double dt = lt - ls, step = -dt / POINTS;
                double s = 0, su = 0;
                for (int p = 0; p < (int)POINTS; p++) {
                    double lp = lt + p * step, sc = noiseScaler(lp);
                    s += 1.0 / sc; su += (lp - ls) / sc;
                }
                s *= step; su *= step;
                double d1 = ls - lam[i - 1];
                for (size_t k = 0; k < sz; k++) denD[k] = (float)((den[k] - oldDen[k]) / d1);
                double c2 = at * (dt + s * noiseScaler(lt));
                for (size_t k = 0; k < sz; k++) x[k] += (float)(c2 * denD[k]);
                if (stage >= 3) {
                    double d2 = (ls - lam[i - 2]) / 2.0;
                    double c3 = at * ((dt * dt) / 2.0 + su * noiseScaler(lt));
                    for (size_t k = 0; k < sz; k++) x[k] += (float)(c3 * ((denD[k] - oldDenD[k]) / d2));
                }
                oldDenD = denD;
            }
            if (sNoise > 0) {
                double v = lt * lt - ls * ls * r * r;
                double amp = at * sNoise * (v > 0 ? std::sqrt(v) : 0.0);
                nz.resize(sz); noise(nz);
                for (size_t k = 0; k < sz; k++) x[k] += (float)(amp * nz[k]);
            }
        }
        oldDen = den;
    }
    return true;
}

// ---- RNG: xoshiro256** seeded by splitmix64
Rng::Rng(unsigned long long seed) {
    unsigned long long z = seed;
    for (int i = 0; i < 4; i++) {
        z += 0x9e3779b97f4a7c15ull;
        unsigned long long t = z;
        t = (t ^ (t >> 30)) * 0xbf58476d1ce4e5b9ull;
        t = (t ^ (t >> 27)) * 0x94d049bb133111ebull;
        s_[i] = t ^ (t >> 31);
    }
}
static inline unsigned long long rotl(unsigned long long x, int k) { return (x << k) | (x >> (64 - k)); }
unsigned long long Rng::next() {
    unsigned long long r = rotl(s_[1] * 5, 7) * 9, t = s_[1] << 17;
    s_[2] ^= s_[0]; s_[3] ^= s_[1]; s_[1] ^= s_[2]; s_[0] ^= s_[3]; s_[2] ^= t; s_[3] = rotl(s_[3], 45);
    return r;
}
void Rng::normal(std::vector<float>& out) {
    for (auto& v : out) {
        if (have_) { v = (float)spare_; have_ = false; continue; }
        double u1 = ((next() >> 11) + 1.0) * (1.0 / 9007199254740993.0), u2 = (next() >> 11) * (1.0 / 9007199254740992.0);
        double m = std::sqrt(-2.0 * std::log(u1)), a = 6.283185307179586 * u2;
        v = (float)(m * std::cos(a)); spare_ = m * std::sin(a); have_ = true;
    }
}

}  // namespace k2
