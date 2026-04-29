#pragma once

#include <cmath>
#include <memory>
#include <string>

namespace rdmft {

// Base class for RDMFT exchange-correlation functionals applied to the HEG.
//
// In the homogeneous electron gas the natural orbitals are plane waves and the
// natural occupation numbers depend only on |k|.  For the spin-unpolarized
// case occupations are normalized so that n(k) in [0, 1] (per spin orbital).
//
// All functionals considered here can be cast in the "JK-only" form
//
//     E_xc[{n}] = -1/2  sum_{ij} F(n_i, n_j) <ij|ij>
//
// where the kernel F factorizes in the Power family as
//
//     F(n_i, n_j) = f(n_i) f(n_j).
//
// Concrete functionals therefore only need to provide:
//   * f(n)        - the function applied to occupations
//   * df(n)       - derivative df/dn (used for the Euler-Lagrange equation)
//
// More general kernels (e.g. BBC corrections that introduce branch-dependent
// signs, or non-factorizable forms) can override `kernel` / `kernel_grad`
// directly.  See `BBC1Functional` further below for an example.
class Functional {
public:
    virtual ~Functional() = default;
    virtual std::string name() const = 0;

    // Factorizable f(n).  By default we assume the kernel factorizes.
    virtual double f(double n) const = 0;
    virtual double df(double n) const = 0;

    // Two-body kernel F(n_i, n_j) entering the exchange-correlation energy.
    // Default = factorized power form.  Override for richer functionals.
    virtual double kernel(double ni, double nj) const {
        return f(ni) * f(nj);
    }
    // Gradient w.r.t. n_i (with n_j fixed).
    virtual double kernel_grad(double ni, double nj) const {
        return df(ni) * f(nj);
    }
};

// f(n) = n   ->  Hartree-Fock
class HFFunctional : public Functional {
public:
    std::string name() const override { return "HF"; }
    double f(double n)  const override { return n; }
    double df(double /*n*/) const override { return 1.0; }
};

// f(n) = sqrt(n) -> Mueller / BB functional (alpha = 1/2).
class MuellerFunctional : public Functional {
public:
    std::string name() const override { return "Mueller"; }
    double f(double n) const override {
        return n > 0.0 ? std::sqrt(n) : 0.0;
    }
    double df(double n) const override {
        // 1 / (2 sqrt n).  Cap near n=0 to avoid divergence in the gradient.
        const double eps = 1.0e-14;
        return 0.5 / std::sqrt(n > eps ? n : eps);
    }
};

// f(n) = n^alpha  -> Power functional (Sharma et al., 2008).
// Recommended values around alpha ~ 0.55 - 0.58 for the HEG.
class PowerFunctional : public Functional {
public:
    explicit PowerFunctional(double alpha) : alpha_(alpha) {}
    std::string name() const override {
        return "Power(alpha=" + std::to_string(alpha_) + ")";
    }
    double f(double n) const override {
        if (n <= 0.0) return 0.0;
        return std::pow(n, alpha_);
    }
    double df(double n) const override {
        const double eps = 1.0e-14;
        if (n <= eps) return alpha_ * std::pow(eps, alpha_ - 1.0);
        return alpha_ * std::pow(n, alpha_ - 1.0);
    }
    double alpha() const { return alpha_; }

private:
    double alpha_;
};

// BBC1 (Gritsenko, Pernal, Baerends 2005).  In a plane-wave basis it reduces
// to using the Mueller kernel everywhere except that pairs of "weakly
// occupied" orbitals (n_i, n_j < 1/2 in the paramagnetic case) get a sign
// flip: F(n_i,n_j) = -sqrt(n_i n_j) instead of +sqrt(n_i n_j).  Useful as a
// demonstration of how to extend the framework to a non-factorizable kernel.
//
// NOTE: The hard sign-switch at n=1/2 makes BBC1 hard to optimize with a
// gradient-based scheme on a fine grid; this implementation is *experimental*
// and provided as a template only.  For production use a smoothed switch
// (e.g. tanh) is recommended.
class BBC1Functional : public Functional {
public:
    explicit BBC1Functional(double smooth = 0.05) : smooth_(smooth) {}
    std::string name() const override { return "BBC1"; }
    double f(double n) const override {
        return n > 0.0 ? std::sqrt(n) : 0.0;
    }
    double df(double n) const override {
        const double eps = 1.0e-14;
        return 0.5 / std::sqrt(n > eps ? n : eps);
    }
    double kernel(double ni, double nj) const override {
        // s = -1 iff both weakly occupied, else +1.
        // s = 1 - 2 * w(ni) * w(nj),  with w(n) ~ smoothed indicator(n < 0.5).
        const double wi = w(ni), wj = w(nj);
        const double s  = 1.0 - 2.0 * wi * wj;
        return s * f(ni) * f(nj);
    }
    double kernel_grad(double ni, double nj) const override {
        const double wi = w(ni), wj = w(nj);
        const double dwi = w_grad(ni);
        const double s   = 1.0 - 2.0 * wi * wj;
        const double dsi = -2.0 * dwi * wj;
        return dsi * f(ni) * f(nj) + s * df(ni) * f(nj);
    }

private:
    double smooth_;  // smoothing width of the n=1/2 indicator
    // Smoothed indicator w(n) ~ 1 for n < 0.5, ~ 0 for n > 0.5.
    double w(double n) const {
        return 0.5 * (1.0 - std::tanh((n - 0.5) / smooth_));
    }
    double w_grad(double n) const {
        const double t = std::tanh((n - 0.5) / smooth_);
        return -0.5 * (1.0 - t * t) / smooth_;
    }
};

// Convenience factory.
inline std::unique_ptr<Functional> make_functional(const std::string& key,
                                                   double alpha = 0.55) {
    if (key == "HF")      return std::make_unique<HFFunctional>();
    if (key == "Mueller" || key == "BB") return std::make_unique<MuellerFunctional>();
    if (key == "Power")   return std::make_unique<PowerFunctional>(alpha);
    if (key == "BBC1")    return std::make_unique<BBC1Functional>();
    return nullptr;
}

}  // namespace rdmft
