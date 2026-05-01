#pragma once

#include <cmath>
#include <cstdio>
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

// Goedecker-Umrigar (GU) functional [S. Goedecker, C. J. Umrigar,
// Phys. Rev. Lett. 81, 866 (1998)].  Uses the same square-root kernel as
// Mueller, f(n) = sqrt(n), but with the orbital self-interaction (i = j)
// terms explicitly removed in the original definition.
//
// In the HEG with plane-wave natural orbitals the two-particle exchange
// integral <ij|ji> for i = j vanishes by translational invariance, so the
// self-interaction terms are zero and GU coincides numerically with the
// Mueller functional in this special case.  We still provide it as a separate
// class so that benchmark tables and plots can label it explicitly, and so
// that the framework can be extended to finite systems in the future.
class GUFunctional : public Functional {
public:
    std::string name() const override { return "GU"; }
    double f(double n) const override {
        return n > 0.0 ? std::sqrt(n) : 0.0;
    }
    double df(double n) const override {
        const double eps = 1.0e-14;
        return 0.5 / std::sqrt(n > eps ? n : eps);
    }
};

// CGA functional [G. Csanyi, S. Goedecker, T. A. Arias,
// Phys. Rev. A 65, 032510 (2002)].  Non-factorizable two-body kernel
//
//     K_CGA(n_i, n_j) = n_i n_j + sqrt(n_i (1 - n_i)) * sqrt(n_j (1 - n_j))
//
// The first piece reproduces the Hartree-Fock exchange while the second
// "hole" piece introduces dynamic correlation that is large for fractional
// occupations.  CGA is known to give an excellent description of the high-
// density HEG correlation energy and to outperform HF at metallic densities.
//
// The kernel is symmetric: K(n_i, n_j) = K(n_j, n_i), so the same
// kernel_grad signature used by BBC1Functional applies (derivative w.r.t.
// the first argument).
class CGAFunctional : public Functional {
public:
    std::string name() const override { return "CGA"; }
    double f(double n) const override {
        return n > 0.0 ? std::sqrt(n) : 0.0;
    }
    double df(double n) const override {
        const double eps = 1.0e-14;
        return 0.5 / std::sqrt(n > eps ? n : eps);
    }
    double kernel(double ni, double nj) const override {
        const double hi = hole(ni);
        const double hj = hole(nj);
        return ni * nj + hi * hj;
    }
    double kernel_grad(double ni, double nj) const override {
        const double hj = hole(nj);
        return nj + dhole(ni) * hj;
    }

private:
    static double hole(double n) {
        if (n <= 0.0 || n >= 1.0) return 0.0;
        return std::sqrt(n * (1.0 - n));
    }
    static double dhole(double n) {
        const double eps = 1.0e-12;
        const double x = n * (1.0 - n);
        if (x <= eps) {
            // Cap the derivative near the endpoints to avoid divergences in
            // the projected-gradient solver.  The cap matches |dh/dn| at
            // n = eps (or 1 - eps).
            const double xc = eps;
            return (1.0 - 2.0 * n) / (2.0 * std::sqrt(xc));
        }
        return (1.0 - 2.0 * n) / (2.0 * std::sqrt(x));
    }
};

// Beta functional (this work).  Generalizes the CGA hole piece via an
// adjustable exponent beta:
//
//     K_beta(n_i, n_j) = n_i n_j + [ n_i (1 - n_i) * n_j (1 - n_j) ]^beta
//
// Special / limiting cases:
//
//   * beta = 1/2  -> exactly the CGA kernel.
//   * beta -> +infinity  -> the bracket vanishes (since 0 <= n(1-n) <= 1/4)
//                         and we recover Hartree-Fock.
//   * beta -> 0          -> the hole saturates at 1 for any fractional
//                          occupation, an extreme over-correlating limit.
//
// Smaller beta therefore enhances the correlation hole and increases |E_c|;
// larger beta suppresses it and tends to HF.  Tuning beta thus interpolates
// between an under-correlated (HF) and over-correlated (Mueller-like) regime
// and lets the user fit the QMC reference for the HEG.
class BetaFunctional : public Functional {
public:
    explicit BetaFunctional(double beta) : beta_(beta) {}
    std::string name() const override {
        // Use a compact, parseable label so that downstream scripts can
        // pretty-print it as "Beta(0.50)" etc.
        char buf[64];
        std::snprintf(buf, sizeof(buf), "Beta(beta=%.3f)", beta_);
        return std::string(buf);
    }
    // f / df are not used for non-factorizable kernels; they are provided
    // only to satisfy the abstract base class.  Returning the Mueller
    // square-root form keeps any accidental reference well-defined.
    double f(double n) const override {
        return n > 0.0 ? std::sqrt(n) : 0.0;
    }
    double df(double n) const override {
        const double eps = 1.0e-14;
        return 0.5 / std::sqrt(n > eps ? n : eps);
    }
    double kernel(double ni, double nj) const override {
        const double xi = clamp_x(ni);
        const double xj = clamp_x(nj);
        if (xi <= 0.0 || xj <= 0.0) return ni * nj;
        return ni * nj + std::pow(xi * xj, beta_);
    }
    double kernel_grad(double ni, double nj) const override {
        const double xi = clamp_x(ni);
        const double xj = clamp_x(nj);
        if (xi <= 0.0 || xj <= 0.0) return nj;
        // d/dn_i [ (n_i(1-n_i) n_j(1-n_j))^beta ]
        //   = beta * (x_i x_j)^(beta-1) * (1 - 2 n_i) * x_j
        //   = beta * x_i^(beta-1) * x_j^beta * (1 - 2 n_i)
        const double pref = beta_ * std::pow(xi, beta_ - 1.0)
                                  * std::pow(xj, beta_);
        return nj + pref * (1.0 - 2.0 * ni);
    }
    double beta() const { return beta_; }

private:
    double beta_;
    // Floor n(1-n) to a tiny positive value so that pow(x, beta-1) is
    // finite for beta < 1 at the endpoints.  This is only relevant for the
    // gradient; energies use beta >= 0 and the contribution at n=0,1 is 0.
    static double clamp_x(double n) {
        const double x = n * (1.0 - n);
        const double eps = 1.0e-12;
        return (x > eps) ? x : eps;
    }
};

// GEO functional (this work).  Multi-power "geometric-mean" kernel built as
// the equally-normalized sum
//
//     K_GEO(n_p, n_q) = [ n_p n_q  +  (n_p n_q)^{1/2}
//                       + 2 * (n_p n_q)^{3/4} ] / 4
//                     = [ f1(n_p) f1(n_q) + f2(n_p) f2(n_q)
//                       + 2 f3(n_p) f3(n_q) ] / 4
//
// with f1(n) = n,  f2(n) = sqrt(n),  f3(n) = n^{3/4}.  This blends the HF
// (alpha = 1), Mueller (alpha = 1/2) and Power(3/4) factorizable kernels
// with weights 1 : 1 : 2, normalized so that K_GEO(1, 1) = 1 (matching HF
// at saturation).  Notable properties:
//
//   * Symmetric: K(n_p, n_q) = K(n_q, n_p).
//   * Vanishes at n_p = 0 (or n_q = 0) and equals 1 at n_p = n_q = 1, so it
//     reproduces the HF saturation limit while adding sqrt-type and 3/4-type
//     correlation contributions in the interior.
//   * Not factorizable, so we override `kernel` / `kernel_grad` rather than
//     supplying a single f / df.  The base-class f / df are still defined
//     (they default to the Mueller form) so that the abstract interface is
//     satisfied; the energy and gradient evaluators only call `kernel` and
//     `kernel_grad`.
class GEOFunctional : public Functional {
public:
    std::string name() const override { return "GEO"; }
    double f(double n) const override {
        return n > 0.0 ? std::sqrt(n) : 0.0;
    }
    double df(double n) const override {
        const double eps = 1.0e-14;
        return 0.5 / std::sqrt(n > eps ? n : eps);
    }
    double kernel(double ni, double nj) const override {
        const double p = ni * nj;
        if (p <= 0.0) return 0.0;
        const double s12 = std::sqrt(p);            // (n_p n_q)^{1/2}
        const double s34 = std::pow(p, 0.75);       // (n_p n_q)^{3/4}
        return 0.25 * (p + 2.0 * s12 +  s34);
    }
    double kernel_grad(double ni, double nj) const override {
        // d K / d n_i with n_j held fixed.  Differentiate term by term:
        //   d/dn_i [ n_i n_j ]                = n_j
        //   d/dn_i [ (n_i n_j)^{1/2} ]        = (1/2) n_j (n_i n_j)^{-1/2}
        //                                     = (1/2) sqrt(n_j / n_i)
        //   d/dn_i [ 2 (n_i n_j)^{3/4} ]      = (3/2) n_j (n_i n_j)^{-1/4}
        const double eps = 1.0e-14;
        const double nic = (ni > eps) ? ni : eps;
        const double njc = (nj > eps) ? nj : eps;
        const double pc  = nic * njc;
        const double ds12 = 0.5 * std::sqrt(njc / nic);
        const double ds34 = 1.5 * njc * std::pow(pc, -0.25);
        return 0.25 * (nj + ds12 + ds34);
    }
};

// optGM (optimized geometric mean).  Same three factorized channels as GEO
// (f1 = n, f2 = sqrt(n), f3 = n^{3/4}) but with tunable nonnegative weights
//
//     K_optGM(n_i, n_j) = w1 n_i n_j + w2 (n_i n_j)^{1/2} + w3 (n_i n_j)^{3/4},
//
// where w1 = alpha^2, w2 = beta^2, w3 = gamma^2 and alpha^2 + beta^2 + gamma^2 = 1,
// so K(1, 1) = 1 (HF saturation).  GEO in this codebase corresponds to
// (w1, w2, w3) = (1/4, 1/2, 1/4).
// To realize nonnegative weights (w1,w2,w3) that sum to 1, pass direction
// (±sqrt(w1), ±sqrt(w2), ±sqrt(w3)); normalization leaves w_i unchanged.
class OptGMFunctional : public Functional {
public:
    explicit OptGMFunctional(double alpha, double beta, double gamma) {
        const double sq = alpha * alpha + beta * beta + gamma * gamma;
        const double nrm = std::sqrt(sq);
        if (nrm < 1.0e-15) {
            // Degenerate input: fall back to equal mixing on the sphere.
            alpha_ = beta_ = gamma_ = 1.0 / std::sqrt(3.0);
        } else {
            alpha_ = alpha / nrm;
            beta_  = beta / nrm;
            gamma_ = gamma / nrm;
        }
        w1_ = alpha_ * alpha_;
        w2_ = beta_ * beta_;
        w3_ = gamma_ * gamma_;
    }

    double w1() const { return w1_; }
    double w2() const { return w2_; }
    double w3() const { return w3_; }
    double alpha() const { return alpha_; }
    double beta() const { return beta_; }
    double gamma() const { return gamma_; }

    std::string name() const override {
        char buf[96];
        std::snprintf(buf, sizeof(buf),
                      "optGM(a=%.4f,b=%.4f,c=%.4f)", alpha_, beta_, gamma_);
        return std::string(buf);
    }
    double f(double n) const override {
        return n > 0.0 ? std::sqrt(n) : 0.0;
    }
    double df(double n) const override {
        const double eps = 1.0e-14;
        return 0.5 / std::sqrt(n > eps ? n : eps);
    }
    double kernel(double ni, double nj) const override {
        const double p = ni * nj;
        if (p <= 0.0) return 0.0;
        const double s12 = std::sqrt(p);
        const double s34 = std::pow(p, 0.75);
        return w1_ * p + w2_ * s12 + w3_ * s34;
    }
    double kernel_grad(double ni, double nj) const override {
        const double eps = 1.0e-14;
        const double nic = (ni > eps) ? ni : eps;
        const double njc = (nj > eps) ? nj : eps;
        const double pc  = nic * njc;
        const double ds12 = 0.5 * std::sqrt(njc / nic);
        const double ds34 = 0.75 * njc * std::pow(pc, -0.25);
        return w1_ * njc + w2_ * ds12 + w3_ * ds34;
    }

private:
    double alpha_;
    double beta_;
    double gamma_;
    double w1_, w2_, w3_;
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
    if (key == "GU")      return std::make_unique<GUFunctional>();
    if (key == "CGA")     return std::make_unique<CGAFunctional>();
    if (key == "Power")   return std::make_unique<PowerFunctional>(alpha);
    if (key == "BBC1")    return std::make_unique<BBC1Functional>();
    if (key == "GEO")     return std::make_unique<GEOFunctional>();
    // OptGM@a;b;c with three floats separated by ';' (commas are reserved for
    // the driver's --funcs list).  Angles are normalized to a^2+b^2+c^2 = 1;
    // weights on the three GEO channels are a^2, b^2, c^2.
    if (key.rfind("OptGM@", 0) == 0) {
        const std::string rest = key.substr(6);
        double a = 0.0, b = 0.0, c = 0.0;
        if (std::sscanf(rest.c_str(), "%lf;%lf;%lf", &a, &b, &c) == 3) {
            return std::make_unique<OptGMFunctional>(a, b, c);
        }
        return nullptr;
    }
    // The Beta functional needs an explicit exponent; callers should use
    // make_functional("Beta", beta) explicitly (alpha is reused as beta).
    if (key == "Beta")    return std::make_unique<BetaFunctional>(alpha);
    return nullptr;
}

}  // namespace rdmft
