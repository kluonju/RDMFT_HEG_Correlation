#pragma once

#include <algorithm>
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

    // True when K(n_i,n_j) = f(n_i) f(n_j) for a scalar f(n).
    virtual bool is_factorized() const { return true; }

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
// Phys. Rev. A 65, 032510 (2002)].  Pairwise kernel in the same JK convention as
// ``EnergyEvaluator``, with an explicit factor 1/2 on the bracket (Csányi–
// Goedecker–Arias tensor-product structure with sqrt(n(2-n)) hole):
//
//     K_CGA(n_i, n_j) = (1/2) * [ n_i n_j
//                               + sqrt(n_i(2-n_i)) * sqrt(n_j(2-n_j)) ].
//
// Occupations are per spin-orbital in [0, 1]; n(2-n) >= 0 on that interval.
//
// The kernel is symmetric; ``kernel_grad`` is dK/dn_i at fixed n_j.
class CGAFunctional : public Functional {
public:
    std::string name() const override { return "CGA"; }
    bool is_factorized() const override { return false; }
    double f(double n) const override {
        return n > 0.0 ? std::sqrt(n) : 0.0;
    }
    double df(double n) const override {
        const double eps = 1.0e-14;
        return 0.5 / std::sqrt(n > eps ? n : eps);
    }
    double kernel(double ni, double nj) const override {
        static constexpr double half = 0.5;
        const double hi = hole(ni);
        const double hj = hole(nj);
        return half * (ni * nj + hi * hj);
    }
    double kernel_grad(double ni, double nj) const override {
        static constexpr double half = 0.5;
        const double hj = hole(nj);
        return half * (nj + dhole(ni) * hj);
    }

private:
    static double hole(double n) {
        if (n <= 0.0 || n >= 1.0) return 0.0;
        const double x = n * (2.0 - n);
        return std::sqrt(std::max(x, 0.0));
    }
    static double dhole(double n) {
        const double eps = 1.0e-12;
        const double x = n * (2.0 - n);
        if (x <= eps) {
            const double xc = eps;
            return (1.0 - n) / std::sqrt(xc);
        }
        return (1.0 - n) / std::sqrt(x);
    }
};

// Beta functional (this work).  Generalizes the CGA hole piece via an
// adjustable exponent beta:
//
//     K_beta(n_i, n_j) = n_i n_j + [ n_i (1 - n_i) * n_j (1 - n_j) ]^beta
//
// Special / limiting cases:
//
//   * beta = 1/2  -> same bracket as the CHF kernel (sqrt(n(1-n)) hole).
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
    bool is_factorized() const override { return false; }
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

// CHF — corrected-HF kernel [G. Csányi, T. A. Arias, Phys. Rev. B 61, 7348 (2000)].
//
//     K_CHF(n_i, n_j) = n_i n_j + [ n_i (1 - n_i) * n_j (1 - n_j) ]^{1/2},
//
// i.e. algebraically ``Beta(beta = 1/2)``.  We inherit that implementation so
// the additive SCF path matches the stable Beta solver (a standalone CHF
// additive branch incorrectly reproduced the CGA momentum distribution).
class CHFFunctional : public BetaFunctional {
public:
    CHFFunctional() : BetaFunctional(0.5) {}
    std::string name() const override { return "CHF"; }
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
    bool is_factorized() const override { return false; }
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

// optGeo: full HF pair kernel plus a sigmoid correlation in the centred pair
// variable
//
//     x_ij = (n_i - 1/2)^2 (n_j - 1/2)^2   (zero when either n = 1/2),
//
//     K_optGeo(n_i, n_j) = n_i n_j
//                        + w * ( 2 sigma(k x_ij) - 1 ),
//
// where sigma(t) = 1 / (1 + exp(-t)).  The correction is 0 at x_ij = 0 and
// approaches +w as x_ij grows (k > 0).  Parameters w (amplitude) and k
// (steepness) are required to be non-negative.  Solved via projected gradient
// (non-factorizable pair coupling).
//
// CLI: ``OptGeo@w;k`` (two semicolon-separated floats; commas reserved in --funcs).
class OptGeoFunctional : public Functional {
public:
    explicit OptGeoFunctional(double w, double k)
        : w_(std::max(0.0, w)), k_(std::max(0.0, k)) {}

    double w() const { return w_; }
    double k() const { return k_; }
    double w1() const { return w_; }
    double w2() const { return k_; }
    bool is_factorized() const override { return false; }

    std::string name() const override {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "optGeo(w=%.4g,k=%.4g)", w_, k_);
        return std::string(buf);
    }
    double f(double n) const override {
        return n > 0.0 ? std::sqrt(n) : 0.0;
    }
    double df(double n) const override {
        const double eps = 1.0e-14;
        return 0.5 / std::sqrt(n > eps ? n : eps);
    }

    static double centre2(double n) {
        const double h = n - 0.5;
        return h * h;
    }
    static double pair_centre2(double ni, double nj) {
        return centre2(ni) * centre2(nj);
    }
    static double d_centre2(double n) { return 2.0 * (n - 0.5); }

    static double sigmoid_stable(double t) {
        if (t >= 0.0) {
            const double et = std::exp(-t);
            return 1.0 / (1.0 + et);
        }
        const double et = std::exp(t);
        return et / (1.0 + et);
    }
    static double dsigmoid_dt(double t) {
        const double s = sigmoid_stable(t);
        return s * (1.0 - s);
    }

    // w * (2 sigma(k x) - 1): 0 at x=0, -> w as k x -> +infty (k > 0).
    static double corr_sigmoid(double x, double w, double k) {
        if (std::abs(w) < 1.0e-15 || std::abs(k) < 1.0e-15) return 0.0;
        const double s = sigmoid_stable(k * x);
        return w * (2.0 * s - 1.0);
    }

    double kernel(double ni, double nj) const override {
        const double hf = ni * nj;
        const double x = pair_centre2(ni, nj);
        return hf + corr_sigmoid(x, w_, k_);
    }
    double kernel_grad(double ni, double nj) const override {
        const double x = pair_centre2(ni, nj);
        const double h2j = centre2(nj);
        double dg = 0.0;
        if (std::abs(w_) >= 1.0e-15 && std::abs(k_) >= 1.0e-15) {
            const double t = k_ * x;
            dg = w_ * 2.0 * dsigmoid_dt(t) * k_ * d_centre2(ni) * h2j;
        }
        return nj + dg;
    }

private:
    double w_;
    double k_;
};

// HybOpt (hybrid HF/Power): convex mixture of the HF pair kernel and Power pair kernel
//
//     K(n_i, n_j) = (1 - lambda) * n_i n_j
//                 + lambda * n_i^alpha * n_j^alpha,
//
// i.e. ``(1-lambda) * HF + lambda * Power(alpha)`` in the same JK convention as
// ``HFFunctional`` / ``PowerFunctional``.  ``lambda`` is clamped to ``[0, 1]``;
// ``alpha`` must be positive (also capped in the ctor for numerical safety).
//
// CLI: ``HybOpt@lambda;alpha`` (one semicolon, two floats).  Legacy ``OptGM@...``
// is accepted as an alias.  Solved via the specialized HF/Power-mix branch in
// ``solve_rdmft``.
class HybOptFunctional : public Functional {
public:
    explicit HybOptFunctional(double lambda, double alpha) {
        if (lambda < 0.0)       lambda_ = 0.0;
        else if (lambda > 1.0) lambda_ = 1.0;
        else                   lambda_ = lambda;
        const double eps = 1.0e-14;
        // α < 1 keeps f'(n) ∝ n^{α-1} decreasing in n so the 2-channel EL
        // bisection in ``solve_rdmft`` is monotone (same regime as Power fits).
        if (alpha <= eps)         alpha_ = eps;
        else if (alpha >= 1.0)    alpha_ = 1.0 - 1.0e-6;
        else                      alpha_ = alpha;
    }

    double lambda_mix() const { return lambda_; }
    double alpha() const { return alpha_; }
    bool is_factorized() const override { return false; }

    std::string name() const override {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "hybopt(lam=%.4g,alpha=%.4g)", lambda_, alpha_);
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
        const double lam = lambda_;
        const double hf = ni * nj;
        const double eps = 1.0e-15;
        const double nic = (ni > 0.0) ? ni : 0.0;
        const double njc = (nj > 0.0) ? nj : 0.0;
        const double nicc = (nic > eps) ? nic : eps;
        const double njcc = (njc > eps) ? njc : eps;
        const double pw =
            std::pow(nicc, alpha_) * std::pow(njcc, alpha_);
        return (1.0 - lam) * hf + lam * pw;
    }
    double kernel_grad(double ni, double nj) const override {
        const double lam = lambda_;
        const double eps = 1.0e-15;
        const double njc = (nj > 0.0) ? nj : 0.0;
        const double njcc = (njc > eps) ? njc : eps;
        const double nic = (ni > 0.0) ? ni : 0.0;
        const double nicc = (nic > eps) ? nic : eps;
        const double d_pw =
            alpha_ * std::pow(nicc, alpha_ - 1.0) * std::pow(njcc, alpha_);
        return (1.0 - lam) * njc + lam * d_pw;
    }

private:
    double lambda_;
    double alpha_;
};


// BOW functional [T. Baldsiefen, A. Cangi, F. G. Eich, and E. K. U. Gross,
// Phys. Rev. A 96, 062508 (2017), Appendix C].  The pair kernel is
//
//     K_BOW(n_i, n_j; alpha) = x^alpha - alpha x + alpha
//                              - alpha (1 - x)^(1 / alpha),
//     x = n_i n_j.
//
// It reduces to Hartree-Fock for alpha = 1, preserves K(0,*) = 0 and
// K(1,1) = 1, and introduces the "bow" shape used in the Baldsiefen et al.
// HEG momentum-distribution fit.  Table IV of that paper recommends
// alpha = 0.61 for the unpolarized 3D HEG; this is the default used here.
class BOWFunctional : public Functional {
public:
    explicit BOWFunctional(double alpha = 0.61)
        : alpha_(std::max(alpha, 1.0e-8)) {}

    double alpha() const { return alpha_; }
    bool is_factorized() const override { return false; }

    std::string name() const override {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "BOW(alpha=%.3f)", alpha_);
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
        return bow_kernel(ni * nj, alpha_);
    }
    double kernel_grad(double ni, double nj) const override {
        return nj * bow_kernel_dx(ni * nj, alpha_);
    }

protected:
    static double bow_kernel(double x, double alpha) {
        const double xc = std::clamp(x, 0.0, 1.0);
        return std::pow(xc, alpha) - alpha * xc + alpha
             - alpha * std::pow(std::max(1.0 - xc, 0.0), 1.0 / alpha);
    }
    static double bow_kernel_dx(double x, double alpha) {
        const double eps = 1.0e-12;
        const double xc = std::clamp(x, eps, 1.0 - eps);
        return alpha * std::pow(xc, alpha - 1.0) - alpha
             + std::pow(1.0 - xc, 1.0 / alpha - 1.0);
    }

private:
    double alpha_;
};

// Particle-hole symmetrized BOW kernel requested as "SymBow": average the
// original BOW pair kernel with the complementary BOW expression evaluated
// after the occupation-hole replacement n -> 1 - n in both arguments,
//
//     K_SymBow(n_i,n_j) = 1/2 [K_BOW(n_i,n_j)
//                            + 1 - K_BOW(1-n_i,1-n_j)].
//
// The complementary form keeps the HF endpoint limits K(0,*) = 0 and
// K(1,1) = 1 while making the pair kernel particle-hole symmetric.
class SymBOWFunctional : public BOWFunctional {
public:
    explicit SymBOWFunctional(double alpha = 0.61) : BOWFunctional(alpha) {}
    bool is_factorized() const override { return false; }

    std::string name() const override {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "SymBow(alpha=%.3f)", alpha());
        return std::string(buf);
    }
    double kernel(double ni, double nj) const override {
        const double holes_i = 1.0 - ni;
        const double holes_j = 1.0 - nj;
        return 0.5 * (bow_kernel(ni * nj, alpha())
                    + 1.0 - bow_kernel(holes_i * holes_j, alpha()));
    }
    double kernel_grad(double ni, double nj) const override {
        const double holes_i = 1.0 - ni;
        const double holes_j = 1.0 - nj;
        const double particle_grad = nj * bow_kernel_dx(ni * nj, alpha());
        const double hole_grad = holes_j * bow_kernel_dx(holes_i * holes_j, alpha());
        return 0.5 * (particle_grad + hole_grad);
    }
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
    bool is_factorized() const override { return false; }
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
    double smooth_;
    double w(double n) const {
        return 0.5 * (1.0 - std::tanh((n - 0.5) / smooth_));
    }
    double w_grad(double n) const {
        const double t = std::tanh((n - 0.5) / smooth_);
        return -0.5 * (1.0 - t * t) / smooth_;
    }
};

// BBC3 in the plane-wave HEG coincides with BBC2 [N. N. Lathiotakis, N. Helbig,
// E. K. U. Gross, Phys. Rev. B 75, 195120 (2007)]: bonding vs anti-bonding
// distinctions in the molecular BBC3 definition collapse, so we
// implement the BBC2 pair rule (Gritsenko–Pernal–Baerends hierarchy as quoted
// in PRB 75):
//
//   weak–weak (n < 1/2):    K = -sqrt(n_i n_j)
//   strong–strong (> 1/2): K =  n_i n_j
//   otherwise:             K = +sqrt(n_i n_j)
//
// The n = 1/2 dividing surface is smoothed like ``BBC1Functional`` for stable
// ``kernel_grad`` in the projected-gradient solver.
class BBC3Functional : public Functional {
public:
    explicit BBC3Functional(double smooth = 0.05) : smooth_(smooth) {}
    std::string name() const override { return "BBC3"; }
    bool is_factorized() const override { return false; }
    double f(double n) const override {
        return n > 0.0 ? std::sqrt(n) : 0.0;
    }
    double df(double n) const override {
        const double eps = 1.0e-14;
        return 0.5 / std::sqrt(n > eps ? n : eps);
    }
    double kernel(double ni, double nj) const override {
        const double w_i = w(ni), w_j = w(nj);
        const double a_ww = w_i * w_j;
        const double a_ss = (1.0 - w_i) * (1.0 - w_j);
        const double a_rm = std::max(0.0, 1.0 - a_ww - a_ss);
        const double s_ij = std::sqrt(std::max(ni * nj, 0.0));
        return a_ww * (-s_ij) + a_ss * (ni * nj) + a_rm * s_ij;
    }
    double kernel_grad(double ni, double nj) const override {
        const double w_i  = w(ni), w_j = w(nj);
        const double dw_i = w_grad(ni);
        const double a_ww = w_i * w_j;
        const double a_ss = (1.0 - w_i) * (1.0 - w_j);
        const double a_rm = std::max(0.0, 1.0 - a_ww - a_ss);

        const double da_ww = dw_i * w_j;
        const double da_ss = -dw_i * (1.0 - w_j);
        const double da_rm = -da_ww - da_ss;

        const double s_ij = std::sqrt(std::max(ni * nj, 0.0));
        const double eps  = 1.0e-14;
        const double dsqrt_dni =
            (s_ij > 0.0) ? (0.5 * nj / s_ij) : (0.5 * std::sqrt(std::max(nj / eps, 0.0)));

        const double K_ww = -s_ij;
        const double K_ss = ni * nj;
        const double K_rm = s_ij;

        const double dKww_dni = -dsqrt_dni;
        const double dKss_dni = nj;
        const double dKrm_dni = dsqrt_dni;

        return da_ww * K_ww + a_ww * dKww_dni + da_ss * K_ss + a_ss * dKss_dni
             + da_rm * K_rm + a_rm * dKrm_dni;
    }

private:
    double smooth_;
    double w(double n) const {
        return 0.5 * (1.0 - std::tanh((n - 0.5) / smooth_));
    }
    double w_grad(double n) const {
        const double t = std::tanh((n - 0.5) / smooth_);
        return -0.5 * (1.0 - t * t) / smooth_;
    }
};

// Load separable NN kernel from JSON (implemented in src/NNFunctional.cpp).
std::unique_ptr<Functional> load_nn_functional(const std::string& json_path);

// Convenience factory.
inline std::unique_ptr<Functional> make_functional(const std::string& key,
                                                   double alpha = 0.55) {
    if (key == "HF")      return std::make_unique<HFFunctional>();
    if (key == "Mueller" || key == "BB") return std::make_unique<MuellerFunctional>();
    if (key == "GU")      return std::make_unique<GUFunctional>();
    if (key == "CGA")     return std::make_unique<CGAFunctional>();
    // Legacy label ``CHF`` maps to CHF (corrected Hartree–Fock kernel).
    if (key == "CHF" || key == "CHF") return std::make_unique<CHFFunctional>();
    if (key == "Power")   return std::make_unique<PowerFunctional>(alpha);
    if (key == "BOW")     return std::make_unique<BOWFunctional>();
    if (key == "SymBow")  return std::make_unique<SymBOWFunctional>();
    if (key == "BBC1")    return std::make_unique<BBC1Functional>();
    if (key == "BBC3")   return std::make_unique<BBC3Functional>();
    if (key == "GEO")     return std::make_unique<GEOFunctional>();
    if (key.rfind("BOW@", 0) == 0) {
        return std::make_unique<BOWFunctional>(std::stod(key.substr(4)));
    }
    if (key.rfind("SymBow@", 0) == 0) {
        return std::make_unique<SymBOWFunctional>(std::stod(key.substr(7)));
    }
    // OptGeo@w;k — HF + sigmoid correlation in (n-1/2)^2 pair variable.
    if (key.rfind("OptGeo@", 0) == 0) {
        const std::string rest = key.substr(7);
        double w = 0.0, k = 0.0;
        if (std::sscanf(rest.c_str(), "%lf;%lf", &w, &k) == 2) {
            return std::make_unique<OptGeoFunctional>(w, k);
        }
        return nullptr;
    }
    // HybOpt@lambda;alpha — HF / Power(alpha) mixture (legacy OptGM@ accepted).
    if (key.rfind("HybOpt@", 0) == 0) {
        const std::string rest = key.substr(8);
        double lam = 0.0, al = 0.55;
        if (std::sscanf(rest.c_str(), "%lf;%lf", &lam, &al) == 2) {
            return std::make_unique<HybOptFunctional>(lam, al);
        }
        return nullptr;
    }
    if (key.rfind("OptGM@", 0) == 0) {
        const std::string rest = key.substr(6);
        double lam = 0.0, al = 0.55;
        if (std::sscanf(rest.c_str(), "%lf;%lf", &lam, &al) == 2) {
            return std::make_unique<HybOptFunctional>(lam, al);
        }
        return nullptr;
    }
    // The Beta functional needs an explicit exponent; callers should use
    // make_functional("Beta", beta) explicitly (alpha is reused as beta).
    if (key == "Beta")    return std::make_unique<BetaFunctional>(alpha);
    // NN@path/to/model.json — auto-detect kernel_type (separable | pair).
    // Legacy separable model.json files continue to load unchanged because
    // ``kernel_type`` defaults to ``"separable"`` when absent.  Setting
    // ``"kernel_type": "pair"`` in the JSON root selects NNPairFunctional
    // (non-separable two-input MLP coupling kernel).
    if (key.rfind("NN@", 0) == 0) {
        const std::string path = key.substr(3);
        if (!path.empty()) {
            return load_nn_functional(path);
        }
        return nullptr;
    }
    // NNPair@path/to/model.json — explicit non-separable pair NN kernel
    // K(n_i, n_j) = sqrt(n_i n_j) * MLP([n_i+n_j, n_i n_j]) (raw output,
    // sign-free).  Equivalent to ``NN@`` with ``"kernel_type": "pair"``.
    if (key.rfind("NNPair@", 0) == 0) {
        const std::string path = key.substr(7);
        if (!path.empty()) {
            return load_nn_functional(path);
        }
        return nullptr;
    }
    return nullptr;
}

}  // namespace rdmft
