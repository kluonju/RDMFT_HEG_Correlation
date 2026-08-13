#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "Energy.hpp"
#include "ExchangeKernel.hpp"
#include "Functional.hpp"
#include "Grid.hpp"
#include "HEG.hpp"

namespace rdmft {

struct SolveOptions {
    int    max_iter      = 800;
    double tol_n         = 1.0e-9;
    double tol_rho       = 1.0e-8;
    double mix           = 0.40;
    int    bisect_iter   = 80;
    double mu_lo         = -50.0;
    double mu_hi         =  50.0;
    bool   verbose       = false;
    // If set, seed n(k) uniformly to this value (clamped to [0,1]) and use a
    // single start instead of the smearing multistart grid.
    std::optional<double> init_uniform_n;
};

struct SolveResult {
    std::vector<double> n;
    double mu        = 0.0;
    double rho       = 0.0;
    double T_per_V   = 0.0;
    double Exc_per_V = 0.0;
    double E_per_V   = 0.0;
    double E_per_N   = 0.0;
    int    iters     = 0;
    bool   converged = false;
    std::string failure_reason;
};

// For a factorized power kernel f(n) = n^alpha, the inner exchange "potential"
// at site i in the EL equation factorizes as
//
//     V_i = f(n_i) * U_i,    U_i = sum_j W_{ij} k_j f(n_j),
//
// (with W from ExchangeKernel) and the Euler-Lagrange equation reads
//
//     epsilon_i := k_i^2/2 - U_i * df/dn_i (with f^2 -> 2 f f') ... 
//
// Working through carefully (see Energy.hpp): with the volume measure
// dn_i  -> (1/pi^2) w_i k_i^2 dn(k_i),
//
//     epsilon_i = k_i^2 / 2  -  alpha * n_i^{alpha-1} * U_i / (pi * k_i),
//
// with U_i = sum_j W_{ij} k_j n_j^alpha.  This must equal the chemical
// potential mu wherever 0 < n_i < 1, while n_i = 0 (1) corresponds to
// epsilon_i > mu (< mu).
//
// The inversion of epsilon_i = mu for n_i is closed-form for alpha != 1
// (and reduces to a step function for alpha = 1, recovering HF).
inline std::vector<double>
update_occupations_power(double alpha,
                         double mu,
                         const std::vector<double>& U,
                         const Grid& g) {
    constexpr double pi = M_PI;
    const std::size_t N = g.n();
    std::vector<double> n(N, 0.0);

    if (alpha >= 1.0 - 1.0e-12) {  // HF: step function (with fractional edge)
        // Compute single-particle energies and find the threshold by
        // continuously interpolating the marginal node.  Specifically: assign
        // n = 1 for eps_i < mu, n = 0 for eps_i > mu, and a linear fraction
        // for the unique node closest in energy to mu.  This makes density
        // a continuous function of mu and the outer bisection converges.
        std::vector<double> eps(N, 0.0);
        for (std::size_t i = 0; i < N; ++i) {
            const double k = g.k[i];
            eps[i] = (k > 0.0) ? 0.5 * k * k - U[i] / (pi * k) : 0.0;
        }
        // Find the index of the smallest eps strictly above mu and the
        // largest strictly below.  Treat anything within a tiny tol as edge.
        const double tol = 1.0e-12;
        std::size_t edge = N;  // sentinel
        double edge_dist = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < N; ++i) {
            const double d = std::abs(eps[i] - mu);
            if (d < edge_dist) { edge_dist = d; edge = i; }
        }
        for (std::size_t i = 0; i < N; ++i) {
            if (i == edge) continue;
            n[i] = (eps[i] < mu) ? 1.0 : 0.0;
        }
        if (edge < N) {
            // Linear ramp around mu of width ~ local energy spacing.
            // Choose width = max(|eps[edge±1] - eps[edge]|).
            double w = 0.0;
            if (edge + 1 < N) w = std::max(w, std::abs(eps[edge+1] - eps[edge]));
            if (edge > 0)     w = std::max(w, std::abs(eps[edge] - eps[edge-1]));
            if (w < tol) w = tol;
            const double frac = std::clamp(0.5 - (eps[edge] - mu) / w, 0.0, 1.0);
            n[edge] = frac;
        }
        return n;
    }

    for (std::size_t i = 0; i < N; ++i) {
        const double k = g.k[i];
        if (k <= 0.0) { n[i] = 1.0; continue; }
        // alpha n_i^{alpha-1} U_i = pi k (k^2/2 - mu)
        const double R = pi * k * (0.5 * k * k - mu);
        if (U[i] <= 0.0) {
            n[i] = (R <= 0.0) ? 1.0 : 0.0;
            continue;
        }
        if (R <= 0.0) { n[i] = 1.0; continue; }
        const double base = R / (alpha * U[i]);  // > 0
        const double cand = std::pow(base, 1.0 / (alpha - 1.0));
        n[i] = std::clamp(cand, 0.0, 1.0);
    }
    return n;
}

inline std::vector<double> initial_step(double rs, const Grid& g) {
    std::vector<double> n(g.n(), 0.0);
    const double kf = HEG::kF(rs);
    for (std::size_t i = 0; i < g.n(); ++i) n[i] = (g.k[i] < kf) ? 1.0 : 0.0;
    return n;
}

inline std::vector<double> initial_constant(const Grid& g, double n0) {
    const double n = std::clamp(n0, 0.0, 1.0);
    return std::vector<double>(g.n(), n);
}

// For CGA, g(n) = sqrt(n(2-n)) on [0,1] and g'(n) = (1-n)/sqrt(n(2-n)).
// Solve g'(n) = s for n in [0,1].  Squaring (1-n)^2 = s^2 n(2-n) gives a
// quadratic; the physical root is
//     n = 1 - s / sqrt(1 + s^2).
inline double invert_h2_prime_for_cga(double s) {
    return std::clamp(1.0 - s / std::sqrt(1.0 + s * s), 0.0, 1.0);
}

// Solve  beta * (n(1-n))^(beta-1) * (1 - 2 n) = s   for n in [0, 1].
// Equivalent (with u = 1 - 2n,  x = n(1-n) = (1 - u^2)/4) to
//
//     u * (1 - u^2)^(beta-1) = B,    B = (s / beta) * 4^(beta-1).
//
// On (-1, 1) the LHS is monotonic for beta < 1 and for the betas we use,
// so a simple bracketed Brent / bisection step converges quickly.
inline double invert_dgbeta(double s, double beta) {
    if (beta <= 0.0) {
        // Degenerate: g_2 is constant in n away from endpoints, so the EL
        // equation is satisfied for any interior n.  Default to 0.5.
        return 0.5;
    }
    const double B = (s / beta) * std::pow(4.0, beta - 1.0);
    auto f = [&](double u) {
        const double y = 1.0 - u * u;
        if (y <= 0.0) return 0.0;
        return u * std::pow(y, beta - 1.0);
    };
    // Bracket: f is monotonically increasing in u for the betas of interest.
    // Endpoints: f(±1) = ±infinity for beta < 1, ±0 for beta > 1, depending.
    // Use a tight bracket of (-1+eps, 1-eps).
    const double eps = 1.0e-10;
    double lo = -1.0 + eps;
    double hi =  1.0 - eps;
    double flo = f(lo);
    double fhi = f(hi);
    // Saturate if outside the bracket.
    if (B <= flo) return std::clamp(0.5 * (1.0 - lo), 0.0, 1.0);
    if (B >= fhi) return std::clamp(0.5 * (1.0 - hi), 0.0, 1.0);
    for (int it = 0; it < 80; ++it) {
        double mid = 0.5 * (lo + hi);
        double fm  = f(mid);
        if (fm < B) lo = mid; else hi = mid;
        if (hi - lo < 1.0e-12) break;
    }
    const double u = 0.5 * (lo + hi);
    return std::clamp(0.5 * (1.0 - u), 0.0, 1.0);
}

// Occupation update for the multi-power "GEO" kernel
//
//   K(n_p, n_q) = c1 f1(n_p) f1(n_q) + c2 f2(n_p) f2(n_q) + c3 f3(n_p) f3(n_q)
//   f1(n) = n,  f2(n) = sqrt(n),  f3(n) = n^{3/4}
//
// with the GEO weights (c1, c2, c3) = (1/4, 1/4, 1/2).  Defining
//
//   U_alpha,i = sum_j W_{ij} k_j f_alpha(n_j),
//
// the Euler-Lagrange equation reads
//
//   c1 f1'(n_i) U1_i + c2 f2'(n_i) U2_i + c3 f3'(n_i) U3_i
//                                       = pi k_i (k_i^2/2 - mu) =: R_i.
//
// f_alpha'(n) = alpha n^{alpha - 1} is strictly decreasing in n on (0, 1)
// for alpha in {1/2, 3/4} and constant for alpha = 1.  Hence whenever the
// U_alpha,i are non-negative the left-hand side is a monotonically
// decreasing function of n_i and the equation has a unique solution in
// [0, 1], found here by bisection.  When all U_alpha vanish (e.g. an HF
// step initial guess at the very first iteration) we fall back to the HF
// step rule R_i >= 0 -> n_i = 0, R_i < 0 -> n_i = 1.
inline std::vector<double>
update_occupations_geo(double mu,
                       const std::vector<double>& U1,
                       const std::vector<double>& U2,
                       const std::vector<double>& U3,
                       const Grid& g) {
    constexpr double pi   = M_PI;
    constexpr double c1   = 0.25;
    constexpr double c2   = 0.25;
    constexpr double c3   = 0.50;
    const std::size_t N   = g.n();
    const double tiny     = 1.0e-14;
    std::vector<double> n(N, 0.0);

    for (std::size_t i = 0; i < N; ++i) {
        const double k = g.k[i];
        if (k <= 0.0) { n[i] = 1.0; continue; }
        const double R = pi * k * (0.5 * k * k - mu);

        const double u1 = U1[i];
        const double u2 = U2[i];
        const double u3 = U3[i];
        const double sumU = std::abs(u1) + std::abs(u2) + std::abs(u3);
        if (sumU < tiny) {
            n[i] = (R >= 0.0) ? 0.0 : 1.0;
            continue;
        }

        // LHS(n) = c1 * 1 * U1
        //       + c2 * (1/(2 sqrt n))      * U2
        //       + c3 * (3/(4 n^{1/4}))     * U3
        // Strictly decreasing in n for n in (0, 1) when U2, U3 >= 0.
        auto lhs = [&](double nn) {
            const double nc = std::clamp(nn, 1.0e-14, 1.0);
            const double t1 = c1 * u1;
            const double t2 = c2 * 0.5  / std::sqrt(nc)         * u2;
            const double t3 = c3 * 0.75 * std::pow(nc, -0.25)   * u3;
            return t1 + t2 + t3;
        };

        const double eps = 1.0e-12;
        const double lo_n = eps;
        const double hi_n = 1.0;
        const double L_lo = lhs(lo_n);  // largest value
        const double L_hi = lhs(hi_n);  // smallest value
        // Saturate on the appropriate boundary if the EL equation has no
        // interior solution.  Note LHS is decreasing in n, so:
        //   R >= L_lo  -> push n down  -> n = 0
        //   R <= L_hi  -> push n up    -> n = 1
        if (R >= L_lo) { n[i] = 0.0; continue; }
        if (R <= L_hi) { n[i] = 1.0; continue; }
        double a = lo_n, b = hi_n;
        for (int it = 0; it < 80; ++it) {
            const double m = 0.5 * (a + b);
            if (lhs(m) > R) a = m; else b = m;
            if (b - a < 1.0e-12) break;
        }
        n[i] = std::clamp(0.5 * (a + b), 0.0, 1.0);
    }
    return n;
}

// HybOpt (HF/Power mix): K = c1 n_i n_j + c2 n_i^alpha n_j^alpha with c1 = 1-lambda, c2 = lambda.
// EL per orbital: c1 f1'(n) U1 + c2 f2'(n) U2 = pi k (k^2/2 - mu), f1=n, f2=n^alpha.
// For 0 < alpha < 1 and U1, U2 >= 0 the LHS is strictly decreasing in n on (0,1).
inline std::vector<double>
update_occupations_hf_power_mix(double mu,
                                const std::vector<double>& U1,
                                const std::vector<double>& U2,
                                const Grid& g,
                                double c1,
                                double c2,
                                double alpha) {
    constexpr double pi = M_PI;
    const std::size_t N = g.n();
    const double tiny = 1.0e-14;
    std::vector<double> n(N, 0.0);

    for (std::size_t i = 0; i < N; ++i) {
        const double k = g.k[i];
        if (k <= 0.0) { n[i] = 1.0; continue; }
        const double R = pi * k * (0.5 * k * k - mu);

        const double u1 = U1[i];
        const double u2 = U2[i];
        const double sumU = std::abs(c1 * u1) + std::abs(c2 * u2);
        if (sumU < tiny) {
            n[i] = (R >= 0.0) ? 0.0 : 1.0;
            continue;
        }

        auto lhs = [&](double nn) {
            const double nc = std::clamp(nn, 1.0e-14, 1.0);
            const double t1 = c1 * u1;
            const double t2 =
                (c2 > 0.0 && alpha > 0.0) ? c2 * alpha * std::pow(nc, alpha - 1.0) * u2 : 0.0;
            return t1 + t2;
        };

        const double eps = 1.0e-12;
        const double lo_n = eps;
        const double hi_n = 1.0;
        const double L_lo = lhs(lo_n);
        const double L_hi = lhs(hi_n);
        if (R >= L_lo) { n[i] = 0.0; continue; }
        if (R <= L_hi) { n[i] = 1.0; continue; }
        double a = lo_n, b = hi_n;
        for (int it = 0; it < 80; ++it) {
            const double m = 0.5 * (a + b);
            if (lhs(m) > R) a = m; else b = m;
            if (b - a < 1.0e-12) break;
        }
        n[i] = std::clamp(0.5 * (a + b), 0.0, 1.0);
    }
    return n;
}

// Same EL structure as `update_occupations_geo` but with tunable channel
// weights (c1, c2, c3) matching OptGeoFunctional / generalized GEO sums.
inline std::vector<double>
update_occupations_optGM(double mu,
                         const std::vector<double>& U1,
                         const std::vector<double>& U2,
                         const std::vector<double>& U3,
                         const Grid& g,
                         double c1,
                         double c2,
                         double c3) {
    constexpr double pi   = M_PI;
    const std::size_t N   = g.n();
    const double tiny     = 1.0e-14;
    std::vector<double> n(N, 0.0);

    for (std::size_t i = 0; i < N; ++i) {
        const double k = g.k[i];
        if (k <= 0.0) { n[i] = 1.0; continue; }
        const double R = pi * k * (0.5 * k * k - mu);

        const double u1 = U1[i];
        const double u2 = U2[i];
        const double u3 = U3[i];
        const double sumU = std::abs(u1) + std::abs(u2) + std::abs(u3);
        if (sumU < tiny) {
            n[i] = (R >= 0.0) ? 0.0 : 1.0;
            continue;
        }

        auto lhs = [&](double nn) {
            const double nc = std::clamp(nn, 1.0e-14, 1.0);
            const double t1 = c1 * u1;
            const double t2 = c2 * 0.5  / std::sqrt(nc)         * u2;
            const double t3 = c3 * 0.75 * std::pow(nc, -0.25)   * u3;
            return t1 + t2 + t3;
        };

        const double eps = 1.0e-12;
        const double lo_n = eps;
        const double hi_n = 1.0;
        const double L_lo = lhs(lo_n);
        const double L_hi = lhs(hi_n);
        if (R >= L_lo) { n[i] = 0.0; continue; }
        if (R <= L_hi) { n[i] = 1.0; continue; }
        double a = lo_n, b = hi_n;
        for (int it = 0; it < 80; ++it) {
            const double m = 0.5 * (a + b);
            if (lhs(m) > R) a = m; else b = m;
            if (b - a < 1.0e-12) break;
        }
        n[i] = std::clamp(0.5 * (a + b), 0.0, 1.0);
    }
    return n;
}

// Closed-form occupation update for kernels with the additive structure
//
//     K(n_i, n_j) = n_i n_j + g(n_i) g(n_j),
//
// or, for CGA only, K = a [ n_i n_j + g(n_i) g(n_j) ] with a = 1/2 (see
// ``el_scale`` in ``update_occupations_additive``).
//
// covering CGA, CHF (additive hole), and Beta
// (g(n) = (n(1-n))^beta).  For each i we solve
//
//     U_HF_i + g'(n_i) U_g_i = pi k_i (k_i^2/2 - mu)
//
// for n_i in [0, 1] using the dedicated 1-D inverters above.  When U_g_i is
// numerically too small we fall back to the HF step occupation, which is
// the correct limit when g_j -> 0 everywhere.
template <class Inverter>
inline std::vector<double>
update_occupations_additive(double mu,
                            const std::vector<double>& U_HF,
                            const std::vector<double>& U_g,
                            const Grid& g,
                            Inverter invert_dg,
                            double el_scale = 1.0) {
    constexpr double pi = M_PI;
    const std::size_t N = g.n();
    std::vector<double> n(N, 0.0);
    const double tiny = 1.0e-14;
    const double esc = (el_scale > 0.0) ? el_scale : 1.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double k  = g.k[i];
        if (k <= 0.0) { n[i] = 1.0; continue; }
        const double R  = pi * k * (0.5 * k * k - mu);
        const double dU = R / esc - U_HF[i];
        if (std::abs(U_g[i]) < tiny) {
            // No hole contribution: revert to the HF step rule.  Since the
            // sign of U_HF compares against R, this matches the HF EL.
            n[i] = (dU >= 0.0) ? 0.0 : 1.0;
            continue;
        }
        const double s = dU / U_g[i];
        n[i] = invert_dg(s);
    }
    return n;
}

// Generic helper that computes U_i = sum_j W(i,j) k_j fn(n_j).  Used to build
// the U_HF and U_g sums above, matching the convention of the existing
// `update_occupations_power` (no `w_j` factor, exactly matching `V_inner`).
template <class Fn>
inline std::vector<double> compute_U_with(const std::vector<double>& nv,
                                          const Grid& g,
                                          const ExchangeKernel& W,
                                          Fn&& fn) {
    const std::size_t N = g.n();
    std::vector<double> U(N, 0.0);
    std::vector<double> kf(N);
    for (std::size_t j = 0; j < N; ++j) {
        kf[j] = g.k[j] * fn(nv[j]);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N > 96)
#endif
    for (std::size_t i = 0; i < N; ++i) {
        const double* row = W.w_row(i);
        double s = 0.0;
        for (std::size_t j = 0; j < N; ++j) {
            s += row[j] * kf[j];
        }
        U[i] = s;
    }
    return U;
}

// Smeared initial guess: a sigmoid centred at the Fermi wave vector, which
// helps Beta, CGA / CHF, GEO, optGeo, and hybopt (HF/Power mix) escape a trivial HF-step-like fixed
// point.  width controls the fractional smearing relative to k_F.
inline std::vector<double>
initial_smeared(double rs, const Grid& g, double width = 0.10) {
    const std::size_t N = g.n();
    const double kf = HEG::kF(rs);
    const double w  = std::max(width, 1.0e-3) * kf;
    std::vector<double> n(N, 0.0);
    for (std::size_t i = 0; i < N; ++i) {
        n[i] = 1.0 / (1.0 + std::exp((g.k[i] - kf) / w));
    }
    return n;
}

// Self-consistent solve for any Functional.  Power-family functionals use the
// closed-form occupation update; CGA, CHF, and Beta use the additive branch
// with a dedicated 1-D inverter.  Other kernels, including arbitrary
// factorized kernels whose df need not be monotone, use projected gradient.
// All branches bisect on mu for particle conservation.
//
// Beta, CGA, and CHF can show a competing HF-like minimum; we therefore try
// several smeared initial guesses and keep the lowest-energy converged result
// when ``needs_multistart`` is set.
inline SolveResult
solve_rdmft(double rs,
            const Functional& F,
            const Grid& g,
            const ExchangeKernel& W,
            const SolveOptions& opt = {}) {
    if (!std::isfinite(rs) || rs <= 0.0) {
        throw std::invalid_argument("solve_rdmft requires finite rs > 0");
    }
    if (opt.max_iter <= 0 || opt.bisect_iter <= 0
        || !std::isfinite(opt.tol_n) || opt.tol_n <= 0.0
        || !std::isfinite(opt.tol_rho) || opt.tol_rho <= 0.0
        || !std::isfinite(opt.mix) || opt.mix <= 0.0 || opt.mix > 1.0
        || !std::isfinite(opt.mu_lo) || !std::isfinite(opt.mu_hi)
        || opt.mu_lo >= opt.mu_hi) {
        throw std::invalid_argument("invalid SolveOptions");
    }
    if (opt.init_uniform_n
        && (!std::isfinite(*opt.init_uniform_n)
            || *opt.init_uniform_n < 0.0 || *opt.init_uniform_n > 1.0)) {
        throw std::invalid_argument("init_uniform_n must be finite and in [0,1]");
    }

    const double rho_target = HEG::density(rs);
    const double rho_tolerance =
        opt.tol_rho * std::max(1.0, std::abs(rho_target));

    auto valid_occupations = [&](const std::vector<double>& nv) {
        if (nv.size() != g.n()) return false;
        for (double value : nv) {
            if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
               return false;
            }
        }
        return true;
    };

    // Detect Power-family functionals.  GU shares the Mueller kernel in the
    // HEG (the orbital self-interaction terms vanish for plane waves), so it
    // takes the closed-form Power(alpha=1/2) update too.
    const PowerFunctional*   pf   = dynamic_cast<const PowerFunctional*>(&F);
    const HFFunctional*      hf   = dynamic_cast<const HFFunctional*>(&F);
    const MuellerFunctional* mu_f = dynamic_cast<const MuellerFunctional*>(&F);
    const GUFunctional*      gu_f = dynamic_cast<const GUFunctional*>(&F);
    const bool power_family = (pf != nullptr) || (hf != nullptr)
                            || (mu_f != nullptr) || (gu_f != nullptr);

    // Additive kernel: CHF inherits ``BetaFunctional(1/2)``; Beta is general
    // exponent; CGA uses K = (1/2) [ n_i n_j + g_i g_j ] (``CGAFunctional``).
    const CGAFunctional*   cga   = dynamic_cast<const CGAFunctional*>(&F);
    const BetaFunctional*  beta_f = dynamic_cast<const BetaFunctional*>(&F);
    const bool additive = (cga != nullptr) || (beta_f != nullptr);

    // GEO uses a non-factorizable multi-power kernel; it is solved via the
    // generic projected-gradient branch below (no special-case here).
    const GEOFunctional*    geo       = dynamic_cast<const GEOFunctional*>(&F);
    const OptGeoFunctional* optgeo    = dynamic_cast<const OptGeoFunctional*>(&F);
    const HybOptFunctional* hybopt = dynamic_cast<const HybOptFunctional*>(&F);
    const BBC1Functional*   bbc1      = dynamic_cast<const BBC1Functional*>(&F);
    const BBC3Functional*   bbc3      = dynamic_cast<const BBC3Functional*>(&F);
    const BOWFunctional*    bow       = dynamic_cast<const BOWFunctional*>(&F);
    const PbeHfMuFunctional* pbehfmu  = dynamic_cast<const PbeHfMuFunctional*>(&F);

    double alpha = 1.0;
    if (pf)        alpha = pf->alpha();
    else if (mu_f) alpha = 0.5;
    else if (gu_f) alpha = 0.5;
    else if (hf)   alpha = 1.0;

    auto density_of = [&](const std::vector<double>& nv) {
        return EnergyEvaluator::density(nv, g);
    };
    const std::vector<double> full_occupations(g.n(), 1.0);
    const double maximum_grid_density = density_of(full_occupations);
    if (!std::isfinite(maximum_grid_density)
        || maximum_grid_density + rho_tolerance < rho_target) {
        SolveResult failed;
        failed.failure_reason =
            "chemical-potential search cannot bracket the target density on this grid";
        return failed;
    }
    auto project_to_target_density = [&](std::vector<double>& nv) {
        if (!valid_occupations(nv)) return false;
        double lo = -1.0;
        double hi = 1.0;
        auto shifted_density = [&](double shift) {
            double sum = 0.0;
            constexpr double pi = M_PI;
            for (std::size_t i = 0; i < g.n(); ++i) {
                const double value = std::clamp(nv[i] + shift, 0.0, 1.0);
                sum += g.w[i] * g.k[i] * g.k[i] * value;
            }
            return sum / (pi * pi);
        };
        for (int b = 0; b < opt.bisect_iter; ++b) {
            const double mid = 0.5 * (lo + hi);
            if (shifted_density(mid) > rho_target) hi = mid;
            else                                    lo = mid;
        }
        const double shift = 0.5 * (lo + hi);
        std::vector<double> projected = nv;
        for (double& value : projected) {
            value = std::clamp(value + shift, 0.0, 1.0);
        }
        const double rho = density_of(projected);
        if (!valid_occupations(projected) || !std::isfinite(rho)
            || std::abs(rho - rho_target) > rho_tolerance) {
            return false;
        }
        nv = std::move(projected);
        return true;
    };

    // Multi-start strategy for additive kernels: probe several smearing
    // widths and keep the lowest-energy result.  CGA / Beta have a stable
    // local minimum at the HF step distribution because g'(n) -> infinity at
    // the endpoints, so we deliberately seed a broad range of fractionally-
    // occupied initial conditions.  For factorized and generic functionals
    // one start (sharp step) is enough.
    // Smeared multi-start helps additive CGA / CHF / Beta and GEO / optGeo /
    // hybopt / BBC3 escape the trivial HF-step fixed point.
    const bool needs_multistart = additive || (geo != nullptr)
                                  || (optgeo != nullptr) || (hybopt != nullptr)
                                  || (bbc1 != nullptr) || (bbc3 != nullptr)
                                  || (bow != nullptr) || (pbehfmu != nullptr);
    const bool uniform_init = opt.init_uniform_n.has_value();
    const std::vector<std::pair<bool, double>> starts = uniform_init
        ? std::vector<std::pair<bool, double>>{{true, 0.0}}
        : (needs_multistart
               ? std::vector<std::pair<bool, double>>{
                     {false, 0.05}, {false, 0.10}, {false, 0.20},
                     {false, 0.40}, {false, 0.80}, {false, 1.50}}
               : std::vector<std::pair<bool, double>>{{true, 0.0}});

    SolveResult best;
    bool best_set = false;
    SolveResult fallback;
    bool fallback_set = false;
    std::string last_failure;
    for (const auto& start : starts) {

    std::vector<double> n =
        uniform_init ? initial_constant(g, *opt.init_uniform_n)
                     : (start.first ? initial_step(rs, g)
                                    : initial_smeared(rs, g, start.second));
    if (!project_to_target_density(n)) {
        last_failure = "failed to project initial occupations to the target density";
        continue;
    }
    double mu = 0.5 * HEG::kF(rs) * HEG::kF(rs);
    int it = 0;
    bool converged = false;
    bool numerical_failure = false;
    std::string failure_reason;

    auto compute_U = [&](const std::vector<double>& nv) {
        const std::size_t N = g.n();
        std::vector<double> U(N, 0.0);
        std::vector<double> kf(N);
        for (std::size_t j = 0; j < N; ++j) {
            kf[j] = g.k[j] * F.f(nv[j]);
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N > 96)
#endif
        for (std::size_t i = 0; i < N; ++i) {
            const double* row = W.w_row(i);
            double s = 0.0;
            for (std::size_t j = 0; j < N; ++j) {
                s += row[j] * kf[j];
            }
            U[i] = s;
        }
        return U;
    };

    // Bisection on mu with the given occupation update functor.
    // density(mu) is monotonically increasing in mu, so a well-formed
    // bracket has rho(lo) <= rho_target <= rho(hi).  If the requested
    // bracket [opt.mu_lo, opt.mu_hi] is too tight (e.g. ill-conditioned NN
    // kernels at large r_s push mu well outside [-50, 50]) we expand it
    // geometrically (up to ``max_expand`` doublings of the span) before
    // falling back to the saturated boundary.  This avoids manual
    // --mu-lo/--mu-hi tuning per functional / r_s and lets the optimizer
    // make progress even on bad NN weights.
    struct BisectionResult {
        bool success = false;
        double mu = 0.0;
        std::string error;
    };
    auto adaptive_bisect = [&](auto update_n) {
        BisectionResult result;
        double lo = opt.mu_lo;
        double hi = opt.mu_hi;
        const int max_expand = 16;
        auto evaluate = [&](double trial_mu, double& rho) {
            const auto trial = update_n(trial_mu);
            if (!valid_occupations(trial)) return false;
            rho = density_of(trial);
            return std::isfinite(rho);
        };
        double rho_lo = 0.0;
        double rho_hi = 0.0;
        if (!evaluate(lo, rho_lo) || !evaluate(hi, rho_hi)) {
            result.error = "chemical-potential bracketing produced non-finite occupations";
            return result;
        }
        for (int e = 0; e < max_expand; ++e) {
            if (rho_lo <= rho_target && rho_hi >= rho_target) break;
            const double span = std::max(hi - lo, 1.0);
            if (rho_lo > rho_target) {
                lo -= span;
                if (!std::isfinite(lo) || !evaluate(lo, rho_lo)) {
                    result.error = "chemical-potential lower-bracket expansion failed";
                    return result;
                }
            }
            if (rho_hi < rho_target) {
                hi += span;
                if (!std::isfinite(hi) || !evaluate(hi, rho_hi)) {
                    result.error = "chemical-potential upper-bracket expansion failed";
                    return result;
                }
            }
        }
        if (rho_lo > rho_hi) {
            result.error = "occupation update is not monotone in chemical potential";
            return result;
        }
        if (rho_lo > rho_target || rho_hi < rho_target) {
            result.error = "chemical-potential search could not bracket the target density";
            return result;
        }
        for (int b = 0; b < opt.bisect_iter; ++b) {
            const double m = 0.5 * (lo + hi);
            double rho_mid = 0.0;
            if (!evaluate(m, rho_mid)) {
                result.error = "chemical-potential bisection produced non-finite occupations";
                return result;
            }
            if (rho_mid > rho_target) hi = m;
            else                       lo = m;
        }
        result.success = true;
        result.mu = 0.5 * (lo + hi);
        return result;
    };

    auto bisect_mu_factorized = [&](const std::vector<double>& U) {
        return adaptive_bisect(
            [&](double m) { return update_occupations_power(alpha, m, U, g); });
    };

    // Pick the appropriate hole inverter for additive kernels.
    auto invert_dg = [&](double s) -> double {
        if (cga)    return invert_h2_prime_for_cga(s);
        if (beta_f) return invert_dgbeta(s, beta_f->beta());
        return 0.5;
    };

    auto identity_id = [](double n_) { return n_; };

    for (it = 0; it < opt.max_iter; ++it) {
        std::vector<double> n_target;

        if (power_family) {
            auto U = compute_U(n);
            const auto bracket = bisect_mu_factorized(U);
            if (!bracket.success) {
                failure_reason = bracket.error;
                numerical_failure = true;
                break;
            }
            mu = bracket.mu;
            n_target = update_occupations_power(alpha, mu, U, g);
        } else if (hybopt) {
            const double lam = hybopt->lambda_mix();
            const double al = hybopt->alpha();
            auto f_pow = [al](double nn) {
                const double eps = 1.0e-14;
                const double x = (nn > 0.0) ? nn : 0.0;
                const double xc = (x > eps) ? x : eps;
                return std::pow(xc, al);
            };
            auto U1 = compute_U_with(n, g, W, identity_id);
            auto U2 = compute_U_with(n, g, W, f_pow);
            const double c1 = 1.0 - lam;
            const double c2 = lam;
            const auto bracket = adaptive_bisect([&](double m) {
                return update_occupations_hf_power_mix(m, U1, U2, g, c1, c2, al);
            });
            if (!bracket.success) {
                failure_reason = bracket.error;
                numerical_failure = true;
                break;
            }
            mu = bracket.mu;
            n_target = update_occupations_hf_power_mix(mu, U1, U2, g, c1, c2, al);
        } else if (geo) {
            auto f1 = [](double nn) { return nn; };
            auto f2 = [](double nn) { return nn > 0.0 ? std::sqrt(nn) : 0.0; };
            auto f3 = [](double nn) { return nn > 0.0 ? std::pow(nn, 0.75) : 0.0; };
            auto U1 = compute_U_with(n, g, W, f1);
            auto U2 = compute_U_with(n, g, W, f2);
            auto U3 = compute_U_with(n, g, W, f3);

            const auto bracket = adaptive_bisect([&](double m) {
                return update_occupations_geo(m, U1, U2, U3, g);
            });
            if (!bracket.success) {
                failure_reason = bracket.error;
                numerical_failure = true;
                break;
            }
            mu = bracket.mu;
            n_target = update_occupations_geo(mu, U1, U2, U3, g);
        } else if (additive) {
            // Additive kernel K = n_i n_j + g(n_i) g(n_j): closed-form update
            // for n_i once mu is found by bisection.  We use a soft floor on n
            // in [n_floor, 1 - n_floor] when computing g(n) so that even
            // near-saturated occupations contribute a finite hole U_g, which
            // lets the SCF escape the metastable HF-like fixed point.  The
            // floor is taken small enough that converged solutions are not
            // affected to leading order (energies use the unmodified n).
            const double n_floor = 1.0e-6;
            const double el_scale = cga ? 0.5 : 1.0;
            auto g_of = [&](double n_) {
                const double nc = std::clamp(n_, n_floor, 1.0 - n_floor);
                if (cga) {
                    return std::sqrt(std::max(nc * (2.0 - nc), 0.0));
                }
                if (beta_f) {
                    return std::pow(nc * (1.0 - nc), beta_f->beta());
                }
                return 0.0;
            };

            auto U_HF = compute_U_with(n, g, W, identity_id);
            auto U_g  = compute_U_with(n, g, W, g_of);

            const auto bracket = adaptive_bisect([&](double m) {
                return update_occupations_additive(
                    m, U_HF, U_g, g, invert_dg, el_scale);
            });
            if (!bracket.success) {
                failure_reason = bracket.error;
                numerical_failure = true;
                break;
            }
            mu = bracket.mu;
            n_target = update_occupations_additive(
                mu, U_HF, U_g, g, invert_dg, el_scale);
        } else {
            // Generic projected-gradient descent with a conservative Armijo
            // backtracking line search.  For each trial step length we bisect
            // the Lagrange multiplier mu so the projected/clamped update obeys
            // the density constraint before the energy decrease is tested.
            auto eps = EnergyEvaluator::pseudo_energy(n, g, W, F);
            const double E0 = EnergyEvaluator::kinetic_per_volume(n, g)
                            + EnergyEvaluator::xc_per_volume(n, g, W, F);
            if (!std::isfinite(E0)) {
                failure_reason = "projected-gradient energy is non-finite";
                numerical_failure = true;
                break;
            }
            for (double value : eps) {
                if (!std::isfinite(value)) {
                    failure_reason = "projected-gradient derivative is non-finite";
                    numerical_failure = true;
                    break;
                }
            }
            if (numerical_failure) break;
            const double c_armijo = 1.0e-4;
            const double shrink = 0.5;
            const double min_step = 1.0e-8;

            auto pgd_at = [&](double step, double m) {
                std::vector<double> n_try(g.n(), 0.0);
                for (std::size_t i = 0; i < g.n(); ++i) {
                    n_try[i] = std::clamp(n[i] - step * (eps[i] - m), 0.0, 1.0);
                }
                return n_try;
            };
            auto project_for_step = [&](double step, double& mu_step,
                                        std::vector<double>& projected) {
                const auto bracket = adaptive_bisect(
                    [&](double m) { return pgd_at(step, m); });
                if (!bracket.success) {
                    failure_reason = bracket.error;
                    return false;
                }
                mu_step = bracket.mu;
                projected = pgd_at(step, mu_step);
                return valid_occupations(projected);
            };
            auto directional_derivative = [&](const std::vector<double>& n_try) {
                constexpr double pi = M_PI;
                double gd = 0.0;
                for (std::size_t i = 0; i < g.n(); ++i) {
                    const double ki = g.k[i];
                    const double pref = g.w[i] * ki * ki / (pi * pi);
                    gd += pref * eps[i] * (n_try[i] - n[i]);
                }
                return gd;
            };

            double step = 1.0;
            double best_E = std::numeric_limits<double>::infinity();
            std::vector<double> best_n = n;
            double best_mu = mu;
            bool accepted = false;
            while (step >= min_step) {
                double trial_mu = mu;
                std::vector<double> trial;
                if (!project_for_step(step, trial_mu, trial)) {
                    numerical_failure = true;
                    break;
                }
                const double E_trial = EnergyEvaluator::kinetic_per_volume(trial, g)
                                     + EnergyEvaluator::xc_per_volume(trial, g, W, F);
                const double gd = directional_derivative(trial);
                if (!std::isfinite(E_trial) || !std::isfinite(gd)) {
                    failure_reason = "projected-gradient line search is non-finite";
                    numerical_failure = true;
                    break;
                }
                const double armijo_rhs = E0 + c_armijo * std::min(gd, 0.0);
                if (E_trial < best_E) {
                    best_E = E_trial;
                    best_n = trial;
                    best_mu = trial_mu;
                }
                if (E_trial <= armijo_rhs) {
                    n_target = std::move(trial);
                    mu = trial_mu;
                    accepted = true;
                    break;
                }
                step *= shrink;
            }
            if (numerical_failure) break;
            if (!accepted) {
                if (best_E < E0) {
                    n_target = std::move(best_n);
                    mu = best_mu;
                } else {
                    // No downhill step survived backtracking; stay put rather
                    // than taking an energy-increasing PGD step.
                    n_target = n;
                }
            }
        }

        if (!valid_occupations(n_target)) {
            failure_reason = "occupation update produced invalid values";
            numerical_failure = true;
            break;
        }
        double dn_max = 0.0;
        for (std::size_t i = 0; i < g.n(); ++i) {
            const double n_new = (1.0 - opt.mix) * n[i] + opt.mix * n_target[i];
            if (!std::isfinite(n_new)) {
                failure_reason = "occupation mixing produced non-finite values";
                numerical_failure = true;
                break;
            }
            dn_max = std::max(dn_max, std::abs(n_new - n[i]));
            n[i] = n_new;
        }
        if (numerical_failure) break;

        const double rho_now = density_of(n);
        if (!std::isfinite(rho_now)) {
            failure_reason = "density became non-finite";
            numerical_failure = true;
            break;
        }
        if (opt.verbose && (it % 20 == 0)) {
            double Eh = EnergyEvaluator::kinetic_per_volume(n, g)
                      + EnergyEvaluator::xc_per_volume(n, g, W, F);
            std::fprintf(stderr,
                "    [iter %4d]  mu=%+.6f  rho=%.6e  dn=%.2e  E/V=%+.6e\n",
                it, mu, rho_now, dn_max, Eh);
        }
        if (dn_max < opt.tol_n
            && std::abs(rho_now - rho_target) <= rho_tolerance) {
            converged = true;
            ++it;
            break;
        }
    }

    SolveResult res;
    res.n         = n;
    res.mu        = mu;
    res.rho       = density_of(n);
    res.T_per_V   = EnergyEvaluator::kinetic_per_volume(n, g);
    res.Exc_per_V = EnergyEvaluator::xc_per_volume(n, g, W, F);
    res.E_per_V   = res.T_per_V + res.Exc_per_V;
    res.E_per_N   = (res.rho > 0.0) ? res.E_per_V / res.rho : 0.0;
    res.iters     = it;
    const bool finite_result =
        valid_occupations(res.n) && std::isfinite(res.mu)
        && std::isfinite(res.rho) && std::isfinite(res.T_per_V)
        && std::isfinite(res.Exc_per_V) && std::isfinite(res.E_per_V)
        && std::isfinite(res.E_per_N);
    const bool density_ok =
        finite_result && std::abs(res.rho - rho_target) <= rho_tolerance;
    res.converged = converged && !numerical_failure && density_ok;
    if (!res.converged) {
        if (failure_reason.empty()) {
            failure_reason = numerical_failure
                ? "numerical failure"
                : "maximum iterations reached before occupation and density convergence";
        }
        res.failure_reason = failure_reason;
        last_failure = failure_reason;
    }

    // Variational selection: among all the start configurations tried,
    // keep the lowest-energy valid, converged result.  A finite unconverged
    // result is retained only as a diagnostic fallback if every start fails.
    if (res.converged && (!best_set || res.E_per_N < best.E_per_N)) {
        best = res;
        best_set = true;
    } else if (!res.converged && finite_result) {
        const double rho_err = std::abs(res.rho - rho_target);
        const double fallback_rho_err = fallback_set
            ? std::abs(fallback.rho - rho_target)
            : std::numeric_limits<double>::infinity();
        if (!fallback_set || rho_err < fallback_rho_err
            || (rho_err == fallback_rho_err && res.E_per_N < fallback.E_per_N)) {
            fallback = res;
            fallback_set = true;
        }
    }
    }  // end multi-start loop

    if (best_set) return best;
    if (fallback_set) {
        fallback.converged = false;
        if (fallback.failure_reason.empty()) {
            fallback.failure_reason = "no valid converged multistart result";
        }
        return fallback;
    }
    SolveResult failed;
    failed.failure_reason = last_failure.empty()
        ? "no finite solver result"
        : last_failure;
    return failed;
}

}  // namespace rdmft
