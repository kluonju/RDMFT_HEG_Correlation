#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
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
    double mix           = 0.40;
    int    bisect_iter   = 80;
    double mu_lo         = -50.0;
    double mu_hi         =  50.0;
    bool   verbose       = false;
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

// Solve dh(n) = s for n in [0,1], where dh(n) = (1 - 2n) / (2 sqrt(n(1-n)))
// (the derivative of the CGA "hole" h(n) = sqrt(n(1-n))).  Closed-form:
//
//     u := 1 - 2n  satisfies  u^2 (1 + s^2) = s^2   =>   u = s / sqrt(1 + s^2),
//
// with the sign of u tracking the sign of s automatically.  Hence
// n = 0.5 (1 - s / sqrt(1 + s^2)).
inline double invert_dh_cga(double s) {
    const double u = s / std::sqrt(1.0 + s * s);
    return std::clamp(0.5 * (1.0 - u), 0.0, 1.0);
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

// Invert  f'(n) = s  for  f(n) = sqrt(n) * (1 - n)^{1/4}  on n in (0, 1).
//
// f'(n) = (2 - 3 n) / [ 4 sqrt(n) (1 - n)^{3/4} ]
//
// is monotonically decreasing from +infinity at n -> 0 to -infinity at n -> 1,
// crossing zero at n = 2/3.  Hence the equation f'(n) = s has a unique
// solution in (0, 1) for every real s, found here by bisection.
inline double invert_dgeo(double s) {
    auto df = [](double n) {
        const double eps = 1.0e-14;
        const double nc  = (n < eps) ? eps
                          : (n > 1.0 - eps ? 1.0 - eps : n);
        return (2.0 - 3.0 * nc) /
               (4.0 * std::sqrt(nc) * std::pow(1.0 - nc, 0.75));
    };
    // df is monotonically decreasing, so bracket via [eps, 1-eps].
    const double eps = 1.0e-10;
    double lo = eps;          // df(lo)  -> +infinity
    double hi = 1.0 - eps;    // df(hi)  -> -infinity
    // Saturate if outside the bracket (numerically unreachable, but safe).
    if (s >= df(lo)) return lo;
    if (s <= df(hi)) return hi;
    for (int it = 0; it < 80; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double fm  = df(mid);
        // df decreasing: if df(mid) > s the root is to the right, else left.
        if (fm > s) lo = mid; else hi = mid;
        if (hi - lo < 1.0e-12) break;
    }
    return std::clamp(0.5 * (lo + hi), 0.0, 1.0);
}

// Occupation update for a generic factorizable kernel K(n_i, n_j) = f(n_i) f(n_j)
// whose derivative f'(n) is monotonic on (0, 1) but not closed-form invertible.
// The Euler-Lagrange equation reads
//
//     f'(n_i) U_i = pi k_i (k_i^2/2 - mu),    U_i = sum_j W_{ij} k_j f(n_j),
//
// solved here for n_i via the user-supplied 1-D inverter `invert_df`.  When
// |U_i| is too small (e.g. before the SCF has built up a non-trivial f(n)
// distribution from a smeared start) we fall back to the HF step rule.
template <class Inverter>
inline std::vector<double>
update_occupations_factor_general(double mu,
                                  const std::vector<double>& U,
                                  const Grid& g,
                                  Inverter invert_df) {
    constexpr double pi = M_PI;
    const std::size_t N = g.n();
    std::vector<double> n(N, 0.0);
    const double tiny = 1.0e-14;
    for (std::size_t i = 0; i < N; ++i) {
        const double k = g.k[i];
        if (k <= 0.0) { n[i] = 1.0; continue; }
        const double R = pi * k * (0.5 * k * k - mu);
        if (std::abs(U[i]) < tiny) {
            // No exchange potential yet: HF-like step rule (eps_i = k^2/2).
            n[i] = (R >= 0.0) ? 0.0 : 1.0;
            continue;
        }
        n[i] = invert_df(R / U[i]);
    }
    return n;
}

// Closed-form occupation update for kernels with the additive structure
//
//     K(n_i, n_j) = n_i n_j + g(n_i) g(n_j),
//
// covering CGA (g(n) = sqrt(n(1-n))) and the Beta family
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
                            Inverter invert_dg) {
    constexpr double pi = M_PI;
    const std::size_t N = g.n();
    std::vector<double> n(N, 0.0);
    const double tiny = 1.0e-14;
    for (std::size_t i = 0; i < N; ++i) {
        const double k  = g.k[i];
        if (k <= 0.0) { n[i] = 1.0; continue; }
        const double R  = pi * k * (0.5 * k * k - mu);
        const double dU = R - U_HF[i];
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
inline std::vector<double>
compute_U_with(const std::vector<double>& nv,
               const Grid& g,
               const ExchangeKernel& W,
               const std::function<double(double)>& fn) {
    const std::size_t N = g.n();
    std::vector<double> U(N, 0.0);
    std::vector<double> kf(N);
    for (std::size_t j = 0; j < N; ++j) kf[j] = g.k[j] * fn(nv[j]);
    for (std::size_t i = 0; i < N; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < N; ++j) s += W(i, j) * kf[j];
        U[i] = s;
    }
    return U;
}

// Smeared initial guess: a sigmoid centred at the Fermi wave vector, which
// is more friendly to non-factorizable additive kernels (CGA / Beta) whose
// EL equation requires non-zero g(n) g'(n) to engage.  width controls the
// fractional smearing relative to k_F.
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
// closed-form occupation update; non-factorizable but additive kernels (CGA,
// Beta) use a dedicated closed-form 1-D inverter; other functionals
// (e.g. BBC1) fall back to a projected-gradient step driven by the analytic
// dE/dn.  All branches use bisection on the chemical potential to enforce
// particle-number conservation at every iteration.
//
// For the additive (CGA / Beta) branch the energy landscape has a competing
// HF minimum (every step occupation is a stationary point of K_CGA) and a
// correlated, fractionally-occupied minimum.  We therefore try several
// initial guesses with different smearing widths and keep the lowest-energy
// converged solution.
inline SolveResult
solve_rdmft(double rs,
            const Functional& F,
            const Grid& g,
            const ExchangeKernel& W,
            const SolveOptions& opt = {}) {
    const double rho_target = HEG::density(rs);

    // Detect Power-family functionals.  GU shares the Mueller kernel in the
    // HEG (the orbital self-interaction terms vanish for plane waves), so it
    // takes the closed-form Power(alpha=1/2) update too.
    const PowerFunctional*   pf   = dynamic_cast<const PowerFunctional*>(&F);
    const HFFunctional*      hf   = dynamic_cast<const HFFunctional*>(&F);
    const MuellerFunctional* mu_f = dynamic_cast<const MuellerFunctional*>(&F);
    const GUFunctional*      gu_f = dynamic_cast<const GUFunctional*>(&F);
    const bool factorized = (pf != nullptr) || (hf != nullptr)
                          || (mu_f != nullptr) || (gu_f != nullptr);

    // Detect additive-kernel functionals K = n_i n_j + g(n_i) g(n_j).
    const CGAFunctional*  cga  = dynamic_cast<const CGAFunctional*>(&F);
    const BetaFunctional* beta_f = dynamic_cast<const BetaFunctional*>(&F);
    const bool additive = (cga != nullptr) || (beta_f != nullptr);

    // Detect the GEO functional (factorizable but with a non-power f, so the
    // Euler-Lagrange step uses a numerical 1-D inverter of f').
    const GEOFunctional*  geo  = dynamic_cast<const GEOFunctional*>(&F);
    const bool factor_general = (geo != nullptr);

    double alpha = 1.0;
    if (pf)        alpha = pf->alpha();
    else if (mu_f) alpha = 0.5;
    else if (gu_f) alpha = 0.5;
    else if (hf)   alpha = 1.0;

    auto density_of = [&](const std::vector<double>& nv) {
        return EnergyEvaluator::density(nv, g);
    };

    // Multi-start strategy for additive kernels: probe several smearing
    // widths and keep the lowest-energy result.  CGA / Beta have a stable
    // local minimum at the HF step distribution because g'(n) -> infinity at
    // the endpoints, so we deliberately seed a broad range of fractionally-
    // occupied initial conditions.  For factorized and generic functionals
    // one start (sharp step) is enough.
    const std::vector<std::pair<bool, double>> starts = (additive || factor_general)
        ? std::vector<std::pair<bool, double>>{
              {false, 0.05}, {false, 0.10}, {false, 0.20},
              {false, 0.40}, {false, 0.80}, {false, 1.50}}
        : std::vector<std::pair<bool, double>>{{true, 0.0}};

    SolveResult best;
    bool best_set = false;
    for (const auto& start : starts) {

    std::vector<double> n = start.first
        ? initial_step(rs, g)
        : initial_smeared(rs, g, start.second);
    double mu = 0.5 * HEG::kF(rs) * HEG::kF(rs);
    int it = 0;
    bool converged = false;

    auto compute_U = [&](const std::vector<double>& nv) {
        const std::size_t N = g.n();
        std::vector<double> U(N, 0.0);
        std::vector<double> kf(N);
        for (std::size_t j = 0; j < N; ++j) kf[j] = g.k[j] * F.f(nv[j]);
        for (std::size_t i = 0; i < N; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < N; ++j) s += W(i, j) * kf[j];
            U[i] = s;
        }
        return U;
    };

    auto bisect_mu_factorized = [&](const std::vector<double>& U) {
        // density(mu) is monotonically *increasing* in mu (larger mu = more
        // states filled), so if rho > target we shrink the upper bound.
        double lo = opt.mu_lo, hi = opt.mu_hi;
        for (int b = 0; b < opt.bisect_iter; ++b) {
            double m = 0.5 * (lo + hi);
            auto n_try = update_occupations_power(alpha, m, U, g);
            if (density_of(n_try) > rho_target) hi = m;
            else                                 lo = m;
        }
        return 0.5 * (lo + hi);
    };

    // Pick the appropriate hole inverter for additive kernels.
    auto invert_dg = [&](double s) -> double {
        if (cga)    return invert_dh_cga(s);
        if (beta_f) return invert_dgbeta(s, beta_f->beta());
        return 0.5;
    };

    auto identity_id = [](double n_) { return n_; };

    for (it = 0; it < opt.max_iter; ++it) {
        std::vector<double> n_target;

        if (factorized) {
            auto U = compute_U(n);
            mu = bisect_mu_factorized(U);
            n_target = update_occupations_power(alpha, mu, U, g);
        } else if (factor_general) {
            // Factorizable kernel K(n_i, n_j) = f(n_i) f(n_j) but with f' not
            // analytically invertible.  Build U_i with the functional's own f
            // and run a 1-D bisection inverter for n_i at every grid point.
            auto U = compute_U(n);
            auto invert_df = [&](double s) -> double {
                if (geo) return invert_dgeo(s);
                return 0.5;
            };
            auto bisect_mu_factor_general = [&]() {
                double lo = opt.mu_lo, hi = opt.mu_hi;
                for (int b = 0; b < opt.bisect_iter; ++b) {
                    double m = 0.5 * (lo + hi);
                    auto n_try = update_occupations_factor_general(
                        m, U, g, invert_df);
                    if (density_of(n_try) > rho_target) hi = m;
                    else                                 lo = m;
                }
                return 0.5 * (lo + hi);
            };
            mu = bisect_mu_factor_general();
            n_target = update_occupations_factor_general(mu, U, g, invert_df);
        } else if (additive) {
            // Additive kernel K = n_i n_j + g(n_i) g(n_j): closed-form update
            // for n_i once mu is found by bisection.  We use a soft floor on
            // n in [n_floor, 1 - n_floor] when computing g(n) so that even
            // near-saturated occupations contribute a finite hole U_g, which
            // lets the SCF escape the metastable HF-like fixed point.  The
            // floor is taken small enough that converged solutions are not
            // affected to leading order (energies use the unmodified n).
            const double n_floor = 1.0e-6;
            auto g_of = [&](double n_) {
                const double nc = std::clamp(n_, n_floor, 1.0 - n_floor);
                if (cga) {
                    return std::sqrt(nc * (1.0 - nc));
                }
                if (beta_f) {
                    return std::pow(nc * (1.0 - nc), beta_f->beta());
                }
                return 0.0;
            };

            auto U_HF = compute_U_with(n, g, W, identity_id);
            auto U_g  = compute_U_with(n, g, W, g_of);

            auto bisect_mu_additive = [&]() {
                double lo = opt.mu_lo, hi = opt.mu_hi;
                for (int b = 0; b < opt.bisect_iter; ++b) {
                    double m = 0.5 * (lo + hi);
                    auto n_try = update_occupations_additive(
                        m, U_HF, U_g, g, invert_dg);
                    if (density_of(n_try) > rho_target) hi = m;
                    else                                 lo = m;
                }
                return 0.5 * (lo + hi);
            };
            mu = bisect_mu_additive();
            n_target = update_occupations_additive(
                mu, U_HF, U_g, g, invert_dg);
        } else {
            // Generic projected-gradient with bisection on mu.
            auto eps = EnergyEvaluator::pseudo_energy(n, g, W, F);
            const double step = 0.10;

            auto pgd_at = [&](double m) {
                std::vector<double> n_try(g.n(), 0.0);
                for (std::size_t i = 0; i < g.n(); ++i) {
                    n_try[i] = std::clamp(n[i] - step * (eps[i] - m), 0.0, 1.0);
                }
                return n_try;
            };
            double lo = opt.mu_lo, hi = opt.mu_hi;
            for (int b = 0; b < opt.bisect_iter; ++b) {
                double m = 0.5 * (lo + hi);
                auto n_try = pgd_at(m);
                if (density_of(n_try) > rho_target) hi = m;
                else                                 lo = m;
            }
            mu = 0.5 * (lo + hi);
            n_target = pgd_at(mu);
        }

        double dn_max = 0.0;
        for (std::size_t i = 0; i < g.n(); ++i) {
            const double n_new = (1.0 - opt.mix) * n[i] + opt.mix * n_target[i];
            dn_max = std::max(dn_max, std::abs(n_new - n[i]));
            n[i] = n_new;
        }

        if (opt.verbose && (it % 20 == 0)) {
            double Eh = EnergyEvaluator::kinetic_per_volume(n, g)
                      + EnergyEvaluator::xc_per_volume(n, g, W, F);
            std::fprintf(stderr,
                "    [iter %4d]  mu=%+.6f  rho=%.6e  dn=%.2e  E/V=%+.6e\n",
                it, mu, density_of(n), dn_max, Eh);
        }
        if (dn_max < opt.tol_n) { converged = true; ++it; break; }
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
    res.converged = converged;

    // Variational selection: among all the start configurations tried,
    // keep the one that minimizes the total energy and is consistent with
    // the density constraint.
    if (!best_set || res.E_per_N < best.E_per_N) {
        best = res;
        best_set = true;
    }
    }  // end multi-start loop

    return best_set ? best : SolveResult{};
}

}  // namespace rdmft
