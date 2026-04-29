#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
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

// Self-consistent solve for any Functional.  Power-family functionals use the
// closed-form occupation update; other functionals (e.g. BBC1) fall back to a
// projected-gradient step driven by the analytic dE/dn.  Both branches use
// bisection on the chemical potential to enforce particle-number conservation
// at every iteration.
inline SolveResult
solve_rdmft(double rs,
            const Functional& F,
            const Grid& g,
            const ExchangeKernel& W,
            const SolveOptions& opt = {}) {
    const double rho_target = HEG::density(rs);

    // Detect Power-family functionals.
    const PowerFunctional* pf  = dynamic_cast<const PowerFunctional*>(&F);
    const HFFunctional*    hf  = dynamic_cast<const HFFunctional*>(&F);
    const MuellerFunctional* mu_f = dynamic_cast<const MuellerFunctional*>(&F);
    const bool factorized = (pf != nullptr) || (hf != nullptr) || (mu_f != nullptr);

    double alpha = 1.0;
    if (pf)        alpha = pf->alpha();
    else if (mu_f) alpha = 0.5;
    else if (hf)   alpha = 1.0;

    auto density_of = [&](const std::vector<double>& nv) {
        return EnergyEvaluator::density(nv, g);
    };

    std::vector<double> n = initial_step(rs, g);
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

    for (it = 0; it < opt.max_iter; ++it) {
        std::vector<double> n_target;

        if (factorized) {
            auto U = compute_U(n);
            mu = bisect_mu_factorized(U);
            n_target = update_occupations_power(alpha, mu, U, g);
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
    return res;
}

}  // namespace rdmft
