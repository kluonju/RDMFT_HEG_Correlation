#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ExchangeKernel.hpp"
#include "Functional.hpp"
#include "Grid.hpp"
#include "HEG.hpp"

namespace rdmft {

// Energy of the paramagnetic homogeneous electron gas as a functional of the
// per-spin momentum occupation n(k) in [0,1].
//
// In Hartree atomic units and per unit volume (factor of 2 for spin already
// absorbed into the prefactors below):
//
//   T/V    =  (1/(2 pi^2)) * \int_0^inf  k^4    n(k) dk
//
//   E_xc/V = -(1/(2 pi^3)) * \int\int    k k'   K(n(k), n(k'))
//                                              * ln|(k+k')/(k-k')|  dk dk'
//
//   rho    =  (1/pi^2)    * \int_0^inf  k^2    n(k) dk.
//
// All inner integrals over k' are evaluated using product integration with
// the precomputed matrix W (see ExchangeKernel.hpp), which absorbs the
// integrable log singularity at k' = k.  The outer integral over k uses the
// grid's trapezoid weights, and is exact whenever n(k) is piecewise linear
// on the grid.
struct EnergyEvaluator {
    static void validate_grid_vector(const std::vector<double>& n,
                                     const Grid& g) {
        if (g.k.empty() || g.k.size() != g.w.size()
            || n.size() != g.k.size()) {
            throw std::invalid_argument(
                "occupation, grid-node, and grid-weight sizes must match");
        }
    }

    static void validate_exchange(const Grid& g, const ExchangeKernel& W) {
        const std::size_t N = g.n();
        if (N == 0 || W.N != N
            || N > std::numeric_limits<std::size_t>::max() / N) {
            throw std::invalid_argument(
                "exchange-kernel dimensions do not match the grid");
        }
        const std::size_t matrix_size = N * N;
        if (W.W.size() != matrix_size || W.Wt.size() != matrix_size) {
            throw std::invalid_argument(
                "exchange-kernel storage does not match its dimensions");
        }
    }

    static double kinetic_per_volume(const std::vector<double>& n,
                                     const Grid& g) {
        validate_grid_vector(n, g);
        constexpr double pi = M_PI;
        double s = 0.0;
        for (std::size_t i = 0; i < g.n(); ++i) {
            const double k = g.k[i];
            s += g.w[i] * k * k * k * k * n[i];
        }
        return s / (2.0 * pi * pi);
    }

    static double density(const std::vector<double>& n, const Grid& g) {
        validate_grid_vector(n, g);
        constexpr double pi = M_PI;
        double s = 0.0;
        for (std::size_t i = 0; i < g.n(); ++i) {
            s += g.w[i] * g.k[i] * g.k[i] * n[i];
        }
        return s / (pi * pi);
    }

    // Inner exchange potential V_i = \int k' K(n_i, n_j) ln|(k_i+k')/(k_i-k')| dk'
    // Uses the product-integration matrix W.  For a factorized kernel
    // K(n_i, n_j) = f(n_i) f(n_j), V_i = f(n_i) * (W * (k . f(n))).
    // For a general kernel, V_i = sum_j W[i,j] * k_j * K(n_i, n_j).
    static std::vector<double> V_inner(const std::vector<double>& n,
                                       const Grid& g,
                                       const ExchangeKernel& W,
                                       const Functional& F) {
        validate_grid_vector(n, g);
        validate_exchange(g, W);
        const std::size_t N = g.n();
        std::vector<double> V(N, 0.0);
        // Fast path: K(n_i,n_j) = f(n_i) f(n_j)  =>  one matvec per SCF step,
        // contiguous row access, optional OpenMP (no N^2 virtual kernel calls).
        if (F.is_factorized()) {
            std::vector<double> kf(N);
            for (std::size_t j = 0; j < N; ++j) {
                kf[j] = g.k[j] * F.f(n[j]);
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N > 96)
#endif
            for (std::size_t i = 0; i < N; ++i) {
                const double* row = W.w_row(i);
                double v = 0.0;
                for (std::size_t j = 0; j < N; ++j) {
                    v += row[j] * kf[j];
                }
                V[i] = F.f(n[i]) * v;
            }
            return V;
        }
        // General symmetric kernel: V_i = sum_j W_ij k_j K(n_i, n_j).
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N > 64)
#endif
        for (std::size_t i = 0; i < N; ++i) {
            const double* row = W.w_row(i);
            double v = 0.0;
            for (std::size_t j = 0; j < N; ++j) {
                v += row[j] * g.k[j] * F.kernel(n[i], n[j]);
            }
            V[i] = v;
        }
        return V;
    }

    // JK-only energy per volume (no density-only piece).
    static double jk_per_volume(const std::vector<double>& n,
                                const Grid& g,
                                const ExchangeKernel& W,
                                const Functional& F) {
        validate_grid_vector(n, g);
        validate_exchange(g, W);
        constexpr double pi = M_PI;
        const std::size_t N = g.n();
        auto V = V_inner(n, g, W, F);
        double s = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            s += g.w[i] * g.k[i] * V[i];
        }
        return -s / (2.0 * pi * pi * pi);
    }

    // E_xc/V using the functional's two-body kernel and product integration.
    // Optional ``W_lr``:
    //   * uses_lr_power_exchange(): Exc = density_xc + E_Power(α)[W_LR]
    //     (no bare-Coulomb JK).
    //   * uses_lr_power_correlation(): add (E_Power(α) − E_HF) on W_LR.
    static double xc_per_volume(const std::vector<double>& n,
                                const Grid& g,
                                const ExchangeKernel& W,
                                const Functional& F,
                                const ExchangeKernel* W_lr = nullptr) {
        const double rho = density(n, g);
        if (W_lr != nullptr && F.uses_lr_power_exchange()) {
            validate_exchange(g, *W_lr);
            const PowerFunctional power(F.lr_power_alpha());
            return F.density_xc_per_volume(rho)
                 + jk_per_volume(n, g, *W_lr, power);
        }
        const double jk = jk_per_volume(n, g, W, F);
        double e = jk + F.density_xc_per_volume(rho);
        if (W_lr != nullptr && F.uses_lr_power_correlation()) {
            validate_exchange(g, *W_lr);
            const PowerFunctional power(F.lr_power_alpha());
            const HFFunctional hf;
            e += jk_per_volume(n, g, *W_lr, power)
               - jk_per_volume(n, g, *W_lr, hf);
        }
        return e;
    }

    // Pseudo orbital energy epsilon_i = dE/dn_i (used in Euler-Lagrange).
    //
    //   d(E_xc)/dn_i =  -(1/(2 pi^3)) * \sum_j w_i k_i W_{ij} k_j (dK/dn_i)
    //                   -(1/(2 pi^3)) * \sum_j w_j k_j W_{ji} k_i (dK/dn_j_at_i)
    // For symmetric kernel K(a,b) = K(b,a), both derivatives can be written
    // with partial_K wrt arg-1 at (n_i,n_j).  We retain both discrete terms
    // rather than assuming weighted symmetry of W, so this is the exact
    // derivative of xc_per_volume on any supported grid.
    // We use the convention: F.kernel_grad(a, b) = dK/da at (a, b), and we
    // assume the kernel is symmetric (true for HF, Mueller, Power, BBC1).
    static std::vector<double> deps_xc(const std::vector<double>& n,
                                       const Grid& g,
                                       const ExchangeKernel& W,
                                       const Functional& F,
                                       const ExchangeKernel* W_lr = nullptr) {
        if (W_lr != nullptr && F.uses_lr_power_exchange()) {
            validate_exchange(g, *W_lr);
            const PowerFunctional power(F.lr_power_alpha());
            return deps_xc(n, g, *W_lr, power, nullptr);
        }
        validate_grid_vector(n, g);
        validate_exchange(g, W);
        constexpr double pi = M_PI;
        const std::size_t N = g.n();
        std::vector<double> de(N, 0.0);
        if (F.is_factorized()) {
            std::vector<double> kf(N);
            std::vector<double> wkf(N);
            for (std::size_t j = 0; j < N; ++j) {
                kf[j] = g.k[j] * F.f(n[j]);
                wkf[j] = g.w[j] * kf[j];
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N > 96)
#endif
            for (std::size_t i = 0; i < N; ++i) {
                const double ki = g.k[i];
                const double* direct_row = W.w_row(i);
                const double* transpose_row = W.wt_row(i);
                double direct = 0.0;
                double transpose = 0.0;
                for (std::size_t j = 0; j < N; ++j) {
                    direct += direct_row[j] * kf[j];
                    transpose += transpose_row[j] * wkf[j];
                }
                de[i] = -0.5 * ki * F.df(n[i])
                      * (g.w[i] * direct + transpose)
                      / (pi * pi * pi);
            }
        } else {
            // General kernel: include both appearances of n_i in the symmetric
            // double sum. W and W^T rows keep both inner traversals contiguous.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N > 64)
#endif
            for (std::size_t i = 0; i < N; ++i) {
                const double ki = g.k[i];
                const double* direct_row = W.w_row(i);
                const double* transpose_row = W.wt_row(i);
                double direct = 0.0;
                double transpose = 0.0;
                for (std::size_t j = 0; j < N; ++j) {
                    const double gradient = F.kernel_grad(n[i], n[j]);
                    direct += direct_row[j] * g.k[j] * gradient;
                    transpose += g.w[j] * g.k[j] * transpose_row[j] * gradient;
                }
                de[i] = -0.5 * ki * (g.w[i] * direct + transpose)
                      / (pi * pi * pi);
            }
        }
        if (W_lr != nullptr && F.uses_lr_power_correlation()) {
            validate_exchange(g, *W_lr);
            const PowerFunctional power(F.lr_power_alpha());
            const HFFunctional hf;
            auto d_pow = deps_xc(n, g, *W_lr, power, nullptr);
            auto d_hf = deps_xc(n, g, *W_lr, hf, nullptr);
            for (std::size_t i = 0; i < N; ++i) {
                de[i] += d_pow[i] - d_hf[i];
            }
        }
        return de;
    }

    // Density-matched Fermi step on the discrete grid (fractional edge node).
    // Used as the numerical HF reference so E_c = E − E_HF is free of the
    // O(Δk) quadrature bias that dominates the analytic HF subtraction at
    // small r_s on coarse uniform meshes.
    static std::vector<double> hf_step_occupation(double rs, const Grid& g) {
        if (g.k.empty() || g.k.size() != g.w.size()) {
            throw std::invalid_argument("hf_step_occupation: invalid grid");
        }
        const double kf = HEG::kF(rs);
        const double rho_target = HEG::density(rs);
        std::vector<double> n(g.n(), 0.0);
        std::size_t edge = 0;
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < g.n(); ++i) {
            n[i] = (g.k[i] < kf) ? 1.0 : 0.0;
            const double d = std::abs(g.k[i] - kf);
            if (d < best) {
                best = d;
                edge = i;
            }
        }
        double lo = 0.0;
        double hi = 1.0;
        for (int it = 0; it < 64; ++it) {
            const double mid = 0.5 * (lo + hi);
            n[edge] = mid;
            if (density(n, g) > rho_target) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        n[edge] = 0.5 * (lo + hi);
        return n;
    }

    // HF energy per electron on the same (g, W) used for the RDMFT solve.
    static double hf_per_electron_on_grid(double rs,
                                          const Grid& g,
                                          const ExchangeKernel& W) {
        const HFFunctional hf;
        const auto n = hf_step_occupation(rs, g);
        const double rho = density(n, g);
        if (!(rho > 0.0) || !std::isfinite(rho)) {
            return HEG::HF_per_electron(rs);
        }
        const double E =
            kinetic_per_volume(n, g) + xc_per_volume(n, g, W, hf);
        const double e = E / rho;
        return std::isfinite(e) ? e : HEG::HF_per_electron(rs);
    }

    // Total epsilon_i = T_i + Vxc_i = k_i^2/2 + d(E_xc)/dn_i.
    static std::vector<double> pseudo_energy(const std::vector<double>& n,
                                             const Grid& g,
                                             const ExchangeKernel& W,
                                             const Functional& F,
                                             const ExchangeKernel* W_lr = nullptr) {
        // Note: T = (1/(2 pi^2)) * \int w_i k_i^4 n(k) dk -> dT/dn_i = (1/(2 pi^2)) w_i k_i^4.
        // But the EL equation balances the *energy density derivative* by the
        // same factor on both sides; absorbing the w_i factor uniformly we
        // simply write epsilon_i = k_i^2/2 + Vxc_i / (constants), as is
        // conventional.  We use the per-orbital definition matching the
        // standard derivation: the variation is constrained at fixed
        // sum w_i k_i^2 n_i, so we multiply by 1/(w_i k_i^2) on both sides:
        //
        //   k_i^2/2 + (1/(w_i k_i^2)) * d E_xc / d n_i = mu.
        //
        constexpr double pi = M_PI;
        const std::size_t N = g.n();
        std::vector<double> eps(N, 0.0);
        auto de = deps_xc(n, g, W, F, W_lr);   // d E_xc/V / d n_i (raw)
        for (std::size_t i = 0; i < N; ++i) {
            const double ki = g.k[i];
            const double pref = (g.w[i] * ki * ki / (pi * pi));
            // dE_xc/dn_i is per-volume already; the kinetic per-volume per dn_i
            // is (1/(2pi^2)) * w_i * k_i^4.  Density per dn_i is (1/pi^2)*w_i*k_i^2.
            // Lagrangian: T + E_xc - mu*rho stationary in n_i =>
            //   (1/(2pi^2)) w_i k_i^4 + de[i] = mu * (1/pi^2) w_i k_i^2.
            // Solve for the "effective epsilon" that should equal mu:
            //   epsilon_i = (k_i^2/2) + de[i] / pref.
            if (pref > 0.0) {
                eps[i] = 0.5 * ki * ki + de[i] / pref;
            } else {
                eps[i] = 0.5 * ki * ki;
            }
        }
        return eps;
    }
};

}  // namespace rdmft
