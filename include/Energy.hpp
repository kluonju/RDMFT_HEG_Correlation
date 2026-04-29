#pragma once

#include <cmath>
#include <vector>

#include "ExchangeKernel.hpp"
#include "Functional.hpp"
#include "Grid.hpp"

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
    static double kinetic_per_volume(const std::vector<double>& n,
                                     const Grid& g) {
        constexpr double pi = M_PI;
        double s = 0.0;
        for (std::size_t i = 0; i < g.n(); ++i) {
            const double k = g.k[i];
            s += g.w[i] * k * k * k * k * n[i];
        }
        return s / (2.0 * pi * pi);
    }

    static double density(const std::vector<double>& n, const Grid& g) {
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
        const std::size_t N = g.n();
        std::vector<double> V(N, 0.0);
        // Generic (and direct) form V_i = sum_j W_ij k_j K(n_i, n_j).  For a
        // factorized kernel this is equivalent to f(n_i)*sum_j W_ij k_j f(n_j)
        // and the cost is identical at O(N^2).  Keeping the general path means
        // adding a new functional with a non-factorizable kernel is plug-and-
        // play.
        for (std::size_t i = 0; i < N; ++i) {
            double v = 0.0;
            for (std::size_t j = 0; j < N; ++j) {
                v += W(i, j) * g.k[j] * F.kernel(n[i], n[j]);
            }
            V[i] = v;
        }
        return V;
    }

    // E_xc/V using the functional's two-body kernel and product integration.
    static double xc_per_volume(const std::vector<double>& n,
                                const Grid& g,
                                const ExchangeKernel& W,
                                const Functional& F) {
        constexpr double pi = M_PI;
        const std::size_t N = g.n();

        // E_xc = -(1/(2 pi^3)) * \sum_i w_i * k_i * V_i,
        // where V_i already encodes K(n_i, n_j) for both arguments.
        auto V = V_inner(n, g, W, F);
        double s = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            s += g.w[i] * g.k[i] * V[i];
        }
        return -s / (2.0 * pi * pi * pi);
    }

    // Pseudo orbital energy epsilon_i = dE/dn_i (used in Euler-Lagrange).
    //
    //   d(E_xc)/dn_i =  -(1/(2 pi^3)) * \sum_j w_i k_i W_{ij} k_j (dK/dn_i)
    //                   -(1/(2 pi^3)) * \sum_j w_j k_j W_{ji} k_i (dK/dn_j_at_i)
    // For symmetric kernel K(a,b) = K(b,a): both pieces equal, so
    //   d(E_xc)/dn_i = -(1/pi^3) * k_i * \sum_j w_j W_{ji} k_j (partial_K wrt arg-1)(n_j, n_i)
    //                 [using K's symmetry to evaluate at n_j, n_i]
    // We use the convention: F.kernel_grad(a, b) = dK/da at (a, b), and we
    // assume the kernel is symmetric (true for HF, Mueller, Power, BBC1).
    static std::vector<double> deps_xc(const std::vector<double>& n,
                                       const Grid& g,
                                       const ExchangeKernel& W,
                                       const Functional& F) {
        constexpr double pi = M_PI;
        const std::size_t N = g.n();
        std::vector<double> de(N, 0.0);
        for (std::size_t i = 0; i < N; ++i) {
            const double ki = g.k[i];
            double s = 0.0;
            for (std::size_t j = 0; j < N; ++j) {
                // 2 contributions (i is first or second argument), times -1/(2pi^3),
                // plus assume symmetric K.
                s += g.w[j] * W(j, i) * g.k[j] * F.kernel_grad(n[i], n[j]);
            }
            de[i] = -ki * s / (pi * pi * pi);
        }
        return de;
    }

    // Total epsilon_i = T_i + Vxc_i = k_i^2/2 + d(E_xc)/dn_i.
    static std::vector<double> pseudo_energy(const std::vector<double>& n,
                                             const Grid& g,
                                             const ExchangeKernel& W,
                                             const Functional& F) {
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
        auto de = deps_xc(n, g, W, F);   // d E_xc/V / d n_i (raw)
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
