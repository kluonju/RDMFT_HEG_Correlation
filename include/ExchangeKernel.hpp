#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "Grid.hpp"

namespace rdmft {

// Accurate quadrature for the inner exchange integral
//
//   V(k_i) = \int_0^L u(k') ln|(k_i+k')/(k_i-k')|  dk'
//
// using product integration with a piecewise-linear ansatz for u(k').  Because
// the only non-smooth feature is the log singularity at k' = k_i (in the
// "ln|k_i-k'|" piece), and that is integrated analytically against the linear
// hat functions, the result is exact whenever u(k') is piecewise linear on
// the grid.  In particular, for the HF step occupation aligned to a grid
// node, this gives the exact analytic exchange energy to machine precision.
//
// For smooth u this is a 2nd-order quadrature with no log-singular error.
//
// We pre-compute the dense matrix W[i,j] such that
//
//   V(k_i) = sum_j W[i,j] * u(k_j),    u(k') = k' * g(k')
//
// and the user multiplies by u_j = k_j * f(n_j) every iteration.
struct ExchangeKernel {
    std::size_t N = 0;
    std::vector<double> W;  // row-major, size N*N

    static ExchangeKernel build(const Grid& g) {
        const std::size_t N = g.n();
        ExchangeKernel K;
        K.N = N;
        K.W.assign(N * N, 0.0);

        // Helper: x*ln|x| with the convention 0*ln(0) = 0.
        auto xlnx = [](double x) {
            const double ax = std::abs(x);
            return (ax > 0.0) ? x * std::log(ax) : 0.0;
        };

        // Indefinite integrals on a generic interval [a, b], wrt the variable
        // k', for the kernel piece P_+(k_i, k') = ln(k_i + k')   (smooth)
        // and                   P_-(k_i, k') = -ln|k_i - k'|     (singular at k' = k_i).
        //
        // We need two moments on each interval:
        //   I0(a,b,c) = \int_a^b ln|c - k'| dk'
        //   I1(a,b,c) = \int_a^b k' ln|c - k'| dk'
        // and the analogous J0, J1 with c -> -k_i (i.e. ln(k_i + k')).
        //
        // I0 and J0 are obtainable from a single primitive
        //   F0(t)  = (t)*ln|t| - t
        //   F1(t)  = (t^2/2) * ln|t| - t^2/4    [for u*ln|u| moments]

        auto I0_indef = [&](double t) {
            // \int ln|t| dt = t ln|t| - t
            return xlnx(t) - t;
        };
        auto I1_aux = [&](double t) {
            // \int t ln|t| dt = (t^2/2) ln|t| - t^2/4
            const double at = std::abs(t);
            const double sq = t * t;
            return (at > 0.0) ? 0.5 * sq * std::log(at) - 0.25 * sq : 0.0;
        };

        auto integ_logabs_c = [&](double a, double b, double c) -> double {
            // \int_a^b ln|c - k'| dk'  (variable u = k' - c)
            return I0_indef(b - c) - I0_indef(a - c);
        };
        auto integ_kp_logabs_c = [&](double a, double b, double c) -> double {
            // \int_a^b k' ln|c - k'| dk' = \int (u + c) ln|u| du
            const double M_main = I1_aux(b - c) - I1_aux(a - c);
            const double M_lin  = c * (I0_indef(b - c) - I0_indef(a - c));
            return M_main + M_lin;
        };

        // For the "+" piece ln(k_i + k') the argument is always positive on
        // [0, L] for k_i >= 0 and k' >= 0, so no singularity.  We can still
        // use the same primitive with the substitution u = k_i + k':
        auto integ_logplus = [&](double a, double b, double ki) -> double {
            // \int_a^b ln(k_i + k') dk' = F0(ki+b) - F0(ki+a)
            return I0_indef(ki + b) - I0_indef(ki + a);
        };
        auto integ_kp_logplus = [&](double a, double b, double ki) -> double {
            // \int_a^b k' ln(k_i + k') dk', using k' = u - k_i:
            const double M_main = I1_aux(ki + b) - I1_aux(ki + a);
            const double M_lin  = -ki * (I0_indef(ki + b) - I0_indef(ki + a));
            return M_main + M_lin;
        };

        // For each row i, accumulate W[i, j] from the two intervals neighbouring
        // node j (left and right) using the local hat-function decomposition.
        for (std::size_t i = 0; i < N; ++i) {
            const double ki = g.k[i];
            for (std::size_t s = 0; s + 1 < N; ++s) {
                // Interval [k_s, k_{s+1}].
                const double a = g.k[s];
                const double b = g.k[s + 1];
                const double h = b - a;
                if (h <= 0.0) continue;

                // Piece P_total(k_i, k') = ln(k_i + k') - ln|k_i - k'|.
                const double M0_plus  = integ_logplus(a, b, ki);
                const double M1_plus  = integ_kp_logplus(a, b, ki);
                const double M0_minus = -integ_logabs_c(a, b, ki);
                const double M1_minus = -integ_kp_logabs_c(a, b, ki);

                const double M0 = M0_plus + M0_minus;
                const double M1 = M1_plus + M1_minus;

                // Hat functions on this interval:
                //   phi_0(k') = (b - k') / h     -> contributes to node s
                //   phi_1(k') = (k' - a) / h     -> contributes to node s+1
                const double w_left  = (b * M0 - M1) / h;
                const double w_right = (M1 - a * M0) / h;
                K.W[i * N + s]       += w_left;
                K.W[i * N + s + 1]   += w_right;
            }
        }
        return K;
    }

    double operator()(std::size_t i, std::size_t j) const {
        return W[i * N + j];
    }
};

}  // namespace rdmft
