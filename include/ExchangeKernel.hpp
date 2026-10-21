#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "Grid.hpp"

namespace rdmft {

namespace exchange_kernel_detail {

// Ein(z) = ∫_0^z (1 - e^{-t})/t dt  for z >= 0 (Ein(0) = 0).
inline double ein(double z) {
    if (!(z > 0.0)) return 0.0;
    if (z < 1.0e-3) {
        // Series: z - z^2/4 + z^3/18 - z^4/96 + ...
        const double z2 = z * z;
        return z * (1.0 - z * (0.25 - z * (1.0 / 18.0 - z2 / 96.0)));
    }
    if (z < 40.0) {
        // Power series ∑_{n=1}^∞ (-1)^{n+1} z^n / (n · n!)
        double term = z;          // (-1)^{2} z^1 / 1!
        double sum = term;        // /1
        for (int n = 2; n < 120; ++n) {
            term *= -z / static_cast<double>(n);
            const double contrib = term / static_cast<double>(n);
            sum += contrib;
            if (std::abs(contrib) < 1.0e-16 * (1.0 + std::abs(sum))) break;
        }
        return sum;
    }
    // Large-z: Ein(z) = γ + ln(z) + E_1(z) ≈ γ + ln(z) (E_1 exponentially small).
    constexpr double euler = 0.5772156649015328606;
    return euler + std::log(z);
}

// Smooth SR angular factor for the erfc Coulomb piece:
//   L_SR(k,k';μ) = (1/2) [Ein(α (k+k')²) - Ein(α (k-k')²)],
//   α = 1/(4 μ²),
// so that L_LR = ln|(k+k')/(k-k')| - L_SR.
inline double L_sr(double k, double kp, double alpha) {
    const double sum = k + kp;
    const double dif = k - kp;
    return 0.5 * (ein(alpha * sum * sum) - ein(alpha * dif * dif));
}

inline void validate_grid(const Grid& g) {
    const std::size_t N = g.n();
    if (N < 2 || g.w.size() != N) {
        throw std::invalid_argument(
            "ExchangeKernel requires matching grid nodes and weights");
    }
    if (N > std::numeric_limits<std::size_t>::max() / N
        || N * N > std::vector<double>().max_size()) {
        throw std::length_error("exchange-kernel matrix size overflows");
    }
    for (std::size_t i = 0; i < N; ++i) {
        if (!std::isfinite(g.k[i]) || !std::isfinite(g.w[i])
            || g.k[i] < 0.0 || g.w[i] < 0.0
            || (i > 0 && g.k[i] <= g.k[i - 1])) {
            throw std::invalid_argument(
                "ExchangeKernel requires finite, increasing non-negative grid data");
        }
    }
}

}  // namespace exchange_kernel_detail

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
//
// For the long-range erf interaction v_LR(q) = (4π/q²) exp(-q²/(4μ²)),
// build_erf_lr returns W_LR = W_Coulomb - W_SR where W_SR product-integrates
// the smooth angular factor L_SR (see exchange_kernel_detail::L_sr).
struct ExchangeKernel {
    std::size_t N = 0;
    std::vector<double> W;   // row-major W[i*N+j]
    // Transpose Wt[i*N+j] = W[j*N+i] so sums over the second W index are
    // contiguous (cache-friendly) when the outer index is fixed.
    std::vector<double> Wt;

    const double* w_row(std::size_t i) const { return W.data() + i * N; }
    const double* wt_row(std::size_t i) const { return Wt.data() + i * N; }

    void fill_transpose() {
        Wt.resize(N * N);
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                Wt[i * N + j] = W[j * N + i];
            }
        }
    }

    static ExchangeKernel build(const Grid& g) {
        exchange_kernel_detail::validate_grid(g);
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
                if (!std::isfinite(w_left) || !std::isfinite(w_right)) {
                    throw std::overflow_error(
                        "non-finite exchange product-integration weight");
                }
                K.W[i * N + s]       += w_left;
                K.W[i * N + s + 1]   += w_right;
            }
        }
        for (double weight : K.W) {
            if (!std::isfinite(weight)) {
                throw std::overflow_error(
                    "exchange product-integration matrix is non-finite");
            }
        }
        K.fill_transpose();
        return K;
    }

    // Long-range erf Coulomb: W_LR = W_Coulomb - W_SR with smooth L_SR.
    // Requires mu > 0.  As mu -> infinity, W_LR -> W_Coulomb.
    static ExchangeKernel build_erf_lr(const Grid& g, double mu) {
        if (!(mu > 0.0) || !std::isfinite(mu)) {
            throw std::invalid_argument(
                "ExchangeKernel::build_erf_lr requires finite mu > 0");
        }
        ExchangeKernel K = build(g);  // start from bare Coulomb
        const std::size_t N = g.n();
        const double alpha = 1.0 / (4.0 * mu * mu);

        // 8-point Gauss–Legendre nodes/weights on [-1,1].
        static constexpr double xgl[8] = {
            -0.9602898564975363, -0.7966664774136267,
            -0.5255324099163290, -0.1834346424956498,
             0.1834346424956498,  0.5255324099163290,
             0.7966664774136267,  0.9602898564975363
        };
        static constexpr double wgl[8] = {
            0.1012285362903763, 0.2223810344533745,
            0.3137066458778873, 0.3626837833783620,
            0.3626837833783620, 0.3137066458778873,
            0.2223810344533745, 0.1012285362903763
        };

        std::vector<double> Wsr(N * N, 0.0);
        for (std::size_t i = 0; i < N; ++i) {
            const double ki = g.k[i];
            for (std::size_t s = 0; s + 1 < N; ++s) {
                const double a = g.k[s];
                const double b = g.k[s + 1];
                const double h = b - a;
                if (h <= 0.0) continue;

                // Moments M0 = ∫_a^b L_SR dk', M1 = ∫_a^b k' L_SR dk'.
                double M0 = 0.0;
                double M1 = 0.0;
                const double mid = 0.5 * (a + b);
                const double half = 0.5 * h;
                for (int q = 0; q < 8; ++q) {
                    const double kp = mid + half * xgl[q];
                    const double L = exchange_kernel_detail::L_sr(ki, kp, alpha);
                    const double wq = wgl[q] * half;
                    M0 += wq * L;
                    M1 += wq * kp * L;
                }

                const double w_left  = (b * M0 - M1) / h;
                const double w_right = (M1 - a * M0) / h;
                if (!std::isfinite(w_left) || !std::isfinite(w_right)) {
                    throw std::overflow_error(
                        "non-finite erf-SR product-integration weight");
                }
                Wsr[i * N + s]     += w_left;
                Wsr[i * N + s + 1] += w_right;
            }
        }

        for (std::size_t idx = 0; idx < N * N; ++idx) {
            K.W[idx] -= Wsr[idx];
            if (!std::isfinite(K.W[idx])) {
                throw std::overflow_error(
                    "erf long-range exchange matrix is non-finite");
            }
        }
        K.fill_transpose();
        return K;
    }

    double operator()(std::size_t i, std::size_t j) const {
        return W[i * N + j];
    }
};

}  // namespace rdmft
