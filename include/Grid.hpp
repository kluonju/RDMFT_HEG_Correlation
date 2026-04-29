#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace rdmft {

// Radial momentum grid k_i in [0, k_max] with the corresponding integration
// weights w_i so that
//
//     \int_0^{k_max} F(k) dk  ~=  \sum_i w_i F(k_i).
//
// The grid uses composite Simpson's rule on a uniform mesh, which is more than
// enough resolution once the grid has a few hundred points.  All routines are
// header-only so they can be reused from tests easily.
struct Grid {
    std::vector<double> k;   // grid points
    std::vector<double> w;   // integration weights
    double k_max = 0.0;
    std::size_t n() const { return k.size(); }

    static Grid uniform_simpson(double k_max_, std::size_t N) {
        assert(N >= 3 && (N % 2) == 1 && "Simpson needs odd #points");
        Grid g;
        g.k.resize(N);
        g.w.resize(N);
        g.k_max = k_max_;
        const double h = k_max_ / static_cast<double>(N - 1);
        for (std::size_t i = 0; i < N; ++i) g.k[i] = i * h;
        for (std::size_t i = 0; i < N; ++i) {
            double c;
            if (i == 0 || i == N - 1) c = 1.0;
            else if (i % 2 == 1)      c = 4.0;
            else                       c = 2.0;
            g.w[i] = c * h / 3.0;
        }
        return g;
    }

    static Grid uniform_trapezoid(double k_max_, std::size_t N) {
        Grid g;
        g.k.resize(N);
        g.w.resize(N);
        g.k_max = k_max_;
        const double h = k_max_ / static_cast<double>(N - 1);
        for (std::size_t i = 0; i < N; ++i) g.k[i] = i * h;
        for (std::size_t i = 0; i < N; ++i) {
            g.w[i] = (i == 0 || i == N - 1) ? 0.5 * h : h;
        }
        return g;
    }

    // Trapezoidal grid with k_F = k_node[m] for integer m, so the Fermi step
    // occupation is exactly representable.  k_max is rounded up to the next
    // grid node beyond the requested k_max_min.
    static Grid trapezoid_with_node_at(double k_F, double k_max_min,
                                       std::size_t N_target) {
        const double h_guess = k_max_min / static_cast<double>(N_target - 1);
        std::size_t m = std::max<std::size_t>(
            8u, static_cast<std::size_t>(std::round(k_F / h_guess)));
        const double h = k_F / static_cast<double>(m);
        std::size_t N = static_cast<std::size_t>(
            std::ceil(k_max_min / h)) + 1;
        if (N < N_target) N = N_target;
        return uniform_trapezoid(h * static_cast<double>(N - 1), N);
    }
};

}  // namespace rdmft
