#pragma once

#include <algorithm>
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
// ``graded_fermi_trapezoid`` concentrates nodes near k_F where n(k) drops from
// ~1 to ~0, using the composite trapezoid rule on a non-uniform mesh (exact
// for piecewise-linear n on the grid).  ``uniform_trapezoid`` / Simpson variants
// remain for tests.  All routines are header-only.
struct Grid {
    std::vector<double> k;   // grid points (strictly increasing)
    std::vector<double> w;   // integration weights
    double k_max = 0.0;
    std::size_t n() const { return k.size(); }

    // Composite trapezoid weights for an arbitrary increasing mesh.
    static std::vector<double> composite_trapezoid_weights(
        const std::vector<double>& x) {
        const std::size_t N = x.size();
        assert(N >= 2);
        std::vector<double> w(N);
        w[0] = 0.5 * (x[1] - x[0]);
        for (std::size_t i = 1; i + 1 < N; ++i) {
            w[i] = 0.5 * (x[i] - x[i - 1]) + 0.5 * (x[i + 1] - x[i]);
        }
        w[N - 1] = 0.5 * (x[N - 1] - x[N - 2]);
        return w;
    }

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

    // Piecewise-uniform mesh: coarse outside an ~0.9 k_F band around the Fermi
    // wavevector, dense inside, with k_F exactly a grid node.  Improves n(k)
    // and energies when the occupation front is narrow compared to k_max.
    static Grid graded_fermi_trapezoid(double k_F, double k_max_min,
                                       std::size_t N_target) {
        assert(k_F > 0.0 && k_max_min > k_F && N_target >= 21u);

        // NM intervals in [k_lo, k_hi]; k_F = k_lo + j_F * h_m with j_F = NM/2.
        // Leave at least ~8 nodes outside the band (coarse wings).
        std::size_t NM = std::max<std::size_t>(
            8u, std::min((N_target * 13u) / 25u, N_target - 8u));
        const double band_width = 0.9 * k_F;
        const double h_m        = band_width / static_cast<double>(NM);
        const std::size_t j_F   = NM / 2;
        const double k_lo       = k_F - static_cast<double>(j_F) * h_m;
        const double k_hi       = k_lo + static_cast<double>(NM) * h_m;

        const std::size_t N_rem = N_target - 1u - NM;
        assert(N_rem >= 4u);

        const double len_left  = k_lo;
        const double len_right = k_max_min - k_hi;
        assert(len_left > 0.0 && len_right > 0.0);

        std::size_t NL = static_cast<std::size_t>(std::llround(
            static_cast<double>(N_rem) * (len_left / (len_left + len_right))));
        NL = std::max<std::size_t>(2u, std::min(N_rem - 2u, NL));
        const std::size_t NR = N_rem - NL;

        std::vector<double> x;
        x.reserve(N_target);
        for (std::size_t i = 0; i <= NL; ++i) {
            x.push_back(static_cast<double>(i) * k_lo / static_cast<double>(NL));
        }
        for (std::size_t t = 1; t <= NM; ++t) {
            x.push_back(k_lo + static_cast<double>(t) * h_m);
        }
        for (std::size_t t = 1; t <= NR; ++t) {
            x.push_back(k_hi + static_cast<double>(t) * (k_max_min - k_hi) /
                                    static_cast<double>(NR));
        }

        assert(x.size() == N_target);
        for (std::size_t i = 1; i < x.size(); ++i) {
            assert(x[i] > x[i - 1] && "graded k mesh must be strictly increasing");
        }

        Grid g;
        g.k     = std::move(x);
        g.k_max = g.k.back();
        g.w     = composite_trapezoid_weights(g.k);
        return g;
    }
};

}  // namespace rdmft
