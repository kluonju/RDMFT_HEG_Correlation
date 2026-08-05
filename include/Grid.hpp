#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rdmft {

// Radial momentum grid k_i in [0, k_max] with the corresponding integration
// weights w_i so that
//
//     \int_0^{k_max} F(k) dk  ~=  \sum_i w_i F(k_i).
//
// The driver uses ``uniform_trapezoid`` on [0, k_max] with composite trapezoid
// weights by default.  Other meshes:
//   - ``graded_fermi_trapezoid``: dense band around k_F (k_F is a node)
//   - ``log_trapezoid``: uniform in ln k (dense near k=0)
//   - ``log_fermi_trapezoid``: log left wing + dense Fermi band + coarse tail
// All routines are header-only.
struct Grid {
    std::vector<double> k;   // grid points (strictly increasing)
    std::vector<double> w;   // integration weights
    double k_max = 0.0;
    std::size_t n() const { return k.size(); }

    // Composite trapezoid weights for an arbitrary increasing mesh.
    static std::vector<double> composite_trapezoid_weights(
        const std::vector<double>& x) {
        const std::size_t N = x.size();
        if (N < 2) {
            throw std::invalid_argument(
                "composite_trapezoid_weights needs at least two nodes");
        }
        if (!std::isfinite(x[0])) {
            throw std::invalid_argument("grid nodes must be finite");
        }
        for (std::size_t i = 1; i < N; ++i) {
            if (!std::isfinite(x[i]) || x[i] <= x[i - 1]) {
                throw std::invalid_argument(
                    "grid nodes must be finite and strictly increasing");
            }
        }
        std::vector<double> w(N);
        w[0] = 0.5 * (x[1] - x[0]);
        for (std::size_t i = 1; i + 1 < N; ++i) {
            w[i] = 0.5 * (x[i] - x[i - 1]) + 0.5 * (x[i + 1] - x[i]);
        }
        w[N - 1] = 0.5 * (x[N - 1] - x[N - 2]);
        for (double weight : w) {
            if (!std::isfinite(weight) || weight < 0.0) {
                throw std::overflow_error(
                    "composite trapezoid weights are not finite");
            }
        }
        return w;
    }

    static Grid uniform_simpson(double k_max_, std::size_t N) {
        if (!std::isfinite(k_max_) || k_max_ <= 0.0) {
            throw std::invalid_argument("uniform_simpson requires finite k_max > 0");
        }
        if (N < 3 || (N % 2) == 0) {
            throw std::invalid_argument(
                "uniform_simpson requires an odd number of at least three nodes");
        }
        if (N > std::vector<double>().max_size()) {
            throw std::length_error("uniform_simpson grid is too large");
        }
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
        if (!std::isfinite(k_max_) || k_max_ <= 0.0) {
            throw std::invalid_argument(
                "uniform_trapezoid requires finite k_max > 0");
        }
        if (N < 2u) {
            throw std::invalid_argument(
                "uniform_trapezoid requires at least two nodes");
        }
        if (N > std::vector<double>().max_size()) {
            throw std::length_error("uniform_trapezoid grid is too large");
        }
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
        if (!std::isfinite(k_F) || !std::isfinite(k_max_min)
            || k_F <= 0.0 || k_max_min <= k_F || N_target < 2u) {
            throw std::invalid_argument(
                "trapezoid_with_node_at requires 0 < k_F < k_max and at least two nodes");
        }
        if (N_target > std::vector<double>().max_size()) {
            throw std::length_error("Fermi-aligned grid is too large");
        }
        const double h_guess = k_max_min / static_cast<double>(N_target - 1);
        const double m_estimate = std::round(k_F / h_guess);
        if (!std::isfinite(m_estimate)
            || m_estimate >= static_cast<double>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::length_error("Fermi-node index is out of range");
        }
        std::size_t m = std::max<std::size_t>(
            8u, static_cast<std::size_t>(m_estimate));
        const double h = k_F / static_cast<double>(m);
        const double n_estimate = std::ceil(k_max_min / h);
        if (!std::isfinite(n_estimate)
            || n_estimate >= static_cast<double>(
                std::numeric_limits<std::size_t>::max() - 1u)) {
            throw std::length_error("Fermi-aligned grid size is out of range");
        }
        std::size_t N = static_cast<std::size_t>(n_estimate) + 1u;
        if (N < N_target) N = N_target;
        return uniform_trapezoid(h * static_cast<double>(N - 1), N);
    }

    // Piecewise-uniform mesh: coarse outside a band around k_F, dense inside,
    // with k_F exactly a grid node.  ~80% of the dense cells sit at k <= k_F,
    // and the left wing [0, k_lo] gets extra weight in the NL/NR split so
    // k < k_F is well resolved for n(k) plots (typical N=401, k_max = 3 k_F).
    static Grid graded_fermi_trapezoid(double k_F, double k_max_min,
                                       std::size_t N_target) {
        if (!std::isfinite(k_F) || !std::isfinite(k_max_min)
            || k_F <= 0.0 || k_max_min <= k_F || N_target < 21u) {
            throw std::invalid_argument(
                "graded_fermi_trapezoid requires 0 < k_F < k_max and at least 21 nodes");
        }
        if (N_target > std::vector<double>().max_size()
            || N_target > std::numeric_limits<std::size_t>::max() / 13u) {
            throw std::length_error("graded grid is too large");
        }

        // NM intervals in [k_lo, k_hi]; k_F = k_lo + j_F * h_m (integer j_F).
        std::size_t NM = std::max<std::size_t>(
            8u, std::min((N_target * 13u) / 25u, N_target - 8u));
        const double band_frac = 1.0;  // (k_hi - k_lo) / k_F
        const double band      = band_frac * k_F;
        const double h_m       = band / static_cast<double>(NM);
        const std::size_t j_F  = std::max<std::size_t>(
            1u, std::min((4u * NM) / 5u, NM - 1u));
        const double k_lo      = k_F - static_cast<double>(j_F) * h_m;
        const double k_hi      = k_lo + static_cast<double>(NM) * h_m;

        const std::size_t N_rem = N_target - 1u - NM;
        if (N_rem < 4u) {
            throw std::logic_error("graded grid has too few wing intervals");
        }

        const double len_left  = k_lo;
        const double len_right = k_max_min - k_hi;
        if (len_left <= 0.0 || len_right <= 0.0) {
            throw std::logic_error("graded grid wing lengths must be positive");
        }

        // Weight k < k_lo (uniform wing, all below k_F) so it keeps more nodes
        // than a pure length-proportional split (k_lo is typically ~0.2 k_F).
        constexpr double k_lo_mesh_weight = 2.0;
        const double wleft = k_lo_mesh_weight * len_left;
        std::size_t NL = static_cast<std::size_t>(std::llround(
            static_cast<double>(N_rem) * (wleft / (wleft + len_right))));
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

        if (x.size() != N_target) {
            throw std::logic_error("graded grid node count mismatch");
        }
        for (std::size_t i = 1; i < x.size(); ++i) {
            if (!std::isfinite(x[i]) || x[i] <= x[i - 1]) {
                throw std::logic_error(
                    "graded k mesh must be finite and strictly increasing");
            }
        }

        Grid g;
        g.k     = std::move(x);
        g.k_max = g.k.back();
        g.w     = composite_trapezoid_weights(g.k);
        return g;
    }

    // Logarithmic / geometric k mesh: k[0]=0, k[1..N-1] uniform in ln(k) from
    // k_eps to k_max.  Resolves small k and the high-k tail; k_F is not an exact
    // node (step n(k) is slightly smeared vs a Fermi-aligned mesh).
    static Grid log_trapezoid(double k_F, double k_max_min, std::size_t N_target) {
        if (!std::isfinite(k_F) || !std::isfinite(k_max_min)
            || k_F <= 0.0 || k_max_min <= k_F || N_target < 3u) {
            throw std::invalid_argument(
                "log_trapezoid requires 0 < k_F < k_max and at least three nodes");
        }
        if (N_target > std::vector<double>().max_size()) {
            throw std::length_error("logarithmic grid is too large");
        }
        std::vector<double> x;
        x.reserve(N_target);
        x.push_back(0.0);
        const double k_eps =
            std::max(k_F * 1.0e-8, k_max_min * 1.0e-14);
        if (!std::isfinite(k_eps) || k_eps <= 0.0 || k_eps >= k_max_min) {
            throw std::invalid_argument("invalid logarithmic grid lower bound");
        }
        const double ln0 = std::log(k_eps);
        const double ln1 = std::log(k_max_min);
        const double inv = 1.0 / static_cast<double>(N_target - 2u);
        for (std::size_t i = 1; i < N_target; ++i) {
            const double t = static_cast<double>(i - 1u) * inv;
            x.push_back(std::exp(ln0 + t * (ln1 - ln0)));
        }
        if (x.size() != N_target) {
            throw std::logic_error("logarithmic grid node count mismatch");
        }
        for (std::size_t i = 1; i < x.size(); ++i) {
            if (!std::isfinite(x[i]) || x[i] <= x[i - 1]) {
                throw std::logic_error(
                    "log k mesh must be finite and strictly increasing");
            }
        }
        Grid g;
        g.k     = std::move(x);
        g.k_max = g.k.back();
        g.w     = composite_trapezoid_weights(g.k);
        return g;
    }

    // Hybrid mesh for n(k): logarithmic wing near k=0, dense uniform band
    // around k_F (with k_F exactly a node), and a coarser uniform high-k tail.
    // Combines the strengths of ``log_trapezoid`` and ``graded_fermi_trapezoid``.
    static Grid log_fermi_trapezoid(double k_F, double k_max_min,
                                    std::size_t N_target) {
        if (!std::isfinite(k_F) || !std::isfinite(k_max_min)
            || k_F <= 0.0 || k_max_min <= k_F || N_target < 31u) {
            throw std::invalid_argument(
                "log_fermi_trapezoid requires 0 < k_F < k_max and at least 31 nodes");
        }
        if (N_target > std::vector<double>().max_size()
            || N_target > std::numeric_limits<std::size_t>::max() / 13u) {
            throw std::length_error("log-fermi grid is too large");
        }

        // Dense mid band [k_lo, k_hi] of width ~ k_F, with k_F a node.
        std::size_t NM = std::max<std::size_t>(
            12u, std::min((N_target * 12u) / 25u, N_target - 16u));
        const double band = 1.0 * k_F;
        const double h_m  = band / static_cast<double>(NM);
        const std::size_t j_F = std::max<std::size_t>(
            1u, std::min((4u * NM) / 5u, NM - 1u));
        const double k_lo = k_F - static_cast<double>(j_F) * h_m;
        const double k_hi = k_lo + static_cast<double>(NM) * h_m;
        if (!(k_lo > 0.0) || !(k_hi < k_max_min)) {
            throw std::logic_error("log-fermi band must lie inside (0, k_max)");
        }

        const std::size_t N_rem = N_target - 1u - NM;
        if (N_rem < 8u) {
            throw std::logic_error("log-fermi grid has too few wing intervals");
        }

        // Prefer the left (log) wing so small-k structure is resolved, but keep
        // enough points on the high-k tail for the n(k>k_F) decay.
        constexpr double left_weight = 2.5;
        const double len_left  = k_lo;
        const double len_right = k_max_min - k_hi;
        const double wleft = left_weight * len_left;
        std::size_t NL = static_cast<std::size_t>(std::llround(
            static_cast<double>(N_rem) * (wleft / (wleft + len_right))));
        NL = std::max<std::size_t>(4u, std::min(N_rem - 4u, NL));
        const std::size_t NR = N_rem - NL;

        const double k_eps = std::max(k_F * 1.0e-6, k_lo * 1.0e-8);
        if (!(k_eps > 0.0) || !(k_eps < k_lo)) {
            throw std::invalid_argument("invalid log-fermi left-wing floor");
        }

        std::vector<double> x;
        x.reserve(N_target);
        x.push_back(0.0);
        // NL intervals on the left wing: k_eps ... k_lo (log), counting k_lo.
        // Nodes: 0, then NL points from k_eps to k_lo inclusive => NL+1 new?
        // graded uses i=0..NL giving NL+1 points from 0 to k_lo.
        // Here: point 0 = 0; points 1..NL-1 log in (k_eps, k_lo); point NL = k_lo.
        if (NL == 1u) {
            x.push_back(k_lo);
        } else {
            const double ln0 = std::log(k_eps);
            const double ln1 = std::log(k_lo);
            const double inv = 1.0 / static_cast<double>(NL - 1u);
            for (std::size_t i = 0; i < NL; ++i) {
                const double t = static_cast<double>(i) * inv;
                x.push_back(std::exp(ln0 + t * (ln1 - ln0)));
            }
        }
        // Mid band (skip k_lo already present).
        for (std::size_t t = 1; t <= NM; ++t) {
            x.push_back(k_lo + static_cast<double>(t) * h_m);
        }
        // Right wing.
        for (std::size_t t = 1; t <= NR; ++t) {
            x.push_back(k_hi + static_cast<double>(t) * (k_max_min - k_hi) /
                                    static_cast<double>(NR));
        }

        if (x.size() != N_target) {
            throw std::logic_error("log-fermi grid node count mismatch");
        }
        // Ensure k_F is (approximately) a node of the mid band.
        bool has_kf = false;
        for (double ki : x) {
            if (std::abs(ki - k_F) <= 1.0e-12 * std::max(1.0, k_F)) {
                has_kf = true;
                break;
            }
        }
        if (!has_kf) {
            throw std::logic_error("log-fermi grid missed k_F node");
        }
        for (std::size_t i = 1; i < x.size(); ++i) {
            if (!std::isfinite(x[i]) || x[i] <= x[i - 1]) {
                throw std::logic_error(
                    "log-fermi k mesh must be finite and strictly increasing");
            }
        }

        Grid g;
        g.k     = std::move(x);
        g.k_max = g.k.back();
        g.w     = composite_trapezoid_weights(g.k);
        return g;
    }
};

}  // namespace rdmft
