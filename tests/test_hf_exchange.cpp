// Sanity test: the discrete log-kernel double integral with n(k) = step(kF-k)
// should converge to the analytic Hartree-Fock exchange energy per electron
//
//     e_x = -(3 kF) / (4 pi)
//
// in Hartree atomic units.  Also checks that the bisection-based solver lands
// on the same step function for the HF functional.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "Energy.hpp"
#include "ExchangeKernel.hpp"
#include "Functional.hpp"
#include "Grid.hpp"
#include "HEG.hpp"
#include "Solver.hpp"

using namespace rdmft;

int main() {
    double rs = 2.0;
    double kf = HEG::kF(rs);

    Grid g            = Grid::graded_fermi_trapezoid(kf, 3.0 * kf, 401);
    ExchangeKernel W  = ExchangeKernel::build(g);

    std::vector<double> n(g.n(), 0.0);
    for (std::size_t i = 0; i < g.n(); ++i) {
        const double k = g.k[i];
        if (k < kf - 1.0e-12)      n[i] = 1.0;
        else if (k < kf + 1.0e-12) n[i] = 0.5;  // midpoint convention at the step
        else                        n[i] = 0.0;
    }

    HFFunctional F;
    double T  = EnergyEvaluator::kinetic_per_volume(n, g);
    double X  = EnergyEvaluator::xc_per_volume(n, g, W, F);
    double rho = EnergyEvaluator::density(n, g);
    double t_per_N = T / rho, x_per_N = X / rho;

    double t_ref = HEG::kinetic_per_electron_HF(rs);
    double x_ref = HEG::exchange_per_electron_HF(rs);

    std::printf("HF step occupations at rs=%.2f, kF=%.6f, N=%zu, kmax=%.3f\n",
                rs, kf, g.n(), g.k_max);
    std::printf("  T/N: numeric=%.8f  analytic=%.8f  err=%.2e\n",
                t_per_N, t_ref, std::abs(t_per_N - t_ref));
    std::printf("  X/N: numeric=%.8f  analytic=%.8f  err=%.2e\n",
                x_per_N, x_ref, std::abs(x_per_N - x_ref));

    double err_T = std::abs(t_per_N - t_ref) / std::abs(t_ref);
    double err_X = std::abs(x_per_N - x_ref) / std::abs(x_ref);
    if (err_T > 5.0e-3 || err_X > 5.0e-3) {
        std::fprintf(stderr, "FAIL: relative error too large.\n");
        return 1;
    }

    SolveOptions opt;
    opt.verbose = false;
    SolveResult r = solve_rdmft(rs, F, g, W, opt);
    std::printf("Self-consistent HF: E/N=%.8f  vs analytic=%.8f\n",
                r.E_per_N, t_ref + x_ref);
    if (std::abs(r.E_per_N - (t_ref + x_ref)) > 5.0e-3) {
        std::fprintf(stderr, "FAIL: HF self-consistent energy mismatch.\n");
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
