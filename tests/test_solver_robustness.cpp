#include <cmath>
#include <iostream>
#include <limits>
#include <string>

#include "ExchangeKernel.hpp"
#include "Functional.hpp"
#include "Grid.hpp"
#include "HEG.hpp"
#include "Solver.hpp"

using namespace rdmft;

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++failures;
    }
}

class GenericSmoothFunctional final : public Functional {
public:
    std::string name() const override { return "GenericSmooth"; }
    double f(double n) const override {
        return n + 0.2 * n * (1.0 - n);
    }
    double df(double n) const override {
        return 1.2 - 0.4 * n;
    }
};

class NonFiniteFunctional final : public Functional {
public:
    std::string name() const override { return "NonFinite"; }
    bool is_factorized() const override { return false; }
    double f(double) const override { return 0.0; }
    double df(double) const override { return 0.0; }
    double kernel(double, double) const override {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double kernel_grad(double, double) const override {
        return std::numeric_limits<double>::quiet_NaN();
    }
};

}  // namespace

int main() {
    const double rs = 2.0;
    const double kf = HEG::kF(rs);

    {
        Grid g = Grid::uniform_trapezoid(3.0 * kf, 81);
        ExchangeKernel W = ExchangeKernel::build(g);
        GenericSmoothFunctional functional;
        SolveOptions options;
        options.max_iter = 1200;
        options.tol_n = 1.0e-7;
        options.tol_rho = 1.0e-7;

        const SolveResult result = solve_rdmft(rs, functional, g, W, options);
        check(result.converged,
              "generic factorized functional should use the robust projected-gradient path: "
                  + result.failure_reason);
        check(std::isfinite(result.E_per_N), "generic factorized energy should be finite");
        check(std::abs(result.rho - HEG::density(rs))
                  <= options.tol_rho * std::max(1.0, HEG::density(rs)),
              "generic factorized result should satisfy the density constraint (rho="
                  + std::to_string(result.rho) + ")");
    }

    {
        Grid g = Grid::uniform_trapezoid(3.0 * kf, 31);
        ExchangeKernel W = ExchangeKernel::build(g);
        NonFiniteFunctional functional;
        SolveOptions options;
        options.max_iter = 5;

        const SolveResult result = solve_rdmft(rs, functional, g, W, options);
        check(!result.converged, "non-finite functional must not converge");
        check(!result.failure_reason.empty(),
              "non-finite functional should report a failure reason");
    }

    {
        Grid g = Grid::uniform_trapezoid(0.5 * kf, 31);
        ExchangeKernel W = ExchangeKernel::build(g);
        HFFunctional functional;
        SolveOptions options;
        options.max_iter = 5;

        const SolveResult result = solve_rdmft(rs, functional, g, W, options);
        check(!result.converged, "an undersized grid must not report convergence");
        check(result.failure_reason.find("bracket") != std::string::npos,
              "failed chemical-potential bracketing should be reported");
    }

    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "test_solver_robustness: all checks passed\n";
    return 0;
}
