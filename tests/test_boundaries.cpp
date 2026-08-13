#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "Energy.hpp"
#include "ExchangeKernel.hpp"
#include "Functional.hpp"
#include "Grid.hpp"

using namespace rdmft;

namespace {

int failures = 0;

void expect_exception(const std::function<void()>& operation,
                      const std::string& message) {
    try {
        operation();
        std::cerr << "FAIL: " << message << "\n";
        ++failures;
    } catch (const std::exception&) {
    }
}

}  // namespace

int main() {
    expect_exception(
        [] { (void)Grid::uniform_trapezoid(1.0, 1); },
        "uniform grid should reject fewer than two nodes");
    expect_exception(
        [] {
            (void)Grid::uniform_trapezoid(
                std::numeric_limits<double>::quiet_NaN(), 3);
        },
        "uniform grid should reject non-finite bounds");
    expect_exception(
        [] {
            (void)Grid::composite_trapezoid_weights({0.0, 1.0, 1.0});
        },
        "quadrature should reject duplicate nodes");

    Grid grid = Grid::uniform_trapezoid(2.0, 5);
    ExchangeKernel exchange = ExchangeKernel::build(grid);
    HFFunctional hf;

    {
        const double kf = 1.0;
        Grid g_logf = Grid::log_fermi_trapezoid(kf, 5.0 * kf, 401u);
        if (g_logf.n() != 401u) {
            std::cerr << "FAIL: log_fermi_trapezoid node count\n";
            ++failures;
        }
        bool has_kf = false;
        for (double k : g_logf.k) {
            if (std::abs(k - kf) <= 1.0e-12) has_kf = true;
        }
        if (!has_kf) {
            std::cerr << "FAIL: log_fermi_trapezoid should place a node at k_F\n";
            ++failures;
        }
        Grid g_grad = Grid::graded_fermi_trapezoid(kf, 5.0 * kf, 401u);
        if (g_grad.n() != 401u) {
            std::cerr << "FAIL: graded_fermi_trapezoid node count\n";
            ++failures;
        }
    }

    expect_exception(
        [&] {
            (void)EnergyEvaluator::density(
                std::vector<double>(grid.n() - 1, 0.5), grid);
        },
        "energy evaluators should reject occupation-size mismatches");

    expect_exception(
        [&] {
            ExchangeKernel malformed = exchange;
            malformed.Wt.pop_back();
            (void)EnergyEvaluator::xc_per_volume(
                std::vector<double>(grid.n(), 0.5), grid, malformed, hf);
        },
        "energy evaluators should reject malformed exchange matrices");

    expect_exception(
        [] {
            Grid malformed;
            malformed.k = {0.0, 1.0};
            malformed.w = {1.0};
            (void)ExchangeKernel::build(malformed);
        },
        "exchange construction should reject mismatched grid storage");

    expect_exception(
        [] { (void)PowerFunctional(0.0); },
        "Power should reject a zero exponent");
    expect_exception(
        [] { (void)BetaFunctional(1.1); },
        "Beta should reject a non-monotone exponent");
    expect_exception(
        [] { (void)HybOptFunctional(1.1, 0.5); },
        "HybOpt should reject lambda outside [0,1]");
    expect_exception(
        [] { (void)OptGeoFunctional(-1.0, 2.0); },
        "OptGeo should reject negative parameters");

    {
        PbeHfMuFunctional pbe;
        if (std::abs(pbe.kernel(1.0, 1.0)) > 1.0e-14) {
            std::cerr << "FAIL: PBEHFmM K(1,1) should be 0\n";
            ++failures;
        }
        if (std::abs(pbe.kernel(0.0, 0.7)) > 1.0e-14) {
            std::cerr << "FAIL: PBEHFmM K(0,n) should be 0\n";
            ++failures;
        }
        const double ni = 0.4, nj = 0.7;
        const double h = 1.0e-8;
        const double fd = (pbe.kernel(ni + h, nj) - pbe.kernel(ni - h, nj)) / (2.0 * h);
        if (std::abs(pbe.kernel_grad(ni, nj) - fd) > 1.0e-6) {
            std::cerr << "FAIL: PBEHFmM kernel_grad vs central difference\n";
            ++failures;
        }
        const double rs = 1.0;
        const double rho = HEG::density(rs);
        const double expected = rho * (HEG::exchange_per_electron_HF(rs)
                                       + PW92::ec_per_electron(rs));
        if (std::abs(pbe.density_xc_per_volume(rho) - expected) > 1.0e-12) {
            std::cerr << "FAIL: PBEHFmM density_xc_per_volume at rs=1\n";
            ++failures;
        }
        if (pbe.name() != "PBEHFmM") {
            std::cerr << "FAIL: PBEHFmM name()\n";
            ++failures;
        }
    }

    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "test_boundaries: all checks passed\n";
    return 0;
}
