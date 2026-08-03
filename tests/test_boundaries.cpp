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

    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "test_boundaries: all checks passed\n";
    return 0;
}
