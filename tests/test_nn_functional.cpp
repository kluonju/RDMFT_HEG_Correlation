// Unit tests for NNFunctional JSON load and f(n)/df(n) consistency.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

#include "Functional.hpp"
#include "NNFunctional.hpp"

using namespace rdmft;

static int failures = 0;

static void check(bool ok, const char* msg) {
    if (!ok) {
        std::cerr << "FAIL: " << msg << "\n";
        ++failures;
    }
}

static void check_near(double a, double b, double tol, const char* msg) {
    if (std::abs(a - b) > tol) {
        std::cerr << "FAIL: " << msg << "  got " << a << " expected " << b << "\n";
        ++failures;
    }
}

int main() {
    const std::string path = "tests/fixtures/nn_power055.json";
    std::unique_ptr<Functional> F = load_nn_functional(path);
    if (!F) {
        std::cerr << "FAIL: could not load " << path << "\n";
        return 1;
    }
    const auto* nn = dynamic_cast<const NNFunctional*>(F.get());
    check(nn != nullptr, "dynamic_cast to NNFunctional");

    check_near(F->f(0.0), 0.0, 1e-14, "f(0)=0");

    const double n_test = 0.42;
    const double h = 1e-7;
    const double fd = (F->f(n_test + h) - F->f(n_test - h)) / (2.0 * h);
    check_near(F->df(n_test), fd, 5e-5, "df vs central difference");

    // Smoke: f(n) positive on (0,1]
    for (double n = 0.05; n <= 1.0; n += 0.1) {
        check(F->f(n) >= 0.0, "f(n) >= 0");
    }

    if (failures == 0) {
        std::cout << "test_nn_functional: all checks passed\n";
        return 0;
    }
    std::cerr << failures << " failure(s)\n";
    return 1;
}
