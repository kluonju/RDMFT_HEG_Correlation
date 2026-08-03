// Unit tests for NNFunctional JSON load and f(n)/df(n) consistency.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "ExchangeKernel.hpp"
#include "Functional.hpp"
#include "Grid.hpp"
#include "HEG.hpp"
#include "NNFunctional.hpp"
#include "Solver.hpp"

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

static void check_model_rejected(const std::string& json, const char* msg) {
    static int counter = 0;
    const std::filesystem::path path =
        std::filesystem::temp_directory_path()
        / ("rdmft_invalid_nn_" + std::to_string(counter++) + ".json");
    {
        std::ofstream out(path);
        out << json;
    }
    bool rejected = false;
    try {
        (void)load_nn_functional(path.string());
    } catch (const std::exception&) {
        rejected = true;
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    check(rejected, msg);
}

int main() {
    {
        std::unique_ptr<Functional> functional =
            make_functional("HybOpt@1.0;0.5");
        const auto* hybrid =
            dynamic_cast<const HybOptFunctional*>(functional.get());
        check(hybrid != nullptr && hybrid->lambda_mix() == 1.0,
              "HybOpt factory preserves lambda=1");
    }

    const std::string path = "tests/fixtures/nn_power055.json";
    std::unique_ptr<Functional> F = load_nn_functional(path);
    if (!F) {
        std::cerr << "FAIL: could not load " << path << "\n";
        return 1;
    }
    const auto* nn = dynamic_cast<const NNFunctional*>(F.get());
    check(nn != nullptr, "dynamic_cast to NNFunctional");

    check_near(F->f(0.0), 0.0, 1e-14, "f(0)=0");
    check(std::isfinite(F->df(0.0)) && F->df(0.0) > 0.0,
          "df(0) is the finite right derivative");

    const double n_test = 0.42;
    const double h = 1e-7;
    const double fd = (F->f(n_test + h) - F->f(n_test - h)) / (2.0 * h);
    check_near(F->df(n_test), fd, 5e-5, "df vs central difference");

    // Smoke: f(n) positive on (0,1]
    for (double n = 0.05; n <= 1.0; n += 0.1) {
        check(F->f(n) >= 0.0, "f(n) >= 0");
    }

    {
        const double rs = 2.0;
        Grid grid = Grid::uniform_trapezoid(3.0 * HEG::kF(rs), 41);
        ExchangeKernel exchange = ExchangeKernel::build(grid);
        SolveOptions options;
        options.max_iter = 400;
        options.tol_n = 1.0e-7;
        options.tol_rho = 1.0e-7;
        const SolveResult result =
            solve_rdmft(rs, *F, grid, exchange, options);
        check(result.converged, "generic NN functional solve converges");
        check(std::isfinite(result.E_per_N), "generic NN solve energy is finite");
        check(std::abs(result.rho - HEG::density(rs))
                  <= options.tol_rho * std::max(1.0, HEG::density(rs)),
              "generic NN solve satisfies the density constraint");
    }

    // -------- NNPairFunctional checks --------------------------------------
    const std::string pair_path = "tests/fixtures/nn_pair.json";
    std::unique_ptr<Functional> P = load_nn_functional(pair_path);
    if (!P) {
        std::cerr << "FAIL: could not load " << pair_path << "\n";
        return 1;
    }
    const auto* nnp = dynamic_cast<const NNPairFunctional*>(P.get());
    check(nnp != nullptr, "dynamic_cast to NNPairFunctional");
    check(P->is_factorized() == false, "NNPair: is_factorized() is false");

    // Zero-boundary K(0, x) = K(x, 0) = 0 over [0, 1].
    for (double x = 0.0; x <= 1.0; x += 0.1) {
        check_near(P->kernel(0.0, x), 0.0, 1e-14, "K(0, x) = 0");
        check_near(P->kernel(x, 0.0), 0.0, 1e-14, "K(x, 0) = 0");
    }

    // Symmetry: K(a, b) = K(b, a) on a representative grid.
    const double pairs[][2] = {{0.1, 0.7}, {0.3, 0.5}, {0.42, 0.42},
                                {0.05, 0.95}, {0.6, 0.2}, {0.8, 0.8}};
    for (const auto& ab : pairs) {
        const double k_ab = P->kernel(ab[0], ab[1]);
        const double k_ba = P->kernel(ab[1], ab[0]);
        check_near(k_ab, k_ba, 1e-14, "K(a, b) = K(b, a)");
    }

    // kernel_grad central-difference consistency at an interior point.
    const double a0 = 0.42, b0 = 0.71;
    const double dh = 1e-5;
    const double fd_ab =
        (P->kernel(a0 + dh, b0) - P->kernel(a0 - dh, b0)) / (2.0 * dh);
    check_near(P->kernel_grad(a0, b0), fd_ab, 5e-3,
               "NNPair: kernel_grad vs central difference");

    check_model_rejected(
        R"({"hidden":[1.5],"kernels":[],"out_bias":0})",
        "fractional hidden dimensions are rejected");
    check_model_rejected(
        R"({"hidden":[],"kernels":[{"in":1,"out":1,"W":[[1e9999]],"b":[0]}],"out_bias":0})",
        "non-finite model weights are rejected");
    check_model_rejected(
        R"({"hidden":[2],"kernels":[{"in":1,"out":3}],"out_bias":0})",
        "declared layer dimensions must match hidden dimensions");

    if (failures == 0) {
        std::cout << "test_nn_functional: all checks passed\n";
        return 0;
    }
    std::cerr << failures << " failure(s)\n";
    return 1;
}
