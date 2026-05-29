#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "Functional.hpp"

namespace rdmft {

// Separable kernel K(n_i, n_j) = f(n_i) f(n_j) with f from a small MLP:
//
//   h_0 = n
//   h_l = tanh(W_l h_{l-1} + b_l)     (l = 1 .. L)
//   raw = W_out h_L + b_out           (scalar)
//   f(n) = n * softplus(raw)
//
// Weights are loaded from a JSON file (see src/NNFunctional.cpp).
class NNFunctional : public Functional {
public:
    explicit NNFunctional(std::string model_path);

    std::string name() const override { return name_; }
    bool is_factorized() const override { return true; }

    double f(double n) const override;
    double df(double n) const override;

    const std::string& model_path() const { return model_path_; }

private:
    struct Layer {
        std::size_t in_dim = 0;
        std::size_t out_dim = 0;
        std::vector<double> W;  // row-major out_dim x in_dim
        std::vector<double> b;  // out_dim
    };

    std::string model_path_;
    std::string name_;
    std::vector<Layer> layers_;
    double out_bias_ = 0.0;

    void eval_forward(double n, std::vector<double>& h_last, double& raw) const;
    double softplus(double x) const;
    double sigmoid(double x) const;
};

// Non-separable pair kernel K(n_i, n_j) parameterised by a small MLP.
//
//   inputs   = symmetric features  [s, p] = [n_i + n_j, n_i * n_j]
//   h_0      = inputs
//   h_l      = tanh(W_l h_{l-1} + b_l)     (l = 1 .. L)
//   raw      = W_out h_L + b_out             (scalar)
//   K(n_i, n_j) = sqrt(max(n_i * n_j, 0)) * softplus(raw)
//
// The construction enforces the physical constraints discussed for the UEG
// coupling function:
//
//   * Symmetry          K(a, b) = K(b, a)  (inputs are symmetric in a, b)
//   * Zero boundary     K(0, b) = K(a, 0) = 0  (sqrt(p) prefactor)
//   * Non-negativity    softplus(.) > 0  =>  K >= 0
//   * HF-like saturation at K(1, 1) = softplus(raw(2, 1))    (controlled by
//     the network output bias; the optimizer is free to drive it toward 1
//     via the same GZ-target loss used for the separable kernel).
//
// is_factorized() returns false, so solve_rdmft routes the SCF through the
// generic projected-gradient branch in Solver.hpp (the kernel does not split
// into a single f(n)).  Pointwise k_i^2/2 + dE_xc/dn_i is evaluated through
// kernel() and kernel_grad() below.
//
// Weights share the same JSON layout as NNFunctional (a list of dense layers
// with row-major W and bias b, plus an out_bias scalar) but with first-layer
// in_dim = 2 and the top-level field "kernel_type": "pair" so the loader can
// disambiguate.  See src/NNFunctional.cpp for the parser.
class NNPairFunctional : public Functional {
public:
    explicit NNPairFunctional(std::string model_path);

    std::string name() const override { return name_; }
    bool is_factorized() const override { return false; }

    // Provided to satisfy the abstract base; not used by the SCF since
    // is_factorized() is false.  We expose Mueller-like sqrt(n) so that any
    // accidental call returns a reasonable, non-divergent number.
    double f(double n) const override {
        return n > 0.0 ? std::sqrt(n) : 0.0;
    }
    double df(double n) const override {
        const double eps = 1.0e-14;
        return 0.5 / std::sqrt(n > eps ? n : eps);
    }

    double kernel(double ni, double nj) const override;
    double kernel_grad(double ni, double nj) const override;

    const std::string& model_path() const { return model_path_; }

private:
    struct Layer {
        std::size_t in_dim = 0;
        std::size_t out_dim = 0;
        std::vector<double> W;
        std::vector<double> b;
    };

    std::string model_path_;
    std::string name_;
    std::vector<Layer> layers_;
    double out_bias_ = 0.0;

    double raw_at(double ni, double nj) const;
    static double softplus(double x);
};

// Load NN weights from JSON; throws std::runtime_error on failure.
//
// The factory reads the optional "kernel_type" string at the JSON root
// (default "separable") and returns either an NNFunctional (separable f(n))
// or an NNPairFunctional (non-separable K(n_i, n_j)).  Callers that want a
// concrete type can use ``make_nn_functional`` (separable only) or
// ``make_nn_pair_functional`` (non-separable only).
std::unique_ptr<Functional> make_nn_functional_any(const std::string& json_path);
std::unique_ptr<NNFunctional> make_nn_functional(const std::string& json_path);
std::unique_ptr<NNPairFunctional> make_nn_pair_functional(
    const std::string& json_path);

}  // namespace rdmft
