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
// To give the optimizer maximum flexibility while keeping the kernel
// physically well-posed, the construction is
//
//   inputs   = symmetric features  [s, p] = [n_i + n_j, n_i * n_j]
//   h_0      = inputs
//   h_l      = tanh(W_l h_{l-1} + b_l)        (l = 1 .. L)
//   raw      = W_out h_L + b_out               (scalar; arbitrary sign)
//   K(n_i, n_j) = sqrt(max(n_i * n_j, 0)) * raw
//
// Notable properties:
//
//   * Symmetry          K(a, b) = K(b, a)  (the MLP only sees the
//                       symmetric features s = a+b, p = a*b).
//   * Zero boundary     K(0, b) = K(a, 0) = 0  (sqrt(p) prefactor; this
//                       is the Pauli-style boundary that all factorisable
//                       RDMFT kernels HF / Mueller / Power obey).
//   * Sign-free output  ``raw`` is the unrestricted MLP output, so K can
//                       attain either sign (cf. BBC1's weak-occupation
//                       sign flip) and is not forced to be positive.
//                       The optimizer therefore has freedom to pick the
//                       magnitude *and* sign of K.
//   * Magnitude         no softplus, no sigmoid; K is unbounded apart
//                       from the sqrt(p) prefactor.  The MLP itself
//                       provides whatever scaling minimises the loss.
//
// is_factorized() returns false, so solve_rdmft routes the SCF through
// the generic projected-gradient branch in Solver.hpp (the kernel does
// not split into a single f(n)).  Pointwise k_i^2/2 + dE_xc/dn_i is
// evaluated through kernel() and kernel_grad() below; the latter is
// computed by centered finite differences (sufficient for the SCF).
//
// Weights share the same JSON layout as NNFunctional (a list of dense
// layers with row-major W and bias b, plus an out_bias scalar) but with
// first-layer in_dim = 2 and the top-level field
//   "kernel_type": "pair"
// so the loader can disambiguate.  See src/NNFunctional.cpp for the
// parser.
class NNPairFunctional : public Functional {
public:
    explicit NNPairFunctional(std::string model_path);

    std::string name() const override { return name_; }
    bool is_factorized() const override { return false; }

    // Provided to satisfy the abstract base; not used by the SCF since
    // is_factorized() is false.  Returns Mueller-like sqrt(n) so any
    // accidental call yields a finite value.
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
};

// Load NN weights from JSON; throws std::runtime_error on failure.
//
// The factory reads the optional "kernel_type" string at the JSON root
// (default "separable") and returns either an NNFunctional (separable
// f(n)) or an NNPairFunctional (non-separable K(n_i, n_j)).
std::unique_ptr<Functional> make_nn_functional_any(const std::string& json_path);
std::unique_ptr<NNFunctional> make_nn_functional(const std::string& json_path);
std::unique_ptr<NNPairFunctional> make_nn_pair_functional(
    const std::string& json_path);

}  // namespace rdmft
