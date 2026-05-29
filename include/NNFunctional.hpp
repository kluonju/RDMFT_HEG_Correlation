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

// Load NN weights from JSON; throws std::runtime_error on failure.
std::unique_ptr<NNFunctional> make_nn_functional(const std::string& json_path);

}  // namespace rdmft
