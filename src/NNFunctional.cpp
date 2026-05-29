#include "NNFunctional.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace rdmft {
namespace {

void skip_ws(const std::string& s, std::size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
}

bool match(const std::string& s, std::size_t& i, char c) {
    skip_ws(s, i);
    if (i < s.size() && s[i] == c) {
        ++i;
        return true;
    }
    return false;
}

double parse_number(const std::string& s, std::size_t& i) {
    skip_ws(s, i);
    std::size_t start = i;
    if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
    while (i < s.size() &&
           (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.' ||
            s[i] == 'e' || s[i] == 'E' || s[i] == '-' || s[i] == '+')) {
        ++i;
    }
    if (start == i) {
        throw std::runtime_error("expected number in NN JSON at pos " + std::to_string(i));
    }
    return std::stod(s.substr(start, i - start));
}

std::vector<double> parse_array(const std::string& s, std::size_t& i) {
    if (!match(s, i, '[')) {
        throw std::runtime_error("expected '[' in NN JSON");
    }
    std::vector<double> out;
    skip_ws(s, i);
    if (match(s, i, ']')) return out;
    for (;;) {
        out.push_back(parse_number(s, i));
        skip_ws(s, i);
        if (match(s, i, ']')) break;
        if (!match(s, i, ',')) {
            throw std::runtime_error("expected ',' or ']' in NN JSON array");
        }
    }
    return out;
}

std::vector<std::vector<double>> parse_matrix(const std::string& s, std::size_t& i) {
    if (!match(s, i, '[')) {
        throw std::runtime_error("expected '[' for matrix in NN JSON");
    }
    std::vector<std::vector<double>> rows;
    skip_ws(s, i);
    if (match(s, i, ']')) return rows;
    for (;;) {
        rows.push_back(parse_array(s, i));
        skip_ws(s, i);
        if (match(s, i, ']')) break;
        if (!match(s, i, ',')) {
            throw std::runtime_error("expected ',' or ']' in NN JSON matrix");
        }
    }
    return rows;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open NN model file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::size_t find_key_after(const std::string& s, std::size_t start, const std::string& key) {
    const std::string quoted = "\"" + key + "\"";
    const auto pos = s.find(quoted, start);
    if (pos == std::string::npos) {
        throw std::runtime_error("NN JSON missing key: " + key);
    }
    return pos + quoted.size();
}

std::size_t find_key(const std::string& s, const std::string& key) {
    return find_key_after(s, 0, key);
}


// Parse a single layer object {"in":..,"out":..,"W":[[...]],"b":[..]}.
// On entry ``pos`` points at the next non-ws char (typically '{').
// On exit it points just past the closing '}'.  ``expected_in`` enforces the
// dim chain and is updated by the caller.
struct LayerData {
    std::size_t in_dim = 0;
    std::size_t out_dim = 0;
    std::vector<double> W;
    std::vector<double> b;
};

LayerData parse_layer_object(const std::string& json,
                             std::size_t& pos,
                             std::size_t expected_in) {
    skip_ws(json, pos);
    const std::size_t obj_start = pos;
    if (!match(json, pos, '{')) {
        throw std::runtime_error("NN JSON: expected kernel object");
    }
    LayerData L;
    std::size_t kp = find_key_after(json, obj_start, "in");
    kp = json.find(':', kp);
    L.in_dim = static_cast<std::size_t>(parse_number(json, ++kp));
    kp = find_key_after(json, obj_start, "out");
    kp = json.find(':', kp);
    L.out_dim = static_cast<std::size_t>(parse_number(json, ++kp));
    if (L.in_dim != expected_in) {
        throw std::runtime_error("NN JSON: layer input dimension mismatch");
    }

    kp = find_key_after(json, obj_start, "W");
    kp = json.find('[', kp);
    const auto Wmat = parse_matrix(json, kp);
    if (Wmat.size() != L.out_dim) {
        throw std::runtime_error("NN JSON: W row count mismatch");
    }
    L.W.resize(L.out_dim * L.in_dim);
    for (std::size_t r = 0; r < L.out_dim; ++r) {
        if (Wmat[r].size() != L.in_dim) {
            throw std::runtime_error("NN JSON: W column count mismatch");
        }
        for (std::size_t c = 0; c < L.in_dim; ++c) {
            L.W[r * L.in_dim + c] = Wmat[r][c];
        }
    }

    kp = find_key_after(json, obj_start, "b");
    kp = json.find('[', kp);
    L.b = parse_array(json, kp);
    if (L.b.size() != L.out_dim) {
        throw std::runtime_error("NN JSON: bias length mismatch");
    }

    pos = json.find('}', kp);
    if (pos == std::string::npos) {
        throw std::runtime_error("NN JSON: expected '}' after kernel");
    }
    ++pos;
    return L;
}

// Parse the MLP block (hidden + kernels + out_bias) into a flat list of
// LayerData.  Used by both NNFunctional and NNPairFunctional; the only
// difference is the input dimension (1 for separable, 2 for pair).
void parse_mlp_block(const std::string& json,
                     std::size_t input_dim,
                     std::vector<LayerData>& layers,
                     double& out_bias) {
    std::size_t pos = find_key(json, "hidden");
    pos = json.find('[', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("NN JSON: hidden sizes not found");
    }
    const auto hidden = parse_array(json, pos);

    pos = find_key(json, "kernels");
    pos = json.find('[', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("NN JSON: kernels array not found");
    }
    if (!match(json, pos, '[')) {
        throw std::runtime_error("NN JSON: malformed kernels");
    }

    std::size_t expected_in = input_dim;
    for (std::size_t ell = 0; ell < hidden.size() + 1; ++ell) {
        LayerData L = parse_layer_object(json, pos, expected_in);
        expected_in = L.out_dim;
        layers.push_back(std::move(L));
        skip_ws(json, pos);
        if (match(json, pos, ',')) continue;
        if (match(json, pos, ']')) break;
        throw std::runtime_error("NN JSON: expected ',' or ']' after kernel");
    }
    if (layers.size() != hidden.size() + 1) {
        throw std::runtime_error("NN JSON: kernel count != hidden.size()+1");
    }
    if (!layers.empty() && layers.back().out_dim != 1) {
        throw std::runtime_error("NN JSON: top-level output must be scalar (out=1)");
    }

    pos = find_key(json, "out_bias");
    pos = json.find(':', pos);
    out_bias = parse_number(json, ++pos);
}

// Read the optional "name" field; returns empty string if absent.
std::string parse_optional_name(const std::string& json) {
    const auto qpos = json.find("\"name\"");
    if (qpos == std::string::npos) return {};
    auto colon = json.find(':', qpos);
    if (colon == std::string::npos) return {};
    auto q1 = json.find('"', colon + 1);
    if (q1 == std::string::npos) return {};
    auto q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return {};
    return json.substr(q1 + 1, q2 - q1 - 1);
}

// Read the optional "kernel_type" string at the JSON root, defaulting to
// ``default_value``.  Used to dispatch separable vs pair NN models.
std::string parse_kernel_type(const std::string& json,
                              const std::string& default_value) {
    const auto qpos = json.find("\"kernel_type\"");
    if (qpos == std::string::npos) return default_value;
    auto colon = json.find(':', qpos);
    if (colon == std::string::npos) return default_value;
    auto q1 = json.find('"', colon + 1);
    if (q1 == std::string::npos) return default_value;
    auto q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return default_value;
    return json.substr(q1 + 1, q2 - q1 - 1);
}

}  // namespace

NNFunctional::NNFunctional(std::string model_path)
    : model_path_(std::move(model_path)) {
    const std::string json = read_file(model_path_);
    std::size_t pos = find_key(json, "name");
    pos = json.find(':', pos);
    if (pos != std::string::npos) {
        pos = json.find('"', pos + 1);
        if (pos != std::string::npos) {
            const std::size_t end = json.find('"', pos + 1);
            if (end != std::string::npos) {
                name_ = json.substr(pos + 1, end - pos - 1);
            }
        }
    }
    if (name_.empty()) name_ = "NN";

    pos = find_key(json, "hidden");
    pos = json.find('[', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("NN JSON: hidden sizes not found");
    }
    const auto hidden = parse_array(json, pos);

    pos = find_key(json, "kernels");
    pos = json.find('[', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("NN JSON: kernels array not found");
    }
    if (!match(json, pos, '[')) {
        throw std::runtime_error("NN JSON: malformed kernels");
    }

    std::size_t expected_in = 1;
    for (std::size_t ell = 0; ell < hidden.size() + 1; ++ell) {
        skip_ws(json, pos);
        const std::size_t obj_start = pos;
        if (!match(json, pos, '{')) {
            throw std::runtime_error("NN JSON: expected kernel object");
        }
        Layer L;
        std::size_t kp = find_key_after(json, obj_start, "in");
        kp = json.find(':', kp);
        L.in_dim = static_cast<std::size_t>(parse_number(json, ++kp));
        kp = find_key_after(json, obj_start, "out");
        kp = json.find(':', kp);
        L.out_dim = static_cast<std::size_t>(parse_number(json, ++kp));
        if (L.in_dim != expected_in) {
            throw std::runtime_error("NN JSON: layer input dimension mismatch");
        }
        expected_in = L.out_dim;

        kp = find_key_after(json, obj_start, "W");
        kp = json.find('[', kp);
        const auto Wmat = parse_matrix(json, kp);
        if (Wmat.size() != L.out_dim) {
            throw std::runtime_error("NN JSON: W row count mismatch");
        }
        L.W.resize(L.out_dim * L.in_dim);
        for (std::size_t r = 0; r < L.out_dim; ++r) {
            if (Wmat[r].size() != L.in_dim) {
                throw std::runtime_error("NN JSON: W column count mismatch");
            }
            for (std::size_t c = 0; c < L.in_dim; ++c) {
                L.W[r * L.in_dim + c] = Wmat[r][c];
            }
        }

        kp = find_key_after(json, obj_start, "b");
        kp = json.find('[', kp);
        L.b = parse_array(json, kp);
        if (L.b.size() != L.out_dim) {
            throw std::runtime_error("NN JSON: bias length mismatch");
        }

        pos = json.find('}', kp);
        if (pos == std::string::npos) {
            throw std::runtime_error("NN JSON: expected '}' after kernel");
        }
        ++pos;
        layers_.push_back(std::move(L));
        skip_ws(json, pos);
        if (match(json, pos, ',')) continue;
        if (match(json, pos, ']')) break;
        throw std::runtime_error("NN JSON: expected ',' or ']' after kernel");
    }

    if (layers_.size() != hidden.size() + 1) {
        throw std::runtime_error("NN JSON: kernel count != hidden.size()+1");
    }

    pos = find_key(json, "out_bias");
    pos = json.find(':', pos);
    out_bias_ = parse_number(json, ++pos);
}

double NNFunctional::softplus(double x) const {
    if (x > 30.0) return x;
    if (x < -30.0) return std::exp(x);
    return std::log1p(std::exp(x));
}

double NNFunctional::sigmoid(double x) const {
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(x);
    return z / (1.0 + z);
}

void NNFunctional::eval_forward(double n,
                                std::vector<double>& h_last,
                                double& raw) const {
    std::vector<double> h_in(1, n);
    const std::size_t nlay = layers_.size();
    for (std::size_t ell = 0; ell < nlay; ++ell) {
        const Layer& L = layers_[ell];
        h_last.assign(L.out_dim, 0.0);
        for (std::size_t o = 0; o < L.out_dim; ++o) {
            double s = L.b[o];
            const double* wrow = &L.W[o * L.in_dim];
            for (std::size_t j = 0; j < L.in_dim; ++j) {
                s += wrow[j] * h_in[j];
            }
            if (ell + 1 < nlay) {
                h_last[o] = std::tanh(s);
            } else {
                h_last[o] = s;
            }
        }
        h_in.swap(h_last);
    }
    raw = out_bias_;
    if (!h_in.empty() && !layers_.empty() && layers_.back().out_dim == 1) {
        raw += h_in[0];
    }
}

double NNFunctional::f(double n) const {
    if (n <= 0.0) return 0.0;
    std::vector<double> h;
    double raw = 0.0;
    eval_forward(n, h, raw);
    return n * softplus(raw);
}

double NNFunctional::df(double n) const {
    if (n <= 0.0) return 0.0;

    constexpr double eps = 1.0e-7;
    const double h = std::max(eps, 1.0e-5 * std::max(n, 1.0));
    return (f(n + h) - f(n - h)) / (2.0 * h);
}

std::unique_ptr<NNFunctional> make_nn_functional(const std::string& json_path) {
    return std::make_unique<NNFunctional>(json_path);
}

// -------------------------------------------------------------------------
// NNPairFunctional: non-separable two-input pair kernel K(n_i, n_j).
// -------------------------------------------------------------------------

NNPairFunctional::NNPairFunctional(std::string model_path)
    : model_path_(std::move(model_path)) {
    const std::string json = read_file(model_path_);
    name_ = parse_optional_name(json);
    if (name_.empty()) name_ = "NNPair";
    std::vector<LayerData> raw;
    parse_mlp_block(json, /*input_dim=*/2, raw, out_bias_);
    layers_.reserve(raw.size());
    for (auto& r : raw) {
        Layer L;
        L.in_dim  = r.in_dim;
        L.out_dim = r.out_dim;
        L.W       = std::move(r.W);
        L.b       = std::move(r.b);
        layers_.push_back(std::move(L));
    }
}

double NNPairFunctional::softplus(double x) {
    if (x > 30.0) return x;
    if (x < -30.0) return std::exp(x);
    return std::log1p(std::exp(x));
}

double NNPairFunctional::raw_at(double ni, double nj) const {
    // Symmetric features: s = ni + nj, p = ni * nj.  Both are invariant
    // under (ni, nj) -> (nj, ni), so the pair kernel is symmetric for any
    // weight vector.
    std::vector<double> h_in(2);
    h_in[0] = ni + nj;
    h_in[1] = ni * nj;
    std::vector<double> h_out;
    const std::size_t nlay = layers_.size();
    for (std::size_t ell = 0; ell < nlay; ++ell) {
        const Layer& L = layers_[ell];
        h_out.assign(L.out_dim, 0.0);
        for (std::size_t o = 0; o < L.out_dim; ++o) {
            double s = L.b[o];
            const double* wrow = &L.W[o * L.in_dim];
            for (std::size_t j = 0; j < L.in_dim; ++j) {
                s += wrow[j] * h_in[j];
            }
            if (ell + 1 < nlay) {
                h_out[o] = std::tanh(s);
            } else {
                h_out[o] = s;
            }
        }
        h_in.swap(h_out);
    }
    return out_bias_ + (h_in.empty() ? 0.0 : h_in[0]);
}

double NNPairFunctional::kernel(double ni, double nj) const {
    if (ni <= 0.0 || nj <= 0.0) return 0.0;
    const double p = ni * nj;
    const double prefac = std::sqrt(p);
    return prefac * softplus(raw_at(ni, nj));
}

double NNPairFunctional::kernel_grad(double ni, double nj) const {
    // dK/dn_i with n_j held fixed.  Computed via centered finite differences
    // for simplicity; the cost is amortised over the SCF projected-gradient
    // step which already does N kernel evaluations per row (the FD step
    // doubles that, which is acceptable for a research functional).
    if (ni <= 0.0 || nj <= 0.0) return 0.0;
    const double eps = 1.0e-6;
    const double h = std::max(eps, 1.0e-5 * std::max(ni, 1.0));
    const double a_lo = std::max(ni - h, 1.0e-12);
    const double a_hi = std::min(ni + h, 1.0);
    const double K_hi = kernel(a_hi, nj);
    const double K_lo = kernel(a_lo, nj);
    const double dn = a_hi - a_lo;
    return (dn > 0.0) ? (K_hi - K_lo) / dn : 0.0;
}

std::unique_ptr<NNPairFunctional> make_nn_pair_functional(
    const std::string& json_path) {
    return std::make_unique<NNPairFunctional>(json_path);
}

// Dispatcher: read the optional "kernel_type" string and return the right
// concrete class.  Defaults to the legacy separable form so existing
// model.json files keep loading unchanged.
std::unique_ptr<Functional> make_nn_functional_any(const std::string& json_path) {
    const std::string json = read_file(json_path);
    const std::string ktype = parse_kernel_type(json, "separable");
    if (ktype == "pair") {
        return std::make_unique<NNPairFunctional>(json_path);
    }
    if (ktype == "separable" || ktype.empty()) {
        return std::make_unique<NNFunctional>(json_path);
    }
    throw std::runtime_error("NN JSON: unknown kernel_type \"" + ktype + "\"");
}

std::unique_ptr<Functional> load_nn_functional(const std::string& json_path) {
    return make_nn_functional_any(json_path);
}

}  // namespace rdmft
