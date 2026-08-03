#include "NNFunctional.hpp"

#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace rdmft {
namespace {

constexpr std::size_t kMaxHiddenLayers = 64;
constexpr std::size_t kMaxModelParameters = 10'000'000;

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
    const std::size_t start = i;
    if (i < s.size() && s[i] == '-') ++i;
    if (i >= s.size()) {
        throw std::runtime_error(
            "expected number in NN JSON at pos " + std::to_string(start));
    }
    if (s[i] == '0') {
        ++i;
    } else {
        const std::size_t integer_start = i;
        while (i < s.size()
               && std::isdigit(static_cast<unsigned char>(s[i]))) {
            ++i;
        }
        if (i == integer_start) {
            throw std::runtime_error(
                "expected number in NN JSON at pos " + std::to_string(start));
        }
    }
    if (i < s.size() && s[i] == '.') {
        ++i;
        const std::size_t fraction_start = i;
        while (i < s.size()
               && std::isdigit(static_cast<unsigned char>(s[i]))) {
            ++i;
        }
        if (i == fraction_start) {
            throw std::runtime_error(
                "invalid fraction in NN JSON at pos " + std::to_string(start));
        }
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        ++i;
        if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
        const std::size_t exponent_start = i;
        while (i < s.size()
               && std::isdigit(static_cast<unsigned char>(s[i]))) {
            ++i;
        }
        if (i == exponent_start) {
            throw std::runtime_error(
                "invalid exponent in NN JSON at pos " + std::to_string(start));
        }
    }
    if (start == i) {
        throw std::runtime_error("expected number in NN JSON at pos " + std::to_string(i));
    }
    std::size_t parsed = 0;
    double value = 0.0;
    try {
        value = std::stod(s.substr(start, i - start), &parsed);
    } catch (const std::exception&) {
        throw std::runtime_error(
            "invalid number in NN JSON at pos " + std::to_string(start));
    }
    if (parsed != i - start || !std::isfinite(value)) {
        throw std::runtime_error(
            "NN JSON numbers must be finite at pos " + std::to_string(start));
    }
    return value;
}

std::size_t parse_dimension(const std::string& s, std::size_t& i,
                            const char* label) {
    const double value = parse_number(s, i);
    if (value < 1.0 || std::floor(value) != value
        || value >= static_cast<double>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            std::string("NN JSON: ") + label + " must be a positive integer");
    }
    return static_cast<std::size_t>(value);
}

std::vector<double> parse_array(
    const std::string& s, std::size_t& i,
    std::size_t max_elements = std::numeric_limits<std::size_t>::max()) {
    if (!match(s, i, '[')) {
        throw std::runtime_error("expected '[' in NN JSON");
    }
    std::vector<double> out;
    skip_ws(s, i);
    if (match(s, i, ']')) return out;
    for (;;) {
        if (out.size() >= max_elements) {
            throw std::runtime_error("NN JSON array has too many elements");
        }
        out.push_back(parse_number(s, i));
        skip_ws(s, i);
        if (match(s, i, ']')) break;
        if (!match(s, i, ',')) {
            throw std::runtime_error("expected ',' or ']' in NN JSON array");
        }
    }
    return out;
}

std::vector<std::vector<double>> parse_matrix(
    const std::string& s, std::size_t& i,
    std::size_t max_rows = std::numeric_limits<std::size_t>::max(),
    std::size_t max_columns = std::numeric_limits<std::size_t>::max()) {
    if (!match(s, i, '[')) {
        throw std::runtime_error("expected '[' for matrix in NN JSON");
    }
    std::vector<std::vector<double>> rows;
    skip_ws(s, i);
    if (match(s, i, ']')) return rows;
    for (;;) {
        if (rows.size() >= max_rows) {
            throw std::runtime_error("NN JSON matrix has too many rows");
        }
        rows.push_back(parse_array(s, i, max_columns));
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
    if (!in.eof() && in.fail()) {
        throw std::runtime_error("cannot read NN model file: " + path);
    }
    return ss.str();
}

std::size_t find_key_after(const std::string& s, std::size_t start,
                           const std::string& key,
                           std::size_t end = std::string::npos) {
    const std::string quoted = "\"" + key + "\"";
    const auto pos = s.find(quoted, start);
    if (pos == std::string::npos || (end != std::string::npos && pos >= end)) {
        throw std::runtime_error("NN JSON missing key: " + key);
    }
    return pos + quoted.size();
}

std::size_t find_key(const std::string& s, const std::string& key) {
    return find_key_after(s, 0, key);
}


// --- shared helpers used by both NNFunctional and NNPairFunctional ---

struct LayerData {
    std::size_t in_dim = 0;
    std::size_t out_dim = 0;
    std::vector<double> W;
    std::vector<double> b;
};

LayerData parse_layer_object(const std::string& json,
                             std::size_t& pos,
                             std::size_t expected_in,
                             std::size_t expected_out) {
    skip_ws(json, pos);
    const std::size_t obj_start = pos;
    if (!match(json, pos, '{')) {
        throw std::runtime_error("NN JSON: expected kernel object");
    }
    const std::size_t obj_end = json.find('}', pos);
    if (obj_end == std::string::npos) {
        throw std::runtime_error("NN JSON: expected '}' after kernel");
    }
    auto value_position = [&](const char* key) {
        std::size_t key_pos =
            find_key_after(json, obj_start, key, obj_end);
        key_pos = json.find(':', key_pos);
        if (key_pos == std::string::npos || key_pos >= obj_end) {
            throw std::runtime_error(
                std::string("NN JSON: missing ':' after ") + key);
        }
        return key_pos + 1;
    };

    LayerData L;
    std::size_t kp = value_position("in");
    L.in_dim = parse_dimension(json, kp, "layer input dimension");
    kp = value_position("out");
    L.out_dim = parse_dimension(json, kp, "layer output dimension");
    if (L.in_dim != expected_in) {
        throw std::runtime_error("NN JSON: layer input dimension mismatch");
    }
    if (L.out_dim != expected_out) {
        throw std::runtime_error("NN JSON: layer output dimension mismatch");
    }

    if (L.out_dim > std::numeric_limits<std::size_t>::max() / L.in_dim
        || L.out_dim * L.in_dim > std::vector<double>().max_size()
        || L.out_dim * L.in_dim > kMaxModelParameters) {
        throw std::runtime_error("NN JSON: layer matrix dimensions are too large");
    }

    kp = value_position("W");
    kp = json.find('[', kp);
    if (kp == std::string::npos || kp >= obj_end) {
        throw std::runtime_error("NN JSON: malformed W matrix");
    }
    const auto Wmat = parse_matrix(json, kp, L.out_dim, L.in_dim);
    if (kp > obj_end) {
        throw std::runtime_error("NN JSON: W matrix extends beyond layer object");
    }
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

    kp = value_position("b");
    kp = json.find('[', kp);
    if (kp == std::string::npos || kp >= obj_end) {
        throw std::runtime_error("NN JSON: malformed bias array");
    }
    L.b = parse_array(json, kp, L.out_dim);
    if (kp > obj_end) {
        throw std::runtime_error("NN JSON: bias array extends beyond layer object");
    }
    if (L.b.size() != L.out_dim) {
        throw std::runtime_error("NN JSON: bias length mismatch");
    }

    pos = obj_end + 1;
    return L;
}

void parse_mlp_block(const std::string& json,
                     std::size_t input_dim,
                     std::vector<LayerData>& layers,
                     double& out_bias) {
    layers.clear();
    std::size_t pos = find_key(json, "hidden");
    pos = json.find('[', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("NN JSON: hidden sizes not found");
    }
    const auto hidden_values =
        parse_array(json, pos, kMaxHiddenLayers);
    if (hidden_values.size() > kMaxHiddenLayers) {
        throw std::runtime_error("NN JSON: too many hidden layers");
    }
    std::vector<std::size_t> hidden;
    hidden.reserve(hidden_values.size());
    for (double value : hidden_values) {
        if (value < 1.0 || std::floor(value) != value
            || value >= static_cast<double>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error(
                "NN JSON: hidden dimensions must be positive integers");
        }
        hidden.push_back(static_cast<std::size_t>(value));
    }

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
        const std::size_t expected_out =
            (ell < hidden.size()) ? hidden[ell] : 1u;
        LayerData L =
            parse_layer_object(json, pos, expected_in, expected_out);
        expected_in = L.out_dim;
        layers.push_back(std::move(L));
        skip_ws(json, pos);
        if (ell + 1 < hidden.size() + 1) {
            if (!match(json, pos, ',')) {
                throw std::runtime_error(
                    "NN JSON: expected ',' between kernel layers");
            }
        } else if (!match(json, pos, ']')) {
            throw std::runtime_error(
                "NN JSON: kernel count does not match hidden dimensions");
        }
    }

    pos = find_key(json, "out_bias");
    pos = json.find(':', pos);
    if (pos == std::string::npos) {
        throw std::runtime_error("NN JSON: malformed out_bias");
    }
    out_bias = parse_number(json, ++pos);
}

std::string parse_optional_string(const std::string& json,
                                  const std::string& key,
                                  const std::string& default_value = {}) {
    const std::string quoted_key = "\"" + key + "\"";
    const std::size_t key_pos = json.find(quoted_key);
    if (key_pos == std::string::npos) return default_value;
    std::size_t pos = key_pos + quoted_key.size();
    skip_ws(json, pos);
    if (pos >= json.size() || json[pos] != ':') {
        throw std::runtime_error("NN JSON: malformed string field " + key);
    }
    ++pos;
    skip_ws(json, pos);
    if (pos >= json.size() || json[pos] != '"') {
        throw std::runtime_error("NN JSON: field " + key + " must be a string");
    }
    const std::size_t value_start = ++pos;
    bool escaped = false;
    for (; pos < json.size(); ++pos) {
        if (!escaped && json[pos] == '"') {
            return json.substr(value_start, pos - value_start);
        }
        if (!escaped && json[pos] == '\\') {
            escaped = true;
        } else {
            escaped = false;
        }
    }
    throw std::runtime_error("NN JSON: unterminated string field " + key);
}

std::string parse_optional_name(const std::string& json) {
    return parse_optional_string(json, "name");
}

std::string parse_kernel_type(const std::string& json,
                              const std::string& default_value) {
    return parse_optional_string(json, "kernel_type", default_value);
}

}  // namespace

NNFunctional::NNFunctional(std::string model_path)
    : model_path_(std::move(model_path)) {
    const std::string json = read_file(model_path_);
    const std::string kernel_type = parse_kernel_type(json, "separable");
    if (kernel_type != "separable" && !kernel_type.empty()) {
        throw std::runtime_error(
            "NN JSON: NNFunctional requires kernel_type \"separable\"");
    }
    name_ = parse_optional_name(json);
    if (name_.empty()) name_ = "NN";
    std::vector<LayerData> raw;
    parse_mlp_block(json, /*input_dim=*/1, raw, out_bias_);
    layers_.reserve(raw.size());
    for (auto& source : raw) {
        Layer layer;
        layer.in_dim = source.in_dim;
        layer.out_dim = source.out_dim;
        layer.W = std::move(source.W);
        layer.b = std::move(source.b);
        layers_.push_back(std::move(layer));
    }
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
    if (n < 0.0) return 0.0;

    std::vector<double> values(1, n);
    std::vector<double> derivatives(1, 1.0);
    std::vector<double> next_values;
    std::vector<double> next_derivatives;
    for (std::size_t ell = 0; ell < layers_.size(); ++ell) {
        const Layer& layer = layers_[ell];
        next_values.assign(layer.out_dim, 0.0);
        next_derivatives.assign(layer.out_dim, 0.0);
        for (std::size_t out = 0; out < layer.out_dim; ++out) {
            double z = layer.b[out];
            double dz = 0.0;
            const double* weights = &layer.W[out * layer.in_dim];
            for (std::size_t in = 0; in < layer.in_dim; ++in) {
                z += weights[in] * values[in];
                dz += weights[in] * derivatives[in];
            }
            if (ell + 1 < layers_.size()) {
                const double activation = std::tanh(z);
                next_values[out] = activation;
                next_derivatives[out] =
                    (1.0 - activation * activation) * dz;
            } else {
                next_values[out] = z;
                next_derivatives[out] = dz;
            }
        }
        values.swap(next_values);
        derivatives.swap(next_derivatives);
    }
    const double raw = out_bias_ + values[0];
    const double draw = derivatives[0];
    return softplus(raw) + n * sigmoid(raw) * draw;
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
    if (parse_kernel_type(json, "separable") != "pair") {
        throw std::runtime_error(
            "NN JSON: NNPairFunctional requires kernel_type \"pair\"");
    }
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

double NNPairFunctional::raw_at(double ni, double nj) const {
    // Symmetric features: (s, p) = (n_i + n_j, n_i n_j).
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
                // Linear output: max flexibility (any sign, any magnitude
                // modulo the sqrt(p) prefactor in kernel()).  The
                // optimizer is free to drive raw to whatever value makes
                // the L1-RMSE-against-GZ loss small.
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
    return std::sqrt(p) * raw_at(ni, nj);
}

double NNPairFunctional::kernel_grad(double ni, double nj) const {
    // dK/dn_i with n_j fixed.  Centered finite difference; the cost is
    // amortised over the SCF projected-gradient step which already does
    // N kernel evaluations per row.
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

// Dispatcher: read the optional "kernel_type" string at the JSON root and
// return the right concrete class.  Defaults to the legacy separable form
// so existing model.json files keep loading unchanged.
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
