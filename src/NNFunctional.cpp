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

std::unique_ptr<Functional> load_nn_functional(const std::string& json_path) {
    return make_nn_functional(json_path);
}

}  // namespace rdmft
