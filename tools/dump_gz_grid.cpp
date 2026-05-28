// Dense ``rs k/kF n(k/kF, rs)`` dump of the Gori-Giorgi/Ziesche
// momentum distribution.  Used by ``scripts/plot_gz.py`` to reproduce
// the upper panel of Fig. 6 in PRB 66, 235116 (2002).
//
// Usage:
//   dump_gz_grid [--rs 1,2,5,10] [--n 401] [--kmax 3.0]
//
// Output columns (whitespace-separated, header lines start with '#'):
//   rs   k/kF   n(k/kF, rs)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "MomentumDistributionGZ.hpp"

using namespace rdmft::gz;

static std::vector<double> parse_rs(const std::string& csv) {
    std::vector<double> out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        out.push_back(std::stod(item));
    }
    return out;
}

int main(int argc, char** argv) {
    std::vector<double> rs_list{1.0, 2.0, 5.0, 10.0};
    int n = 401;
    double kmax = 3.0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* k) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", k);
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (a == "--rs") rs_list = parse_rs(need("--rs"));
        else if (a == "--n") n = std::atoi(need("--n").c_str());
        else if (a == "--kmax") kmax = std::atof(need("--kmax").c_str());
        else {
            std::fprintf(stderr, "Unknown option: %s\n", a.c_str());
            return 2;
        }
    }
    if (n < 2) n = 2;

    std::printf("# rs    k/kF    n(k/kF, rs)\n");
    for (double rs : rs_list) {
        for (int i = 0; i < n; ++i) {
            double x = kmax * static_cast<double>(i) / static_cast<double>(n - 1);
            std::printf("%g  %.8f  %.10f\n", rs, x, n_of_k_over_kf(x, rs));
        }
    }
    return 0;
}
