#include <cmath>
#include <cstdio>

#include "MomentumDistributionGZ.hpp"

using namespace rdmft::gz;

int main() {
    const double rs = 5.0;

    const double n0_ref = n0(rs);
    const double n0_num = n_of_k_over_kf(0.0, rs);
    if (std::abs(n0_num - n0_ref) > 1.0e-10) {
        std::fprintf(stderr, "FAIL: n(0) mismatch: %.12f vs %.12f\n", n0_num, n0_ref);
        return 1;
    }

    const double eps = 1.0e-6;
    const double n1m_ref = n1_minus(rs);
    const double n1p_ref = n1_plus(rs);
    const double n_left = n_of_k_over_kf(1.0 - eps, rs);
    const double n_right = n_of_k_over_kf(1.0 + eps, rs);
    if (std::abs(n_left - n1m_ref) > 5.0e-4) {
        std::fprintf(stderr, "FAIL: n(1-) mismatch: %.12f vs %.12f\n", n_left, n1m_ref);
        return 1;
    }
    if (std::abs(n_right - n1p_ref) > 5.0e-4) {
        std::fprintf(stderr, "FAIL: n(1+) mismatch: %.12f vs %.12f\n", n_right, n1p_ref);
        return 1;
    }

    const double n2 = n_of_k_over_kf(2.0, rs);
    const double n5 = n_of_k_over_kf(5.0, rs);
    if (!(n5 < n2 && n2 < n_right)) {
        std::fprintf(stderr, "FAIL: high-k tail monotonicity check failed.\n");
        return 1;
    }

    for (double k : {0.0, 0.5, 0.9, 1.1, 2.0, 5.0}) {
        const double nk = n_of_k_over_kf(k, rs);
        if (nk < -1.0e-8 || nk > 1.0 + 1.0e-8) {
            std::fprintf(stderr, "FAIL: n(k)=%.12f out of [0,1] at k=%.3f\n", nk, k);
            return 1;
        }
    }

    std::printf("OK\n");
    return 0;
}

