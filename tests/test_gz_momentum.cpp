#include <cmath>
#include <cstdio>
#include <initializer_list>

#include "MomentumDistributionGZ.hpp"

using namespace rdmft::gz;

// Composite-trapezoid integral of 3 x^2 n(x, rs) on [0, xmax].
static double sum_rule_norm(double rs, double xmax, int n) {
    double s = 0.0;
    const double h = xmax / static_cast<double>(n);
    for (int i = 0; i <= n; ++i) {
        const double x = h * i;
        const double w = (i == 0 || i == n) ? 0.5 : 1.0;
        s += w * 3.0 * x * x * n_of_k_over_kf(x, rs);
    }
    return s * h;
}

int main() {
    // ---- end-point matching (n(0) = n0, n(1-+) = n_{-,+}) ----
    {
        const double rs = 5.0;

        const double n0_ref = n0(rs);
        const double n0_num = n_of_k_over_kf(0.0, rs);
        if (std::abs(n0_num - n0_ref) > 1.0e-10) {
            std::fprintf(stderr, "FAIL: n(0) mismatch: %.12f vs %.12f\n", n0_num, n0_ref);
            return 1;
        }

        const double eps = 1.0e-6;
        const double n_left = n_of_k_over_kf(1.0 - eps, rs);
        const double n_right = n_of_k_over_kf(1.0 + eps, rs);
        if (std::abs(n_left - n1_minus(rs)) > 5.0e-4) {
            std::fprintf(stderr, "FAIL: n(1-) mismatch: %.12f vs %.12f\n",
                         n_left, n1_minus(rs));
            return 1;
        }
        if (std::abs(n_right - n1_plus(rs)) > 5.0e-4) {
            std::fprintf(stderr, "FAIL: n(1+) mismatch: %.12f vs %.12f\n",
                         n_right, n1_plus(rs));
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
    }

    // ---- particle-number sum rule (Eq. 3 of PRB 66, 235116):
    //      I_2 = \int_0^\infty 3 x^2 n(x, rs) dx = 1.
    // Regression guard for the historical typo `rs^6` in a_coeff: with
    // that bug the sum rule is violated by ~11% at rs=5 and ~35% at rs=10.
    {
        struct Case { double rs; double tol; };
        for (Case c : {Case{0.5, 5e-3}, {1.0, 5e-3}, {2.0, 5e-3}, {3.0, 5e-3},
                       {5.0, 7e-3}, {7.0, 7e-3}, {10.0, 8e-3}, {12.0, 1.0e-2}}) {
            const double I2 = sum_rule_norm(c.rs, 12.0, 4000);
            if (std::abs(I2 - 1.0) > c.tol) {
                std::fprintf(stderr,
                             "FAIL: sum rule I_2(rs=%g) = %.6f, expected 1 +/- %.0e\n",
                             c.rs, I2, c.tol);
                return 1;
            }
        }
    }

    // ---- a(rs) sanity (regression guard for the rs^6/rs^4 typo).
    // From the original FORTRAN nk_GZ.f (and consistent with the GZ
    // Fig. 5 of A(rs)), a(rs) decreases gently from ~1.65 at rs=1 to
    // ~0.48 at rs=10; the published Eq. (19) crushes it to 0.012 at rs=10.
    {
        struct AT { double rs; double a_min; double a_max; };
        for (AT t : {AT{1.0, 1.5, 1.8}, {5.0, 0.95, 1.15}, {10.0, 0.40, 0.55}}) {
            const double av = a_coeff(t.rs);
            if (av < t.a_min || av > t.a_max) {
                std::fprintf(stderr,
                             "FAIL: a(rs=%g) = %.4f outside [%.3f, %.3f]\n",
                             t.rs, av, t.a_min, t.a_max);
                return 1;
            }
        }
    }

    std::printf("OK\n");
    return 0;
}

