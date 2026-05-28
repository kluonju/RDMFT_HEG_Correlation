#include "MomentumDistributionGZ.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "HEG.hpp"

namespace rdmft::gz {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kG0 = 3.353337;      // Kulik function at the origin.
constexpr double kFpp0 = 17.968746;   // F''(0) from Gori-Giorgi/Ziesche.
constexpr double kKulikSmallX = 1.4e-3;

double r_of_u(double u) {
    return 1.0 - u * std::atan(1.0 / u);
}

double dr_du(double u) {
    return -std::atan(1.0 / u) + u / (u * u + 1.0);
}

double xkul_integrand(double u, double x) {
    const double ru = r_of_u(u);
    const double rud = dr_du(u);
    const double y = x / std::sqrt(ru);
    const double at_u = std::atan(1.0 / u);

    double at_y = 0.0;
    if (std::abs(y) < 1.0e-15) {
        at_y = 0.5 * kPi;
    } else {
        at_y = std::atan(1.0 / y);
    }

    const double den = ru - (x * x) / (u * u);
    return -rud / den * (at_u - (y / u) * at_y);
}

template <std::size_t N>
std::array<double, N> integrate_xint6(const std::array<double, N>& x,
                                      const std::array<double, N>& wr,
                                      double h,
                                      double y0) {
    static_assert(N >= 6, "xint6 requires at least 6 points");
    std::array<double, N> y{};

    double ca = wr[0] * x[0];
    double cb = wr[1] * x[1];
    double cc = wr[2] * x[2];
    double cd = wr[3] * x[3];
    double ce = wr[4] * x[4];
    double cf = wr[5] * x[5];

    const double wa = h * 11.0 / 1440.0;
    const double wb = -h * 93.0 / 1440.0;
    const double wc = h * 802.0 / 1440.0;

    y[0] = y0;
    y[1] = y[0] + h * (475.0 * ca + 1427.0 * cb - 798.0 * cc + 482.0 * cd
                     - 173.0 * ce + 27.0 * cf) / 1440.0;
    y[2] = y[1] + h * (-27.0 * ca + 637.0 * cb + 1022.0 * cc - 258.0 * cd
                     + 77.0 * ce - 11.0 * cf) / 1440.0;

    for (std::size_t i = 3; i + 2 < N; ++i) {
        cf = wr[i + 2] * x[i + 2];
        y[i] = y[i - 1] + wa * (ca + cf) + wb * (cb + ce) + wc * (cc + cd);
        ca = cb;
        cb = cc;
        cc = cd;
        cd = ce;
        ce = cf;
    }

    y[N - 2] = y[N - 3]
             + h * (-11.0 * wr[N - 6] * x[N - 6] + 77.0 * ca - 258.0 * cb
                    + 1022.0 * cc + 637.0 * cd - 27.0 * ce) / 1440.0;
    y[N - 1] = y[N - 2]
             + h * (27.0 * wr[N - 6] * x[N - 6] - 173.0 * ca + 482.0 * cb
                    - 798.0 * cc + 1427.0 * cd + 475.0 * ce) / 1440.0;
    return y;
}

}  // namespace

double g0_on_top(double rs) {
    // Gori-Giorgi and Perdew, PRB 64, 155102 (2001), as used by GZ.
    const double c = 0.08183;
    const double d = -0.01277;
    const double e = 0.001859;
    const double dex = 0.7524;
    const double b = 0.7317 - dex;
    return 0.5 * (1.0 - b * rs + c * rs * rs + d * rs * rs * rs
                + e * std::pow(rs, 4.0)) * std::exp(-dex * rs);
}

double b_coeff(double rs) {
    // Eq. (15) in PRB 66, 235116 (2002).
    return std::sqrt(1.0 + 0.0009376925 * std::pow(rs, 13.0 / 4.0));
}

double a_coeff(double rs) {
    // Eq. (17) in PRB 66, 235116 (2002).
    const double p1 = -78.8682;
    const double p2 = -0.0989941;
    const double p3 = -68.5997;
    const double p4 = 38.1159;
    const double p5 = -17.6829;
    const double p6 = -0.01136759;
    const double q = std::pow(rs, 0.25);
    const double s = std::sqrt(rs);
    return (1.0 + p1 * q + p2 * s)
         / (1.0 + p3 * q + p4 * s + p5 * rs + p6 * std::pow(rs, 6.0));
}

double n0(double rs) {
    // Eq. (13) in PRB 66, 235116 (2002).
    const double t1 = 0.003438169;
    const double t2 = 0.00725313666;
    const double t3 = 0.014900367;
    const double t4 = 0.00113244364;
    const double rs2 = rs * rs;
    return (1.0 + t1 * rs2 + t2 * std::pow(rs, 2.5))
         / (1.0 + t3 * rs2 + t4 * std::pow(rs, 13.0 / 4.0));
}

double n1_minus(double rs) {
    // Eq. (16) in PRB 66, 235116 (2002).
    const double v1 = -0.0679793;
    const double v2 = -0.00102846;
    const double v3 = 0.000189111;
    const double v4 = 0.0205397;
    const double v5 = -0.0086838;
    const double v6 = 6.87109e-5;
    const double v7 = 4.868047e-5;
    const double rs2 = rs * rs;
    const double rs3 = rs2 * rs;
    return (1.0 + v1 * rs + v2 * rs2 + v3 * rs3)
         / (1.0 + v4 * rs + v5 * rs2 + v6 * rs3 + v7 * std::pow(rs, 15.0 / 4.0));
}

double n1_plus(double rs) {
    // Eq. (16) in PRB 66, 235116 (2002).
    const double q1 = 0.088519;
    const double q2 = 0.45;
    const double q3 = 0.022786335;
    return (q1 * rs) / (1.0 + q2 * std::sqrt(rs) + q3 * std::pow(rs, 7.0 / 4.0));
}

double kulik_G(double x) {
    if (x <= 0.0) return kG0;

    // Small-x expansion from Eq. (A9): G(x) = G(0) + c x ln x + O(x),
    // where c = pi (pi/4 + sqrt(3)).
    if (x <= kKulikSmallX) {
        const double c = kPi * (kPi / 4.0 + std::sqrt(3.0));
        return kG0 + c * x * std::log(x);
    }

    constexpr std::size_t n = 900;
    constexpr double rho_min = -6.3;
    constexpr double h = 0.0218;

    std::array<double, n> integrand{};
    std::array<double, n> weights{};

    double u = std::exp(rho_min);
    for (std::size_t i = 0; i < n; ++i) {
        const double rho = rho_min + static_cast<double>(i) * h;
        // Keep the same Newton reconstruction used in the original FORTRAN
        // routine (it is exact here because alpha=0).
        for (int it = 0; it < 10; ++it) {
            const double rho_u = std::log(u);
            u += (rho - rho_u) * u;
        }
        integrand[i] = xkul_integrand(u, x);
        weights[i] = u;
    }

    const auto y = integrate_xint6(integrand, weights, h, 0.0);
    return y.back();
}

double n_of_k_over_kf(double k_ratio, double rs) {
    if (rs <= 0.0) {
        throw std::invalid_argument("n_of_k_over_kf requires rs > 0");
    }

    const double k = std::abs(k_ratio);
    const double alph = std::cbrt(4.0 / (9.0 * kPi));
    const double n0v = n0(rs);
    const double n1m = n1_minus(rs);
    const double n1p = n1_plus(rs);
    const double av = a_coeff(rs);
    const double bv = b_coeff(rs);
    const double g0 = g0_on_top(rs);
    const double fac = std::sqrt(4.0 * alph * rs / kPi);
    const double one_minus_ln2 = 1.0 - std::log(2.0);

    if (k <= 1.0e-14) return n0v;

    if (k <= 1.0) {
        const double x1 = av * (alph * rs / (2.0 * kPi * kPi)) * (kG0 / (n0v - n1m))
                        * ((1.0 - k) / fac);
        const double x2 = bv * ((kPi * kPi) / (alph * rs))
                        * std::sqrt((kPi / 3.0) * (one_minus_ln2 / kFpp0)
                                  * ((n0v - n1m) / kG0))
                        * ((1.0 - k) * (1.0 - k) / k);
        const double gx = kulik_G(x1 + x2);
        return n0v - ((n0v - n1m) / kG0) * gx;
    }

    const double x1 = av * (alph * rs / (2.0 * kPi * kPi)) * (kG0 / n1p)
                    * ((k - 1.0) / fac);
    const double x2 = std::sqrt((3.0 * kPi * one_minus_ln2 / g0) * (n1p / kG0))
                    * (kPi / (4.0 * alph * rs))
                    * std::pow(k - 1.0, 4.0);
    const double gx = kulik_G(x1 + x2);
    return (n1p / kG0) * gx;
}

double n_of_k(double k_abs, double rs) {
    const double kf = HEG::kF(rs);
    return n_of_k_over_kf(k_abs / kf, rs);
}

}  // namespace rdmft::gz

