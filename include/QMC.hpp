#pragma once

#include <cmath>
#include <limits>

namespace rdmft {

// Perdew-Wang 1992 (PW92) parameterization of the correlation energy per
// electron of the paramagnetic 3D HEG.
//
// Reference: J.P. Perdew & Y. Wang, Phys. Rev. B 45, 13244 (1992).
// In Hartree atomic units; valid over essentially all rs of physical interest.
struct PW92 {
    // Paramagnetic parameters from PW92 Table I.
    static constexpr double A     = 0.0310907;
    static constexpr double alpha1 = 0.21370;
    static constexpr double beta1 = 7.5957;
    static constexpr double beta2 = 3.5876;
    static constexpr double beta3 = 1.6382;
    static constexpr double beta4 = 0.49294;

    // ec(rs) per electron.
    static double ec_per_electron(double rs) {
        const double sqrt_rs = std::sqrt(rs);
        const double denom = 2.0 * A *
            (beta1 * sqrt_rs + beta2 * rs +
             beta3 * rs * sqrt_rs + beta4 * rs * rs);
        const double G = -2.0 * A * (1.0 + alpha1 * rs) *
                          std::log(1.0 + 1.0 / denom);
        return G;
    }
};

// Ortiz-Ballone (1997) DMC parameterization of the paramagnetic 3D HEG
// momentum distribution n(q).
//
// Reference: G. Ortiz & P. Ballone, Phys. Rev. B 56, 9970 (1997)
//   [Erratum to Phys. Rev. B 50, 1391 (1994)].
//
// With x = q/k_F, the per-spin-orbital occupation rho_tilde(q) in [0, 1]
// reads (Eq. 57 of the erratum)
//
//     rho_tilde(q) = A + B x^2 + C x^3 + D x^4,           q <  k_F
//                  = E (k_F/q)^8 + F (k_F/q)^10,          q >  k_F.
//
// The Fermi-surface discontinuity is
//
//     Z_F = rho_tilde(k_F^-) - rho_tilde(k_F^+)
//         = (A + B + C + D) - (E + F)
//
// and is reported separately in Table I.  Coefficients are tabulated only
// at r_s in {0.8, 1, 2, 3, 5, 8, 10}; values outside this set are not
// interpolated here -- callers are expected to use ``find`` first or pass
// an exact tabulated r_s.
struct OrtizBallone1997 {
    struct Coeffs {
        double rs;
        double A, B, C, D, E, F;
        double ZF_DMC;   // DMC Fermi-surface discontinuity (Table I)
        double ZF_RPA;   // RPA value where given; NaN otherwise
    };

    static constexpr int    N_RS   = 7;
    static constexpr double rs_tol = 1.0e-6;

    static const Coeffs* table() {
        static const Coeffs T[N_RS] = {
            { 0.8,
               0.99980391,  0.03458472, -0.13518065,  0.08581977,
              -0.00318200,  0.02838474,
               0.960,       std::numeric_limits<double>::quiet_NaN() },
            { 1.0,
               0.99981053,  0.00292176, -0.08682315,  0.05605365,
               0.05297960, -0.03257978,
               0.952,       0.859 },
            { 2.0,
               0.99316251, -0.02996622, -0.09518898,  0.06002633,
               0.17476271, -0.13573347,
               0.889,       std::numeric_limits<double>::quiet_NaN() },
            { 3.0,
               0.98279143, -0.06853916, -0.05238832,  0.02586864,
               0.31353239, -0.26758613,
               0.8420,      0.700 },
            { 5.0,
               0.96589033,  0.04297206, -0.50414293,  0.31844905,
               0.47767371, -0.37959715,
               0.725,       0.602 },
            { 8.0,
               0.90116912, -0.01209355, -0.37233983,  0.22718771,
               0.90382824, -0.81061442,
               0.651,       std::numeric_limits<double>::quiet_NaN() },
            {10.0,
               0.90134023, -0.08057040, -0.31006521,  0.18479946,
               1.04270800, -0.93984323,
               0.593,       std::numeric_limits<double>::quiet_NaN() },
        };
        return T;
    }

    // Locate coefficients for an r_s exactly in the table (within rs_tol).
    // Returns nullptr if r_s is not tabulated.
    static const Coeffs* find(double rs) {
        const Coeffs* T = table();
        for (int i = 0; i < N_RS; ++i) {
            if (std::fabs(T[i].rs - rs) <= rs_tol) return &T[i];
        }
        return nullptr;
    }

    // Per-spin-orbital n(q) evaluated at x = q/k_F.  Returns NaN if r_s
    // is not in the table.  At x = 1 the lower (q < k_F) branch is used,
    // so the returned value is the "below-Fermi" limit n(k_F^-).
    static double n_of_x(double rs, double x) {
        const Coeffs* c = find(rs);
        if (!c) return std::numeric_limits<double>::quiet_NaN();
        if (x <= 1.0) {
            const double x2 = x * x;
            const double x3 = x2 * x;
            const double x4 = x2 * x2;
            return c->A + c->B * x2 + c->C * x3 + c->D * x4;
        }
        const double y  = 1.0 / x;             // y = k_F / q
        const double y2 = y * y;
        const double y4 = y2 * y2;
        const double y8 = y4 * y4;
        const double y10 = y8 * y2;
        return c->E * y8 + c->F * y10;
    }
};

}  // namespace rdmft
