#pragma once

#include <cmath>

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

}  // namespace rdmft
