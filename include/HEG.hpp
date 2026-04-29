#pragma once

#include <cmath>

namespace rdmft {

// Physical / geometrical helpers for the spin-unpolarized homogeneous
// electron gas (HEG) in Hartree atomic units.
//
// We work in 3D, paramagnetic case (n_up = n_down = n / 2) so that the
// momentum distribution n(k) is the same for both spin channels and is
// bounded by 1 (occupations per spin orbital).
struct HEG {
    // Number density rho corresponding to a given Wigner-Seitz radius rs.
    //   rho = 3 / (4 pi rs^3)
    static double density(double rs) {
        constexpr double pi = M_PI;
        return 3.0 / (4.0 * pi * rs * rs * rs);
    }

    // Free-electron Fermi wave-vector for a paramagnetic HEG.
    //   kF = (9 pi / 4)^{1/3} / rs
    static double kF(double rs) {
        constexpr double pi = M_PI;
        return std::cbrt(9.0 * pi / 4.0) / rs;
    }

    // Hartree-Fock kinetic + exchange energy per electron (paramagnetic).
    //   t_s  =  3/10 * kF^2          (Hartree)
    //   e_x  = -3/(4 pi) * kF        (Hartree)
    static double kinetic_per_electron_HF(double rs) {
        double kf = kF(rs);
        return 0.3 * kf * kf;
    }
    static double exchange_per_electron_HF(double rs) {
        constexpr double pi = M_PI;
        return -0.75 * kF(rs) / pi;
    }
    static double HF_per_electron(double rs) {
        return kinetic_per_electron_HF(rs) + exchange_per_electron_HF(rs);
    }
};

}  // namespace rdmft
