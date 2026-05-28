#pragma once

#include <string>

namespace rdmft::gz {

// Gori-Giorgi/Ziesche parametrization of the momentum distribution n(k, rs)
// for the unpolarized 3D electron gas (Phys. Rev. B 66, 235116 (2002),
// arXiv:cond-mat/0205342v3).
//
// Conventions:
//   * k_ratio = k / k_F (dimensionless momentum).
//   * rs is the Wigner-Seitz radius in Bohr.
//   * Validity range in the original paper: rs \lesssim 12.
//
// The implementation follows Eqs. (9)-(12), (14)-(19) of the paper and uses
// the Kulik function G(x) from Appendix A.
//
// NOTE: The published Eq. (19) (parametrization of a(rs)) contains a typo
// (last denominator term reads p6 * rs^6).  The original FORTRAN reference
// implementation by Gori-Giorgi (``nk_GZ.f``) uses p6 * rs^4, which is the
// form required to satisfy the particle-number sum rule (Eq. 3) and to
// reproduce paper Fig. 6 over the whole validity range rs <= 12.  We follow
// the FORTRAN here; see ``a_coeff`` in ``src/MomentumDistributionGZ.cpp``
// for details.
double n_of_k_over_kf(double k_ratio, double rs);

// Convenience overload for dimensional k (a.u.^-1).
double n_of_k(double k_abs, double rs);

// Fitted rs-dependent ingredients entering Eqs. (9)-(12).
double n0(double rs);         // n(k=0, rs)
double n1_minus(double rs);   // n(k=1-, rs)
double n1_plus(double rs);    // n(k=1+, rs)
double a_coeff(double rs);    // Fermi-edge logarithmic coefficient prefactor
double b_coeff(double rs);    // Curvature-control parameter near k=0
double g0_on_top(double rs);  // on-top pair density g(0, rs)

// Kulik function used in the ansatz.
double kulik_G(double x);

}  // namespace rdmft::gz

