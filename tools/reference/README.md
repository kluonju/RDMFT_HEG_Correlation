# Third-party reference implementations

Read-only copies of upstream reference implementations that the C++ code in
this repository is intentionally faithful to.  We keep them in-tree so the
mapping between paper, reference code, and our port is unambiguous (and so
later contributors do not have to re-discover which form of an equation is
"the real one" when the literature contains typos).

Nothing here is built or linked by the rdmft_heg targets.

## `nk_GZ.f`

Original FORTRAN 77 implementation of the Gori-Giorgi / Ziesche parametrized
momentum distribution `n(k, rs)` of the unpolarized 3D uniform electron gas,
covering `rs <= 12`.  Cited as:

* P. Gori-Giorgi and P. Ziesche,
  *Momentum distribution of the uniform electron gas: improved parametrization
  and exact limits of the cumulant expansion*,
  **Phys. Rev. B 66, 235116 (2002)**
  (preprint: arXiv:cond-mat/0205342v3).

### Provenance

Public mirror used as the upstream:

* URL:  <https://csclub.uwaterloo.ca/~pbarfuss/nk_GZ.f>
* SHA-256: `accc2863604daa670c1eaab344db87b620b9db8c50d6c47f59b0bdfbc32e7882`
* Retrieved: 2026-05-28.

The file is the reference implementation released by Gori-Giorgi
herself (re-hosted on the above mirror) and is the authoritative
form for the parametrization.  We treat its formulas as ground truth
whenever they disagree with the published paper.

### Mapping FORTRAN <-> C++ port

The C++ port lives in
`src/MomentumDistributionGZ.cpp` (declarations in
`include/MomentumDistributionGZ.hpp`).  Routine-by-routine:

| FORTRAN routine       | C++ counterpart                                | What it computes                                    |
|-----------------------|-----------------------------------------------|-----------------------------------------------------|
| `subroutine nofk`     | `rdmft::gz::n_of_k_over_kf` (and `n_of_k`)    | `n(k/kF, rs)` for `k <= kF` and `k > kF` branches   |
| `subroutine kulik`    | `rdmft::gz::kulik_G`                          | Kulik function `G(x)` (Eq. 47 of the paper)         |
| `double function xkul`| anonymous `xkul_integrand` in the .cpp file   | Integrand of `G(x)`                                 |
| `subroutine xint`     | `integrate_xint6` template in the .cpp file   | 6-point Adams-style indefinite integral on log grid |
| `function g0GP`       | `rdmft::gz::g0_on_top`                        | On-top pair density `g(0, rs)` (Gori-Giorgi/Perdew 2001) |
| `function bpar`       | `rdmft::gz::b_coeff`                          | `b(rs)` interpolating `k=0` curvature (Eq. 16)      |
| `function apar`       | `rdmft::gz::a_coeff`                          | `a(rs)`, Fermi-edge log-singularity strength (Eq. 19) |
| `function an0`        | `rdmft::gz::n0`                               | `n(k=0, rs)` (Eq. 15)                               |
| `subroutine parn1`    | `rdmft::gz::n1_minus`, `rdmft::gz::n1_plus`   | `n(k=1±, rs)` (Eqs. 17, 18)                         |

The numerical integration constants are matched verbatim
(`rho_min = -6.3`, `h = 0.0218`, `ndm = 900`, the same 6-point Adams-Moulton
weights `11/1440, -93/1440, 802/1440`, and the small-x cutoff
`x <= 1.4e-3`).

### Known discrepancies between the published paper and this FORTRAN

These were discovered while porting and are documented here so the next reader
does not chase them as bugs:

1. **`a(rs)` exponent, Eq. (19).**  The paper prints the last term of the
   denominator as `p6 * rs^6`; the FORTRAN uses `gg2 * |gg6| * x^4`, i.e.
   `rs^4`.  Numerically `gg2 * |gg6| = -0.0989941 * 0.114831 = -0.01136759`,
   matching the paper's `p6` value, but the exponent is 4, not 6.  With
   `rs^6` the particle-number sum rule (Eq. 3) is violated by ~11 % at
   `rs = 5` and ~35 % at `rs = 10`, and `n(k, rs)` no longer matches paper
   Fig. 6.  We follow the FORTRAN.  See the comment in
   `rdmft::gz::a_coeff` and the regression test
   `tests/test_gz_momentum.cpp`.

2. **`g(0, rs)`, `c` coefficient.**  The FORTRAN uses `c = 0.08183d0`;
   the source paper (Gori-Giorgi & Perdew, PRB 64, 155102 (2001)) prints
   `C = 0.08193`.  We follow the FORTRAN so the on-top pair density agrees
   bit-for-bit with the upstream implementation.  The numerical effect over
   `rs <= 12` is below 0.1 % and does not affect the sum rule.

### Licence / copyright

`nk_GZ.f` is included verbatim from the upstream mirror; we do not modify it.
Original copyright belongs to the authors (P. Gori-Giorgi / P. Ziesche).  It
is redistributed here under fair-use for archival purposes; remove the file
if the authors object.
