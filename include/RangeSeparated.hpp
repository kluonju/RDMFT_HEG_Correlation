#pragma once

#include <cmath>

#include "HEG.hpp"
#include "QMC.hpp"

namespace rdmft {

// Short-range LDA exchange and correlation for the erf/erfc Coulomb split
//
//     1/r = erfc(μ r)/r + erf(μ r)/r
//
// in the paramagnetic 3-D HEG (ζ = 0; PBE ≡ LDA when ∇ρ = 0).
//
// Exchange: analytic Toulouse–Savin short-range complement
//   [Int. J. Quantum Chem. 100, 1047 (2004); cond-mat/0611559, Eq. (29)].
//   Matches Molpro EXERF (Savin 1996) for ζ = 0.
//
// Correlation: Molpro ECERF / Paziani et al., Phys. Rev. B 73, 155111 (2006)
//   — short-range complement ε_c^SR(μ) = ε_c^PW92 − (long-range piece).
//   Limits: μ → 0 → PW92; μ → ∞ → 0.
struct RangeSeparatedLDA {
    // Short-range exchange per electron (Hartree).
    // A = μ / (2 k_F); recovers Slater exchange as μ → 0 and 0 as μ → ∞.
    static double ex_sr_per_electron(double rs, double mu) {
        if (!(rs > 0.0) || !std::isfinite(rs)) return 0.0;
        if (!(mu > 0.0) || !std::isfinite(mu)) {
            return HEG::exchange_per_electron_HF(rs);
        }
        constexpr double pi = M_PI;
        const double pref = std::cbrt(18.0 / (pi * pi)) / rs;  // (18/π²)^{1/3}/rs
        const double kf = HEG::kF(rs);
        const double A = mu / (2.0 * kf);
        if (!(A > 0.0) || !std::isfinite(A)) {
            return HEG::exchange_per_electron_HF(rs);
        }
        // Large-μ asymptotic: −3/(16 rs³ μ²) + …
        if (A > 50.0) {
            return -3.0 / (16.0 * rs * rs * rs * mu * mu);
        }
        const double inv2A = 0.5 / A;
        const double A2 = A * A;
        const double A3 = A2 * A;
        const double exp_term = std::exp(-inv2A * inv2A);
        const double bracket =
            std::sqrt(pi) * std::erf(inv2A)
            + (2.0 * A - 4.0 * A3) * exp_term
            - 3.0 * A + 4.0 * A3;
        // ε̄_x = −pref * [3/8 − A * bracket]
        return -pref * (0.375 - A * bracket);
    }

    // Paziani / Molpro ECERF short-range correlation (paramagnetic ζ = 0).
    static double ec_sr_per_electron(double rs, double mu) {
        if (!(rs > 0.0) || !std::isfinite(rs)) return 0.0;
        const double ec0 = PW92::ec_per_electron(rs);
        if (!(mu > 0.0) || !std::isfinite(mu)) return ec0;

        constexpr double pi = M_PI;
        constexpr double alpha = 0.5210617611978481;  // (4/(9π))^{1/3}
        constexpr double ln2 = 0.6931471805599453;

        // On-top pair-distribution g(0, rs, ζ=0) [Gori-Giorgi–Perdew].
        constexpr double B = -0.0207;
        constexpr double C = 0.08193;
        constexpr double D = -0.01277;
        constexpr double E = 0.001859;
        constexpr double d_exp = 0.7524;
        const double rs2 = rs * rs;
        const double rs3 = rs2 * rs;
        const double rs4 = rs2 * rs2;
        // Molpro: ½ (1 − B rs + C rs² + …) e^{−d rs} with B = −0.0207.
        const double g0 = 0.5 * (1.0 - B * rs + C * rs2 + D * rs3 + E * rs4)
                          * std::exp(-d_exp * rs);
        const double gc0 = g0 - 0.5;

        // g''(0, rs, ζ=1) and D2, D3.
        auto gpp_zeta1 = [alpha](double rs1) {
            const double den = 1.0 + 0.4319 * rs1 + 0.04 * rs1 * rs1;
            const double pref = std::pow(2.0, 5.0 / 3.0)
                                / (5.0 * alpha * alpha * rs1 * rs1);
            return pref * (1.0 - 0.02267 * rs1) / den;
        };
        const double rs_p = rs * std::cbrt(2.0);
        const double gpp = gpp_zeta1(rs_p);
        const double D2 = std::exp(-0.547 * rs) / rs2
                          * (-0.388 * rs + 0.676 * rs2);
        const double D3 = std::exp(-0.31 * rs) / rs3
                          * (-4.95 * rs + rs2);

        // φ_n(0) = 1.  c4, c5 at ζ = 0.
        const double inv_a2_rs2 = 1.0 / (alpha * alpha * rs2);
        const double c4 = 0.5 * gpp + D2 - inv_a2_rs2 / 5.0;
        const double c5 = 0.5 * gpp + D3;

        const double C2 = -3.0 * gc0 / (8.0 * rs3);
        const double C3 = -g0 / (std::sqrt(2.0 * pi) * rs3);
        const double C4 = -9.0 * c4 / (64.0 * rs3);
        const double C5 = -9.0 * c5 / (40.0 * std::sqrt(2.0 * pi) * rs3);

        const double b0 = 0.784949 * rs;
        const double b02 = b0 * b0;
        const double b04 = b02 * b02;
        const double b06 = b04 * b02;
        const double b08 = b04 * b04;

        const double a1 = 4.0 * b06 * C3 + b08 * C5;
        const double a2 = 4.0 * b06 * C2 + b08 * C4 + 6.0 * b04 * ec0;
        const double a3 = b08 * C3;
        const double a4 = b08 * C2 + 4.0 * b06 * ec0;
        const double a5 = b08 * ec0;

        // Q(x) with x = μ √rs (φ2(0) = 1).
        constexpr double qa = 5.84605;
        constexpr double qc = 3.91744;
        constexpr double qd = 3.44851;
        const double qb = qd - 3.0 * pi * alpha / (4.0 * ln2 - 4.0);
        const double x = mu * std::sqrt(rs);
        const double x2 = x * x;
        const double x3 = x2 * x;
        const double Qnum = 1.0 + qa * x + qb * x2 + qc * x3;
        const double Qden = 1.0 + qa * x + qd * x2;
        double Q = 0.0;
        if (Qnum > 0.0 && Qden > 0.0) {
            Q = (2.0 * ln2 - 2.0) / (pi * pi) * std::log(Qnum / Qden);
        }

        const double mu2 = mu * mu;
        const double mu3 = mu2 * mu;
        const double mu4 = mu2 * mu2;
        const double mu5 = mu4 * mu;
        const double mu6 = mu4 * mu2;
        const double mu8 = mu4 * mu4;
        const double numer = Q + a1 * mu3 + a2 * mu4 + a3 * mu5
                             + a4 * mu6 + a5 * mu8;
        const double denom = std::pow(1.0 + b02 * mu2, 4.0);
        if (!(denom > 0.0) || !std::isfinite(denom) || !std::isfinite(numer)) {
            return ec0;
        }
        const double ec = ec0 - numer / denom;
        // Numerical guard: SR correlation is between full PW92 and 0.
        if (!std::isfinite(ec)) return ec0;
        if (ec > 0.0) return 0.0;
        if (ec < ec0) return ec0;
        return ec;
    }

    // Density-only XC energy per volume: ρ (e_x^SR + e_c^SR).
    static double xc_per_volume(double rho, double mu) {
        if (!(rho > 0.0) || !std::isfinite(rho)) return 0.0;
        constexpr double pi = M_PI;
        const double rs = std::cbrt(3.0 / (4.0 * pi * rho));
        return rho * (ex_sr_per_electron(rs, mu) + ec_sr_per_electron(rs, mu));
    }

    // Density-only short-range correlation energy per volume: ρ e_c^SR.
    static double c_per_volume(double rho, double mu) {
        if (!(rho > 0.0) || !std::isfinite(rho)) return 0.0;
        constexpr double pi = M_PI;
        const double rs = std::cbrt(3.0 / (4.0 * pi * rho));
        return rho * ec_sr_per_electron(rs, mu);
    }

    // Molpro ECERFPBE: Toulouse–Colonna–Savin short-range PBE correlation
    // [J. Chem. Phys. 122, 014110 (2005), Appendix B, Eqs. (B7)–(B12)].
    //
    //   ε_c = ε_c^{SR,LDA}(r_s, μ) + H(r_s, t, μ),
    //   t = |∇n| / (2 k_s n).
    //
    // On the uniform gas t = 0, so H = 0 and this is the μ-dependent
    // short-range LDA (Molpro ECERF / Paziani), not full PW92.
    // ``t`` is the reduced gradient; the HEG driver passes 0.
    static double ec_sr_pbe_per_electron(double rs, double mu, double t = 0.0) {
        const double ec_sr = ec_sr_per_electron(rs, mu);
        if (!(t > 0.0) || !std::isfinite(t)) return ec_sr;

        constexpr double gamma = 0.031090690869654895;  // (1 − ln 2) / π²
        const double beta = sr_pbe_beta(rs, mu);
        if (!(beta > 0.0) || !std::isfinite(beta) || !(gamma > 0.0)) {
            return ec_sr;
        }
        const double t2 = t * t;
        const double exp_arg = -ec_sr / gamma;
        double denom_A = std::expm1(exp_arg);
        if (!(denom_A > 0.0) || !std::isfinite(denom_A)) return ec_sr;
        const double A = (beta / gamma) / denom_A;
        const double At2 = A * t2;
        const double frac_den = 1.0 + At2 + At2 * At2;
        if (!(frac_den > 0.0)) return ec_sr;
        const double arg = 1.0 + (beta * t2 / gamma) * ((1.0 + At2) / frac_den);
        if (!(arg > 0.0) || !std::isfinite(arg)) return ec_sr;
        const double H = gamma * std::log(arg);
        if (!std::isfinite(H)) return ec_sr;
        return ec_sr + H;
    }

private:
    // β(r_s, μ) from the erf exchange gradient expansion, Eq. (18):
    //   β = −ε_x^{SR} b(μ̃) (k_s / k_F)²,  μ̃ = μ / (2 k_F).
    static double sr_pbe_beta(double rs, double mu) {
        if (!(rs > 0.0) || !(mu > 0.0)) return 0.0;
        const double kf = HEG::kF(rs);
        const double mu_tilde = mu / (2.0 * kf);
        const double b = erf_exchange_gea_b(mu_tilde);
        const double ex = ex_sr_per_electron(rs, mu);
        // k_s² = 4 k_F / π  ⇒  (k_s / k_F)² = 4 / (π k_F).
        constexpr double pi = M_PI;
        const double ks_over_kf_sq = 4.0 / (pi * kf);
        const double beta = -ex * b * ks_over_kf_sq;
        return (beta > 0.0 && std::isfinite(beta)) ? beta : 0.0;
    }

    // b_erf(μ̃), Toulouse et al. Eq. (A18).  Rewritten as
    //   (−c1 e^{−1/(4μ̃²)} + c2) / (c3 e^{−1/(4μ̃²)} + 54 c4)
    // so the μ̃ → 0 limit is Sham's 7/81 without overflow.
    static double erf_exchange_gea_b(double mu_tilde) {
        constexpr double b_coulomb = 7.0 / 81.0;
        if (!(mu_tilde > 0.0) || !std::isfinite(mu_tilde)) return b_coulomb;
        if (mu_tilde < 1.0e-4) return b_coulomb;
        const double nu = mu_tilde;
        const double nu2 = nu * nu;
        const double nu4 = nu2 * nu2;
        const double c1 = 1.0 + 22.0 * nu2 + 144.0 * nu4;
        const double c2 = 2.0 * nu2 * (-7.0 + 72.0 * nu2);
        const double c3 = -864.0 * nu4 * (-1.0 + 2.0 * nu2);
        const double c4 =
            nu2 * (-3.0 - 24.0 * nu2 + 32.0 * nu4
                   + 8.0 * nu * std::sqrt(M_PI) * std::erf(0.5 / nu));
        const double e = std::exp(-0.25 / nu2);
        const double num = -c1 * e + c2;
        const double den = c3 * e + 54.0 * c4;
        if (!(std::abs(den) > 0.0) || !std::isfinite(num) || !std::isfinite(den)) {
            return b_coulomb;
        }
        const double b = num / den;
        return std::isfinite(b) ? b : b_coulomb;
    }
};

}  // namespace rdmft
