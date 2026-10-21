#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "Energy.hpp"
#include "ExchangeKernel.hpp"
#include "Functional.hpp"
#include "Grid.hpp"
#include "HEG.hpp"
#include "QMC.hpp"
#include "RangeSeparated.hpp"

using namespace rdmft;

namespace {

int failures = 0;

void expect_exception(const std::function<void()>& operation,
                      const std::string& message) {
    try {
        operation();
        std::cerr << "FAIL: " << message << "\n";
        ++failures;
    } catch (const std::exception&) {
    }
}

}  // namespace

int main() {
    expect_exception(
        [] { (void)Grid::uniform_trapezoid(1.0, 1); },
        "uniform grid should reject fewer than two nodes");
    expect_exception(
        [] {
            (void)Grid::uniform_trapezoid(
                std::numeric_limits<double>::quiet_NaN(), 3);
        },
        "uniform grid should reject non-finite bounds");
    expect_exception(
        [] {
            (void)Grid::composite_trapezoid_weights({0.0, 1.0, 1.0});
        },
        "quadrature should reject duplicate nodes");

    Grid grid = Grid::uniform_trapezoid(2.0, 5);
    ExchangeKernel exchange = ExchangeKernel::build(grid);
    HFFunctional hf;

    {
        const double kf = 1.0;
        Grid g_logf = Grid::log_fermi_trapezoid(kf, 5.0 * kf, 401u);
        if (g_logf.n() != 401u) {
            std::cerr << "FAIL: log_fermi_trapezoid node count\n";
            ++failures;
        }
        bool has_kf = false;
        for (double k : g_logf.k) {
            if (std::abs(k - kf) <= 1.0e-12) has_kf = true;
        }
        if (!has_kf) {
            std::cerr << "FAIL: log_fermi_trapezoid should place a node at k_F\n";
            ++failures;
        }
        Grid g_grad = Grid::graded_fermi_trapezoid(kf, 5.0 * kf, 401u);
        if (g_grad.n() != 401u) {
            std::cerr << "FAIL: graded_fermi_trapezoid node count\n";
            ++failures;
        }
    }

    expect_exception(
        [&] {
            (void)EnergyEvaluator::density(
                std::vector<double>(grid.n() - 1, 0.5), grid);
        },
        "energy evaluators should reject occupation-size mismatches");

    expect_exception(
        [&] {
            ExchangeKernel malformed = exchange;
            malformed.Wt.pop_back();
            (void)EnergyEvaluator::xc_per_volume(
                std::vector<double>(grid.n(), 0.5), grid, malformed, hf);
        },
        "energy evaluators should reject malformed exchange matrices");

    expect_exception(
        [] {
            Grid malformed;
            malformed.k = {0.0, 1.0};
            malformed.w = {1.0};
            (void)ExchangeKernel::build(malformed);
        },
        "exchange construction should reject mismatched grid storage");

    expect_exception(
        [] { (void)PowerFunctional(0.0); },
        "Power should reject a zero exponent");
    expect_exception(
        [] { (void)BetaFunctional(1.1); },
        "Beta should reject a non-monotone exponent");
    expect_exception(
        [] { (void)HybOptFunctional(1.1, 0.5); },
        "HybOpt should reject lambda outside [0,1]");
    expect_exception(
        [] { (void)OptGeoFunctional(-1.0, 2.0); },
        "OptGeo should reject negative parameters");

    {
        PbeHfMuFunctional pbe;
        if (std::abs(pbe.kernel(1.0, 1.0)) > 1.0e-14) {
            std::cerr << "FAIL: PBEHFmM K(1,1) should be 0\n";
            ++failures;
        }
        if (std::abs(pbe.kernel(0.0, 0.7)) > 1.0e-14) {
            std::cerr << "FAIL: PBEHFmM K(0,n) should be 0\n";
            ++failures;
        }
        const double ni = 0.4, nj = 0.7;
        const double h = 1.0e-8;
        const double fd = (pbe.kernel(ni + h, nj) - pbe.kernel(ni - h, nj)) / (2.0 * h);
        if (std::abs(pbe.kernel_grad(ni, nj) - fd) > 1.0e-6) {
            std::cerr << "FAIL: PBEHFmM kernel_grad vs central difference\n";
            ++failures;
        }
        const double rs = 1.0;
        const double rho = HEG::density(rs);
        const double expected = rho * (HEG::exchange_per_electron_HF(rs)
                                       + PW92::ec_per_electron(rs));
        if (std::abs(pbe.density_xc_per_volume(rho) - expected) > 1.0e-12) {
            std::cerr << "FAIL: PBEHFmM density_xc_per_volume at rs=1\n";
            ++failures;
        }
        if (pbe.name() != "PBEHFmM") {
            std::cerr << "FAIL: PBEHFmM name()\n";
            ++failures;
        }
    }

    expect_exception(
        [] { (void)RsFunctional(0.0); },
        "rs should reject non-positive mu");
    expect_exception(
        [] { (void)RsFunctional(0.5, 0.0); },
        "rs should reject non-positive alpha");
    expect_exception(
        [] { (void)ExchangeKernel::build_erf_lr(
                 Grid::uniform_trapezoid(1.0, 5), -1.0); },
        "build_erf_lr should reject non-positive mu");

    {
        RsFunctional hole(0.5, 0.55);
        if (std::abs(hole.range_mu() - 0.5) > 1.0e-15
            || !hole.uses_lr_power_correlation()
            || std::abs(hole.lr_power_alpha() - 0.55) > 1.0e-15) {
            std::cerr << "FAIL: rs range_mu / LR Power flag\n";
            ++failures;
        }
        if (!hole.is_factorized()
            || std::abs(hole.f(0.4) - 0.4) > 1.0e-15
            || std::abs(hole.df(0.4) - 1.0) > 1.0e-15) {
            std::cerr << "FAIL: rs primary kernel should be HF\n";
            ++failures;
        }
        if (std::abs(hole.kernel(1.0, 1.0) - 1.0) > 1.0e-14
            || std::abs(hole.kernel(0.0, 0.7)) > 1.0e-14
            || std::abs(hole.kernel(0.25, 0.64) - 0.16) > 1.0e-14) {
            std::cerr << "FAIL: rs HF kernel values\n";
            ++failures;
        }
        const double rs = 1.0;
        const double rho = HEG::density(rs);
        const double c_only = rho * RangeSeparatedLDA::ec_sr_per_electron(rs, 0.5);
        if (std::abs(hole.density_xc_per_volume(rho) - c_only) > 1.0e-12) {
            std::cerr << "FAIL: rs density piece should be SR correlation only\n";
            ++failures;
        }
        // Integer occupations: LR Power−HF vanishes; Exc = Ex^HF + ε_c^SR.
        {
            Grid g = Grid::uniform_trapezoid(2.0, 17);
            ExchangeKernel Wc = ExchangeKernel::build(g);
            ExchangeKernel Wlr = ExchangeKernel::build_erf_lr(g, 0.5);
            std::vector<double> n(g.n(), 0.0);
            const double kf = HEG::kF(rs);
            for (std::size_t i = 0; i < g.n(); ++i) {
                n[i] = (g.k[i] <= kf) ? 1.0 : 0.0;
            }
            const double Exc = EnergyEvaluator::xc_per_volume(n, g, Wc, hole, &Wlr);
            const HFFunctional hf;
            const double Ex_hf = EnergyEvaluator::xc_per_volume(n, g, Wc, hf);
            const double expect = Ex_hf + hole.density_xc_per_volume(
                EnergyEvaluator::density(n, g));
            if (std::abs(Exc - expect) > 1.0e-10 * (1.0 + std::abs(expect))) {
                std::cerr << "FAIL: integer-n rs should be HF + SR-c"
                          << " (Exc=" << Exc << " expect=" << expect << ")\n";
                ++failures;
            }
        }
        auto made = make_functional("rs@0.4;0.55");
        if (!made || std::abs(made->range_mu() - 0.4) > 1.0e-15
            || !made->uses_lr_power_correlation()
            || std::abs(made->lr_power_alpha() - 0.55) > 1.0e-15) {
            std::cerr << "FAIL: make_functional(rs@0.4;0.55)\n";
            ++failures;
        }
    }

    expect_exception(
        [] { (void)Rs2Functional(0.0); },
        "rs2 should reject non-positive mu");
    expect_exception(
        [] { (void)Rs2Functional(0.5, 0.0); },
        "rs2 should reject non-positive alpha");

    {
        Rs2Functional rs2(0.5, 0.55);
        if (std::abs(rs2.range_mu() - 0.5) > 1.0e-15
            || !rs2.uses_lr_power_exchange()
            || rs2.uses_lr_power_correlation()
            || std::abs(rs2.lr_power_alpha() - 0.55) > 1.0e-15) {
            std::cerr << "FAIL: rs2 range_mu / LR Power exchange flag\n";
            ++failures;
        }
        const double rs = 1.0;
        const double rho = HEG::density(rs);
        const double xc = RangeSeparatedLDA::xc_per_volume(rho, 0.5);
        if (std::abs(rs2.density_xc_per_volume(rho) - xc) > 1.0e-12) {
            std::cerr << "FAIL: rs2 density piece should be SR LDA x+c\n";
            ++failures;
        }
        // Integer occupations: n^α = n ⇒ Power ≡ HF on W_LR.
        {
            Grid g = Grid::uniform_trapezoid(2.0, 17);
            ExchangeKernel Wc = ExchangeKernel::build(g);
            ExchangeKernel Wlr = ExchangeKernel::build_erf_lr(g, 0.5);
            std::vector<double> n(g.n(), 0.0);
            const double kf = HEG::kF(rs);
            for (std::size_t i = 0; i < g.n(); ++i) {
                n[i] = (g.k[i] <= kf) ? 1.0 : 0.0;
            }
            const double Exc =
                EnergyEvaluator::xc_per_volume(n, g, Wc, rs2, &Wlr);
            const HFFunctional hf;
            const double Ex_lr =
                EnergyEvaluator::jk_per_volume(n, g, Wlr, hf);
            const double expect =
                rs2.density_xc_per_volume(EnergyEvaluator::density(n, g))
                + Ex_lr;
            if (std::abs(Exc - expect) > 1.0e-10 * (1.0 + std::abs(expect))) {
                std::cerr << "FAIL: integer-n rs2 should be SR xc + HF[W_LR]"
                          << " (Exc=" << Exc << " expect=" << expect << ")\n";
                ++failures;
            }
        }
        auto made2 = make_functional("rs2@0.4;0.55");
        if (!made2 || std::abs(made2->range_mu() - 0.4) > 1.0e-15
            || !made2->uses_lr_power_exchange()
            || made2->uses_lr_power_correlation()
            || std::abs(made2->lr_power_alpha() - 0.55) > 1.0e-15) {
            std::cerr << "FAIL: make_functional(rs2@0.4;0.55)\n";
            ++failures;
        }
        // Ensure rs2@ is not mis-parsed as rs.
        auto not_rs = make_functional("rs2@0.3");
        if (!not_rs || !not_rs->uses_lr_power_exchange()
            || not_rs->uses_lr_power_correlation()) {
            std::cerr << "FAIL: rs2@0.3 must resolve to Rs2Functional\n";
            ++failures;
        }
    }

    {
        Rs3Functional rs3(0.5, 0.55);
        const double rs = 1.0;
        const double rho = HEG::density(rs);
        const double ec_sr = RangeSeparatedLDA::ec_sr_per_electron(rs, 0.5);
        const double ec_pbe =
            RangeSeparatedLDA::ec_sr_pbe_per_electron(rs, 0.5, 0.0);
        const double ec_pw = PW92::ec_per_electron(rs);
        if (std::abs(ec_pbe - ec_sr) > 1.0e-14
            || std::abs(rs3.density_xc_per_volume(rho) - rho * ec_sr) > 1.0e-12) {
            std::cerr << "FAIL: rs3 on the HEG should be mu-dependent ECERF\n";
            ++failures;
        }
        if (std::abs(ec_pbe - ec_pw) < 1.0e-4) {
            std::cerr << "FAIL: ECERFPBE at mu=0.5 should differ from full PW92\n";
            ++failures;
        }
        const double ec_grad =
            RangeSeparatedLDA::ec_sr_pbe_per_electron(rs, 0.5, 0.5);
        if (!std::isfinite(ec_grad) || std::abs(ec_grad - ec_sr) < 1.0e-8) {
            std::cerr << "FAIL: ECERFPBE gradient correction H should be nonzero at t=0.5\n";
            ++failures;
        }
        auto made3 = make_functional("rs3@0.4;0.55");
        if (!made3 || std::abs(made3->range_mu() - 0.4) > 1.0e-15
            || !made3->uses_lr_power_correlation()
            || made3->uses_lr_power_exchange()
            || std::abs(made3->lr_power_alpha() - 0.55) > 1.0e-15) {
            std::cerr << "FAIL: make_functional(rs3@0.4;0.55)\n";
            ++failures;
        }
    }

    {
        const double rs = 2.0;
        const double ex0 = HEG::exchange_per_electron_HF(rs);
        const double ec0 = PW92::ec_per_electron(rs);
        const double ex_small =
            RangeSeparatedLDA::ex_sr_per_electron(rs, 1.0e-8);
        const double ec_small =
            RangeSeparatedLDA::ec_sr_per_electron(rs, 1.0e-8);
        if (std::abs(ex_small - ex0) > 1.0e-6 * std::abs(ex0)
            || std::abs(ec_small - ec0) > 1.0e-6 * std::abs(ec0)) {
            std::cerr << "FAIL: SR LDA XC should recover Slater+PW92 as mu->0\n";
            ++failures;
        }
        const double ex_large =
            RangeSeparatedLDA::ex_sr_per_electron(rs, 1.0e3);
        const double ec_large =
            RangeSeparatedLDA::ec_sr_per_electron(rs, 1.0e3);
        if (std::abs(ex_large) > 1.0e-4 || std::abs(ec_large) > 1.0e-4) {
            std::cerr << "FAIL: SR LDA XC should vanish as mu->infinity\n";
            ++failures;
        }
    }

    {
        Grid g = Grid::uniform_trapezoid(3.0, 21);
        ExchangeKernel Wc = ExchangeKernel::build(g);
        ExchangeKernel Wlr = ExchangeKernel::build_erf_lr(g, 50.0);
        double max_abs = 0.0;
        double max_diff = 0.0;
        for (std::size_t i = 0; i < Wc.W.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(Wc.W[i]));
            max_diff = std::max(max_diff, std::abs(Wc.W[i] - Wlr.W[i]));
        }
        if (!(max_abs > 0.0) || max_diff > 5.0e-3 * max_abs) {
            std::cerr << "FAIL: W_LR(mu=50) should approach W_Coulomb"
                      << " (rel=" << (max_diff / max_abs) << ")\n";
            ++failures;
        }
        ExchangeKernel Wtiny = ExchangeKernel::build_erf_lr(g, 1.0e-3);
        double max_lr = 0.0;
        for (double w : Wtiny.W) max_lr = std::max(max_lr, std::abs(w));
        if (max_lr > 1.0e-2 * max_abs) {
            std::cerr << "FAIL: W_LR(mu->0) should vanish"
                      << " (max=" << max_lr << " vs Coulomb " << max_abs << ")\n";
            ++failures;
        }
    }

    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "test_boundaries: all checks passed\n";
    return 0;
}
