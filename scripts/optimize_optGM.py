#!/usr/bin/env python3
"""Optimize ``OptGM@lambda;alpha`` vs PW92 correlation energy (same target as ``Ec_QMC`` in TSVs).

The C++ kernel is::

    K(n_i, n_j) = (1 - lambda) * n_i n_j + lambda * n_i^alpha * n_j^alpha,

i.e. **(1−λ)·HF + λ·Power(α)** in the driver JK convention.  ``lambda`` is clamped to
``[0, 1]`` in the C++ functional; ``alpha`` must be positive.

This script minimizes RMSE of model ``Ec_per_N`` vs **PW92** ``ec(rs)`` (the QMC
parameterization used throughout the repo and printed as ``Ec_QMC`` by
``rdmft_heg``).

**SciPy is required.**  Install: ``pip install -r scripts/requirements-optgm.txt``.

For **optGeo** angles use ``scripts/optimize_optGeo.py``.
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def pw92_ec_per_electron(rs: float) -> float:
    a = 0.0310907
    alpha1 = 0.21370
    beta1 = 7.5957
    beta2 = 3.5876
    beta3 = 1.6382
    beta4 = 0.49294
    sqrt_rs = math.sqrt(rs)
    denom = 2.0 * a * (beta1 * sqrt_rs + beta2 * rs + beta3 * rs * sqrt_rs + beta4 * rs * rs)
    g = -2.0 * a * (1.0 + alpha1 * rs) * math.log(1.0 + 1.0 / denom)
    return g


def parse_ec_from_tsv(tsv_path: Path) -> dict[float, float]:
    out: dict[float, float] = {}
    with tsv_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            try:
                rs = float(parts[0])
                ec = float(parts[3])
            except ValueError:
                continue
            out[rs] = ec
    return out


def tsv_stem_for_func_key(key: str) -> str:
    s = key.replace("@", "_").replace(";", "_").replace("/", "_").replace(" ", "_")
    return s + ".tsv"


def run_rdmft_optgm(
    exe: Path,
    lam: float,
    alpha: float,
    rs_list: list[float],
    out_dir: Path,
    n_grid: int | None,
    kmax: float | None,
    verbose: bool,
) -> dict[float, float]:
    """Run rdmft_heg once for ``OptGM@lambda;alpha``; return {rs: Ec}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    key = f"OptGM@{lam:.12g};{alpha:.12g}"
    cmd = [
        str(exe),
        "--funcs",
        key,
        "--rs",
        ",".join(str(x) for x in rs_list),
        "--out-dir",
        str(out_dir),
        "--force",
    ]
    if n_grid is not None:
        cmd.extend(["--N", str(n_grid)])
    if kmax is not None:
        cmd.extend(["--kmax", str(kmax)])
    proc = subprocess.run(
        cmd,
        cwd=str(exe.parent.parent) if exe.parent.name == "build" else None,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"rdmft_heg failed (code {proc.returncode}):\n{proc.stderr}\n{proc.stdout}"
        )
    if verbose and proc.stdout:
        print(proc.stdout, end="")

    tsv = out_dir / tsv_stem_for_func_key(key)
    if not tsv.is_file():
        raise FileNotFoundError(f"Expected output {tsv}, stdout:\n{proc.stdout}")
    return parse_ec_from_tsv(tsv)


def rmse_ec(
    ec_model: dict[float, float],
    rs_list: list[float],
    ec_ref: dict[float, float],
) -> float:
    errs = []
    for rs in rs_list:
        if rs not in ec_model or rs not in ec_ref:
            continue
        errs.append(ec_model[rs] - ec_ref[rs])
    if not errs:
        return float("inf")
    return math.sqrt(sum(e * e for e in errs) / len(errs))


def default_rs() -> list[float]:
    return [0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


def clamp_la(
    lam: float, alpha: float, lam_min: float, lam_max: float, a_min: float, a_max: float
) -> tuple[float, float]:
    return (
        max(lam_min, min(lam_max, lam)),
        max(a_min, min(a_max, alpha)),
    )


def prescreen_pairs() -> list[tuple[str, float, float]]:
    return [
        ("mid (0.5, 0.55)", 0.5, 0.55),
        ("more HF (0.25, 0.56)", 0.25, 0.56),
        ("more Power (0.65, 0.54)", 0.65, 0.54),
        ("high alpha (0.45, 0.62)", 0.45, 0.62),
        ("low alpha (0.5, 0.48)", 0.5, 0.48),
        ("edge HF (0.05, 0.57)", 0.05, 0.57),
        ("edge Pow (0.92, 0.53)", 0.92, 0.53),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Optimize OptGM@lambda;alpha vs PW92 E_c (2D box; SciPy required). "
            "Kernel: (1-lambda)*HF + lambda*Power(alpha)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 scripts/optimize_optGM.py --rs 2,3,5\n"
            "  python3 scripts/optimize_optGM.py --alpha-min 0.45 --alpha-max 0.65 --method differential-evolution\n"
            "Writes build/optgm_best/ and a quoted --funcs line."
        ),
    )
    ap.add_argument("--exe", type=Path, default=Path("build/rdmft_heg"), help="rdmft_heg path")
    ap.add_argument(
        "--rs",
        type=str,
        default="",
        help="Comma-separated r_s list (default: built-in benchmark grid)",
    )
    ap.add_argument("--N", type=int, default=801, help="Grid points for main / final run")
    ap.add_argument("--kmax", type=float, default=3.0, help="k_max in units of k_F")
    ap.add_argument("--lam-min", type=float, default=0.0, help="Lower bound on lambda")
    ap.add_argument("--lam-max", type=float, default=1.0, help="Upper bound on lambda")
    ap.add_argument("--alpha-min", type=float, default=0.35, help="Lower bound on alpha")
    ap.add_argument(
        "--alpha-max",
        type=float,
        default=0.72,
        help="Upper bound on alpha (must stay < 1 for the C++ EL solver; values ≥1 are clamped)",
    )
    ap.add_argument("--maxiter", type=int, default=35, help="Optimizer max iterations / DE generations")
    ap.add_argument(
        "--method",
        type=str,
        default="Nelder-Mead",
        choices=(
            "Nelder-Mead",
            "L-BFGS-B",
            "Powell",
            "TNC",
            "SLSQP",
            "differential-evolution",
        ),
        help="SciPy optimizer",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="ftol (L-BFGS-B/Powell/TNC/SLSQP) or atol (DE); ignored for Nelder-Mead",
    )
    ap.add_argument("--seed", type=int, default=0, help="DE seed (0 = nondeterministic)")
    ap.add_argument("--de-popsize", type=int, default=15, help="DE population size")
    ap.add_argument("--no-de-polish", action="store_true", help="Skip L-BFGS-B polish after DE")
    ap.add_argument("--nm-xatol", type=float, default=0.015, help="Nelder-Mead xatol")
    ap.add_argument("--nm-fatol", type=float, default=5e-4, help="Nelder-Mead fatol")
    ap.add_argument("--decimals", type=int, default=6, help="Decimals for rounded CLI / final run")
    ap.add_argument("--no-prescreen", action="store_true", help="Skip prescreen; start (0.5, 0.55)")
    ap.add_argument(
        "--prescreen-rs",
        type=str,
        default="0.2,1.0,5.0,10.0",
        help="Comma-separated r_s for prescreen",
    )
    ap.add_argument("--prescreen-N", type=int, default=101, help="N during prescreen")
    ap.add_argument("--prescreen-kmax", type=float, default=2.0, help="k_max during prescreen")
    ap.add_argument("--quiet", action="store_true", help="Less progress output")
    ap.add_argument("--verbose", action="store_true", help="Print rdmft_heg stdout")
    ap.add_argument("--no-clean-tmp", action="store_true", help="Keep temp eval dirs")
    args = ap.parse_args()

    try:
        import numpy as np  # noqa: F401
    except ImportError as e:
        print("This script requires numpy.", file=sys.stderr)
        raise SystemExit(1) from e

    try:
        from scipy.optimize import (  # type: ignore[import-untyped]
            differential_evolution,
            minimize as scipy_minimize,
        )
    except ImportError as e:
        print("SciPy is required. Install:  pip install scipy", file=sys.stderr)
        raise SystemExit(1) from e

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    repo_root = Path(__file__).resolve().parents[1]
    exe = (repo_root / args.exe).resolve() if not args.exe.is_absolute() else args.exe
    if not exe.is_file():
        print(f"Executable not found: {exe}", file=sys.stderr)
        print("Run `make` from the repository root first.", file=sys.stderr)
        raise SystemExit(1)

    lam_min, lam_max = float(args.lam_min), float(args.lam_max)
    a_min, a_max = float(args.alpha_min), float(args.alpha_max)
    if not (lam_min < lam_max) or not (a_min < a_max):
        print("--lam-min < --lam-max and --alpha-min < --alpha-max required.", file=sys.stderr)
        raise SystemExit(1)

    rs_list = (
        [float(x) for x in args.rs.split(",") if x.strip()]
        if args.rs.strip()
        else default_rs()
    )
    ec_ref = {rs: pw92_ec_per_electron(rs) for rs in rs_list}
    prescreen_rs = [float(x) for x in args.prescreen_rs.split(",") if x.strip()]
    ec_ref_pre = {rs: pw92_ec_per_electron(rs) for rs in prescreen_rs}

    tmp_root = Path(tempfile.mkdtemp(prefix="optgm_la_", dir=repo_root / "build"))
    log("")
    log("=== OptGM (lambda, alpha) vs PW92 — RMSE(E_c) ===")
    log(f"  executable: {exe}")
    log(f"  box: lambda in [{lam_min}, {lam_max}], alpha in [{a_min}, {a_max}]")
    log(f"  main grid:  N={args.N}  kmax={args.kmax}")
    log(f"  main r_s:   {rs_list}")
    log(f"  work dir:   {tmp_root}")

    n_eval = [0]

    def objective_vec(vec: Any, *, phase: str = "main") -> float:
        lam, alpha = clamp_la(float(vec[0]), float(vec[1]), lam_min, lam_max, a_min, a_max)
        run_dir = tmp_root / f"eval_{n_eval[0]:05d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        n_eval[0] += 1
        t0 = time.perf_counter()
        try:
            ec_model = run_rdmft_optgm(
                exe, lam, alpha, rs_list, run_dir, args.N, args.kmax, args.verbose
            )
        except (RuntimeError, FileNotFoundError) as exc:
            log(f"  [{phase}] eval #{n_eval[0]} FAILED after {time.perf_counter()-t0:.1f}s: {exc}")
            return 1e3

        val = rmse_ec(ec_model, rs_list, ec_ref)
        dt = time.perf_counter() - t0
        log(f"  [{phase}] eval #{n_eval[0]}  lam={lam:.5f} alpha={alpha:.5f}  RMSE={val:.5e}  ({dt:.1f}s)")
        return val

    if args.no_prescreen:
        lam0, a0 = 0.5, 0.55
        log("\n--- prescreen: skipped; start (0.5, 0.55) ---")
    else:
        log("\n--- prescreen: (lambda, alpha) probes ---")
        log(f"  N={args.prescreen_N}  kmax={args.prescreen_kmax}  r_s={prescreen_rs}")
        best_rmse = float("inf")
        lam0, a0 = 0.5, 0.55
        best_name = ""
        pre_root = tmp_root / "prescreen"
        pre_root.mkdir(parents=True, exist_ok=True)
        for idx, (name, lam, alpha) in enumerate(prescreen_pairs()):
            lam, alpha = clamp_la(lam, alpha, lam_min, lam_max, a_min, a_max)
            run_dir = pre_root / f"cand_{idx:02d}"
            t0 = time.perf_counter()
            try:
                ec_m = run_rdmft_optgm(
                    exe,
                    lam,
                    alpha,
                    prescreen_rs,
                    run_dir,
                    args.prescreen_N,
                    args.prescreen_kmax,
                    args.verbose,
                )
            except (RuntimeError, FileNotFoundError) as exc:
                log(f"  [{idx+1:2d}] {name[:36]:<36}  FAIL  ({exc})")
                continue
            rmse = rmse_ec(ec_m, prescreen_rs, ec_ref_pre)
            dt = time.perf_counter() - t0
            log(f"  [{idx+1:2d}] {name[:40]:<40}  RMSE={rmse:.5e}  ({dt:.1f}s)  lam,a=({lam:.3f},{alpha:.3f})")
            if rmse < best_rmse:
                best_rmse = rmse
                lam0, a0 = lam, alpha
                best_name = name
        log(f"\n  prescreen best: {best_name}")
        log(f"  -> start lam={lam0:.4f}  alpha={a0:.4f}  (coarse RMSE={best_rmse:.5e})")

    log("\n--- main optimization ---")
    import numpy as np

    x0 = np.array([lam0, a0], dtype=float)
    n_main_start = n_eval[0]
    bounds_scipy = [(lam_min, lam_max), (a_min, a_max)]

    def obj_scipy(v: np.ndarray) -> float:
        return objective_vec(np.asarray(v, dtype=float), phase="main")

    method = args.method
    log(f"  SciPy: method={method}")

    if method == "differential-evolution":
        seed = None if args.seed == 0 else int(args.seed)
        pop = max(int(args.de_popsize), 2)
        res = differential_evolution(
            obj_scipy,
            bounds_scipy,
            maxiter=max(int(args.maxiter), 1),
            popsize=pop,
            seed=seed,
            polish=not args.no_de_polish,
            atol=float(args.tol),
            workers=1,
        )
        lam_opt, a_opt = clamp_la(float(res.x[0]), float(res.x[1]), lam_min, lam_max, a_min, a_max)
        final_rmse = float(res.fun)
        success = bool(getattr(res, "success", True))
        msg = str(getattr(res, "message", ""))
    else:
        opts: dict[str, float | int] = {"maxiter": int(args.maxiter)}
        if method == "L-BFGS-B":
            opts["ftol"] = float(args.tol)
        elif method == "Nelder-Mead":
            opts["xatol"] = float(args.nm_xatol)
            opts["fatol"] = float(args.nm_fatol)
        elif method in ("TNC", "SLSQP", "Powell"):
            opts["ftol"] = float(args.tol)

        kwargs: dict[str, Any] = {
            "fun": obj_scipy,
            "x0": np.clip(
                x0, np.array([lam_min, a_min], dtype=float), np.array([lam_max, a_max], dtype=float)
            ),
            "method": method,
            "options": opts,
        }
        if method in ("L-BFGS-B", "TNC", "SLSQP", "Nelder-Mead"):
            kwargs["bounds"] = bounds_scipy
        try:
            res = scipy_minimize(**kwargs)
        except ValueError as exc:
            if method == "Nelder-Mead" and "bounds" in kwargs:
                log(f"  Note: retrying Nelder-Mead without bounds ({exc}); objective still clamps.")
                kwargs.pop("bounds", None)
                res = scipy_minimize(**kwargs)
            else:
                raise
        lam_opt, a_opt = clamp_la(float(res.x[0]), float(res.x[1]), lam_min, lam_max, a_min, a_max)
        final_rmse = float(res.fun)
        success = bool(res.success)
        msg = str(res.message)

    nfev_main = n_eval[0] - n_main_start
    d = int(args.decimals)
    lam_r = round(lam_opt, d)
    a_r = round(a_opt, d)
    lam_r, a_r = clamp_la(lam_r, a_r, lam_min, lam_max, a_min, a_max)

    log("\n=== optimization finished ===")
    log(f"  message: {msg}")
    log(f"  success={success}  rdmft_evals={n_eval[0]}  main-phase={nfev_main}")
    log(f"  final RMSE: {final_rmse:.5e}")
    log(f"  refined lambda={lam_opt:.8f}  alpha={a_opt:.8f}")
    key_q = f"OptGM@{lam_r:.{d}f};{a_r:.{d}f}"
    log(f"\n  Recommended: {key_q}")
    log(f"  rdmft_heg: --funcs '{key_q}'")

    final_dir = repo_root / "build" / "optgm_best"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True)
    log(f"\n--- final validation -> {final_dir} ---")
    ec_final = run_rdmft_optgm(exe, lam_r, a_r, rs_list, final_dir, args.N, args.kmax, args.verbose)
    for rs in rs_list:
        em = ec_final.get(rs, float("nan"))
        er = ec_ref[rs]
        log(f"  rs={rs:4g}  Ec={em: .8f}  PW92={er: .8f}  diff={em - er: .8e}")
    log(f"  RMSE after rounding: {rmse_ec(ec_final, rs_list, ec_ref):.5e}")

    if not args.no_clean_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)
        log(f"\nRemoved temp dir {tmp_root}")
    else:
        log(f"\n[debug] kept {tmp_root}")


if __name__ == "__main__":
    main()
