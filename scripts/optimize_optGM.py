#!/usr/bin/env python3
"""Optimize optGM mixing angles (alpha, beta, gamma) on the unit sphere.

The C++ functional ``OptGM@a;b;c`` (semicolons: commas are reserved in ``--funcs``)
normalizes (a,b,c) so that a^2+b^2+c^2=1 and
uses kernel weights w1=a^2, w2=b^2, w3=c^2 on the three GEO channels:

    K = w1 * n_i n_j + w2 * (n_i n_j)^{1/2} + w3 * (n_i n_j)^{3/4}.

Flow (default):
  1) **Prescreen** — run several named directions on a cheap grid (fewer r_s,
     smaller N/kmax) and pick the lowest RMSE vs PW92 as the starting angles.
  2) **Main** — Nelder-Mead (or SciPy method) on the full r_s list with loose
     tolerances so ~1 decimal on (a,b,c) is enough for practice.
  3) **Final** — one full ``rdmft_heg`` with **rounded** (a,b,c) written under
     ``build/optgm_best/``.

Progress is printed by default; use ``--quiet`` to reduce noise, ``--verbose``
to include C++ stdout.  Requires NumPy; SciPy optional for L-BFGS-B / Powell.
"""
from __future__ import annotations

import argparse
import time
from typing import Any, Callable
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# PW92 paramagnetic correlation energy per electron (hartree), same as QMC.hpp.
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


def spherical_to_unit(theta: float, phi: float) -> tuple[float, float, float]:
    """Map (theta, phi) to (alpha, beta, gamma) on the unit sphere."""
    st = math.sin(theta)
    alpha = st * math.cos(phi)
    beta = st * math.sin(phi)
    gamma = math.cos(theta)
    return alpha, beta, gamma


def angles_for_weights(w1: float, w2: float, w3: float) -> tuple[float, float, float]:
    """Return (alpha, beta, gamma) with squares (w1,w2,w3), w1+w2+w3 ~ 1."""
    a = math.sqrt(max(w1, 0.0))
    b = math.sqrt(max(w2, 0.0))
    c = math.sqrt(max(w3, 0.0))
    nrm = math.hypot(math.hypot(a, b), c)
    if nrm < 1e-15:
        s = 1.0 / math.sqrt(3.0)
        return s, s, s
    return a / nrm, b / nrm, c / nrm


def guess_spherical_from_weights(w1: float, w2: float, w3: float) -> tuple[float, float]:
    """Initial (theta, phi) matching GEO-like (w1,w2,w3) on the unit sphere."""
    a, b, c = angles_for_weights(w1, w2, w3)
    theta = math.acos(max(-1.0, min(1.0, c)))
    phi = math.atan2(b, a)
    return theta, phi


def parse_ec_from_tsv(tsv_path: Path) -> dict[float, float]:
    """Map r_s -> E_c (column 3) from a single per-functional TSV."""
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


def run_rdmft(
    exe: Path,
    alpha: float,
    beta: float,
    gamma: float,
    rs_list: list[float],
    out_dir: Path,
    n_grid: int | None,
    kmax: float | None,
    verbose: bool,
) -> dict[float, float]:
    """Run rdmft_heg once; return {rs: Ec}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    key = f"OptGM@{alpha:.12g};{beta:.12g};{gamma:.12g}"
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

    # Filename stem: replace @ -> _ and ';' -> _ (matches main.cpp filename_for).
    stem = (
        key.replace("@", "_")
        .replace(";", "_")
        .replace("/", "_")
        .replace(" ", "_")
        + ".tsv"
    )
    tsv = out_dir / stem
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


def prescreen_candidate_thetas_phis() -> list[tuple[str, float, float]]:
    """Named (theta, phi) directions on the sphere for cheap coarse RMSE screening."""
    g = guess_spherical_from_weights
    out: list[tuple[str, float, float]] = [
        ("GEO-like (w≈0.25,0.5,0.25)", *g(0.25, 0.5, 0.25)),
        ("equal weights w1=w2=w3", *g(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)),
        ("HF-heavy", *g(0.55, 0.28, 0.17)),
        ("sqrt-heavy (Mueller-like)", *g(0.18, 0.58, 0.24)),
        ("3/4-power-heavy", *g(0.22, 0.28, 0.50)),
        ("octant mix", *g(0.45, 0.45, 0.35)),
        ("polar-ish (large |gamma|)", *g(0.2, 0.2, 0.94)),
    ]
    # A few fixed spherical probes
    out.append(("theta=π/4, phi=0", math.pi / 4.0, 0.0))
    out.append(("theta=π/3, phi=π/2", math.pi / 3.0, math.pi / 2.0))
    return out


def round_on_sphere(a: float, b: float, c: float, ndigits: int) -> tuple[float, float, float]:
    """Round components to ndigits decimals, then re-project onto the unit sphere."""
    a, b, c = round(a, ndigits), round(b, ndigits), round(c, ndigits)
    nrm = math.hypot(math.hypot(a, b), c)
    if nrm < 1e-14:
        s = 1.0 / math.sqrt(3.0)
        return s, s, s
    return a / nrm, b / nrm, c / nrm


def nelder_mead_2d(
    f: Callable[[Any], float],
    x0: Any,
    *,
    maxiter: int = 100,
    xatol: float = 1e-5,
    fatol: float = 1e-8,
    init_step: float = 0.05,
) -> tuple[Any, float, int, bool]:
    """Tiny Nelder-Mead for 2D (no SciPy). Returns (x_best, f_best, nfev, success)."""
    import numpy as np

    alpha_r = 1.0
    gamma_e = 2.0
    rho_c = 0.5
    sigma_s = 0.5

    x0 = np.asarray(x0, dtype=float).reshape(2,)
    # Initial simplex: x0 plus axis steps
    step = init_step
    simplex = np.vstack([x0, x0 + np.array([step, 0.0]), x0 + np.array([0.0, step])])
    vals = np.array([f(simplex[i]) for i in range(3)])
    nfev = 3

    def sort_simplex():
        nonlocal simplex, vals
        order = np.argsort(vals)
        simplex = simplex[order]
        vals = vals[order]

    sort_simplex()
    for _it in range(maxiter):
        best, second, worst = vals[0], vals[1], vals[2]
        simp_size = float(np.max(np.linalg.norm(simplex - simplex[0], axis=1)))
        if simp_size < xatol and abs(worst - best) < fatol:
            break
        if abs(worst - best) < fatol:
            break

        centroid = 0.5 * (simplex[0] + simplex[1])
        xr = centroid + alpha_r * (centroid - simplex[2])
        fr = f(xr)
        nfev += 1

        if fr < vals[0]:
            xe = centroid + gamma_e * (centroid - simplex[2])
            fe = f(xe)
            nfev += 1
            if fe < fr:
                simplex[2], vals[2] = xe, fe
            else:
                simplex[2], vals[2] = xr, fr
        elif fr < vals[1]:
            simplex[2], vals[2] = xr, fr
        else:
            if fr < vals[2]:
                simplex[2], vals[2] = xr, fr
            xc = centroid + rho_c * (centroid - simplex[2])
            fc = f(xc)
            nfev += 1
            if fc < vals[2]:
                simplex[2], vals[2] = xc, fc
            else:
                simplex[1:] = simplex[0] + sigma_s * (simplex[1:] - simplex[0])
                vals[1] = f(simplex[1])
                vals[2] = f(simplex[2])
                nfev += 2

        sort_simplex()

    return simplex[0], float(vals[0]), nfev, True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optimize optGM angles vs PW92 E_c (loose mode by default).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--exe",
        type=Path,
        default=Path("build/rdmft_heg"),
        help="Path to rdmft_heg executable (default: build/rdmft_heg)",
    )
    ap.add_argument(
        "--rs",
        type=str,
        default="",
        help="Comma-separated r_s list (default: built-in benchmark grid)",
    )
    ap.add_argument(
        "--N",
        type=int,
        default=401,
        help="Grid points for the main optimization / final run (Makefile: 401).",
    )
    ap.add_argument(
        "--kmax",
        type=float,
        default=3.0,
        help="k_max in units of k_F for the main run (Makefile: 3).",
    )
    ap.add_argument(
        "--method",
        default="Nelder-Mead",
        choices=("Nelder-Mead", "Powell", "L-BFGS-B"),
        help="scipy.optimize.minimize method (default: Nelder-Mead)",
    )
    ap.add_argument(
        "--maxiter",
        type=int,
        default=22,
        help="Max optimizer iterations (loose default; increase for tighter search).",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=5e-3,
        help="Tolerance for L-BFGS-B (ignored for Nelder-Mead / built-in NM).",
    )
    ap.add_argument(
        "--nm-xatol",
        type=float,
        default=0.07,
        help="Nelder-Mead simplex size stop (radians); ~0.07 is coarse (~4°).",
    )
    ap.add_argument(
        "--nm-fatol",
        type=float,
        default=5e-4,
        help="Nelder-Mead objective spread stop (RMSE units).",
    )
    ap.add_argument(
        "--nm-init-step",
        type=float,
        default=0.12,
        help="Built-in Nelder-Mead initial simplex edge length in (theta, phi).",
    )
    ap.add_argument(
        "--decimals",
        type=int,
        default=1,
        help="Round final (a,b,c) to this many decimals (then renormalize on sphere).",
    )
    ap.add_argument(
        "--no-prescreen",
        action="store_true",
        help="Skip coarse prescreen; start from GEO-like weights only.",
    )
    ap.add_argument(
        "--prescreen-rs",
        type=str,
        default="0.2,0.5,1,2,4",
        help="Comma-separated r_s for cheap prescreen runs.",
    )
    ap.add_argument(
        "--prescreen-N",
        type=int,
        default=101,
        help="Grid points during prescreen (smaller => faster).",
    )
    ap.add_argument(
        "--prescreen-kmax",
        type=float,
        default=2.0,
        help="k_max factor during prescreen (coarser than main --kmax).",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Less progress output (errors still print).",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Also print rdmft_heg stdout on each subprocess run.",
    )
    ap.add_argument(
        "--no-clean-tmp",
        action="store_true",
        help="Keep temporary output directories (for debugging).",
    )
    args = ap.parse_args()

    try:
        import numpy as np
    except ImportError as e:
        print("This script requires numpy.", file=sys.stderr)
        raise SystemExit(1) from e

    try:
        from scipy.optimize import minimize as scipy_minimize  # type: ignore[import-untyped]

        have_scipy = True
    except ImportError:
        scipy_minimize = None
        have_scipy = False

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    repo_root = Path(__file__).resolve().parents[1]
    exe = (repo_root / args.exe).resolve() if not args.exe.is_absolute() else args.exe
    if not exe.is_file():
        print(f"Executable not found: {exe}", file=sys.stderr)
        print("Run `make` from the repository root first.", file=sys.stderr)
        raise SystemExit(1)

    rs_list = (
        [float(x) for x in args.rs.split(",") if x.strip()]
        if args.rs.strip()
        else default_rs()
    )
    ec_ref = {rs: pw92_ec_per_electron(rs) for rs in rs_list}
    prescreen_rs = [float(x) for x in args.prescreen_rs.split(",") if x.strip()]
    ec_ref_pre = {rs: pw92_ec_per_electron(rs) for rs in prescreen_rs}

    tmp_root = Path(tempfile.mkdtemp(prefix="optgm_", dir=repo_root / "build"))
    log("")
    log("=== optGM angle search (RMSE of E_c vs PW92) ===")
    log(f"  executable: {exe}")
    log(f"  main grid:  N={args.N}  kmax={args.kmax}")
    log(f"  main r_s:   {rs_list}")
    log(f"  method:     {args.method}  maxiter={args.maxiter}")
    log(f"  work dir:   {tmp_root}")

    n_eval = [0]

    def objective(vec: np.ndarray, *, phase: str = "opt") -> float:
        theta, phi = float(vec[0]), float(vec[1])
        alpha, beta, gamma = spherical_to_unit(theta, phi)
        run_dir = tmp_root / f"eval_{n_eval[0]:05d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        n_eval[0] += 1
        t0 = time.perf_counter()
        try:
            ec_model = run_rdmft(
                exe,
                alpha,
                beta,
                gamma,
                rs_list,
                run_dir,
                args.N,
                args.kmax,
                args.verbose,
            )
        except (RuntimeError, FileNotFoundError) as exc:
            log(f"  [{phase}] eval #{n_eval[0]} FAILED after {time.perf_counter()-t0:.1f}s: {exc}")
            return 1e3

        val = rmse_ec(ec_model, rs_list, ec_ref)
        dt = time.perf_counter() - t0
        log(
            f"  [{phase}] eval #{n_eval[0]}  "
            f"theta={theta:.4f} phi={phi:.4f}  "
            f"a={alpha:.4f} b={beta:.4f} c={gamma:.4f}  "
            f"RMSE={val:.5e}  ({dt:.1f}s)"
        )
        return val

    # --- Prescreen: cheap rdmft runs to pick a good (theta0, phi0) ---
    if args.no_prescreen:
        theta0, phi0 = guess_spherical_from_weights(0.25, 0.5, 0.25)
        log("\n--- prescreen: skipped (--no-prescreen); start = GEO-like ---")
        log(f"  initial theta={theta0:.4f}  phi={phi0:.4f}")
    else:
        log("\n--- prescreen: coarse rdmft cases for a better starting (theta, phi) ---")
        log(
            f"  prescreen uses N={args.prescreen_N}  kmax={args.prescreen_kmax}  "
            f"r_s={prescreen_rs}  (faster than main run)"
        )
        best_rmse = float("inf")
        best_name = ""
        best_theta, best_phi = guess_spherical_from_weights(0.25, 0.5, 0.25)
        pre_root = tmp_root / "prescreen"
        pre_root.mkdir(parents=True, exist_ok=True)
        for idx, (name, th, ph) in enumerate(prescreen_candidate_thetas_phis()):
            a, b, c = spherical_to_unit(th, ph)
            run_dir = pre_root / f"cand_{idx:02d}"
            t0 = time.perf_counter()
            try:
                ec_m = run_rdmft(
                    exe,
                    a,
                    b,
                    c,
                    prescreen_rs,
                    run_dir,
                    args.prescreen_N,
                    args.prescreen_kmax,
                    args.verbose,
                )
            except (RuntimeError, FileNotFoundError) as exc:
                log(f"  [{idx+1:2d}] {name[:40]:<40}  FAIL  ({exc})")
                continue
            rmse = rmse_ec(ec_m, prescreen_rs, ec_ref_pre)
            dt = time.perf_counter() - t0
            log(
                f"  [{idx+1:2d}] {name[:44]:<44}  RMSE={rmse:.5e}  ({dt:.1f}s)  "
                f"a,b,c=({a:.3f},{b:.3f},{c:.3f})"
            )
            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name
                best_theta, best_phi = th, ph
        theta0, phi0 = best_theta, best_phi
        log(f"\n  prescreen best: {best_name}")
        log(f"  -> starting theta={theta0:.4f}  phi={phi0:.4f}  (coarse RMSE={best_rmse:.5e})")

    x0 = np.array([theta0, phi0], dtype=float)

    log("\n--- main optimization (full r_s, full N/kmax) ---")
    log(
        f"  stopping loosely: ~{args.decimals} decimal(s) on (a,b,c) after round+renorm; "
        f"NM xatol={args.nm_xatol}  fatol={args.nm_fatol}"
    )

    if args.method == "L-BFGS-B" and not have_scipy:
        print("L-BFGS-B requires SciPy; install scipy or use Nelder-Mead/Powell.", file=sys.stderr)
        raise SystemExit(1)

    if have_scipy and args.method in ("Nelder-Mead", "Powell", "L-BFGS-B"):
        bounds = None
        if args.method == "L-BFGS-B":
            bounds = [(0.0, math.pi), (-math.pi, math.pi)]

        opts: dict[str, Any] = {"maxiter": args.maxiter}
        if args.method == "L-BFGS-B":
            opts["ftol"] = args.tol
        # Nelder-Mead tolerances (SciPy >= 1.7); ignored for Powell.
        if args.method == "Nelder-Mead":
            opts["xatol"] = args.nm_xatol
            opts["fatol"] = args.nm_fatol

        res = scipy_minimize(
            lambda v: objective(v, phase="main"),
            x0,
            method=args.method,
            bounds=bounds,
            options=opts,
        )

        theta_opt, phi_opt = float(res.x[0]), float(res.x[1])
        a_opt, b_opt, c_opt = spherical_to_unit(theta_opt, phi_opt)
        final_rmse = float(res.fun)
        nfev = int(res.nfev)
        nit = getattr(res, "nit", None)
        success = bool(res.success)
        msg = getattr(res, "message", "")
    else:
        if args.method != "Nelder-Mead":
            log(
                f"SciPy not available; using built-in Nelder-Mead instead of {args.method}."
            )
        x_best, final_rmse, nfev, success = nelder_mead_2d(
            lambda v: objective(v, phase="main"),
            x0,
            maxiter=args.maxiter,
            xatol=args.nm_xatol,
            fatol=args.nm_fatol,
            init_step=args.nm_init_step,
        )
        theta_opt, phi_opt = float(x_best[0]), float(x_best[1])
        a_opt, b_opt, c_opt = spherical_to_unit(theta_opt, phi_opt)
        nit = None
        msg = "built-in Nelder-Mead"

    a_r, b_r, c_r = round_on_sphere(a_opt, b_opt, c_opt, args.decimals)

    log("\n=== optimization finished ===")
    log(f"  message: {msg}")
    nit_str = f"  iterations={nit}" if nit is not None else ""
    log(f"  success={success}{nit_str}  rdmft_evals={nfev}  final RMSE={final_rmse:.5e}")
    log(
        f"  refined (a,b,c) on unit sphere:  "
        f"{a_opt:.6f}  {b_opt:.6f}  {c_opt:.6f}"
    )
    log(
        f"  weights w1,w2,w3 = a^2,b^2,c^2:  "
        f"{a_opt*a_opt:.5f}  {b_opt*b_opt:.5f}  {c_opt*c_opt:.5f}"
    )
    log(f"  spherical: theta={theta_opt:.5f}  phi={phi_opt:.5f}")
    log(
        f"\n  Recommended CLI (rounded to {args.decimals} dp, renormalized): "
        f"OptGM@{a_r:.{args.decimals}f};{b_r:.{args.decimals}f};{c_r:.{args.decimals}f}"
    )

    # Final detailed run in a persistent folder under build/ (rounded angles)
    final_dir = repo_root / "build" / "optgm_best"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True)
    log(f"\n--- final validation run -> {final_dir} (rounded angles, full grid) ---")
    ec_final = run_rdmft(
        exe,
        a_r,
        b_r,
        c_r,
        rs_list,
        final_dir,
        args.N,
        args.kmax,
        args.verbose,
    )
    log("Per r_s (model vs PW92):")
    for rs in rs_list:
        em = ec_final.get(rs, float("nan"))
        er = ec_ref[rs]
        log(f"  rs={rs:4g}  Ec={em: .8f}  PW92={er: .8f}  diff={em - er: .8e}")
    val_round = rmse_ec(ec_final, rs_list, ec_ref)
    log(f"  RMSE after rounding: {val_round:.5e}")

    if not args.no_clean_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)
        log(f"\nRemoved temp dir {tmp_root}")
    else:
        log(f"\n[debug] temp runs kept under {tmp_root}")


if __name__ == "__main__":
    main()
