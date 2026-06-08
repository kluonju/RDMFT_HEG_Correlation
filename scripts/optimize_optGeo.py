#!/usr/bin/env python3
"""Optimize OptGeo sigmoid correlation (w, k) vs PW92 QMC E_c.

The C++ functional ``OptGeo@w;k`` uses

    x_ij = (n_i - 1/2)^2 (n_j - 1/2)^2,

    K = n_i n_j + w * ( 2 sigma(k x_ij) - 1 ),

with full Hartree-Fock (first term) plus a sigmoid correlation that is zero
when either occupation is 1/2 and approaches +w at large x_ij (k > 0).
Parameters w (amplitude) and k (steepness) must be non-negative.
Semicolons separate w and k in ``--funcs`` (commas are reserved).

Flow (default):
  1) **Prescreen** — cheap grid over named (w, k) pairs; pick lowest RMSE vs PW92.
  2) **Main** — Nelder-Mead / Powell / L-BFGS-B on the full r_s list.
  3) **Final** — one full ``rdmft_heg`` with rounded (w, k) under ``build/optgeo_best/``.

Requires NumPy; SciPy recommended for Powell / L-BFGS-B.
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
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def run_rdmft(
    exe: Path,
    w1: float,
    w2: float,
    rs_list: list[float],
    out_dir: Path,
    n_grid: int | None,
    kmax: float | None,
    verbose: bool,
) -> dict[float, float]:
    """Run rdmft_heg once; return {rs: Ec}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    key = f"OptGeo@{w1:.12g};{w2:.12g}"
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
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"rdmft_heg failed (code {proc.returncode}):\n{proc.stderr}\n{proc.stdout}"
        )
    if verbose and proc.stdout:
        print(proc.stdout, end="")

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


def prescreen_candidates() -> list[tuple[str, float, float]]:
    """Named (w, k) starting points for coarse RMSE screening (w, k >= 0)."""
    return [
        ("HF only (0,0)", 0.0, 0.0),
        ("w=0.5,k=1", 0.5, 1.0),
        ("w=1,k=1", 1.0, 1.0),
        ("w=1,k=5", 1.0, 5.0),
        ("w=1,k=10", 1.0, 10.0),
        ("w=2,k=2", 2.0, 2.0),
        ("w=0.5,k=0.5", 0.5, 0.5),
        ("w=1,k=0.1", 1.0, 0.1),
        ("w=5,k=1", 5.0, 1.0),
        ("w=2,k=20", 2.0, 20.0),
        ("w=5,k=10", 5.0, 10.0),
        ("w=10,k=5", 10.0, 5.0),
    ]


def clamp_wk(
    w: float, k: float, *, w_min: float, w_max: float
) -> tuple[float, float]:
    lo = max(0.0, w_min)
    hi = w_max
    return max(lo, min(hi, w)), max(lo, min(hi, k))


def round_weights(
    w1: float, w2: float, ndigits: int, *, w_min: float, w_max: float
) -> tuple[float, float]:
    w, k = clamp_wk(w1, w2, w_min=w_min, w_max=w_max)
    return round(w, ndigits), round(k, ndigits)


def nelder_mead_2d(
    f: Callable[[Any], float],
    x0: Any,
    *,
    maxiter: int = 100,
    xatol: float = 1e-5,
    fatol: float = 1e-8,
    init_step: float = 0.05,
) -> tuple[Any, float, int, bool]:
    """Nelder-Mead in 2D when SciPy is unavailable."""
    import numpy as np

    alpha_r = 1.0
    gamma_e = 2.0
    rho_c = 0.5
    sigma_s = 0.5

    x0 = np.asarray(x0, dtype=float).reshape(2,)
    step = init_step
    simplex = np.vstack([x0, x0 + np.array([step, 0.0]), x0 + np.array([0.0, step])])
    vals = np.array([f(simplex[i]) for i in range(3)])
    nfev = 3

    def sort_simplex() -> None:
        nonlocal simplex, vals
        order = np.argsort(vals)
        simplex = simplex[order]
        vals = vals[order]

    sort_simplex()
    for _it in range(maxiter):
        best, _, worst = vals[0], vals[1], vals[2]
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
        description="Optimize OptGeo (w,k) vs PW92 E_c.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--exe", type=Path, default=Path("build/rdmft_heg"))
    ap.add_argument("--rs", type=str, default="", help="Comma-separated r_s list")
    ap.add_argument("--N", type=int, default=401)
    ap.add_argument("--kmax", type=float, default=3.0)
    ap.add_argument(
        "--method",
        default="Nelder-Mead",
        choices=("Nelder-Mead", "Powell", "L-BFGS-B"),
    )
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--tol", type=float, default=5e-3, help="L-BFGS-B ftol")
    ap.add_argument(
        "--nm-xatol",
        type=float,
        default=0.05,
        help="Nelder-Mead simplex size stop in (w,k) space",
    )
    ap.add_argument(
        "--nm-fatol",
        type=float,
        default=5e-4,
        help="Nelder-Mead objective spread stop (RMSE Ha)",
    )
    ap.add_argument(
        "--nm-init-step",
        type=float,
        default=0.25,
        help="Initial Nelder-Mead simplex edge in weight space",
    )
    ap.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Round final (w,k) to this many decimals",
    )
    ap.add_argument(
        "--w-min",
        type=float,
        default=0.0,
        help="Lower bound for w, k (>= 0; L-BFGS-B only)",
    )
    ap.add_argument(
        "--w-max",
        type=float,
        default=20.0,
        help="Upper bound for w, k in L-BFGS-B",
    )
    ap.add_argument("--no-prescreen", action="store_true")
    ap.add_argument("--prescreen-rs", type=str, default="0.2,1.0,5.0,10.0")
    ap.add_argument("--prescreen-N", type=int, default=101)
    ap.add_argument("--prescreen-kmax", type=float, default=2.0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-clean-tmp", action="store_true")
    args = ap.parse_args()

    try:
        import numpy as np
    except ImportError as e:
        print("This script requires numpy.", file=sys.stderr)
        raise SystemExit(1) from e

    try:
        from scipy.optimize import minimize as scipy_minimize  # type: ignore

        have_scipy = True
    except ImportError:
        scipy_minimize = None
        have_scipy = False

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    exe = (REPO_ROOT / args.exe).resolve() if not args.exe.is_absolute() else args.exe
    if not exe.is_file():
        print(f"Executable not found: {exe}", file=sys.stderr)
        raise SystemExit(1)

    rs_list = (
        [float(x) for x in args.rs.split(",") if x.strip()]
        if args.rs.strip()
        else default_rs()
    )
    ec_ref = {rs: pw92_ec_per_electron(rs) for rs in rs_list}
    prescreen_rs = [float(x) for x in args.prescreen_rs.split(",") if x.strip()]
    ec_ref_pre = {rs: pw92_ec_per_electron(rs) for rs in prescreen_rs}

    tmp_root = Path(tempfile.mkdtemp(prefix="optgeo_", dir=REPO_ROOT / "build"))
    log("")
    log("=== OptGeo (w,k) search — RMSE of E_c vs PW92 ===")
    log(f"  executable: {exe}")
    log(f"  main grid:  N={args.N}  kmax={args.kmax}")
    log(f"  main r_s:   {rs_list}")
    log(f"  method:     {args.method}  maxiter={args.maxiter}")
    log(f"  bounds:     w,k in [{max(0.0, args.w_min)}, {args.w_max}]")
    log(f"  work dir:   {tmp_root}")

    n_eval = [0]

    def objective(vec: np.ndarray, *, phase: str = "opt") -> float:
        w1, w2 = clamp_wk(float(vec[0]), float(vec[1]), w_min=args.w_min, w_max=args.w_max)
        run_dir = tmp_root / f"eval_{n_eval[0]:05d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        n_eval[0] += 1
        t0 = time.perf_counter()
        try:
            ec_model = run_rdmft(
                exe,
                w1,
                w2,
                rs_list,
                run_dir,
                args.N,
                args.kmax,
                args.verbose,
            )
        except (RuntimeError, FileNotFoundError) as exc:
            log(f"  [{phase}] eval #{n_eval[0]} FAILED ({time.perf_counter()-t0:.1f}s): {exc}")
            return 1e3

        val = rmse_ec(ec_model, rs_list, ec_ref)
        dt = time.perf_counter() - t0
        log(
            f"  [{phase}] eval #{n_eval[0]}  "
            f"w={w1:.4g}  k={w2:.4g}  RMSE={val:.5e}  ({dt:.1f}s)"
        )
        return val

    if args.no_prescreen:
        w1_0, w2_0 = 0.0, 0.0
        log("\n--- prescreen: skipped; start = HF (0,0) ---")
    else:
        log("\n--- prescreen: coarse (w,k) cases ---")
        log(
            f"  N={args.prescreen_N}  kmax={args.prescreen_kmax}  r_s={prescreen_rs}"
        )
        best_rmse = float("inf")
        best_name = ""
        w1_0, w2_0 = 0.0, 0.0
        pre_root = tmp_root / "prescreen"
        pre_root.mkdir(parents=True, exist_ok=True)
        for idx, (name, w1, w2) in enumerate(prescreen_candidates()):
            run_dir = pre_root / f"cand_{idx:02d}"
            t0 = time.perf_counter()
            try:
                ec_m = run_rdmft(
                    exe,
                    w1,
                    w2,
                    prescreen_rs,
                    run_dir,
                    args.prescreen_N,
                    args.prescreen_kmax,
                    args.verbose,
                )
            except (RuntimeError, FileNotFoundError) as exc:
                log(f"  [{idx+1:2d}] {name:<36}  FAIL  ({exc})")
                continue
            rmse = rmse_ec(ec_m, prescreen_rs, ec_ref_pre)
            dt = time.perf_counter() - t0
            log(f"  [{idx+1:2d}] {name:<36}  RMSE={rmse:.5e}  ({dt:.1f}s)")
            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name
                w1_0, w2_0 = w1, w2
        log(f"\n  prescreen best: {best_name}")
        log(f"  -> starting w={w1_0:.4g}  k={w2_0:.4g}  (coarse RMSE={best_rmse:.5e})")

    x0 = np.array([w1_0, w2_0], dtype=float)

    log("\n--- main optimization (full r_s, full N/kmax) ---")

    if args.method == "L-BFGS-B" and not have_scipy:
        print("L-BFGS-B requires SciPy.", file=sys.stderr)
        raise SystemExit(1)

    bounds = [(max(0.0, args.w_min), args.w_max), (max(0.0, args.w_min), args.w_max)]

    if have_scipy and args.method in ("Nelder-Mead", "Powell", "L-BFGS-B"):
        opts: dict[str, Any] = {"maxiter": args.maxiter}
        if args.method == "L-BFGS-B":
            opts["ftol"] = args.tol
        if args.method == "Nelder-Mead":
            opts["xatol"] = args.nm_xatol
            opts["fatol"] = args.nm_fatol

        res = scipy_minimize(
            lambda v: objective(v, phase="main"),
            x0,
            method=args.method,
            bounds=bounds if args.method == "L-BFGS-B" else None,
            options=opts,
        )
        w1_opt, w2_opt = clamp_wk(
            float(res.x[0]), float(res.x[1]), w_min=args.w_min, w_max=args.w_max
        )
        final_rmse = float(res.fun)
        nfev = int(res.nfev)
        nit = getattr(res, "nit", None)
        success = bool(res.success)
        msg = getattr(res, "message", "")
    else:
        if args.method != "Nelder-Mead":
            log(f"SciPy unavailable; built-in Nelder-Mead instead of {args.method}.")
        x_best, final_rmse, nfev, success = nelder_mead_2d(
            lambda v: objective(v, phase="main"),
            x0,
            maxiter=args.maxiter,
            xatol=args.nm_xatol,
            fatol=args.nm_fatol,
            init_step=args.nm_init_step,
        )
        w1_opt, w2_opt = clamp_wk(
            float(x_best[0]), float(x_best[1]), w_min=args.w_min, w_max=args.w_max
        )
        nit = None
        msg = "built-in Nelder-Mead"

    w1_r, w2_r = round_weights(
        w1_opt, w2_opt, args.decimals, w_min=args.w_min, w_max=args.w_max
    )

    log("\n=== optimization finished ===")
    log(f"  message: {msg}")
    nit_str = f"  iterations={nit}" if nit is not None else ""
    log(f"  success={success}{nit_str}  rdmft_evals={nfev}  final RMSE={final_rmse:.5e}")
    log(f"  refined weights:  w={w1_opt:.6g}  k={w2_opt:.6g}")
    log(f"\n  Recommended CLI:  OptGeo@{w1_r:.{args.decimals}f};{w2_r:.{args.decimals}f}")

    final_dir = REPO_ROOT / "build" / "optgeo_best"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True)
    log(f"\n--- final validation -> {final_dir} ---")
    ec_final = run_rdmft(
        exe,
        w1_r,
        w2_r,
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

    log_path = final_dir / "optimize.log"
    with log_path.open("w") as f:
        f.write(f"w={w1_opt}\n")
        f.write(f"k={w2_opt}\n")
        f.write(f"w_round={w1_r}\n")
        f.write(f"k_round={w2_r}\n")
        f.write(f"rmse={final_rmse}\n")
        f.write(f"rmse_round={val_round}\n")
        f.write(f"success={success}\n")
        f.write(f"method={args.method}\n")
        f.write(f"rs={rs_list}\n")
    log(f"  log: {log_path}")

    if not args.no_clean_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)
    else:
        log(f"\n[debug] temp runs kept under {tmp_root}")


if __name__ == "__main__":
    main()
