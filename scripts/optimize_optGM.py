#!/usr/bin/env python3
"""Optimize optGM mixing angles (alpha, beta, gamma) on the unit sphere.

The C++ functional ``OptGM@a;b;c`` (semicolons: commas are reserved in ``--funcs``)
normalizes (a,b,c) so that a^2+b^2+c^2=1 and
uses kernel weights w1=a^2, w2=b^2, w3=c^2 on the three GEO channels:

    K = w1 * n_i n_j + w2 * (n_i n_j)^{1/2} + w3 * (n_i n_j)^{3/4}.

This script minimizes the RMSE of RDMFT correlation energy E_c vs PW92 (QMC
parameterization) over a list of r_s values by varying two spherical angles.

Requires: numpy.  SciPy is optional (used if installed for L-BFGS-B / Powell;
otherwise a small built-in Nelder-Mead is used).
"""
from __future__ import annotations

import argparse
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


def nelder_mead_2d(
    f: Callable[[Any], float],
    x0: Any,
    *,
    maxiter: int = 100,
    xatol: float = 1e-5,
    fatol: float = 1e-8,
) -> tuple[Any, float, int, bool]:
    """Tiny Nelder-Mead for 2D (no SciPy). Returns (x_best, f_best, nfev, success)."""
    import numpy as np

    alpha_r = 1.0
    gamma_e = 2.0
    rho_c = 0.5
    sigma_s = 0.5

    x0 = np.asarray(x0, dtype=float).reshape(2,)
    # Initial simplex: x0 plus axis steps
    step = 0.05
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
    ap = argparse.ArgumentParser(description="Optimize optGM angles vs PW92 E_c.")
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
        help="Grid points (default 401, same as Makefile run/geo).",
    )
    ap.add_argument(
        "--kmax",
        type=float,
        default=6.0,
        help="k_max in units of k_F (default 6, same as Makefile run/geo).",
    )
    ap.add_argument(
        "--method",
        default="Nelder-Mead",
        choices=("Nelder-Mead", "Powell", "L-BFGS-B"),
        help="scipy.optimize.minimize method (default: Nelder-Mead)",
    )
    ap.add_argument("--maxiter", type=int, default=60, help="Max optimizer iterations.")
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Tolerance for L-BFGS-B (ignored for derivative-free methods).",
    )
    ap.add_argument("--verbose", action="store_true", help="Print rdmft_heg stdout.")
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

    tmp_root = Path(tempfile.mkdtemp(prefix="optgm_", dir=repo_root / "build"))

    n_eval = [0]

    def objective(vec: np.ndarray) -> float:
        theta, phi = float(vec[0]), float(vec[1])
        alpha, beta, gamma = spherical_to_unit(theta, phi)
        run_dir = tmp_root / f"eval_{n_eval[0]:05d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        n_eval[0] += 1
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
            if args.verbose:
                print(f"[warn] eval failed: {exc}", file=sys.stderr)
            return 1e3

        val = rmse_ec(ec_model, rs_list, ec_ref)
        if args.verbose:
            print(
                f"eval {n_eval[0]-1}: theta={theta:.5f} phi={phi:.5f} "
                f"a={alpha:.5f} b={beta:.5f} c={gamma:.5f} RMSE={val:.6e}"
            )
        return val

    # Start near GEO-like weights (1/4, 1/2, 1/4) implemented in this codebase.
    theta0, phi0 = guess_spherical_from_weights(0.25, 0.5, 0.25)
    x0 = np.array([theta0, phi0], dtype=float)

    if args.method == "L-BFGS-B" and not have_scipy:
        print("L-BFGS-B requires SciPy; install scipy or use Nelder-Mead/Powell.", file=sys.stderr)
        raise SystemExit(1)

    if have_scipy and args.method in ("Nelder-Mead", "Powell", "L-BFGS-B"):
        bounds = None
        if args.method == "L-BFGS-B":
            bounds = [(0.0, math.pi), (-math.pi, math.pi)]

        opts = {"maxiter": args.maxiter}
        if args.method == "L-BFGS-B":
            opts["ftol"] = args.tol

        res = scipy_minimize(
            objective,
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
            print(
                f"SciPy not available; using built-in Nelder-Mead instead of {args.method}.",
                file=sys.stderr,
            )
        x_best, final_rmse, nfev, success = nelder_mead_2d(
            objective, x0, maxiter=args.maxiter
        )
        theta_opt, phi_opt = float(x_best[0]), float(x_best[1])
        a_opt, b_opt, c_opt = spherical_to_unit(theta_opt, phi_opt)
        nit = None
        msg = "built-in Nelder-Mead"

    print("Optimization finished:", msg)
    nit_str = f"  nit={nit}" if nit is not None else ""
    print(f"  success={success}{nit_str}  nfev={nfev}  final RMSE={final_rmse:.8e}")
    print(f"  optimal angles (unit sphere): alpha={a_opt:.8f}  beta={b_opt:.8f}  gamma={c_opt:.8f}")
    print(f"  weights w=a^2,b^2,c^2: {a_opt*a_opt:.8f}  {b_opt*b_opt:.8f}  {c_opt*c_opt:.8f}")
    print(f"  spherical params: theta={theta_opt:.8f}  phi={phi_opt:.8f}")
    print(f"  OptGM CLI key: OptGM@{a_opt:.10f};{b_opt:.10f};{c_opt:.10f}")

    # Final detailed run in a persistent folder under build/
    final_dir = repo_root / "build" / "optgm_best"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True)
    ec_final = run_rdmft(
        exe,
        a_opt,
        b_opt,
        c_opt,
        rs_list,
        final_dir,
        args.N,
        args.kmax,
        False,
    )
    print("\nPer r_s (model vs PW92):")
    for rs in rs_list:
        em = ec_final.get(rs, float("nan"))
        er = ec_ref[rs]
        print(f"  rs={rs:4g}  Ec={em: .8f}  PW92={er: .8f}  diff={em - er: .8e}")

    if not args.no_clean_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)
    else:
        print(f"\n[debug] temp runs kept under {tmp_root}")


if __name__ == "__main__":
    main()
