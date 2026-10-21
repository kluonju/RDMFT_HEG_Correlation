#!/usr/bin/env python3
"""Optimize ``rs@mu`` (range-separated: SR LDA-c + LR Power) vs PW92 Ec.

Minimizes RMSE of converged ``Ec_per_N`` against PW92 on a chosen r_s grid.
Uses a 1-D μ grid (no SciPy required).  Example::

    python3 scripts/optimize_rs.py --N 201
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def pw92_ec_per_electron(rs: float) -> float:
    a = 0.0310907
    alpha1 = 0.21370
    beta1 = 7.5957
    beta2 = 3.5876
    beta3 = 1.6382
    beta4 = 0.49294
    sqrt_rs = math.sqrt(rs)
    denom = 2.0 * a * (
        beta1 * sqrt_rs + beta2 * rs + beta3 * rs * sqrt_rs + beta4 * rs * rs
    )
    return -2.0 * a * (1.0 + alpha1 * rs) * math.log(1.0 + 1.0 / denom)


def parse_ec_from_tsv(tsv_path: Path, *, converged_only: bool = True) -> dict[float, float]:
    out: dict[float, float] = {}
    with tsv_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            if converged_only and len(parts) >= 10:
                try:
                    if int(float(parts[9])) != 1:
                        continue
                except ValueError:
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


def run_rs(
    exe: Path,
    mu: float,
    rs_list: list[float],
    out_dir: Path,
    n_grid: int,
    kmax: float,
    init_uniform: float,
) -> dict[float, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    key = f"rs@{mu:.12g}"
    cmd = [
        str(exe),
        "--funcs",
        key,
        "--rs",
        ",".join(str(x) for x in rs_list),
        "--out-dir",
        str(out_dir),
        "--force",
        "--N",
        str(n_grid),
        "--kmax",
        str(kmax),
        "--init-uniform",
        str(init_uniform),
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    proc = subprocess.run(
        cmd,
        cwd=str(exe.parent.parent) if exe.parent.name == "build" else None,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"rdmft_heg failed (code {proc.returncode}):\n{proc.stderr}\n{proc.stdout}"
        )
    tsv = out_dir / tsv_stem_for_func_key(key)
    if not tsv.is_file():
        raise FileNotFoundError(f"Expected {tsv}\n{proc.stdout}")
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
    if len(errs) < max(3, len(rs_list) // 2):
        return float("inf")
    return math.sqrt(sum(e * e for e in errs) / len(errs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, default=Path("build/rdmft_heg"))
    ap.add_argument(
        "--rs",
        default="0.01,0.02,0.03,0.05,0.08,0.1,0.2,0.3,0.5,1,2,3,4,5,6,8,10,20,30,50,80,100",
        help="Comma-separated r_s fit grid",
    )
    ap.add_argument("--N", type=int, default=201, help="Grid size during optimization")
    ap.add_argument("--N-final", type=int, default=401, help="Grid size for final report")
    ap.add_argument("--kmax", type=float, default=3.0)
    ap.add_argument("--init-uniform", type=float, default=0.5)
    ap.add_argument("--mu-min", type=float, default=0.1)
    ap.add_argument("--mu-max", type=float, default=2.0)
    ap.add_argument("--mu-steps", type=int, default=20, help="Number of μ grid points")
    ap.add_argument("--decimals", type=int, default=4)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("build/rs_opt"),
        help="Scratch directory for intermediate TSVs",
    )
    args = ap.parse_args()

    if not args.exe.is_file():
        raise SystemExit(f"Missing {args.exe}; run make first")

    rs_list = [float(x) for x in args.rs.split(",") if x.strip()]
    ec_ref = {rs: pw92_ec_per_electron(rs) for rs in rs_list}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="rs_", dir=str(args.out_dir)))

    best: dict[str, float] = {"rmse": float("inf"), "mu": 0.5}
    n_steps = max(2, args.mu_steps)
    mus = [
        args.mu_min + (args.mu_max - args.mu_min) * i / (n_steps - 1)
        for i in range(n_steps)
    ]

    print(f"μ grid ({n_steps} pts in [{args.mu_min:g}, {args.mu_max:g}]):")
    for i, mu in enumerate(mus, start=1):
        t0 = time.time()
        try:
            ec = run_rs(
                args.exe,
                mu,
                rs_list,
                scratch / f"eval_{i:03d}",
                args.N,
                args.kmax,
                args.init_uniform,
            )
            val = rmse_ec(ec, rs_list, ec_ref)
        except Exception as exc:  # noqa: BLE001 — keep scan alive
            print(f"  [{i:02d}] FAIL mu={mu:.4f}: {exc}")
            continue
        dt = time.time() - t0
        print(f"  [{i:02d}] mu={mu:.4f}  RMSE={val:.5e}  ({dt:.1f}s, npts={len(ec)})")
        if val < best["rmse"]:
            best.update(rmse=val, mu=mu)

    mu_r = round(float(best["mu"]), args.decimals)
    print("\nFinal production run:")
    final_dir = args.out_dir / "final"
    ec_final = run_rs(
        args.exe,
        mu_r,
        rs_list,
        final_dir,
        args.N_final,
        args.kmax,
        args.init_uniform,
    )
    final_rmse = rmse_ec(ec_final, rs_list, ec_ref)
    key = f"rs@{mu_r}"
    print(f"  best CLI: --funcs '{key}'")
    print(f"  RMSE (N={args.N_final}): {final_rmse:.5e}")
    for rs in rs_list:
        if rs in ec_final:
            print(
                f"    rs={rs:g}:  Ec={ec_final[rs]:+.6f}  "
                f"QMC={ec_ref[rs]:+.6f}  err={ec_final[rs]-ec_ref[rs]:+.6f}"
            )

    summary = args.out_dir / "best.txt"
    summary.write_text(
        f"funcs={key}\nmu={mu_r}\nrmse={final_rmse:.8e}\nN={args.N_final}\n"
    )
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
