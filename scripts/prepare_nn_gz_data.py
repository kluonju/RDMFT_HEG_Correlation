#!/usr/bin/env python3
"""Prepare cached objective data for NN-vs-GZ kernel training.

Writes under ``build/nn_data/`` (by default):

  gz_targets.npz   — GZ reference n(k/kF, r_s) on a common k grid
  manifest.json    — r_s list, kmax, grid size, loss weight
  power_sweep.json — optional RMSE vs GZ for Power(alpha) baselines
  baselines/       — optional n(k) TSVs from named functionals

The optimizer (``optimize_nn_gz.py``) loads ``gz_targets.npz`` so it does not
call ``dump_gz_grid`` on every run.  **Full training still runs RDMFT SCF** for
each NN weight vector; only the GZ *targets* are precomputed here.

Usage::

    make prepare-nn-data
    # or
    python3 scripts/prepare_nn_gz_data.py --data-dir build/nn_data --power-sweep
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from nn_gz_common import (  # noqa: E402
    DEFAULT_RS,
    GZ_TARGETS_FILE,
    Log,
    MANIFEST_FILE,
    POWER_SWEEP_FILE,
    REPO_ROOT,
    aggregate_rmse,
    ensure_gz_targets,
    load_gz_targets,
    load_nk_tsv,
    rmse_vs_gz,
)


def run_power_sweep(
    exe: Path,
    data_dir: Path,
    rs_list: list[float],
    alphas: list[float],
    *,
    n_grid: int,
    kmax: float,
    gz_n: int,
    weight_x2: bool,
    init_uniform: float,
    log: Log,
) -> dict[str, Any]:
    gz = load_gz_targets(data_dir / GZ_TARGETS_FILE)
    results: list[dict[str, Any]] = []
    work = data_dir / "_power_sweep_work"
    work.mkdir(parents=True, exist_ok=True)

    for alpha in alphas:
        label = f"Power@{alpha:g}"
        log.info(f"\n>>> power sweep: {label}")
        key = f"Power@{alpha}"
        cmd = [
            str(exe.resolve()),
            "--funcs",
            key,
            "--rs",
            ",".join(f"{r:g}" for r in rs_list),
            "--N",
            str(n_grid),
            "--kmax",
            str(kmax),
            "--nk-out",
            str(work / "nk"),
            "--force",
            "--out-dir",
            str(work / "energy"),
            "--init-uniform",
            str(init_uniform),
        ]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            log.info(f"  {label} FAILED: {proc.stderr[:500]}")
            results.append({"alpha": alpha, "rmse": None, "per_rs": {}})
            continue
        stem = key.replace("@", "_")
        per_rs: dict[str, float] = {}
        for rs in rs_list:
            nk_path = work / "nk" / f"{stem}_rs{rs:.4f}.tsv"
            if not nk_path.is_file():
                per_rs[str(rs)] = float("inf")
                continue
            k_m, n_m, conv = load_nk_tsv(nk_path)
            if conv is False:
                per_rs[str(rs)] = float("inf")
                continue
            per_rs[str(rs)] = rmse_vs_gz(
                k_m, n_m, gz, rs, kmax=kmax, nk=gz_n, weight_x2=weight_x2
            )
        rmse = aggregate_rmse({float(k): v for k, v in per_rs.items()})
        log.info(f"  {label}: RMSE={rmse:.6f}  ({time.time() - t0:.1f}s)")
        results.append({"alpha": alpha, "rmse": rmse, "per_rs": per_rs})

    finite = [
        r for r in results if r["rmse"] is not None and math.isfinite(r["rmse"])
    ]
    best = min(finite, key=lambda r: r["rmse"]) if finite else None

    out: dict[str, Any] = {
        "rs": rs_list,
        "alphas": alphas,
        "n_grid": n_grid,
        "kmax": kmax,
        "results": results,
        "best_alpha": (best["alpha"] if best else None),
        "best_rmse": (best["rmse"] if best else None),
    }
    return out


def run_baselines(
    exe: Path,
    data_dir: Path,
    funcs: list[str],
    rs_list: list[float],
    *,
    n_grid: int,
    kmax: float,
    init_uniform: float,
    log: Log,
) -> None:
    out = data_dir / "baselines" / "nk"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(exe.resolve()),
        "--funcs",
        ",".join(funcs),
        "--rs",
        ",".join(f"{r:g}" for r in rs_list),
        "--N",
        str(n_grid),
        "--kmax",
        str(kmax),
        "--nk-out",
        str(out),
        "--force",
        "--out-dir",
        str(data_dir / "baselines" / "energy"),
        "--init-uniform",
        str(init_uniform),
    ]
    log.info(f"Baseline n(k) export: {funcs}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "build" / "nn_data")
    ap.add_argument("--dump-gz", type=Path, default=REPO_ROOT / "build" / "dump_gz_grid")
    ap.add_argument("--exe", type=Path, default=REPO_ROOT / "build" / "rdmft_heg")
    ap.add_argument("--rs", default=",".join(f"{r:g}" for r in DEFAULT_RS))
    ap.add_argument("--kmax", type=float, default=3.0)
    ap.add_argument("--gz-n", type=int, default=401)
    ap.add_argument("--uniform-weight", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="Rebuild gz_targets.npz")
    ap.add_argument(
        "--power-sweep",
        action="store_true",
        help="Run Power(alpha) SCF sweep and save RMSE vs GZ (slow).",
    )
    ap.add_argument(
        "--alphas",
        default="0.48,0.50,0.52,0.55,0.58,0.60",
        help="Comma-separated Power exponents for --power-sweep.",
    )
    ap.add_argument("--power-N", type=int, default=401, help="k-grid for power sweep.")
    ap.add_argument(
        "--baselines",
        default="",
        help="Optional comma-separated functionals for n(k) export (e.g. Mueller,Power@0.55).",
    )
    ap.add_argument("--init-uniform", type=float, default=0.5)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rs_list = [float(x) for x in args.rs.split(",") if x.strip()]
    log = Log(quiet=args.quiet, verbose=args.verbose)
    weight_x2 = not args.uniform_weight

    if not args.dump_gz.is_file():
        raise SystemExit(f"Build dump_gz_grid first: {args.dump_gz}")

    log.info("=" * 60)
    log.info("Prepare NN / GZ objective data")
    log.info("=" * 60)
    log.info(f"  data_dir : {args.data_dir.resolve()}")
    log.info(f"  r_s      : {rs_list}")
    log.info(f"  kmax     : {args.kmax}  gz_n={args.gz_n}")
    log.info("")

    ensure_gz_targets(
        args.data_dir,
        args.dump_gz,
        rs_list,
        args.gz_n,
        args.kmax,
        weight_x2,
        log,
        refresh=args.refresh,
    )

    if args.power_sweep:
        if not args.exe.is_file():
            raise SystemExit(f"Build rdmft_heg for --power-sweep: {args.exe}")
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        log.info("\nPower functional sweep vs cached GZ targets...")
        sweep = run_power_sweep(
            args.exe,
            args.data_dir,
            rs_list,
            alphas,
            n_grid=args.power_N,
            kmax=args.kmax,
            gz_n=args.gz_n,
            weight_x2=weight_x2,
            init_uniform=args.init_uniform,
            log=log,
        )
        path = args.data_dir / POWER_SWEEP_FILE
        path.write_text(json.dumps(sweep, indent=2))
        log.info(f"Wrote {path}")
        if sweep.get("best_alpha") is not None:
            log.info(
                f"  best Power alpha ≈ {sweep['best_alpha']}  "
                f"(RMSE={sweep['best_rmse']:.6f})"
            )

    if args.baselines.strip():
        if not args.exe.is_file():
            raise SystemExit(f"Build rdmft_heg for --baselines: {args.exe}")
        funcs = [x.strip() for x in args.baselines.split(",") if x.strip()]
        run_baselines(
            args.exe,
            args.data_dir,
            funcs,
            rs_list,
            n_grid=args.power_N,
            kmax=args.kmax,
            init_uniform=args.init_uniform,
            log=log,
        )
        log.info(f"  baselines under {args.data_dir / 'baselines'}")

    log.info("\nDone.  Train with:")
    log.info(f"  python3 scripts/optimize_nn_gz.py --data-dir {args.data_dir}")


if __name__ == "__main__":
    main()
