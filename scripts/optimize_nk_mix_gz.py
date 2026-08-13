#!/usr/bin/env python3
"""Find the best post-hoc mix of two RDMFT n(k) curves vs GZ reference.

For each pair of functionals A,B with existing ``data/nk`` TSVs, optimize
``λ ∈ [0,1]`` in

    n_mix(k) = λ n_A(k) + (1-λ) n_B(k)

to minimise the aggregate L1 error vs Gori-Giorgi/Ziesche n(k)
(same objective as ``optimize_nn_gz.py``):

    L = sqrt( mean_rs  [∫_0^{kmax} |n_mix - n_GZ| dx]^2 ),  x = k/k_F.

Also reports an optional per-r_s λ(r_s).

Usage::

    make build/dump_gz_grid
    python3 scripts/optimize_nk_mix_gz.py
    python3 scripts/optimize_nk_mix_gz.py --plot figures/nk_mix_gz.png
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

_SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from nn_gz_common import (  # noqa: E402
    Log,
    aggregate_rmse,
    ensure_gz_targets,
    l1_error_vs_gz,
)
from plot_common import legend_label, pretty_functional_name  # noqa: E402
from plot_nk import load_nk_tsv  # noqa: E402

# Functionals considered for pairing (must have nk TSVs).
CANDIDATE_KEYS = (
    "Mueller",
    "CGA",
    "CHF",
    "Power(0.55)",
    "Power(0.58)",
    "hybopt",
)


def discover_nk_by_key(nk_dir: Path, rs_list: list[float], rs_tol: float) -> dict[str, dict[float, Path]]:
    """Map pretty key -> {rs: path} for converged nk TSVs."""
    out: dict[str, dict[float, Path]] = {}
    for path in sorted(nk_dir.glob("*_rs*.tsv")):
        m = re.search(r"_rs([0-9.]+)\.tsv$", path.name)
        if not m:
            continue
        rs_file = float(m.group(1))
        rs_match = None
        for rs in rs_list:
            if abs(rs_file - rs) <= rs_tol:
                rs_match = rs
                break
        if rs_match is None:
            continue
        d = load_nk_tsv(path)
        if d.get("nk_converged") is False:
            continue
        if d["kF"] is None or d["k"].size == 0:
            continue
        key = pretty_functional_name(d["functional"])
        if key not in CANDIDATE_KEYS:
            continue
        out.setdefault(key, {})[rs_match] = path
    return out


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = load_nk_tsv(path)
    kf = float(d["kF"])
    return d["k"] / kf, d["n"]


def mix_l1(
    xa: np.ndarray,
    na: np.ndarray,
    xb: np.ndarray,
    nb: np.ndarray,
    lam: float,
    gz: dict[float, tuple[np.ndarray, np.ndarray]],
    rs: float,
    kmax: float,
    nk: int,
) -> float:
    x = np.linspace(0.0, kmax, nk)
    n_mix = lam * np.interp(x, xa, na) + (1.0 - lam) * np.interp(x, xb, nb)
    return l1_error_vs_gz(x, n_mix, gz, rs, kmax=kmax, nk=nk)


def fit_pair_global(
    curves: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    gz: dict[float, tuple[np.ndarray, np.ndarray]],
    kmax: float,
    nk: int,
) -> tuple[float, float, dict[float, float]]:
    """Return (λ*, L*, per-rs L1) for a shared λ across all r_s."""

    def objective(lam: float) -> float:
        per: dict[float, float] = {}
        for rs, (xa, na, xb, nb) in curves.items():
            per[rs] = mix_l1(xa, na, xb, nb, lam, gz, rs, kmax, nk)
        return aggregate_rmse(per)

    res = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-5})
    lam = float(res.x)
    per = {
        rs: mix_l1(xa, na, xb, nb, lam, gz, rs, kmax, nk)
        for rs, (xa, na, xb, nb) in curves.items()
    }
    return lam, aggregate_rmse(per), per


def fit_pair_per_rs(
    curves: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    gz: dict[float, tuple[np.ndarray, np.ndarray]],
    kmax: float,
    nk: int,
) -> tuple[dict[float, float], float, dict[float, float]]:
    """Independent λ(r_s); return (λ_map, L*, per-rs L1)."""
    lam_map: dict[float, float] = {}
    per: dict[float, float] = {}
    for rs, (xa, na, xb, nb) in curves.items():
        def obj(lam: float, rs=rs, xa=xa, na=na, xb=xb, nb=nb) -> float:
            return mix_l1(xa, na, xb, nb, lam, gz, rs, kmax, nk)

        res = minimize_scalar(obj, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-5})
        lam_map[rs] = float(res.x)
        per[rs] = float(res.fun)
    return lam_map, aggregate_rmse(per), per


def baseline_l1(
    key: str,
    by_key: dict[str, dict[float, Path]],
    rs_list: list[float],
    gz: dict[float, tuple[np.ndarray, np.ndarray]],
    kmax: float,
    nk: int,
) -> float | None:
    if key not in by_key or any(rs not in by_key[key] for rs in rs_list):
        return None
    per: dict[float, float] = {}
    for rs in rs_list:
        x, n = load_curve(by_key[key][rs])
        per[rs] = l1_error_vs_gz(x, n, gz, rs, kmax=kmax, nk=nk)
    return aggregate_rmse(per)


def plot_best(
    out: Path,
    key_a: str,
    key_b: str,
    lam: float,
    curves: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    gz: dict[float, tuple[np.ndarray, np.ndarray]],
    kmax: float,
) -> None:
    rs_list = sorted(curves)
    n = len(rs_list)
    ncols = 2 if n >= 2 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    for ax, rs in zip(axes.flat, rs_list):
        xa, na, xb, nb = curves[rs]
        x_g, n_g = gz[rs]
        x = np.linspace(0.0, kmax, 401)
        n_mix = lam * np.interp(x, xa, na) + (1.0 - lam) * np.interp(x, xb, nb)
        ax.plot(x_g, n_g, "k-", lw=1.8, label="GZ", zorder=5)
        ax.plot(xa, na, "--", color="#21c7d6", lw=1.2, alpha=0.85, label=legend_label(key_a))
        ax.plot(xb, nb, "--", color="#d62728", lw=1.2, alpha=0.85, label=legend_label(key_b))
        ax.plot(
            x,
            n_mix,
            "-",
            color="#1f77b4",
            lw=2.0,
            label=rf"${lam:.3f}\,$" + legend_label(key_a) + rf"$+{1-lam:.3f}\,$" + legend_label(key_b),
        )
        ax.set_xlim(0.0, kmax)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(r"$k/k_F$")
        ax.set_ylabel(r"$n(k)$")
        ax.set_title(rf"$r_s={rs:g}$", loc="left")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.92)
    for ax in axes.flat[n:]:
        ax.set_axis_off()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nk-dir", type=Path, default=REPO_ROOT / "data" / "nk")
    ap.add_argument("--dump-gz", type=Path, default=REPO_ROOT / "build" / "dump_gz_grid")
    ap.add_argument("--rs", default="0.5,1,5,10")
    ap.add_argument("--kmax", type=float, default=3.0)
    ap.add_argument("--n", type=int, default=401)
    ap.add_argument("--rs-tol", type=float, default=0.0005)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--plot", type=Path, default=REPO_ROOT / "figures" / "nk_mix_gz.png")
    ap.add_argument("--json-out", type=Path, default=REPO_ROOT / "data" / "nk_mix_gz_best.json")
    args = ap.parse_args()

    rs_list = [float(x) for x in args.rs.split(",") if x.strip()]
    if not args.dump_gz.is_file():
        raise SystemExit(f"Missing {args.dump_gz}; run make build/dump_gz_grid")

    log = Log()
    gz = ensure_gz_targets(
        REPO_ROOT / "build" / "nn_data",
        args.dump_gz.resolve(),
        rs_list,
        args.n,
        args.kmax,
        weight_x2=False,
        log=log,
        refresh=False,
    )
    by_key = discover_nk_by_key(args.nk_dir, rs_list, args.rs_tol)
    available = [k for k in CANDIDATE_KEYS if all(rs in by_key.get(k, {}) for rs in rs_list)]
    if len(available) < 2:
        raise SystemExit(
            f"Need at least two functionals with nk at r_s={rs_list}; found {available}"
        )

    print("Baselines (single functional, aggregate L1-RMSE vs GZ):")
    baselines: dict[str, float] = {}
    for key in available:
        L = baseline_l1(key, by_key, rs_list, gz, args.kmax, args.n)
        if L is not None:
            baselines[key] = L
            print(f"  {legend_label(key):16s}  L={L:.5f}")

    results = []
    for a, b in itertools.combinations(available, 2):
        curves = {}
        for rs in rs_list:
            xa, na = load_curve(by_key[a][rs])
            xb, nb = load_curve(by_key[b][rs])
            curves[rs] = (xa, na, xb, nb)
        lam, L, per = fit_pair_global(curves, gz, args.kmax, args.n)
        lam_rs, L_rs, per_rs = fit_pair_per_rs(curves, gz, args.kmax, args.n)
        results.append(
            {
                "A": a,
                "B": b,
                "lambda": lam,
                "L": L,
                "per_rs": {str(rs): per[rs] for rs in rs_list},
                "lambda_per_rs": {str(rs): lam_rs[rs] for rs in rs_list},
                "L_per_rs_opt": L_rs,
                "per_rs_opt": {str(rs): per_rs[rs] for rs in rs_list},
            }
        )

    results.sort(key=lambda r: r["L"])
    print(f"\nTop {min(args.top, len(results))} pairs (shared λ, n_mix = λ A + (1-λ) B):")
    print(f"{'rank':>4}  {'A':16s} {'B':16s}  {'λ':>7}  {'L':>8}  {'L(λ(rs))':>9}")
    for i, r in enumerate(results[: args.top], 1):
        print(
            f"{i:4d}  {legend_label(r['A']):16s} {legend_label(r['B']):16s}  "
            f"{r['lambda']:7.4f}  {r['L']:8.5f}  {r['L_per_rs_opt']:9.5f}"
        )

    best = results[0]
    print("\nBest shared-λ mix:")
    print(
        f"  n_mix = {best['lambda']:.4f} * {legend_label(best['A'])} "
        f"+ {1 - best['lambda']:.4f} * {legend_label(best['B'])}"
    )
    print(f"  L = {best['L']:.5f}  (best single baseline L = {min(baselines.values()):.5f})")
    print("  per-r_s L1:", "  ".join(f"rs={rs:g}:{best['per_rs'][str(rs)]:.4f}" for rs in rs_list))
    print("  per-r_s λ*:", "  ".join(f"rs={rs:g}:{best['lambda_per_rs'][str(rs)]:.3f}" for rs in rs_list))

    payload = {
        "rs": rs_list,
        "kmax": args.kmax,
        "nk": args.n,
        "baselines": baselines,
        "best": best,
        "top": results[: args.top],
        "note": "Post-hoc occupation mix (not an in-solver kernel). "
        "n_mix = λ A + (1-λ) B. L = sqrt(mean L1(rs)^2) vs GZ on x=k/k_F in [0,kmax].",
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.json_out}")

    if args.plot:
        curves = {}
        for rs in rs_list:
            xa, na = load_curve(by_key[best["A"]][rs])
            xb, nb = load_curve(by_key[best["B"]][rs])
            curves[rs] = (xa, na, xb, nb)
        plot_best(
            args.plot,
            best["A"],
            best["B"],
            best["lambda"],
            curves,
            gz,
            args.kmax,
        )


if __name__ == "__main__":
    main()
