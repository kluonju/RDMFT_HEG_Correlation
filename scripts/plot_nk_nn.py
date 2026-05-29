#!/usr/bin/env python3
"""Overlay NN RDMFT n(k) vs GZ reference (Fig. 6 style).

Expects ``build/nn_best/nk/*`` from ``optimize_nn_gz.py`` or a manual run with
``--nk-out`` and ``NN@build/nn_best/model.json``.

Usage::

    make plot-nk-nn
    python3 scripts/plot_nk_nn.py --nk-dir build/nn_best/nk --out figures/nk_nn_vs_gz.png
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from plot_nk import load_nk_tsv  # noqa: E402


def run_dump(binary: Path, rs_list: list[float], nk: int, kmax: float) -> np.ndarray:
    rs_arg = ",".join(f"{r:g}" for r in rs_list)
    out = subprocess.run(
        [str(binary), "--rs", rs_arg, "--n", str(nk), "--kmax", str(kmax)],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    ).stdout
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return np.array(rows, dtype=float)


def find_nn_tsvs(nk_dir: Path) -> dict[float, Path]:
    """Map r_s -> nk TSV for NN exports (stem contains 'NN_')."""
    by_rs: dict[float, Path] = {}
    for path in sorted(nk_dir.glob("*_rs*.tsv")):
        if "NN_" not in path.name:
            continue
        m = re.search(r"_rs([0-9.]+)\.tsv$", path.name)
        if not m:
            continue
        rs = float(m.group(1))
        by_rs[rs] = path
    return by_rs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rs", default="0.2,0.5,1,2,3,5,7,10,15")
    ap.add_argument("--nk-dir", type=Path, default=REPO_ROOT / "build" / "nn_best" / "nk")
    ap.add_argument("--dump-gz", type=Path, default=REPO_ROOT / "build" / "dump_gz_grid")
    ap.add_argument("--n", type=int, default=401)
    ap.add_argument("--kmax", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "figures" / "nk_nn_vs_gz.png")
    args = ap.parse_args()

    rs_list = [float(x) for x in args.rs.split(",") if x.strip()]
    if not args.dump_gz.is_file():
        raise SystemExit(f"Missing {args.dump_gz}; run make build/dump_gz_grid")

    gz = run_dump(args.dump_gz.resolve(), rs_list, args.n, args.kmax)
    nk_map = find_nn_tsvs(args.nk_dir)
    if not nk_map:
        raise SystemExit(
            f"No NN nk TSVs under {args.nk_dir}.  Run optimize_nn_gz.py or "
            f"rdmft_heg with --nk-out and NN@... first."
        )

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(rs_list)))
    for rs, c in zip(rs_list, colors):
        mask = np.isclose(gz[:, 0], rs)
        x_gz = gz[mask, 1]
        n_gz = gz[mask, 2]
        ax.plot(x_gz, n_gz, color=c, lw=1.2, ls="--", alpha=0.85, label=f"$r_s={rs:g}$ GZ")

        path = nk_map.get(rs)
        if path is None:
            for k, p in nk_map.items():
                if abs(k - rs) < 1e-6:
                    path = p
                    break
        if path is None:
            continue
        data = load_nk_tsv(path)
        kf = data["kF"] or 1.0
        x = data["k"] / kf if kf > 0 else data["k"]
        ax.plot(x, data["n"], color=c, lw=1.8, label=f"$r_s={rs:g}$ NN")

    ax.set_xlim(0.0, args.kmax)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"$k/k_F$")
    ax.set_ylabel(r"$n(k)$")
    ax.set_title("NN separable kernel vs GZ reference")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
