#!/usr/bin/env python3
"""Plot the Gori-Giorgi/Ziesche momentum distribution n(k/kF, rs).

Calls the small driver ``build/dump_gz_grid`` (built from
``tools/dump_gz_grid.cpp``) which prints ``rs k/kF n(k/kF, rs)`` rows
for the requested rs list and a dense k/kF grid.  This mirrors Fig. 6
(upper panel) of P. Gori-Giorgi and P. Ziesche, *Phys. Rev. B* 66,
235116 (2002).

Usage::

    make build/dump_gz_grid
    python3 scripts/plot_gz.py [--rs 1,2,5,10] [--out figures/nk_gz.png]

The default rs list ``0.2,0.5,1,2,3,5,7,10,20, 50`` extends Fig. 6 with low-density
points and keeps the paper's rs values.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run_dump(binary: Path, rs_list: list[float], nk: int) -> np.ndarray:
    rs_arg = ",".join(f"{r:g}" for r in rs_list)
    out = subprocess.run(
        [str(binary), "--rs", rs_arg, "--n", str(nk)],
        check=True, capture_output=True, text=True,
    ).stdout
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return np.array(rows, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # ap.add_argument("--rs", default="0.2,0.5,1,2,3,5,7,10,20,50",
    # help="Comma-separated rs values (default 0.2,0.5,1,2,3,5,7,10,20,50)")

    ap.add_argument("--rs", default="0.2,0.5,1,2,3,5,7,10,15",
                    help="Comma-separated rs values (default 0.2,0.5,1,2,3,5,7,10,15)")
    ap.add_argument("--bin", default="build/dump_gz_grid",
                    help="Path to dump_gz_grid binary (built from tools/dump_gz_grid.cpp)")
    ap.add_argument("--n", type=int, default=401,
                    help="Number of k/kF grid points on [0, kmax]")
    ap.add_argument("--kmax", type=float, default=3.0,
                    help="Max k/kF to plot")
    ap.add_argument("--out", type=Path, default=Path("figures/nk_gz.png"))
    args = ap.parse_args()

    rs_list = [float(x) for x in args.rs.split(",") if x.strip()]
    binary = Path(args.bin).resolve()
    if not binary.exists():
        raise SystemExit(
            f"Driver binary not found: {binary}. Build it first via\n"
            f"  g++ -O2 -std=c++17 -Iinclude tools/dump_gz_grid.cpp \\\n"
            f"      src/MomentumDistributionGZ.cpp -o {binary}\n"
            f"or, from the workspace root, `make build/dump_gz_grid` "
            f"(or `make plot-gz`)."
        )

    data = run_dump(binary, rs_list, args.n)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(rs_list)))
    for rs, c in zip(rs_list, colors):
        mask = np.isclose(data[:, 0], rs)
        x = data[mask, 1]
        n = data[mask, 2]
        ax.plot(x, n, color=c, lw=1.6, label=f"$r_s = {rs:g}$")

    ax.set_xlabel(r"$k / k_{\rm F}$")
    ax.set_ylabel(r"$n(k, r_s)$")
    ax.set_xlim(0.0, args.kmax)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.0, color="0.75", lw=0.7, ls="--")
    ax.axhline(1.0, color="0.75", lw=0.7, ls="--")
    ax.axvline(1.0, color="0.75", lw=0.7, ls="--")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Gori-Giorgi / Ziesche $n(k, r_s)$ — paper Fig. 6")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
