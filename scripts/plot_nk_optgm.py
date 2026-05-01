#!/usr/bin/env python3
"""Plot optGM ``n(k)`` only: one curve per ``r_s`` on a single axes (``k/k_F``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import plot_nk as nk  # noqa: E402
from plot_common import SUBSET_STYLE, pretty_functional_name  # noqa: E402


def discover_rs_optgm(nk_dir: Path) -> list[float]:
    """Sorted ``r_s`` for which an nk TSV exists with functional optGM."""
    seen: set[float] = set()
    for p in sorted(nk_dir.glob("*_rs*.tsv")):
        d = nk.load_nk_tsv(p)
        if pretty_functional_name(d["functional"]) != "optGM":
            continue
        if d["rs"] is None or d["kF"] is None or d["k"].size == 0:
            continue
        seen.add(float(d["rs"]))
    return sorted(seen)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay optGM n(k) vs k/k_F for several r_s on one figure."
    )
    ap.add_argument("--dir", type=Path, default=Path("data/nk"), help="nk TSV directory")
    ap.add_argument(
        "--rs",
        type=str,
        default="auto",
        help="'auto' = every r_s that has an optGM nk TSV; else comma-separated r_s.",
    )
    ap.add_argument("--rs-tol", type=float, default=0.0005, help="File match tolerance")
    ap.add_argument("--out", type=Path, default=Path("figures/nk_optgm.png"))
    ap.add_argument("--xmax", type=float, default=3.0, help="Upper limit for k/k_F")
    args = ap.parse_args()

    if not args.dir.is_dir():
        raise SystemExit(f"Not a directory: {args.dir}")

    rs_arg = args.rs.strip()
    if rs_arg.lower() == "auto":
        rs_list = discover_rs_optgm(args.dir)
        if not rs_list:
            raise SystemExit(
                f"No optGM nk TSVs under {args.dir}. Run e.g. "
                "`make nk-data` with OptGM in NK_FUNCS."
            )
    else:
        rs_list = [float(x.strip()) for x in args.rs.split(",") if x.strip()]
        if not rs_list:
            raise SystemExit("Provide at least one --rs value.")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    _c_opt, ls_opt, mk_opt = SUBSET_STYLE["optGM"]
    n = len(rs_list)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(0.12 + 0.78 * (i / max(n - 1, 1))) for i in range(n)]

    plotted = 0
    for i, rs_want in enumerate(rs_list):
        files = nk.discover_files(args.dir, rs_want, args.rs_tol)
        d = None
        for p in files:
            di = nk.load_nk_tsv(p)
            if pretty_functional_name(di["functional"]) != "optGM":
                continue
            d = di
            break
        if d is None:
            print(
                f"Warning: no optGM nk TSV for r_s≈{rs_want} under {args.dir}",
                file=sys.stderr,
            )
            continue
        kf = float(d["kF"])
        x = d["k"] / kf
        lbl = rf"$r_s = {nk._fmt_rs_label(rs_want)}$"
        ax.plot(
            x,
            d["n"],
            ls_opt,
            color=colors[i],
            marker=mk_opt,
            markevery=max(1, len(x) // 25),
            markersize=3.5,
            linewidth=1.35,
            label=lbl,
        )
        plotted += 1

    if plotted == 0:
        raise SystemExit("No curves drawn; check --dir and --rs.")

    ax.set_xlabel(r"$k / k_{\mathrm{F}}$")
    ax.set_ylabel(r"$n(k)$")
    ax.set_xlim(0.0, float(args.xmax))
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(1.0, color="0.75", linewidth=0.8, linestyle="--")
    ax.axhline(0.0, color="0.75", linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92, title="optGM")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
