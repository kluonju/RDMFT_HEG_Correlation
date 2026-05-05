#!/usr/bin/env python3
"""Plot natural occupation n(k) vs k/k_F from rdmft_heg --nk-out TSVs.

Each file is produced by the driver with two metadata header lines::

    # rs=2 kF=0.959579 k_max=5.757...
    # functional: GEO
    # k\\tn

Example::

    make nk-data
    python3 scripts/plot_nk.py --dir data/nk --rs auto --out figures/nk.png

Only functionals listed in ``plot_common.WANTED_SERIES`` are drawn (same as
the correlation-energy figure, including CGA and CHF). The x-axis is fixed to
``0 <= k/k_F <= 3``.  Driver exports use ``--N 801 --kmax 3`` (801 odd points,
``k_max = 3 k_F``) on a uniform k grid (composite trapezoid).
nk TSVs that declare ``# converged: 0`` are excluded; legacy exports without
that line are still plotted.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from plot_common import (  # noqa: E402
    SUBSET_STYLE,
    WANTED_SERIES,
    WANTED_SET,
    pretty_functional_name,
)


def _fmt_rs_label(rs: float) -> str:
    """Compact r_s for annotations (match tick style: integers without .0)."""
    r = float(rs)
    if abs(r - round(r)) < 1e-9 * max(1.0, abs(r)):
        return str(int(round(r)))
    return f"{r:g}"


def load_nk_tsv(path: Path) -> dict:
    """Return dict rs, kF, k_max, functional, k, n, nk_converged.

    ``nk_converged`` is True/False from ``# converged:`` in the header, or
    None if absent (legacy nk files: treated as usable).
    """
    rs = kf = k_max = None
    functional = ""
    nk_converged: bool | None = None
    k_list: list[float] = []
    n_list: list[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# rs="):
                m = re.search(r"rs=([^\s]+)", line)
                if m:
                    rs = float(m.group(1))
                m = re.search(r"kF=([^\s]+)", line)
                if m:
                    kf = float(m.group(1))
                m = re.search(r"k_max=([^\s]+)", line)
                if m:
                    k_max = float(m.group(1))
            elif line.startswith("# functional:"):
                functional = line.split(":", 1)[1].strip()
            elif line.startswith("# converged:"):
                v = line.split(":", 1)[1].strip()
                try:
                    nk_converged = int(float(v)) == 1
                except ValueError:
                    nk_converged = None
            elif line.startswith("#"):
                continue
            else:
                parts = line.split()
                if len(parts) >= 2:
                    k_list.append(float(parts[0]))
                    n_list.append(float(parts[1]))
    return {
        "path": path,
        "rs": rs,
        "kF": kf,
        "k_max": k_max,
        "functional": functional,
        "nk_converged": nk_converged,
        "k": np.array(k_list, dtype=float),
        "n": np.array(n_list, dtype=float),
    }


def discover_files(nk_dir: Path, rs_want: float, rs_tol: float) -> list[Path]:
    files = sorted(nk_dir.glob("*_rs*.tsv"))
    out: list[Path] = []
    for p in files:
        m = re.search(r"_rs([0-9.]+)\.tsv$", p.name, re.I)
        if not m:
            continue
        if abs(float(m.group(1)) - rs_want) <= rs_tol:
            out.append(p)
    return out


def peek_nk_meta(path: Path) -> tuple[float | None, str, bool | None]:
    """Read only headers (no full k,n scan) for discovery."""
    rs_val = None
    functional = ""
    nk_converged: bool | None = None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# rs="):
                m = re.search(r"rs=([^\s]+)", line)
                if m:
                    rs_val = float(m.group(1))
            elif line.startswith("# functional:"):
                functional = line.split(":", 1)[1].strip()
            elif line.startswith("# converged:"):
                v = line.split(":", 1)[1].strip()
                try:
                    nk_converged = int(float(v)) == 1
                except ValueError:
                    nk_converged = None
            elif line.startswith("#"):
                continue
            else:
                break
    return rs_val, functional, nk_converged


def discover_rs_auto(nk_dir: Path) -> list[float]:
    """Sorted r_s values that have at least one nk TSV for a plotted functional."""
    seen: set[float] = set()
    for p in sorted(nk_dir.glob("*_rs*.tsv")):
        rs_h, fn, conv = peek_nk_meta(p)
        if rs_h is None:
            continue
        if conv is False:
            continue
        if pretty_functional_name(fn) not in WANTED_SET:
            continue
        seen.add(rs_h)
    return sorted(seen)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot n(k) vs k/k_F from --nk-out TSVs.")
    ap.add_argument("--dir", type=Path, default=Path("data/nk"), help="Directory of nk_*.tsv")
    ap.add_argument(
        "--rs",
        type=str,
        default="auto",
        help="Comma-separated r_s values, or 'auto' to use only r_s present under "
        "--dir with a plotted functional (default: auto).",
    )
    ap.add_argument(
        "--rs-tol",
        type=float,
        default=0.0005,
        help="Match files whose embedded rs differs by at most this much.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("figures/nk.png"),
        help="Output PNG (one file; multi-rs uses subplots).",
    )
    args = ap.parse_args()

    if not args.dir.is_dir():
        raise SystemExit(f"Not a directory: {args.dir}")

    rs_arg = args.rs.strip()
    if not rs_arg:
        raise SystemExit("Provide --rs values or 'auto'.")

    if rs_arg.lower() == "auto":
        rs_list = discover_rs_auto(args.dir)
        if not rs_list:
            raise SystemExit(
                f"No nk TSVs under {args.dir} for plotted functionals "
                f"({', '.join(WANTED_SERIES)}). Run: make nk-data"
            )
    else:
        rs_list = [float(x.strip()) for x in args.rs.split(",") if x.strip()]
        if not rs_list:
            raise SystemExit("Provide at least one --rs value.")

    missing_no_files: list[float] = []
    missing_filtered: list[float] = []

    n_rows = len(rs_list)
    # Keep multi-r_s figures from growing without bound (Makefile uses full RS_LIST).
    max_total_h = 28.0
    row_h = max(2.0, min(3.2, max_total_h / max(n_rows, 1)))
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(7.0, row_h * n_rows),
        squeeze=False,
    )

    for ax_row, rs_want in zip(axes.flat, rs_list):
        files = discover_files(args.dir, rs_want, args.rs_tol)
        if not files:
            missing_no_files.append(rs_want)
            ax_row.text(0.5, 0.5, f"No data for $r_s$={rs_want}", ha="center", va="center")
            ax_row.set_axis_off()
            continue

        order = {n: i for i, n in enumerate(WANTED_SERIES)}
        loaded: list[tuple[int, Path, dict]] = []
        for p in files:
            d = load_nk_tsv(p)
            if d["kF"] is None or d["k"].size == 0:
                print(f"Warning: skip empty or bad {p}", file=sys.stderr)
                continue
            if d.get("nk_converged") is False:
                continue
            key = pretty_functional_name(d["functional"])
            if key not in WANTED_SET:
                continue
            loaded.append((order.get(key, 99), p, d))
        loaded.sort(key=lambda t: t[0])

        if not loaded:
            missing_filtered.append(rs_want)
            ax_row.text(
                0.5,
                0.5,
                f"No n(k) data for $r_s$={rs_want}",
                ha="center",
                va="center",
            )
            ax_row.set_axis_off()
            continue

        for _rank, _p, d in loaded:
            kf = float(d["kF"])
            x = d["k"] / kf
            key = pretty_functional_name(d["functional"])
            c, ls, mk = SUBSET_STYLE[key]
            ax_row.plot(
                x,
                d["n"],
                ls,
                color=c,
                marker=mk,
                markevery=max(1, len(x) // 20),
                markersize=5 if key == "optGM" else 4,
                linewidth=1.4,
                label=key,
            )

        ax_row.set_xlabel(r"$k / k_{\mathrm{F}}$")
        ax_row.set_ylabel(r"$n(k)$")
        ax_row.set_xlim(0.0, 3.0)
        ax_row.set_ylim(-0.02, 1.05)
        ax_row.axhline(1.0, color="0.75", linewidth=0.8, linestyle="--")
        ax_row.axhline(0.0, color="0.75", linewidth=0.8, linestyle="--")
        ax_row.grid(True, alpha=0.25)
        ax_row.legend(loc="upper right", fontsize=8, framealpha=0.92)

        # r_s as subplot title (above axes) so it never sits on top of n(k) curves.
        rs_lbl = _fmt_rs_label(rs_want)
        ax_row.set_title(rf"$r_s = {rs_lbl}$", fontsize=10, loc="left", pad=7)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"Wrote {args.out}")

    if missing_no_files:
        print(
            "Warning: no nk TSV for "
            + ", ".join(str(r) for r in missing_no_files)
            + f" under {args.dir} (run: make nk-data)",
            file=sys.stderr,
        )
    if missing_filtered:
        print(
            "Warning: nk files exist for r_s "
            + ", ".join(str(r) for r in missing_filtered)
            + " but none match plotted functionals ("
            + ", ".join(WANTED_SERIES)
            + ").",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
