#!/usr/bin/env python3
"""Plot natural occupation n(k) vs k/k_F from rdmft_heg --nk-out TSVs.

Each file is produced by the driver with two metadata header lines::

    # rs=2 kF=0.959579 k_max=5.757...
    # functional: GEO
    # k\\tn

Example::

    make nk-data
    python3 scripts/plot_nk.py --dir data/nk --rs 0.5,1,5,10 --layout 2x2 \
        --out figures/nk.png

Only functionals listed in ``plot_common.WANTED_SERIES`` are drawn (same as
the correlation-energy figure, including CGA and CHF). By default each panel
also overlays the Gori-Giorgi/Ziesche (GZ) reference n(k) from
``build/dump_gz_grid``. The default x-axis is ``0 <= k/k_F <= 3`` (override
with ``--xmax``); driver exports may use a larger ``--kmax``.
nk TSVs that declare ``# converged: 0`` are excluded; legacy exports without
that line are still plotted.
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
_REPO_ROOT = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from plot_common import (  # noqa: E402
    SUBSET_STYLE,
    WANTED_SERIES,
    WANTED_SET,
    legend_label,
    pretty_functional_name,
)

# GZ reference: solid black, no markers (stands apart from dashed RDMFT curves).
GZ_STYLE = ("black", "-", None)
GZ_LABEL = "GZ"


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


def load_gz_grid(
    binary: Path,
    rs_list: list[float],
    nk: int = 401,
    kmax: float = 3.0,
) -> np.ndarray:
    """Run dump_gz_grid; return array of shape (N, 3): rs, k/kF, n."""
    rs_arg = ",".join(f"{r:g}" for r in rs_list)
    out = subprocess.run(
        [str(binary), "--rs", rs_arg, "--n", str(nk), "--kmax", str(kmax)],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    ).stdout
    rows: list[tuple[float, float, float]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return np.array(rows, dtype=float)


def gz_curve_for_rs(gz: np.ndarray, rs: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (k/kF, n) for the closest rs row-block in a dump_gz_grid array."""
    if gz.size == 0:
        return None
    mask = np.isclose(gz[:, 0], rs)
    if not np.any(mask):
        # Fall back to nearest rs in the dump (should not happen if dump used rs_list).
        uniq = np.unique(gz[:, 0])
        nearest = float(uniq[np.argmin(np.abs(uniq - rs))])
        if abs(nearest - rs) > 1e-3:
            return None
        mask = np.isclose(gz[:, 0], nearest)
    return gz[mask, 1], gz[mask, 2]


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
    ap.add_argument(
        "--dump-gz",
        type=Path,
        default=_REPO_ROOT / "build" / "dump_gz_grid",
        help="Path to dump_gz_grid binary for GZ reference overlay.",
    )
    ap.add_argument(
        "--no-gz",
        action="store_true",
        help="Do not overlay the Gori-Giorgi/Ziesche n(k) reference.",
    )
    ap.add_argument(
        "--gz-n",
        type=int,
        default=401,
        help="Number of k/kF points for the GZ overlay grid (default: 401).",
    )
    ap.add_argument(
        "--layout",
        choices=("col", "2x2"),
        default="col",
        help="Subplot arrangement: stacked column (default) or 2x2 grid "
        "(requires exactly 4 --rs values).",
    )
    ap.add_argument(
        "--xmax",
        type=float,
        default=3.0,
        help="Max k/k_F on the plot x-axis (default: 3).",
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

    if args.layout == "2x2" and len(rs_list) != 4:
        raise SystemExit(
            f"--layout 2x2 needs exactly 4 r_s values (got {len(rs_list)})."
        )

    gz_data: np.ndarray | None = None
    if not args.no_gz:
        dump_gz = Path(args.dump_gz)
        if not dump_gz.is_file():
            raise SystemExit(
                f"Missing GZ driver {dump_gz}; run `make build/dump_gz_grid` "
                f"or pass --no-gz."
            )
        gz_data = load_gz_grid(
            dump_gz.resolve(), rs_list, nk=args.gz_n, kmax=float(args.xmax)
        )

    missing_no_files: list[float] = []
    missing_filtered: list[float] = []

    if args.layout == "2x2":
        n_rows, n_cols = 2, 2
        figsize = (10.0, 7.5)
        legend_fs = 7
    else:
        n_rows, n_cols = len(rs_list), 1
        # Keep multi-r_s figures from growing without bound (Makefile uses full RS_LIST).
        max_total_h = 28.0
        row_h = max(2.0, min(3.2, max_total_h / max(n_rows, 1)))
        figsize = (7.0, row_h * n_rows)
        legend_fs = 8

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
    )

    for ax, rs_want in zip(axes.flat, rs_list):
        files = discover_files(args.dir, rs_want, args.rs_tol)
        if not files:
            missing_no_files.append(rs_want)
            ax.text(0.5, 0.5, f"No data for $r_s$={rs_want}", ha="center", va="center")
            ax.set_axis_off()
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
            ax.text(
                0.5,
                0.5,
                f"No n(k) data for $r_s$={rs_want}",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
            continue

        for _rank, _p, d in loaded:
            kf = float(d["kF"])
            x = d["k"] / kf
            key = pretty_functional_name(d["functional"])
            c, ls, mk = SUBSET_STYLE[key]
            ax.plot(
                x,
                d["n"],
                ls,
                color=c,
                marker=mk,
                markevery=max(1, len(x) // 60),
                markersize=5 if key == "hybopt" else 4,
                linewidth=1.4,
                label=legend_label(key),
            )

        if gz_data is not None:
            curve = gz_curve_for_rs(gz_data, rs_want)
            if curve is not None:
                x_gz, n_gz = curve
                c, ls, mk = GZ_STYLE
                ax.plot(
                    x_gz,
                    n_gz,
                    ls,
                    color=c,
                    marker=mk,
                    linewidth=1.8,
                    label=GZ_LABEL,
                    zorder=5,
                )

        ax.set_xlabel(r"$k / k_{\mathrm{F}}$")
        ax.set_ylabel(r"$n(k)$")
        ax.set_xlim(0.0, float(args.xmax))
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(1.0, color="0.75", linewidth=0.8, linestyle="--")
        ax.axhline(0.0, color="0.75", linewidth=0.8, linestyle="--")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=legend_fs, framealpha=0.92)

        # r_s as subplot title (above axes) so it never sits on top of n(k) curves.
        rs_lbl = _fmt_rs_label(rs_want)
        ax.set_title(rf"$r_s = {rs_lbl}$", fontsize=10, loc="left", pad=7)

    # Hide any unused axes (e.g. if layout larger than len(rs_list) in future).
    for ax in axes.flat[len(rs_list) :]:
        ax.set_axis_off()

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
