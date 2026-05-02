#!/usr/bin/env python3
"""Plot RDMFT correlation energies vs rs from per-functional TSVs.

Reads ``*.tsv`` files under ``--in`` and overlays the PW92 QMC reference with
Müller, CGA, CHF, optGM, and Power(0.55/0.58) (see ``plot_common.WANTED_SERIES``).
The directory layout matches the rdmft_heg driver:

    data/
        HF.tsv
        Mueller.tsv
        GEO.tsv
        Power_0.55.tsv
        ...

For backwards compatibility ``--in`` may also point at a single ``.tsv``
file (the previous monolithic ``data/results.tsv`` layout) and in that case
all rows are read from that one file.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from plot_common import (  # noqa: E402
    MONTE_CARLO_LABEL,
    SUBSET_STYLE,
    WANTED_SERIES,
    legend_key_for_subset,
)


def _fmt_rs_tick(rs: float) -> str:
    """Compact r_s label for log-axis ticks (e.g. 0.2, 1, 10).

    Do not use ``rstrip('0')`` on integers written without a dot: ``"10"`` would
    become ``"1"``.
    """
    r = float(rs)
    if abs(r - round(r)) < 1e-9 * max(1.0, abs(r)):
        return str(int(round(r)))
    s = f"{r:.4f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else str(r)


def apply_rs_ticks(ax, series: dict, qmc_rs: list[float]) -> None:
    """Label every distinct r_s from data on the (log) x-axis; span 0.15–10."""
    tick_vals = {float(r) for r in qmc_rs}
    for pts in series.values():
        for r, _ in pts:
            tick_vals.add(float(r))
    ticks = sorted(tick_vals)
    if len(ticks) < 2:
        ax.set_xlim(0.15, 11.0)
        return
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt_rs_tick(t) for t in ticks])
    ax.tick_params(axis="x", which="major", labelsize=8)
    for lab in ax.get_xticklabels():
        lab.set_horizontalalignment("center")
    ax.set_xlim(0.15, 11.0)


def parse_tsv(path: Path):
    """Yield (rs, fn, ec, ec_qmc) tuples from a single TSV file."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            try:
                rs  = float(parts[0])
                fn  = parts[1]
                ec  = float(parts[3])
                ecq = float(parts[4])
            except ValueError:
                continue
            yield rs, fn, ec, ecq


def collect_rows(in_path: Path):
    """Read all rows from either a directory of *.tsv or a single TSV."""
    if in_path.is_dir():
        files = sorted(in_path.glob("*.tsv"))
        if not files:
            raise SystemExit(f"No .tsv files in {in_path}/")
        for f in files:
            yield from parse_tsv(f)
    else:
        yield from parse_tsv(in_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_path",  default="data",
                    help="Directory of per-functional .tsv files "
                         "(or a single .tsv).  Default: data")
    ap.add_argument("--out", dest="out_path",
                    default="figures/correlation_energy.png")
    args = ap.parse_args()

    in_path = Path(args.in_path)

    series: dict[str, list[tuple[float, float]]] = {}
    qmc_pairs: list[tuple[float, float]] = []
    for rs, fn, ec, ecq in collect_rows(in_path):
        key = legend_key_for_subset(fn)
        if key is None:
            continue
        series.setdefault(key, []).append((rs, ec))
        qmc_pairs.append((rs, ecq))

    if not qmc_pairs:
        raise SystemExit(
            f"No usable data points for plotted functionals in {in_path}"
        )

    qmc = sorted(set(qmc_pairs))
    qmc_rs = [p[0] for p in qmc]
    qmc_e = [p[1] for p in qmc]

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.set_xscale("log")
    mc_c, mc_ls, _mc_mk = SUBSET_STYLE[MONTE_CARLO_LABEL]
    ax.plot(
        qmc_rs,
        qmc_e,
        mc_ls,
        color=mc_c,
        linewidth=2.2,
        label=MONTE_CARLO_LABEL,
    )
    for name in WANTED_SERIES:
        if name not in series:
            continue
        pts = sorted(series[name])
        rs = [p[0] for p in pts]
        ec = [p[1] for p in pts]
        c, ls, mk = SUBSET_STYLE[name]
        ax.plot(
            rs,
            ec,
            ls,
            color=c,
            marker=mk,
            label=name,
            linewidth=1.5,
            markersize=5,
        )

    apply_rs_ticks(ax, series, qmc_rs)
    ax.set_xlabel(r"$r_s$ (a.u.)")
    ax.set_ylabel(r"Correlation energy per electron $E_c$ (hartree)")
    ax.axhline(0, color="0.6", linewidth=0.6, linestyle="-")
    ax.grid(True, which="both", alpha=0.25)
    # Y-axis from data + QMC (padding); keeps all wanted curves visible.
    ec_vals = [e for pts in series.values() for (_, e) in pts] + qmc_e
    if ec_vals:
        lo, hi = min(ec_vals), max(ec_vals)
        span = max(hi - lo, 1e-6)
        pad = 0.06 * span + 0.005
        ax.set_ylim(lo - pad, hi + pad)
    else:
        ax.set_ylim(-0.20, 0.02)
    ax.legend(loc="lower right", fontsize=9, ncol=2, framealpha=0.92)

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
