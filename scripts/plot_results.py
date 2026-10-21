#!/usr/bin/env python3
"""Plot RDMFT correlation energies vs rs from per-functional TSVs.

Reads ``*.tsv`` files under ``--in`` and overlays the PW92 QMC reference with
Müller, CGA, CHF, hybopt, Power(0.55/0.58), PBEHFmM (see ``plot_common.WANTED_SERIES``),
plus the post-hoc mix ``λ Ec_CGA + (1-λ) Ec_Power(0.58)`` with λ from
``data/nk_mix_gz_best.json`` (0.243 CGA + 0.757 Power(0.58)).
Only rows with ``converged == 1`` are plotted (non-converged SCF points are dropped).
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
import json
import re
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
    is_rs2_series,
    is_rs3_series,
    is_rs_series,
    legend_key_for_subset,
    legend_label,
    style_for_series,
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


def apply_rs_ticks(
    ax,
    series: dict,
    qmc_rs: list[float],
    *,
    xlim: tuple[float, float] | None = None,
) -> None:
    """Place a sparse set of major ticks on the log r_s axis."""
    data_rs = {float(r) for r in qmc_rs}
    for pts in series.values():
        for r, _ in pts:
            data_rs.add(float(r))
    x0, x1 = xlim if xlim is not None else (0.008, 120.0)
    # Prefer decade-style majors; fall back to a thinned subset of data r_s.
    preferred = (0.01, 0.1, 1.0, 10.0, 100.0)
    ticks = [t for t in preferred if x0 <= t <= x1]
    if len(ticks) < 2:
        all_rs = sorted(r for r in data_rs if x0 <= r <= x1)
        if len(all_rs) >= 2:
            step = max(1, len(all_rs) // 4)
            ticks = all_rs[::step]
            if ticks[-1] != all_rs[-1]:
                ticks.append(all_rs[-1])
    if len(ticks) < 2:
        ax.set_xlim(x0, x1)
        return
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt_rs_tick(t) for t in ticks])
    ax.tick_params(axis="x", which="major", labelsize=9)
    ax.minorticks_off()
    for lab in ax.get_xticklabels():
        lab.set_horizontalalignment("center")
    ax.set_xlim(x0, x1)


def parse_tsv(path: Path):
    """Yield (rs, fn, ec, ec_qmc) tuples from a single TSV file (converged rows only)."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            # Driver layout: ... rho_err, converged, iters (column index 9 = converged).
            if len(parts) < 10:
                continue
            try:
                if int(float(parts[9].strip())) != 1:
                    continue
            except ValueError:
                continue
            try:
                rs  = float(parts[0])
                fn  = parts[1]
                ec  = float(parts[3])
                ecq = float(parts[4])
            except ValueError:
                continue
            yield rs, fn, ec, ecq


def add_cga_power_mix(
    series: dict[str, list[tuple[float, float]]],
    mix_json: Path,
) -> str | None:
    """Add λ Ec_CGA + (1-λ) Ec_Power(0.58) using the n(k)-mix weight.

    Returns the legend label, or None if the parent curves or JSON are missing.
    """
    if "CGA" not in series or "Power(0.58)" not in series or not mix_json.is_file():
        return None
    best = json.loads(mix_json.read_text()).get("best", {})
    if best.get("A") != "CGA" or best.get("B") != "Power(0.58)":
        return None
    lam = float(best["lambda"])
    by_rs_a = {rs: ec for rs, ec in series["CGA"]}
    by_rs_b = {rs: ec for rs, ec in series["Power(0.58)"]}
    shared = sorted(set(by_rs_a) & set(by_rs_b))
    if not shared:
        return None
    label = rf"${lam:.3f}$ CGA $+{1.0 - lam:.3f}$ Power(0.58)"
    series[label] = [
        (rs, lam * by_rs_a[rs] + (1.0 - lam) * by_rs_b[rs]) for rs in shared
    ]
    return label


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
    ap.add_argument(
        "--rs-min",
        type=float,
        default=None,
        help="Only plot points with r_s >= this value (zoom).",
    )
    ap.add_argument(
        "--rs-max",
        type=float,
        default=None,
        help="Only plot points with r_s <= this value.",
    )
    ap.add_argument(
        "--no-rs",
        action="store_true",
        help="Omit rs, rs2, and rs3 curves.",
    )
    ap.add_argument(
        "--no-hybopt",
        action="store_true",
        help="Omit the hybopt curve.",
    )
    ap.add_argument(
        "--mix-json",
        type=Path,
        default=Path("data/nk_mix_gz_best.json"),
        help="Best n(k) mix; its λ weights the CGA and Power(0.58) Ec curves.",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    rs_min = args.rs_min
    rs_max = args.rs_max

    # Per-series map rs -> Ec so a mislabeled extra TSV (e.g. stale CHA.tsv
    # whose functional column still said "CHF") cannot double-plot and zigzag.
    series_map: dict[str, dict[float, float]] = {}
    qmc_pairs: list[tuple[float, float]] = []
    for rs, fn, ec, ecq in collect_rows(in_path):
        if rs_min is not None and rs < rs_min:
            continue
        if rs_max is not None and rs > rs_max:
            continue
        key = legend_key_for_subset(fn)
        if key is None:
            continue
        if args.no_rs and (
            is_rs_series(key) or is_rs2_series(key) or is_rs3_series(key)
        ):
            continue
        if args.no_hybopt and key == "hybopt":
            continue
        series_map.setdefault(key, {})[rs] = ec
        qmc_pairs.append((rs, ecq))

    series: dict[str, list[tuple[float, float]]] = {
        key: sorted(pts.items()) for key, pts in series_map.items()
    }

    mix_label = None  # post-hoc CGA+Power mix omitted from the main figure

    if not qmc_pairs:
        raise SystemExit(
            f"No usable data points for plotted functionals in {in_path}"
            + (f" with r_s filter [{rs_min}, {rs_max}]" if rs_min or rs_max else "")
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
    def _mu_sort_key(k: str) -> float:
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", k)
        return float(m.group(1)) if m else 0.0

    rs_keys = sorted((k for k in series if is_rs_series(k)), key=_mu_sort_key)
    rs2_keys = sorted((k for k in series if is_rs2_series(k)), key=_mu_sort_key)
    rs3_keys = sorted((k for k in series if is_rs3_series(k)), key=_mu_sort_key)
    plot_order = list(WANTED_SERIES) + rs_keys + rs2_keys + rs3_keys
    if mix_label is not None:
        plot_order.append(mix_label)
    for name in plot_order:
        if name not in series:
            continue
        pts = sorted(series[name])
        rs = [p[0] for p in pts]
        ec = [p[1] for p in pts]
        c, ls, mk = style_for_series(name)
        if name == mix_label and name not in SUBSET_STYLE:
            c, ls, mk = SUBSET_STYLE["CGA+Power(0.58)"]
        ax.plot(
            rs,
            ec,
            ls,
            color=c,
            marker=mk,
            label=legend_label(name),
            linewidth=2.0
            if (name == mix_label or is_rs_series(name) or is_rs2_series(name)
                or is_rs3_series(name))
            else 1.5,
            markersize=7 if name == mix_label else (6 if name == "hybopt" else 5),
        )

    # X limits: pad slightly around filtered data (or full default span).
    if rs_min is not None or rs_max is not None:
        lo_x = min(qmc_rs)
        hi_x = max(qmc_rs)
        pad_x = max(0.05 * (hi_x - lo_x), 0.05 * lo_x)
        xlim = (max(0.005, lo_x * 0.8), hi_x * 1.2)
    else:
        xlim = (0.008, 120.0)
    apply_rs_ticks(ax, series, qmc_rs, xlim=xlim)
    ax.set_xlabel(r"$r_s$ (a.u.)")
    ax.set_ylabel(r"$e_c$ (hartree) per electron")
    ax.axhline(0, color="0.6", linewidth=0.6, linestyle="-")
    ax.grid(True, which="both", alpha=0.25)
    # Y-axis from data + QMC (padding); keeps all wanted curves visible.
    ec_vals = [e for pts in series.values() for (_, e) in pts] + qmc_e
    if ec_vals:
        lo, hi = min(ec_vals), max(ec_vals)
        span = max(hi - lo, 1e-6)
        pad = 0.06 * span + 0.002
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
