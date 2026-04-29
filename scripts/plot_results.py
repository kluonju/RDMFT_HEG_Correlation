#!/usr/bin/env python3
"""Plot RDMFT correlation energies vs rs from data/results.tsv.

Reproduces the typical comparison plot of E_c(r_s) against the QMC reference
for HEG using HF (zero), Mueller, BBC1 and the Power functional with several
alphas, plus the PW92 QMC parameterization.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


COLORS = {
    "Monte Carlo":   ("black",   "-",  None),
    "HF":            ("#3a6cff", "--", None),
    "CHF":           ("#3a6cff", "--", None),
    "Mueller":       ("#21c7d6", "-",  "o"),
    "GU":            ("#1f77b4", ":",  "x"),
    "CGA":           ("#ff7f0e", "-",  "D"),
    "GEO":           ("#000080", "-",  "*"),
    "BBC1":          ("#9b59b6", "-",  ">"),
    "Power(0.55)":   ("#d62728", "-",  "s"),
    "Power(0.58)":   ("#2ca02c", "-",  "o"),
    "Beta(0.45)":    ("#e377c2", "-",  "v"),
    "Beta(0.55)":    ("#8c564b", "-",  "^"),
    "Beta(0.65)":    ("#17becf", "-",  "P"),
}


def read_tsv(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            rs  = float(parts[0])
            fn  = parts[1]
            ec  = float(parts[3])
            ecq = float(parts[4])
            rows.append((rs, fn, ec, ecq))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_path",  default="data/results.tsv")
    ap.add_argument("--out", dest="out_path", default="figures/correlation_energy.png")
    args = ap.parse_args()

    rows = read_tsv(Path(args.in_path))

    series: dict[str, list[tuple[float, float]]] = {}
    qmc_pairs: list[tuple[float, float]] = []
    for rs, fn, ec, ecq in rows:
        # Pretty short names: collapse "Power(alpha=0.550000)" ->
        # "Power(0.55)" and "Beta(beta=0.400)" -> "Beta(0.40)" so the legend
        # stays compact and the COLORS lookup hits.
        if fn.startswith("Power(alpha="):
            alpha = fn.split("=")[1].rstrip(")")
            try:
                fn = f"Power({float(alpha):.2f})"
            except ValueError:
                pass
        elif fn.startswith("Beta(beta="):
            beta = fn.split("=")[1].rstrip(")")
            try:
                fn = f"Beta({float(beta):.2f})"
            except ValueError:
                pass
        series.setdefault(fn, []).append((rs, ec))
        qmc_pairs.append((rs, ecq))

    qmc = sorted(set(qmc_pairs))
    qmc_rs = [p[0] for p in qmc]
    qmc_e  = [p[1] for p in qmc]

    # Group functionals so the legend reads HF/Mueller/GU/CGA/Power.../Beta...
    # rather than alphabetical, which scatters families.
    def order_key(name: str) -> tuple:
        if name == "HF":      return (0, 0, name)
        if name == "Mueller": return (1, 0, name)
        if name == "GU":      return (2, 0, name)
        if name == "CGA":     return (3, 0, name)
        if name == "GEO":     return (3, 1, name)
        if name.startswith("Power("):
            try:
                a = float(name[6:-1])
            except ValueError:
                a = 0.0
            return (4, a, name)
        if name.startswith("Beta("):
            try:
                b = float(name[5:-1])
            except ValueError:
                b = 0.0
            return (5, b, name)
        return (9, 0, name)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.set_xscale("log")
    ax.plot(qmc_rs, qmc_e, "k-", linewidth=2.2, label="Monte Carlo (PW92)")
    for name in sorted(series.keys(), key=order_key):
        pts = sorted(series[name])
        rs = [p[0] for p in pts]
        ec = [p[1] for p in pts]
        c, ls, mk = COLORS.get(name, (None, "-", "o"))
        ax.plot(rs, ec, ls, color=c, marker=mk, label=name, linewidth=1.5,
                markersize=5)

    ax.set_xlabel(r"$r_s$ (a.u.)")
    ax.set_ylabel(r"Correlation energy per electron $E_c$ (hartree)")
    ax.set_title("RDMFT correlation energy of the HEG: GU, CGA, GEO and Beta functionals")
    ax.axhline(0, color="0.6", linewidth=0.6, linestyle="-")
    ax.grid(True, which="both", alpha=0.25)
    # Clip to a sensible window so over-correlating outliers don't dominate.
    ax.set_ylim(-0.20, 0.02)
    ax.legend(loc="lower right", fontsize=9, ncol=2, framealpha=0.92)

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
