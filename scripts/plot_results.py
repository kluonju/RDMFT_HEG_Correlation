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
    "Monte Carlo":   ("black", "-",  None),
    "HF":            ("#3a6cff", "--", None),
    "CHF":           ("#3a6cff", "--", None),
    "Mueller":       ("#21c7d6", "-", "o"),
    "BBC1":          ("#9b59b6", "-", ">"),
    "Power(0.55)":   ("#d62728", "-", "s"),
    "Power(0.58)":   ("#2ca02c", "-", "o"),
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
        # Pretty short name for Power(alpha=...).
        if fn.startswith("Power(alpha="):
            alpha = fn.split("=")[1].rstrip(")")
            try:
                a = float(alpha)
                fn = f"Power({a:.2f})"
            except ValueError:
                pass
        series.setdefault(fn, []).append((rs, ec))
        qmc_pairs.append((rs, ecq))

    qmc = sorted(set(qmc_pairs))
    qmc_rs = [p[0] for p in qmc]
    qmc_e  = [p[1] for p in qmc]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.set_xscale("log")
    for name, pts in sorted(series.items()):
        pts.sort()
        rs = [p[0] for p in pts]
        ec = [p[1] for p in pts]
        c, ls, mk = COLORS.get(name, (None, "-", "o"))
        ax.plot(rs, ec, ls, color=c, marker=mk, label=name, linewidth=1.5,
                markersize=5)
    ax.plot(qmc_rs, qmc_e, "k-", linewidth=2.0, label="Monte Carlo (PW92)")

    ax.set_xlabel(r"$r_s$ (a.u.)")
    ax.set_ylabel("Correlation energy (hartree)")
    ax.set_title("RDMFT correlation energy of the HEG")
    ax.axhline(0, color="0.7", linewidth=0.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
