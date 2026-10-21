"""Shared curve list and styles for correlation-energy and n(k) figures.

``WANTED_SERIES`` / ``WANTED_SET`` define which functionals appear in both plots
(Müller, CGA, CHF, Power curves, hybopt, PBEHFmM).  Any ``rs(μ=…)``
series present in the data is also plotted (see ``legend_key_for_subset``).
"""

from __future__ import annotations

import re

# Curves (after Monte Carlo / PW92), fixed order for legends.
# hybopt last so it is drawn on top (z-order highlight) and appears last in the legend.
WANTED_SERIES = (
    "Mueller",
    "CGA",
    "CHF",
    "Power(0.55)",
    "Power(0.58)",
    "hybopt",
)

WANTED_SET = frozenset(WANTED_SERIES)

MONTE_CARLO_LABEL = "QMC"

# (color, linestyle, marker); marker None => no markers on line.
# RDMFT curves use dashed lines; Monte Carlo solid black; hybopt solid blue (highlight).
SUBSET_STYLE: dict[str, tuple[str | None, str, str | None]] = {
    MONTE_CARLO_LABEL: ("black", "-", None),
    "Mueller": ("#21c7d6", "--", "o"),
    "CGA": ("#ff7f0e", "--", "D"),
    "CHF": ("#9467bd", "--", "^"),
    "BOW": ("#8c564b", "--", "v"),
    "optGeo": ("#006400", "--", "h"),
    "PBEHFmM": ("#e377c2", "--", "P"),
    # Solid blue + diamonds + drawn last (stands out vs dashed RDMFT curves).
    "hybopt": ("#1f77b4", "-", "D"),
    "Power(0.55)": ("#d62728", "--", "s"),
    "Power(0.58)": ("#2ca02c", "--", "o"),
    # Post-hoc 0.243 CGA + 0.757 Power(0.58) energy mix (see nk_mix_gz_best.json).
    "CGA+Power(0.58)": ("#393b79", "-.", "*"),
}

# Distinct styles for rs(μ=…) curves (cycled by increasing μ).
_RS_STYLE_CYCLE: list[tuple[str, str, str]] = [
    ("#8c564b", "-.", "X"),   # μ=0.2, α=0.55
    ("#bcbd22", "-.", "s"),   # μ=0.2, α=0.5
    ("#17becf", "-.", "D"),   # μ=2
    ("#e377c2", "-.", "P"),   # μ=0.5
]

# Styles for rs2(μ=…) (SR LDA x+c + LR Power).
_RS2_STYLE_CYCLE: list[tuple[str, str, str]] = [
    ("#e377c2", ":", "P"),
    ("#7f7f7f", ":", "v"),
    ("#c49c94", ":", "^"),
]

# Only these μ values are drawn when present (keeps the main figure uncluttered).
# Plot these μ values when present (HF + SR LDA-c + LR Power−HF).
RS_PLOT_MUS = frozenset({0.2, 0.5, 2.0})
# rs2 omitted from the main Ec figure for now.
RS2_PLOT_MUS = frozenset()
# rs3 (Molpro ECERFPBE + LR Power−HF).  On the HEG this matches rs.
RS3_PLOT_MUS = frozenset({0.2, 0.5, 2.0})

_RS3_STYLE_CYCLE: list[tuple[str, str, str]] = [
    ("#e377c2", ":", "P"),
    ("#7f7f7f", ":", "v"),
    ("#c49c94", ":", "^"),
]


def pretty_functional_name(fn: str) -> str:
    """Match TSV / nk header labels to short legend keys (plotted functionals only)."""
    if fn.startswith("Power(alpha="):
        alpha = fn.split("=", 1)[1].rstrip(")")
        try:
            return f"Power({float(alpha):.2f})"
        except ValueError:
            return fn
    if fn.startswith("BOW(alpha="):
        return "BOW"
    if fn.startswith("SymBow(alpha="):
        return "SymBow"
    if fn.startswith("optGeo("):
        return "optGeo"
    if fn.startswith("hybopt("):
        return "hybopt"
    if fn.startswith("rs3(") or fn == "rs3":
        m = re.search(r"mu=([0-9.eE+-]+)", fn)
        a = re.search(r"alpha=([0-9.eE+-]+)", fn)
        if m:
            try:
                mu = float(m.group(1))
                if a:
                    return rf"rs3($\mu={mu:g}$,$\alpha={float(a.group(1)):g}$)"
                return rf"rs3($\mu={mu:g}$)"
            except ValueError:
                pass
        return "rs3"
    if fn.startswith("rs2(") or fn == "rs2":
        m = re.search(r"mu=([0-9.eE+-]+)", fn)
        a = re.search(r"alpha=([0-9.eE+-]+)", fn)
        if m:
            try:
                mu = float(m.group(1))
                if a:
                    return rf"rs2($\mu={mu:g}$,$\alpha={float(a.group(1)):g}$)"
                return rf"rs2($\mu={mu:g}$)"
            except ValueError:
                pass
        return "rs2"
    if fn.startswith("rs(") or fn == "rs":
        m = re.search(r"mu=([0-9.eE+-]+)", fn)
        a = re.search(r"alpha=([0-9.eE+-]+)", fn)
        if m:
            try:
                mu = float(m.group(1))
                if a:
                    return rf"rs($\mu={mu:g}$,$\alpha={float(a.group(1)):g}$)"
                return rf"rs($\mu={mu:g}$)"
            except ValueError:
                pass
        return "rs"
    if fn.startswith("optGMw("):
        return "hybopt"
    if fn.startswith("optGM("):
        # hybopt(lam=...,alpha=...) legacy optGM label; optGM(a=,b=,c=) optGeo-style.
        if "lam=" in fn or "alpha=" in fn or "w_hf=" in fn or ("hole=" in fn and "lam=" not in fn):
            return "hybopt"
        return "optGeo"
    return fn


# Display names for figure legends (internal keys stay ASCII for TSV matching).
_LEGEND_DISPLAY = {
    "Mueller": "Müller",
}


def legend_label(key: str) -> str:
    """Human-readable legend text for an internal series key."""
    return _LEGEND_DISPLAY.get(key, key)


def rs_mu_from_key(key: str) -> float | None:
    """Parse μ from a pretty key like ``rs($\\mu=0.8$,$\\alpha=0.55$)``."""
    m = re.search(r"\\mu=([0-9.eE+-]+)", key)
    if not m:
        # Fallback: first number (legacy μ-only labels).
        m = re.search(r"([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)", key)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def rs_alpha_from_key(key: str) -> float | None:
    """Parse α from a pretty rs legend key."""
    m = re.search(r"\\alpha=([0-9.eE+-]+)", key)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def is_rs_series(key: str) -> bool:
    if key.startswith("rs2") or key.startswith("rs3"):
        return False
    if not key.startswith("rs"):
        return False
    mu = rs_mu_from_key(key)
    return mu is not None and mu in RS_PLOT_MUS


def is_rs2_series(key: str) -> bool:
    if not key.startswith("rs2"):
        return False
    mu = rs_mu_from_key(key)
    return mu is not None and mu in RS2_PLOT_MUS


def is_rs3_series(key: str) -> bool:
    if not key.startswith("rs3"):
        return False
    mu = rs_mu_from_key(key)
    return mu is not None and mu in RS3_PLOT_MUS


def style_for_series(key: str) -> tuple[str | None, str, str | None]:
    """Return (color, linestyle, marker) for a plotted series key."""
    if key in SUBSET_STYLE:
        return SUBSET_STYLE[key]
    presets = {
        (0.2, 0.55): 0,
        (0.2, 0.5): 1,
        (0.5, 0.55): 3,
        (0.5, None): 3,
        (2.0, 0.55): 2,
        (2.0, None): 2,
        (0.2, None): 0,
    }
    if key.startswith("rs3"):
        mu = rs_mu_from_key(key)
        alpha = rs_alpha_from_key(key)
        idx = presets.get((mu, alpha), presets.get((mu, None), 0))
        return _RS3_STYLE_CYCLE[idx % len(_RS3_STYLE_CYCLE)]
    if key.startswith("rs2"):
        mu = rs_mu_from_key(key)
        alpha = rs_alpha_from_key(key)
        idx = presets.get((mu, alpha), presets.get((mu, None), 0))
        return _RS2_STYLE_CYCLE[idx % len(_RS2_STYLE_CYCLE)]
    if key.startswith("rs"):
        mu = rs_mu_from_key(key)
        alpha = rs_alpha_from_key(key)
        idx = presets.get((mu, alpha), presets.get((mu, None), 0))
        return _RS_STYLE_CYCLE[idx % len(_RS_STYLE_CYCLE)]
    return ("#7f7f7f", "--", "o")


def legend_key_for_subset(fn: str) -> str | None:
    """Return legend series name if ``fn`` is one of the wanted curves, else None."""
    key = pretty_functional_name(fn)
    if key in WANTED_SET or is_rs_series(key) or is_rs2_series(key) or is_rs3_series(key):
        return key
    return None
