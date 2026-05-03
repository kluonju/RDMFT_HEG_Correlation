"""Shared curve list and styles for correlation-energy and n(k) figures.

``WANTED_SERIES`` / ``WANTED_SET`` define which functionals appear in both plots.
"""

from __future__ import annotations

# Curves (after Monte Carlo / PW92), fixed order for legends.
# optGM last so it is drawn on top (z-order highlight) and appears last in the legend.
WANTED_SERIES = (
    "Mueller",
    "CGA",
    "CHF",
    "Power(0.55)",
    "Power(0.58)",
    "optGM",
)

WANTED_SET = frozenset(WANTED_SERIES)

MONTE_CARLO_LABEL = "Monte Carlo (PW92)"

# (color, linestyle, marker); marker None => no markers on line.
# RDMFT curves use dashed lines; Monte Carlo solid black; optGM solid blue (highlight).
SUBSET_STYLE: dict[str, tuple[str | None, str, str | None]] = {
    MONTE_CARLO_LABEL: ("black", "-", None),
    "Mueller": ("#21c7d6", "--", "o"),
    "CGA": ("#ff7f0e", "--", "D"),
    "CHF": ("#9467bd", "--", "^"),
    "optGeo": ("#006400", "--", "h"),
    # Solid blue + diamonds + drawn last (stands out vs dashed RDMFT curves).
    "optGM": ("#1f77b4", "-", "D"),
    "Power(0.55)": ("#d62728", "--", "s"),
    "Power(0.58)": ("#2ca02c", "--", "o"),
}


def pretty_functional_name(fn: str) -> str:
    """Match TSV / nk header labels to short legend keys (plotted functionals only)."""
    if fn.startswith("Power(alpha="):
        alpha = fn.split("=", 1)[1].rstrip(")")
        try:
            return f"Power({float(alpha):.2f})"
        except ValueError:
            return fn
    if fn.startswith("optGeo("):
        return "optGeo"
    if fn.startswith("optGMw("):
        return "optGM"
    if fn.startswith("optGM("):
        # optGM(lam=...,alpha=...). Old: optGM(a=,b=,c=) optGeo-style labels; w_hf= obsolete triple-weight.
        if "lam=" in fn or "alpha=" in fn or "w_hf=" in fn or ("hole=" in fn and "lam=" not in fn):
            return "optGM"
        return "optGeo"
    return fn


def legend_key_for_subset(fn: str) -> str | None:
    """Return legend series name if ``fn`` is one of the wanted curves, else None."""
    key = pretty_functional_name(fn)
    return key if key in WANTED_SET else None
