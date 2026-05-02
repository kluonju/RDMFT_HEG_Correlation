"""Shared curve list and styles for correlation-energy and n(k) figures.

``WANTED_SERIES`` / ``WANTED_SET`` define which functionals appear in both plots.
"""

from __future__ import annotations

# Curves (after Monte Carlo / PW92), fixed order for legends.
WANTED_SERIES = (
    "Mueller",
    "CGA",
    "CHF",
    "optGM",
    "Power(0.55)",
    "Power(0.58)",
)

WANTED_SET = frozenset(WANTED_SERIES)

MONTE_CARLO_LABEL = "Monte Carlo (PW92)"

# (color, linestyle, marker); marker None => no markers on line.
SUBSET_STYLE: dict[str, tuple[str | None, str, str | None]] = {
    MONTE_CARLO_LABEL: ("black", "-", None),
    "Mueller": ("#21c7d6", "-", "o"),
    "CGA": ("#ff7f0e", "-", "D"),
    "CHF": ("#9467bd", "--", "^"),
    "optGM": ("#006400", "-", "h"),
    "Power(0.55)": ("#d62728", "-", "s"),
    "Power(0.58)": ("#2ca02c", "-", "o"),
}


def pretty_functional_name(fn: str) -> str:
    """Match TSV / nk header labels to short legend keys (plotted functionals only)."""
    if fn.startswith("Power(alpha="):
        alpha = fn.split("=", 1)[1].rstrip(")")
        try:
            return f"Power({float(alpha):.2f})"
        except ValueError:
            return fn
    if fn.startswith("optGM("):
        return "optGM"
    # Older TSVs used ``CHF``; kernel is now labeled ``CHF`` in the driver.
    if fn == "CHF":
        return "CHF"
    return fn


def legend_key_for_subset(fn: str) -> str | None:
    """Return legend series name if ``fn`` is one of the wanted curves, else None."""
    key = pretty_functional_name(fn)
    return key if key in WANTED_SET else None
