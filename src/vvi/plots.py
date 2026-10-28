"""Publication-style figures (matplotlib; install the ``plot`` extra)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from vvi.model import HybridImmunityFit
from vvi.sequence import parse

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
PLATFORM_COLORS = {"B": "#2a78d6", "S": "#eb6834", None: "#52514e"}
PLATFORM_NAMES = {"B": "Comirnaty (B)", "S": "CoronaVac (S)"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    return ax


def hr_axis(ax, ticks=(0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3)):
    """Log-scaled hazard-ratio axis with plain decimal tick labels."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    ax.xaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    return ax


def forest(
    fit: HybridImmunityFit,
    ax=None,
    *,
    profiles: Sequence[str] | None = None,
    truth: Mapping[str, float] | None = None,
    title: str | None = None,
):
    """Hazard ratios versus the reference profile, one row per profile."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    style(ax)
    profiles = [p for p in (profiles or fit.profiles) if p != fit.reference]
    profiles = sorted(profiles, key=lambda p: (parse(p).platform or "", parse(p).infected, len(p)))
    y = np.arange(len(profiles))[::-1]
    for yi, p in zip(y, profiles, strict=False):
        e = fit.profile_effect(p)
        lo, hi = e.ci
        color = PLATFORM_COLORS.get(parse(p).platform, INK_2)
        ax.plot([lo, hi], [yi, yi], color=color, lw=2, solid_capstyle="round")
        ax.plot(e.hr, yi, "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=3)
        if truth is not None and p in truth:
            ax.plot(
                np.exp(truth[p] - truth.get(fit.reference, 0.0)),
                yi,
                marker="|",
                ms=14,
                mew=2,
                color=INK,
                zorder=4,
            )
        ax.text(
            1.02,
            yi,
            f"{e.hr:.2f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=9,
            color=INK_2,
        )
    ax.axvline(1, color=MUTED, lw=1, ls=(0, (3, 3)))
    hr_axis(ax)
    ax.set_yticks(y, profiles, fontsize=10, color=INK, family="monospace")
    ax.set_xlabel(f"Hazard ratio vs {fit.reference} (log scale, 95% CI)", color=INK_2, fontsize=9)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    return ax


def trend(fit: HybridImmunityFit, agg: pd.DataFrame, ax=None, *, title: str | None = None):
    """Observed daily incidence (reference cell) with the fitted calendar-time curve."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.2), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.grid(axis="x", visible=False)
    ref = agg[
        (agg["profile"] == fit.reference) & (agg["sex"] == "F") & (agg["age_band"] == "18-59")
    ]
    daily = ref.groupby("day")[["events", "at_risk"]].sum()
    rate = daily["events"] / daily["at_risk"] * 1e4
    ax.bar(rate.index, rate.values, width=1.0, color="#c9dcf3", lw=0, label="observed")
    days = np.arange(agg["day"].min(), agg["day"].max() + 1)
    ax.plot(
        days,
        fit.baseline_hazard(days) * 1e4,
        color=PLATFORM_COLORS["B"],
        lw=2,
        label="fitted B-spline",
    )
    ax.set_ylabel("Daily infections per 10,000 at risk", color=INK_2, fontsize=9)
    ax.set_xlabel("Calendar day", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    return ax
