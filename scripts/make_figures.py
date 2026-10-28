"""Regenerate the README figures from a synthetic cohort (no real data involved).

python scripts/make_figures.py            # writes docs/figures/*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vvi import aggregate, fit, simulate  # noqa: E402
from vvi.plots import (  # noqa: E402
    INK,
    INK_2,
    PLATFORM_COLORS,
    SURFACE,
    forest,
    hr_axis,
    style,
    trend,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": SURFACE})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = simulate(600_000, seed=2024)
    agg = aggregate(cohort.people, cohort.events, horizon=cohort.horizon)
    model = fit(agg, reference="BB")

    # 1. Recovery forest plot
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    forest(model, ax, truth=cohort.truth.profile_effects)
    ax.set_title(
        "Estimated hazard ratios recover the planted truth",
        loc="left",
        color=INK,
        fontsize=12,
        fontweight="bold",
        pad=22,
    )
    ax.text(
        0,
        1.03,
        "Synthetic cohort, 600,000 people · dots = estimate with 95% CI · black tick = true value",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_2,
    )
    handles = [
        plt.Line2D([], [], color=PLATFORM_COLORS[k], marker="o", lw=2, label=v)
        for k, v in (("B", "Comirnaty (B)"), ("S", "CoronaVac (S)"))
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "forest_recovery.png", dpi=200)
    plt.close(fig)

    # 2. Calendar-time trend
    fig, ax = plt.subplots(figsize=(7.2, 3.4), facecolor=SURFACE)
    trend(model, agg, ax)
    ax.set_title(
        "Calendar time absorbs the epidemic waves",
        loc="left",
        color=INK,
        fontsize=12,
        fontweight="bold",
        pad=22,
    )
    ax.text(
        0,
        1.03,
        "Reference cell (BB, female, 18-59) · cubic B-spline, df = 7",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_2,
    )
    fig.tight_layout()
    fig.savefig(OUT / "calendar_trend.png", dpi=200)
    plt.close(fig)

    # 3. Imprinting contrasts
    fig, ax = plt.subplots(figsize=(7.2, 2.4), facecolor=SURFACE)
    style(ax)
    rows = [("B", "BBBI", "BBI"), ("S", "SSSI", "SSI")]
    for y, (plat, a, b) in zip((1, 0), rows, strict=False):
        c = model.contrast(a, b)
        lo, hi = c.ci
        ax.plot([lo, hi], [y, y], color=PLATFORM_COLORS[plat], lw=2, solid_capstyle="round")
        ax.plot(c.hr, y, "o", ms=8, color=PLATFORM_COLORS[plat], mec=SURFACE, mew=2)
        t = cohort.truth.profile_effects
        ax.plot(np.exp(t[a] - t[b]), y, marker="|", ms=14, mew=2, color=INK)
        ax.text(
            1.02,
            y,
            f"{c.hr:.2f} ({lo:.2f}-{hi:.2f})",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=9,
            color=INK_2,
        )
    ax.axvline(1, color="#8a8984", lw=1, ls=(0, (3, 3)))
    hr_axis(ax, ticks=(0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75))
    ax.set_yticks([1, 0], ["BBBI vs BBI", "SSSI vs SSI"], family="monospace", color=INK)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel(
        "Hazard ratio of reinfection (> 1: extra pre-infection dose, more reinfection)",
        color=INK_2,
        fontsize=9,
    )
    ax.set_title(
        "The imprinting question is a single contrast",
        loc="left",
        color=INK,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "imprinting_contrast.png", dpi=200)
    plt.close(fig)
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
