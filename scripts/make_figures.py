"""Draw every README figure as an Atelier plate, from synthetic cohorts only.

    python scripts/make_figures.py          # writes docs/figures/fig*.png

Reads docs/results/{benchmark,power}.json (written by benchmark.py and
power_study.py) for the last two plates; everything else is simulated here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.lines
import matplotlib.patches

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import vvi  # noqa: E402
from vvi import plots as P  # noqa: E402
from vvi.simulate import epidemic_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
RES = ROOT / "docs" / "results"


def fig_histories() -> None:
    c = vvi.simulate(6_000, seed=21)
    ex = vvi.histories(c, seed=3)
    ev = c.events
    with_ev = set(ev["id"])
    picks = []
    for prof in ("BB", "BBB", "BBI", "BBBI", "SS", "SSS", "SSI", "SSSI"):
        g = c.people[c.people["profile"] == prof]
        a = g[g["id"].isin(with_ev)].head(1)
        b = g[~g["id"].isin(with_ev)].head(1)
        picks += [a, b]
    people = pd.concat(picks)
    fig = P.plate(
        (9.2, 6.2),
        kicker="Fig. 1 · the data, as the model sees it",
        title="Exposure histories become calendar-time risk sets",
        note="one lane per person",
    )
    ax = fig.add_axes([0.13, 0.16, 0.84, 0.63])
    P.histories(
        people,
        ex,
        ev,
        ax,
        blank_days=c.blank_days,
        horizon=c.horizon,
        curve=np.exp(epidemic_curve(np.arange(c.horizon))),
    )
    y_top = len(people) + 2.4
    P.hand(ax, 150, y_top + 0.2, "two epidemic waves", size=12, color=P.GRAPH)
    legend = [
        ("dose · Comirnaty", P.BLUE, "|"),
        ("dose · CoronaVac", P.ORANGE, "|"),
        ("first infection", P.INK, "o"),
        ("reinfection (event)", P.ORANGE, "o"),
    ]
    for k, (lab, col, m) in enumerate(legend):
        x0 = 0.11 + k * 0.165
        fig.text(x0 + 0.014, 0.06, lab, fontproperties=P._fp("label", 10), color=P.INK, va="center")
        fig.lines.append(
            matplotlib.lines.Line2D(
                [x0 + 0.004],
                [0.062],
                transform=fig.transFigure,
                marker=m,
                ls="",
                color=col,
                ms=7 if m == "o" else 9,
                mew=2.0 if m == "|" else 1,
                mec=P.PAPER if m == "o" else col,
            )
        )
    x0 = 0.11 + 4 * 0.165
    fig.patches.append(
        matplotlib.patches.Rectangle(
            (x0 - 0.004, 0.054),
            0.022,
            0.016,
            transform=fig.transFigure,
            facecolor="none",
            edgecolor=P.GRAPH,
            hatch="//////",
            lw=0,
        )
    )
    fig.text(
        x0 + 0.024,
        0.06,
        "28-day episode window",
        fontproperties=P._fp("label", 10),
        color=P.INK,
        va="center",
    )
    P.save(fig, OUT / "fig1_histories.png")


def fit_main():
    c = vvi.simulate(600_000, seed=2024)
    agg = vvi.aggregate(c.people, c.events, horizon=c.horizon)
    return c, agg, vvi.fit(agg, reference="BB")


def fig_trend(c, agg, model) -> None:
    fig = P.plate(
        (9.2, 4.2),
        kicker="Fig. 2 · calendar time",
        title="The epidemic curve is absorbed by a smooth calendar baseline",
        note="cubic B-spline, df = 7",
    )
    ax = fig.add_axes([0.1, 0.14, 0.86, 0.58])
    P.trend(model, agg, ax)
    P.hand(
        ax,
        0.99,
        0.86,
        "dots: observed daily rate, reference cell",
        transform=ax.transAxes,
        size=10.5,
        color=P.GRAPH,
        ha="right",
    )
    P.save(fig, OUT / "fig2_calendar.png")


def fig_recovery(c, agg, model) -> None:
    fig = P.plate(
        (9.2, 7.0),
        kicker="Fig. 3 · validation against a planted truth",
        title="Every hazard ratio lands on its true value",
        note="600,000 synthetic people",
    )
    ax = fig.add_axes([0.13, 0.40, 0.72, 0.43])
    P.forest(model, ax, truth=c.truth.profile_effects)
    P.hand(
        ax,
        0.995,
        1.02,
        "dot: estimate · bar: 95% CI · ink tick: truth",
        transform=ax.transAxes,
        size=10.5,
        color=P.GRAPH,
        ha="right",
    )
    ax2 = fig.add_axes([0.13, 0.085, 0.72, 0.13])
    P.contrasts(
        model,
        [("BBBI", "BBI"), ("SSSI", "SSI")],
        ax2,
        truth=c.truth.profile_effects,
        ticks=(0.7, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75),
    )
    P.caps(
        ax2,
        0.0,
        1.38,
        "the imprinting contrast · extra pre-infection dose",
        transform=ax2.transAxes,
        color=P.INK,
    )
    P.hand(
        ax2,
        0.995,
        1.34,
        "planted: HR 1.28 for B, 0.90 for S",
        transform=ax2.transAxes,
        size=10.5,
        color=P.ORANGE,
        ha="right",
    )
    P.axis_label(ax2, "")
    P.save(fig, OUT / "fig3_recovery.png")


def fig_waning() -> None:
    c = vvi.simulate(600_000, seed=77, truth=vvi.Truth(waning=vvi.WANING_EXAMPLE))
    b = vvi.Bands()
    agg = vvi.aggregate(c.people, c.events, horizon=c.horizon, bands=b)
    f = vvi.fit(agg, reference="BB", covariates=("sex", "age_band", b.name))
    fig = P.plate(
        (9.2, 4.4),
        kicker="Fig. 4 · a covariate that changes during follow-up",
        title="Protection wanes with time since the last exposure",
        note="still exact: people move between cells",
    )
    ax = fig.add_axes([0.1, 0.21, 0.86, 0.5])
    P.waning(f, ax, truth=vvi.WANING_EXAMPLE)
    P.hand(
        ax,
        0.02,
        0.88,
        "dotted: planted truth · orange: 95% CI",
        transform=ax.transAxes,
        size=10.5,
        color=P.GRAPH,
    )
    P.save(fig, OUT / "fig4_waning.png")


def fig_power() -> None:
    res = json.loads((RES / "power.json").read_text())
    sizes = np.array(res["sizes"])
    curves = {k: np.array(v["power"]) for k, v in res["curves"].items()}
    sim = {
        k: (np.array(v["sizes"]), np.array(v["power"]), np.array(v["half_width"]))
        for k, v in res["monte_carlo"].items()
    }
    fig = P.plate(
        (9.2, 4.6),
        kicker="Fig. 5 · why sampling-free matters",
        title="The whole cohort can see effects a sub-sample cannot",
        note="BBBI vs BBI · two-sided test at 5%",
    )
    ax = fig.add_axes([0.1, 0.2, 0.86, 0.52])
    P.power_curves(
        sizes,
        curves,
        ax,
        simulated=sim,
        markers={"300k sub-sample": 300_000, "~1.5M full cohort": 1_500_000},
    )
    P.hand(
        ax,
        0.02,
        0.9,
        f"curves: analytic, from one fit · dots: {res['reps']} simulations each",
        transform=ax.transAxes,
        size=10.5,
        color=P.GRAPH,
    )
    P.save(fig, OUT / "fig5_power.png")


def fig_benchmark() -> None:
    rows = pd.DataFrame(json.loads((RES / "benchmark.json").read_text())["rows"])
    fig = P.plate(
        (9.2, 4.4),
        kicker="Fig. 6 · cost",
        title="Same likelihood, a table that does not grow with the cohort",
        note="2 vCPU · 7 GB RAM",
    )
    a1 = fig.add_axes([0.09, 0.2, 0.38, 0.47])
    a2 = fig.add_axes([0.58, 0.2, 0.38, 0.47])
    P.benchmark(rows, [a1, a2])
    P.save(fig, OUT / "fig6_benchmark.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    P.fonts()
    fig_histories()
    c, agg, model = fit_main()
    fig_trend(c, agg, model)
    fig_recovery(c, agg, model)
    fig_waning()
    if (RES / "power.json").exists():
        fig_power()
    fig_benchmark()
    print("wrote", *sorted(p.name for p in OUT.glob("fig*.png")))


if __name__ == "__main__":
    main()
