"""Re-render every image in docs/img from synthetic data (light and dark themes).

    python scripts/render_assets.py

Nothing here reads real data: cohorts are simulated with known parameters, fitted with
vvi, and only the resulting figures are written. The power and benchmark figures read
docs/results/{power,benchmark}.json, written by scripts/power_study.py and
scripts/benchmark.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import vvi
from vvi import plots

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
RES = ROOT / "docs" / "results"
MARKERS = {"300k sub-sample": 300_000, "full cohort ≈ 1.5M": 1_500_000}


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)

    cohort = vvi.simulate(600_000, seed=2024)
    agg = vvi.aggregate(cohort.people, cohort.events, horizon=cohort.horizon)
    fit = vvi.fit(agg, reference="BB")
    truth = cohort.truth.profile_effects

    wc = vvi.simulate(600_000, seed=77, truth=vvi.Truth(waning=vvi.WANING_EXAMPLE))
    wagg = vvi.aggregate(wc.people, wc.events, horizon=wc.horizon, bands=vvi.Bands())
    wfit = vvi.fit(wagg, reference="BB", covariates=("sex", "age_band", "since"))

    small = vvi.simulate(6_000, seed=21)
    exposures = vvi.histories(small, seed=3)
    with_events = set(small.events["id"])
    picks = []
    for prof in ("BB", "BBB", "BBI", "BBBI", "SS", "SSS", "SSI", "SSSI"):
        g = small.people[small.people["profile"] == prof]
        picks += [g[g["id"].isin(with_events)].head(1), g[~g["id"].isin(with_events)].head(1)]
    lanes = pd.concat(picks)

    power = json.loads((RES / "power.json").read_text())
    sizes = np.array(power["sizes"])
    curves = {k: np.array(v["power"]) for k, v in power["curves"].items()}
    sim = {
        k: (np.array(v["sizes"]), np.array(v["power"]), np.array(v["half_width"]))
        for k, v in power["monte_carlo"].items()
    }
    bench = pd.DataFrame(json.loads((RES / "benchmark.json").read_text())["rows"])

    for theme in ("light", "dark"):
        plots.set_theme(theme)
        plots.hero(fit, IMG / f"hero-{theme}.png")
        plots.recovery(fit, truth, IMG / f"recovery-{theme}.png")
        plots.calendar(fit, agg, IMG / f"calendar-{theme}.png")
        plots.waning(wfit, IMG / f"waning-{theme}.png", truth=vvi.WANING_EXAMPLE)
        plots.power_curves(
            sizes, curves, IMG / f"power-{theme}.png", simulated=sim, markers=MARKERS
        )
        plots.histories(
            lanes,
            exposures,
            small.events,
            IMG / f"histories-{theme}.png",
            blank_days=small.blank_days,
            horizon=small.horizon,
        )
        plots.benchmark(bench, IMG / f"benchmark-{theme}.png")

        plots.animate(
            lambda buf, p: plots.hero(fit, buf, progress=p), IMG / f"hero-{theme}.gif", frames=64
        )
        plots.animate(
            lambda buf, p: plots.recovery(fit, truth, buf, progress=p),
            IMG / f"recovery-{theme}.gif",
        )
        plots.animate(
            lambda buf, p: plots.power_curves(
                sizes, curves, buf, simulated=sim, markers=MARKERS, progress=p
            ),
            IMG / f"power-{theme}.gif",
        )
        plots.animate(
            lambda buf, p: plots.benchmark(bench, buf, progress=p),
            IMG / f"benchmark-{theme}.gif",
            frames=48,
        )
        print(f"{theme}: done")

    plots.set_theme("light")
    plots.social_preview(fit, IMG / "social-preview.png")
    print("wrote", len(list(IMG.iterdir())), "files to", IMG)


if __name__ == "__main__":
    main()
