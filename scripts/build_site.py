"""Precompute the interactive explorer's data (site/data.json) — synthetic only.

    python scripts/build_site.py

For each planted imprinting strength (true HR of BBBI vs BBI) and cohort size, a
cohort is simulated, aggregated and fitted with vvi; the page then lets readers move
between scenarios. Power curves and the benchmark are read from docs/results.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

import vvi
from vvi import Truth, aggregate, fit, simulate

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
HRS = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)
SIZES = (100_000, 300_000, 1_000_000)
PROFILES = ("BBB", "BBI", "BBBI", "SS", "SSS", "SSI", "SSSI")


def scenario(hr: float, n: int, seed: int) -> dict:
    t = Truth()
    t.profile_effects = dict(t.profile_effects)
    t.profile_effects["BBBI"] = t.profile_effects["BBI"] + float(np.log(hr))
    c = simulate(n, seed=seed, truth=t)
    agg = aggregate(c.people, c.events, horizon=c.horizon)
    f = fit(agg, reference="BB")
    rows = []
    for p in PROFILES:
        e = f.profile_effect(p)
        lo, hi = e.ci
        rows.append(
            dict(
                profile=p,
                hr=round(e.hr, 4),
                lo=round(lo, 4),
                hi=round(hi, 4),
                truth=round(float(np.exp(t.profile_effects[p])), 4),
            )
        )
    con = []
    for a, b in (("BBBI", "BBI"), ("SSSI", "SSI")):
        e = f.contrast(a, b)
        lo, hi = e.ci
        con.append(
            dict(
                pair=f"{a} vs {b}",
                hr=round(e.hr, 4),
                lo=round(lo, 4),
                hi=round(hi, 4),
                p=float(f"{e.p:.3g}"),
                truth=round(float(np.exp(t.profile_effects[a] - t.profile_effects[b])), 4),
            )
        )
    return dict(
        hr=hr,
        n=n,
        events=int(len(c.events)),
        person_days=int(agg["at_risk"].sum()),
        rows=len(agg),
        profiles=rows,
        contrasts=con,
    )


def main() -> None:
    SITE.mkdir(exist_ok=True)
    scenarios = []
    for i, hr in enumerate(HRS):
        for j, n in enumerate(SIZES):
            scenarios.append(scenario(hr, n, seed=7000 + 10 * i + j))
            print(f"HR {hr} n {n:,} done")
    res = ROOT / "docs" / "results"
    data = {
        "version": vvi.__version__,
        "scenarios": scenarios,
        "power": json.loads((res / "power.json").read_text()),
        "benchmark": json.loads((res / "benchmark.json").read_text()),
    }
    (SITE / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    assets = SITE / "assets"
    assets.mkdir(exist_ok=True)
    for name in ("plate.svg", "pipeline.svg", "fig1_histories.png", "fig4_waning.png"):
        shutil.copy(ROOT / "docs" / "figures" / name, assets / name)
    shutil.copy(ROOT / "src" / "vvi" / "assets" / "grain.png", assets / "grain.png")
    print("wrote", SITE / "data.json")


if __name__ == "__main__":
    main()
