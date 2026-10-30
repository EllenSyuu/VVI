"""Build the static explorer site from synthetic data.

    python scripts/build_site.py [OUT_DIR]      # default: build/site

For each planted imprinting strength (true HR of BBBI vs BBI) and cohort size, a cohort
is simulated, aggregated and fitted with vvi. The page template is copied from
docs/explorer/index.html and the results written next to it as data.json, together
with the power and benchmark records in docs/results. No real data is read.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

import vvi
from vvi import Truth, aggregate, fit, simulate

ROOT = Path(__file__).resolve().parents[1]
HRS = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)
SIZES = (100_000, 300_000, 1_000_000)
PROFILES = ("BBB", "BBI", "BBBI", "SS", "SSS", "SSI", "SSSI")


def _effect(e: vvi.Effect, truth: float) -> dict:
    lo, hi = e.ci
    return dict(hr=round(e.hr, 4), lo=round(lo, 4), hi=round(hi, 4), truth=round(truth, 4))


def scenario(hr: float, n: int, seed: int) -> dict:
    t = Truth()
    t.profile_effects = dict(t.profile_effects)
    t.profile_effects["BBBI"] = t.profile_effects["BBI"] + float(np.log(hr))
    c = simulate(n, seed=seed, truth=t)
    agg = aggregate(c.people, c.events, horizon=c.horizon)
    f = fit(agg, reference="BB")
    eff = t.profile_effects
    profiles = [
        {"profile": p, **_effect(f.profile_effect(p), float(np.exp(eff[p])))} for p in PROFILES
    ]
    contrasts = []
    for a, b in (("BBBI", "BBI"), ("SSSI", "SSI")):
        e = f.contrast(a, b)
        contrasts.append(
            {
                "pair": f"{a} vs {b}",
                "p": float(f"{e.p:.3g}"),
                **_effect(e, float(np.exp(eff[a] - eff[b]))),
            }
        )
    return dict(
        hr=hr,
        n=n,
        events=int(len(c.events)),
        person_days=int(agg["at_risk"].sum()),
        rows=len(agg),
        profiles=profiles,
        contrasts=contrasts,
    )


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = []
    for i, hr in enumerate(HRS):
        for j, n in enumerate(SIZES):
            scenarios.append(scenario(hr, n, seed=7000 + 10 * i + j))
            print(f"HR {hr} · {n:,} people")
    res = ROOT / "docs" / "results"
    data = {
        "version": vvi.__version__,
        "scenarios": scenarios,
        "power": json.loads((res / "power.json").read_text()),
        "benchmark": json.loads((res / "benchmark.json").read_text()),
    }
    (out_dir / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    shutil.copy(ROOT / "docs" / "explorer" / "index.html", out_dir / "index.html")
    print(f"site written to {out_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "site")
