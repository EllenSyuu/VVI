"""Command line: ``vvi demo`` runs the full pipeline on a synthetic cohort."""

from __future__ import annotations

import argparse
import time

from vvi import aggregate, fit, simulate


def demo(n: int, seed: int, df: int) -> None:
    t0 = time.perf_counter()
    cohort = simulate(n, seed=seed)
    t1 = time.perf_counter()
    agg = aggregate(
        cohort.people, cohort.events, horizon=cohort.horizon, blank_days=cohort.blank_days
    )
    t2 = time.perf_counter()
    model = fit(agg, reference=cohort.truth.reference, df=df)
    t3 = time.perf_counter()

    person_days = int(agg["at_risk"].sum())
    print(f"synthetic cohort : {n:,} people, {len(cohort.events):,} infections")
    print(
        f"risk set         : {person_days:,} person-days -> {len(agg):,} rows "
        f"({person_days / len(agg):,.0f}x smaller)"
    )
    print(
        f"timings          : simulate {t1 - t0:.1f}s | aggregate {t2 - t1:.2f}s | "
        f"fit {t3 - t2:.2f}s\n"
    )

    table = model.table()
    truth = cohort.truth.profile_effects
    table["true_hr"] = [
        f"{2.718281828 ** truth[t]:.3f}" if t in truth else "" for t in table["term"]
    ]
    print(table.round(3).to_string(index=False))
    for platform in ("B", "S"):
        c = model.contrast(platform * 3 + "I", platform * 2 + "I")
        lo, hi = c.ci
        print(
            f"\nimprinting contrast {c.term}: HR {c.hr:.2f} (95% CI {lo:.2f}-{hi:.2f}), p={c.p:.2g}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vvi", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="simulate a cohort, aggregate it and fit the model")
    d.add_argument("--n", type=int, default=300_000)
    d.add_argument("--seed", type=int, default=2024)
    d.add_argument("--df", type=int, default=7, help="B-spline degrees of freedom")
    args = parser.parse_args(argv)
    if args.cmd == "demo":
        demo(args.n, args.seed, args.df)


if __name__ == "__main__":  # pragma: no cover
    main()
