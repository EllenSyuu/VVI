"""Synthetic cohort with a known data-generating process.

Every number in this module is invented. The simulator exists so the estimation
pipeline can be tested end-to-end against ground truth, and so the repository can
demonstrate the method without touching any restricted record.

Data-generating process (daily discrete time, Andersen–Gill style recurrent events)::

    cloglog P(event on day t | at risk) = eta(t) + alpha' x_i + beta[profile_i]
                                          + gamma[band(t - anchor_i)]

* ``eta(t)`` is a two-wave epidemic curve (a winter wave and a summer–autumn wave);
* ``x_i`` holds sex and age band; ``profile_i`` is fixed at enrolment;
* ``anchor_i`` is the day of the last exposure (dose or infection) before entry, and
  ``gamma`` an optional waning effect by time since that exposure (zero by default);
* after each event the person is not at risk for ``blank_days`` days (same episode);
* people enter on a staggered date and may be censored (death, emigration, ...).

:func:`histories` back-fills dated doses and infections consistent with each
person's profile and anchor, so the full pipeline — :func:`vvi.sequence.encode` on
raw dated events, then aggregation, then the fit — can be exercised end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

AGE_BANDS = ("18-59", "60-79", "80+")
SEXES = ("F", "M")


def epidemic_curve(days: np.ndarray, *, base: float = -10.0) -> np.ndarray:
    """Log daily hazard ``eta(t)`` for an illustrative two-wave year."""
    t = np.asarray(days, dtype=float)
    wave1 = 3.6 * np.exp(-0.5 * ((t - 70) / 18) ** 2)
    wave2 = 2.6 * np.exp(-0.5 * ((t - 250) / 45) ** 2)
    return base + np.log(np.exp(wave1) + np.exp(wave2) - 1.0)


@dataclass
class Truth:
    """Ground-truth parameters of the simulated cohort (log hazard ratios)."""

    profile_effects: dict[str, float] = field(
        default_factory=lambda: {
            # reference: two doses, no prior infection
            "BB": 0.00,
            "BBB": -0.45,
            "BBI": -1.60,
            "BBBI": -1.35,  # a booster *weaker* than BBI: an imprinting-like pattern
            "SS": 0.25,
            "SSS": -0.10,
            "SSI": -1.30,
            "SSSI": -1.40,
        }
    )
    profile_weights: dict[str, float] = field(
        default_factory=lambda: {
            "BB": 0.12,
            "BBB": 0.24,
            "BBI": 0.06,
            "BBBI": 0.10,
            "SS": 0.12,
            "SSS": 0.22,
            "SSI": 0.06,
            "SSSI": 0.08,
        }
    )
    sex_male: float = 0.08
    age_effects: dict[str, float] = field(
        default_factory=lambda: {"18-59": 0.0, "60-79": -0.20, "80+": -0.30}
    )
    age_weights: dict[str, float] = field(
        default_factory=lambda: {"18-59": 0.62, "60-79": 0.30, "80+": 0.08}
    )
    reference: str = "BB"
    #: Log hazard ratio by time since the last exposure; keys are band labels of
    #: ``waning_edges`` (see :class:`vvi.riskset.Bands`). ``None`` = no waning.
    waning: dict[str, float] | None = None
    waning_edges: tuple[int, ...] = (0, 90, 180, 270)


WANING_EXAMPLE: dict[str, float] = {"0-89d": 0.0, "90-179d": 0.25, "180-269d": 0.45, "270d+": 0.60}
"""A waning pattern for demonstrations: protection fades as time since exposure grows."""


@dataclass
class Cohort:
    """A simulated cohort: one row per person plus a table of event days."""

    people: pd.DataFrame  # id, sex, age_band, profile, anchor, entry, exit
    events: pd.DataFrame  # id, day
    horizon: int
    blank_days: int
    truth: Truth


def simulate(
    n: int = 100_000,
    *,
    horizon: int = 365,
    blank_days: int = 28,
    max_entry_day: int = 90,
    annual_censoring: float = 0.03,
    truth: Truth | None = None,
    seed: int | None = 2024,
) -> Cohort:
    """Simulate ``n`` people over ``horizon`` calendar days.

    The loop runs over days (vectorised over people), so cost is
    ``O(n * horizon)`` time but only ``O(n)`` memory.
    """
    truth = truth or Truth()
    rng = np.random.default_rng(seed)

    profiles = np.array(list(truth.profile_weights))
    pw = np.array([truth.profile_weights[p] for p in profiles])
    profile = rng.choice(profiles, size=n, p=pw / pw.sum())

    bands = np.array(AGE_BANDS)
    aw = np.array([truth.age_weights[a] for a in bands])
    age_band = rng.choice(bands, size=n, p=aw / aw.sum())
    sex = rng.choice(np.array(SEXES), size=n)

    lin = (
        np.vectorize(truth.profile_effects.__getitem__)(profile)
        + np.vectorize(truth.age_effects.__getitem__)(age_band)
        + truth.sex_male * (sex == "M")
    )

    entry = rng.integers(0, max_entry_day + 1, size=n)
    daily_censor = -np.log1p(-annual_censoring) / 365.0
    exit_ = np.minimum(entry + rng.geometric(1 - np.exp(-daily_censor), size=n), horizon)
    # Last exposure 14-240 days before entry (14 days: a dose or infection needs time).
    anchor = entry - rng.integers(14, 241, size=n)

    edges = np.asarray(truth.waning_edges)
    if truth.waning:
        labels = band_labels(truth.waning_edges)
        gamma = np.array([truth.waning.get(lab, 0.0) for lab in labels])
    else:
        gamma = np.zeros(len(edges))

    eta = epidemic_curve(np.arange(horizon))
    blocked_until = np.full(n, -1)  # last day of the current not-at-risk window
    ev_id: list[np.ndarray] = []
    ev_day: list[np.ndarray] = []
    for t in range(horizon):
        at_risk = (entry <= t) & (t < exit_) & (blocked_until < t)
        idx = np.flatnonzero(at_risk)
        if idx.size == 0:
            continue
        lp = eta[t] + lin[idx]
        if truth.waning:
            lp = lp + gamma[np.searchsorted(edges, t - anchor[idx], side="right") - 1]
        p = -np.expm1(-np.exp(lp))
        hit = idx[rng.random(idx.size) < p]
        if hit.size:
            ev_id.append(hit)
            ev_day.append(np.full(hit.size, t))
            blocked_until[hit] = t + blank_days

    people = pd.DataFrame(
        {
            "id": np.arange(n),
            "sex": pd.Categorical(sex, categories=SEXES),
            "age_band": pd.Categorical(age_band, categories=AGE_BANDS),
            "profile": profile,
            "anchor": anchor,
            "entry": entry,
            "exit": exit_,
        }
    )
    events = pd.DataFrame(
        {
            "id": np.concatenate(ev_id) if ev_id else np.array([], dtype=int),
            "day": np.concatenate(ev_day) if ev_day else np.array([], dtype=int),
        }
    )
    return Cohort(people, events, horizon, blank_days, truth)


def band_labels(edges: tuple[int, ...] | list[int]) -> list[str]:
    """``(0, 90, 180)`` -> ``["0-89d", "90-179d", "180d+"]``."""
    e = list(edges)
    return [f"{a}-{b - 1}d" for a, b in zip(e[:-1], e[1:], strict=True)] + [f"{e[-1]}d+"]


_PRODUCTS = {"B": "Comirnaty", "S": "CoronaVac", "O": "Comirnaty bivalent"}


def histories(cohort: Cohort, *, seed: int | None = 0) -> pd.DataFrame:
    """Dated exposure events consistent with each person's profile and anchor.

    Returns ``id, day, kind`` where ``kind`` is a product name or ``"infection"``
    and ``day`` is relative to the start of follow-up (negative = before it). The
    last exposure falls on the person's ``anchor``; earlier doses are spaced like a
    primary series (21-35 days) and a booster (120-270 days). Encoding these events
    with :func:`vvi.sequence.encode` recovers each person's profile.
    """
    rng = np.random.default_rng(seed)
    ppl = cohort.people
    rows_id: list[np.ndarray] = []
    rows_day: list[np.ndarray] = []
    rows_kind: list[np.ndarray] = []
    for profile, grp in ppl.groupby("profile", sort=False):
        ids = grp["id"].to_numpy()
        day = grp["anchor"].to_numpy().copy()
        k = len(ids)
        symbols = list(profile)
        # walk backwards from the anchor
        for pos in range(len(symbols) - 1, -1, -1):
            sym = symbols[pos]
            kind = "infection" if sym in "IPA" else _PRODUCTS[sym]
            rows_id.append(ids)
            rows_day.append(day.copy())
            rows_kind.append(np.full(k, kind, dtype=object))
            if pos == 0:
                break
            prev_is_dose = symbols[pos - 1] not in "IPA"
            if sym in "IPA":  # infection some months after the last dose
                gap = rng.integers(60, 300, size=k)
            elif pos == 1 and prev_is_dose:  # second dose of the primary series
                gap = rng.integers(21, 36, size=k)
            else:  # booster
                gap = rng.integers(120, 271, size=k)
            day = day - gap
    out = pd.DataFrame(
        {
            "id": np.concatenate(rows_id),
            "day": np.concatenate(rows_day),
            "kind": np.concatenate(rows_kind),
        }
    )
    return out.sort_values(["id", "day"], ignore_index=True)
