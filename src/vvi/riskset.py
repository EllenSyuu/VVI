"""Calendar-time risk sets: the naive person-day panel and its sufficient statistics.

The model in :mod:`vvi.model` is a Bernoulli GLM on person-days. For a cohort of
1.5 million adults followed for a year that is ~5×10^8 rows — the original study
had to sub-sample and write one file per calendar day to fit it at all.

But the likelihood only depends on the data through, for each calendar day ``t``
and covariate cell ``c`` (profile × sex × age band),

* ``n[c, t]`` — people at risk, and
* ``d[c, t]`` — events.

A binomial GLM on those counts has *exactly* the same log-likelihood (up to a
constant), MLE and Fisher information as the person-day Bernoulli GLM. The table
has at most ``cells × days`` rows (8 × 2 × 3 × 365 ≈ 17 500), independent of the
cohort size, so the full population can be analysed without sampling.

Covariates that *change during follow-up* keep the identity exact as long as they
are piecewise constant: time since the last exposure is cut into :class:`Bands`,
each at-risk interval is split where a person crosses a band edge, and the band
becomes part of the cell key. A person then moves between cells over calendar
time — still with no person-day row.

:func:`aggregate` builds the table in ``O(people + events + cells × days)`` with
difference arrays over at-risk intervals. :func:`person_days` builds the naive
table and exists for testing and benchmarking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vvi.simulate import band_labels

DEFAULT_CELLS: tuple[str, ...] = ("profile", "sex", "age_band")


@dataclass(frozen=True)
class Bands:
    """Time-since-exposure bands, a covariate that changes during follow-up.

    ``edges`` are band starts in days since ``people[anchor]`` (the last exposure
    before entry). The default gives 0-89, 90-179, 180-269 and 270+ days.
    """

    edges: tuple[int, ...] = (0, 90, 180, 270)
    anchor: str = "anchor"
    name: str = "since"

    @property
    def labels(self) -> list[str]:
        return band_labels(self.edges)

    def index(self, offset: np.ndarray) -> np.ndarray:
        """Band index of each offset (days since anchor); offsets before the first edge -> 0."""
        idx = np.searchsorted(np.asarray(self.edges), np.asarray(offset), side="right") - 1
        return np.clip(idx, 0, len(self.edges) - 1)

    def split(self, intervals: pd.DataFrame, anchor: np.ndarray) -> pd.DataFrame:
        """Cut ``[start, stop)`` intervals at band edges; adds an integer ``band`` column.

        ``anchor`` is aligned with ``intervals`` (one value per row).
        """
        start = intervals["start"].to_numpy()
        stop = intervals["stop"].to_numpy()
        lows = np.r_[np.iinfo(np.int64).min // 4, np.asarray(self.edges[1:])]
        highs = np.r_[np.asarray(self.edges[1:]), np.iinfo(np.int64).max // 4]
        parts = []
        for k, (lo, hi) in enumerate(zip(lows, highs, strict=True)):
            a = np.maximum(start, anchor + lo)
            b = np.minimum(stop, anchor + hi)
            keep = b > a
            parts.append(
                pd.DataFrame(
                    {
                        "id": intervals["id"].to_numpy()[keep],
                        "start": a[keep],
                        "stop": b[keep],
                        "band": k,
                    }
                )
            )
        return pd.concat(parts, ignore_index=True)


def at_risk_intervals(people: pd.DataFrame, events: pd.DataFrame, blank_days: int) -> pd.DataFrame:
    """Split each person's follow-up ``[entry, exit)`` into at-risk intervals.

    The event day itself is at risk; the following ``blank_days`` days are not
    (same infection episode). Returns columns ``id, start, stop`` (half-open).
    """
    fu = people[["id", "entry", "exit"]].to_numpy()
    ev = events[["id", "day"]].to_numpy()
    if len(ev):
        ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]

    # Breakpoints per person: [entry, (e+1, e+1+blank) for each event, exit]
    entry = dict(zip(fu[:, 0], fu[:, 1], strict=False))
    exit_ = dict(zip(fu[:, 0], fu[:, 2], strict=False))
    ids, starts, stops = [], [], []
    ev_by_id: dict[int, list[int]] = {}
    for i, d in ev:
        ev_by_id.setdefault(int(i), []).append(int(d))

    # People without events: one interval, vectorised.
    has_ev = np.isin(fu[:, 0], np.fromiter(ev_by_id, dtype=fu.dtype, count=len(ev_by_id)))
    quiet = fu[~has_ev]
    ids.append(quiet[:, 0])
    starts.append(quiet[:, 1])
    stops.append(quiet[:, 2])

    # People with events (a small minority): explicit splitting.
    s_i, s_a, s_b = [], [], []
    for i, days in ev_by_id.items():
        a, end = entry[i], exit_[i]
        for d in days:
            s_i.append(i)
            s_a.append(a)
            s_b.append(min(d + 1, end))
            a = d + 1 + blank_days
        if a < end:
            s_i.append(i)
            s_a.append(a)
            s_b.append(end)
    ids.append(np.asarray(s_i, dtype=fu.dtype))
    starts.append(np.asarray(s_a, dtype=fu.dtype))
    stops.append(np.asarray(s_b, dtype=fu.dtype))

    out = pd.DataFrame(
        {"id": np.concatenate(ids), "start": np.concatenate(starts), "stop": np.concatenate(stops)}
    )
    return out[out["stop"] > out["start"]].sort_values(["id", "start"], ignore_index=True)


def aggregate(
    people: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon: int,
    blank_days: int = 28,
    cells: Sequence[str] = DEFAULT_CELLS,
    window: tuple[int, int] | None = None,
    bands: Bands | None = None,
) -> pd.DataFrame:
    """Collapse a cohort to ``(cell, day) -> (at_risk, events)``.

    Parameters
    ----------
    window:
        Optional analysis window ``[first_day, last_day)``; e.g. a variant era.
    bands:
        Optional time-since-exposure :class:`Bands`; adds a ``bands.name`` column
        (ordered categorical) and makes it part of the cell key.
    """
    cells = list(cells)
    keys = people[cells].astype(str)
    cell_index, cell_labels = pd.factorize(pd.MultiIndex.from_frame(keys))
    n_static = len(cell_labels)
    cell_of = pd.Series(cell_index, index=people["id"].to_numpy())
    n_bands = len(bands.edges) if bands else 1

    iv = at_risk_intervals(people, events, blank_days)
    if bands:
        anchor_of = pd.Series(people[bands.anchor].to_numpy(), index=people["id"].to_numpy())
        iv = bands.split(iv, anchor_of.loc[iv["id"].to_numpy()].to_numpy())
        band_iv = iv["band"].to_numpy()
    else:
        band_iv = 0
    c = cell_of.loc[iv["id"].to_numpy()].to_numpy() * n_bands + band_iv
    n_cells = n_static * n_bands
    diff = np.zeros((n_cells, horizon + 1), dtype=np.int64)
    np.add.at(diff, (c, iv["start"].to_numpy()), 1)
    np.add.at(diff, (c, iv["stop"].to_numpy()), -1)
    at_risk = np.cumsum(diff[:, :horizon], axis=1)

    d = np.zeros((n_cells, horizon), dtype=np.int64)
    if len(events):
        ev_id = events["id"].to_numpy()
        ev_day = events["day"].to_numpy()
        ev_c = cell_of.loc[ev_id].to_numpy() * n_bands
        if bands:
            ev_c = ev_c + bands.index(ev_day - anchor_of.loc[ev_id].to_numpy())
        np.add.at(d, (ev_c, ev_day), 1)

    cc, tt = np.nonzero(at_risk)
    static = cc // n_bands
    columns = {
        name: cell_labels.get_level_values(k).to_numpy()[static] for k, name in enumerate(cells)
    }
    if bands:
        columns[bands.name] = pd.Categorical.from_codes(
            cc % n_bands, categories=bands.labels, ordered=True
        )
    out = pd.DataFrame({**columns, "day": tt, "at_risk": at_risk[cc, tt], "events": d[cc, tt]})
    if window is not None:
        lo, hi = window
        out = out[(out["day"] >= lo) & (out["day"] < hi)]
    if (out["events"] > out["at_risk"]).any():  # pragma: no cover - invariant
        raise AssertionError("more events than people at risk")
    return out.reset_index(drop=True)


def person_days(
    people: pd.DataFrame,
    events: pd.DataFrame,
    *,
    blank_days: int = 28,
    cells: Sequence[str] = DEFAULT_CELLS,
    bands: Bands | None = None,
) -> pd.DataFrame:
    """Materialise the naive one-row-per-person-per-at-risk-day table."""
    iv = at_risk_intervals(people, events, blank_days)
    lengths = (iv["stop"] - iv["start"]).to_numpy()
    rep = np.repeat(np.arange(len(iv)), lengths)
    offset = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    pd_ = pd.DataFrame(
        {"id": iv["id"].to_numpy()[rep], "day": iv["start"].to_numpy()[rep] + offset}
    )
    hit = pd.MultiIndex.from_frame(events[["id", "day"]])
    pd_["event"] = pd.MultiIndex.from_frame(pd_[["id", "day"]]).isin(hit).astype(np.int8)
    attrs = people.set_index("id")[list(cells)]
    pd_ = pd_.join(attrs, on="id")
    if bands:
        anchor = people.set_index("id")[bands.anchor].loc[pd_["id"].to_numpy()].to_numpy()
        pd_[bands.name] = pd.Categorical.from_codes(
            bands.index(pd_["day"].to_numpy() - anchor), categories=bands.labels, ordered=True
        )
    return pd_
