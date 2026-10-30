"""Encode individual exposure histories as vaccination–infection *profiles*.

A profile is a short string read left to right in calendar order, e.g. ``"BBBI"``
= three Comirnaty doses followed by a first infection. The alphabet is:

========  =================================================
``B``     Comirnaty (BNT162b2, mRNA)
``S``     CoronaVac (Sinovac, inactivated)
``O``     Comirnaty Original/Omicron BA.4-5 bivalent booster
``I``     confirmed infection (variant-agnostic)
``P``     infection while BA.1/2 dominated
``A``     infection while BA.4/5 dominated
========  =================================================

The encoder applies the two cleaning rules used in the original analysis:

1. a positive test within ``episode_gap`` days (default 28) of the previous one
   belongs to the same episode and is dropped;
2. infections *after* the first one are outcomes, not exposures, so a profile
   keeps at most one trailing infection symbol (``"BBIII" -> "BBI"``).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

#: Default mapping from product names (as they appear in registries) to symbols.
PRODUCT_CODES: Mapping[str, str] = {
    "comirnaty": "B",
    "biontech": "B",
    "bnt162b2": "B",
    "pfizer-biontech": "B",
    "coronavac": "S",
    "sinovac": "S",
    "biontech-bv": "O",
    "biontech/fosun bivalent_o": "O",
    "comirnaty bivalent": "O",
}

VACCINE_SYMBOLS = frozenset("BSO")
INFECTION_SYMBOLS = frozenset("IPA")

_PROFILE_RE = re.compile(r"^[BSO]*[IPA]?$")


@dataclass(frozen=True)
class Event:
    """One dated exposure: a vaccine dose (product name) or an infection."""

    day: date
    kind: str  # product name, or "infection"


@dataclass(frozen=True)
class Profile:
    """Parsed view of a profile string."""

    code: str
    doses: int
    products: str
    infected: bool
    variant: str | None

    @property
    def homologous(self) -> bool:
        """True when every dose is the same product (mixed schedules excluded)."""
        return len(set(self.products)) <= 1

    @property
    def platform(self) -> str | None:
        """Single product symbol for homologous schedules, else ``None``."""
        return self.products[0] if self.products and self.homologous else None


def product_symbol(name: str, codes: Mapping[str, str] = PRODUCT_CODES) -> str:
    """Map a product name to its symbol, case- and whitespace-insensitively."""
    key = " ".join(name.strip().lower().split())
    try:
        return codes[key]
    except KeyError as err:
        raise ValueError(f"unknown vaccine product: {name!r}") from err


def encode(
    events: Iterable[Event],
    *,
    as_of: date | None = None,
    episode_gap: int = 28,
    variant_switch: date | None = None,
    codes: Mapping[str, str] = PRODUCT_CODES,
) -> str:
    """Encode one person's dated events into a profile string.

    Parameters
    ----------
    events:
        Doses and infections in any order.
    as_of:
        Only events on or before this date are encoded (``None`` = all).
    episode_gap:
        Positives within this many days of the previous kept positive are the
        same episode and are dropped.
    variant_switch:
        If given, infections before this date are coded ``P`` (BA.1/2) and on or
        after it ``A`` (BA.4/5); otherwise infections are coded ``I``.
    """
    ordered = sorted(
        (e for e in events if as_of is None or e.day <= as_of),
        # doses before infections on the same day: a same-day dose cannot protect
        key=lambda e: (e.day, e.kind == "infection"),
    )
    out: list[str] = []
    last_infection: date | None = None
    for e in ordered:
        if e.kind == "infection":
            if last_infection is not None and (e.day - last_infection).days < episode_gap:
                continue
            last_infection = e.day
            if variant_switch is None:
                out.append("I")
            else:
                out.append("P" if e.day < variant_switch else "A")
        else:
            out.append(product_symbol(e.kind, codes))
    return collapse("".join(out))


def encode_table(
    events,
    *,
    id_col: str = "id",
    day_col: str = "day",
    kind_col: str = "kind",
    origin: date = date(2022, 1, 1),
    as_of: date | int | None = None,
    episode_gap: int = 28,
    variant_switch: date | None = None,
    codes: Mapping[str, str] = PRODUCT_CODES,
):
    """Encode a long table of dated events (one row per dose or positive) per person.

    ``day_col`` may hold dates or integer day offsets from ``origin``. Returns a
    ``pandas.Series`` of profile strings indexed by person id.
    """
    import pandas as pd

    df = events[[id_col, day_col, kind_col]]
    days = df[day_col]
    if pd.api.types.is_integer_dtype(days):
        days = [origin + timedelta(days=int(d)) for d in days]
    if isinstance(as_of, int):
        as_of = origin + timedelta(days=as_of)
    rows = zip(df[id_col].to_numpy(), days, df[kind_col].to_numpy(), strict=True)
    by_id: dict = {}
    for pid, d, kind in rows:
        by_id.setdefault(pid, []).append(Event(d, str(kind)))
    out = {
        pid: encode(
            evs, as_of=as_of, episode_gap=episode_gap, variant_switch=variant_switch, codes=codes
        )
        for pid, evs in by_id.items()
    }
    return pd.Series(out, name="profile").rename_axis(id_col)


def collapse(code: str) -> str:
    """Keep exposures up to and including the first infection.

    ``"BBIII" -> "BBI"``; ``"BIB" -> "BI"`` (a post-infection dose is follow-up,
    not part of the baseline exposure history).
    """
    for i, ch in enumerate(code):
        if ch in INFECTION_SYMBOLS:
            return code[: i + 1]
    return code


def parse(code: str) -> Profile:
    """Validate and decompose a profile string."""
    if not _PROFILE_RE.match(code):
        raise ValueError(f"not a valid profile: {code!r}")
    products = "".join(c for c in code if c in VACCINE_SYMBOLS)
    last = code[-1:] if code else ""
    infected = last in INFECTION_SYMBOLS
    variant = {"P": "BA.1/2", "A": "BA.4/5"}.get(last) if infected else None
    return Profile(code, len(products), products, infected, variant)


def generic(code: str) -> str:
    """Replace product symbols by ``V``: ``"BBBI" -> "VVVI"`` (pooled analyses)."""
    return "".join("V" if c in VACCINE_SYMBOLS else c for c in code)


def select(
    codes: Sequence[str],
    *,
    homologous_only: bool = True,
    max_doses: int | None = 4,
    min_doses: int = 0,
) -> list[bool]:
    """Boolean mask implementing the study-population filters."""
    keep = []
    for c in codes:
        p = parse(c)
        ok = p.doses >= min_doses and (max_doses is None or p.doses <= max_doses)
        if homologous_only:
            ok = ok and p.homologous
        keep.append(ok)
    return keep
