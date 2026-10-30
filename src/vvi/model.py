"""Discrete-time (calendar-day) cloglog hazard model for hybrid immunity.

For person ``i`` at risk on calendar day ``t``::

    log(-log(1 - p_it)) = eta(t) + alpha' x_i + beta[profile_i]

The complementary log-log link makes ``exp(beta)`` a hazard ratio — the discrete-
time counterpart of an Andersen–Gill (recurrent-event Cox) model with calendar
time as the time scale. Calendar time absorbs the epidemic curve, so profiles are
compared on the *same* days (no immortal-time or wave-timing bias between groups).

``eta(t)`` is a cubic B-spline with ``df`` degrees of freedom: smooth, numerically
well conditioned, and far cheaper than one intercept per day.

Protection effectiveness of profile ``j`` relative to the reference is
``PE_j = 1 - exp(beta_j)``. Immune imprinting is assessed with a *contrast*
between two infected profiles, e.g. ``BBBI`` vs ``BBI``: ``HR > 1`` means the extra
pre-infection dose is associated with *more* reinfection.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats


@dataclass
class Effect:
    """A hazard ratio with a Wald confidence interval."""

    term: str
    log_hr: float
    se: float
    level: float = 0.95

    @property
    def z(self) -> float:
        return self.log_hr / self.se

    @property
    def p(self) -> float:
        return float(2 * stats.norm.sf(abs(self.z)))

    @property
    def hr(self) -> float:
        return float(np.exp(self.log_hr))

    @property
    def ci(self) -> tuple[float, float]:
        q = stats.norm.ppf(0.5 + self.level / 2)
        return float(np.exp(self.log_hr - q * self.se)), float(np.exp(self.log_hr + q * self.se))

    @property
    def pe(self) -> float:
        """Protection effectiveness ``1 - HR``."""
        return 1.0 - self.hr


class HybridImmunityFit:
    """Fitted model; see :func:`fit`."""

    def __init__(
        self,
        result,
        design_info,
        reference: str,
        profiles: Sequence[str],
        covariate_refs: Mapping[str, str] | None = None,
        covariate_levels: Mapping[str, Sequence[str]] | None = None,
    ):
        self.result = result
        self.design_info = design_info
        self.reference = reference
        self.profiles = list(profiles)
        self.covariate_refs = dict(covariate_refs or {})
        self.covariate_levels = {k: list(v) for k, v in (covariate_levels or {}).items()}
        self._names = list(design_info.column_names)

    # --- coefficients -------------------------------------------------------
    def _profile_term(self, profile: str) -> str:
        return f"C(profile, Treatment({self.reference!r}))[T.{profile}]"

    def _vector(self, weights: dict[str, float]) -> np.ndarray:
        L = np.zeros(len(self._names))
        for term, w in weights.items():
            L[self._names.index(term)] += w
        return L

    def _effect(self, label: str, L: np.ndarray) -> Effect:
        b = float(L @ self.result.params)
        se = float(np.sqrt(L @ self.result.cov_params() @ L))
        return Effect(label, b, se)

    def profile_effect(self, profile: str) -> Effect:
        """Hazard ratio of ``profile`` versus the reference profile."""
        if profile == self.reference:
            return Effect(profile, 0.0, np.nan)
        return self._effect(profile, self._vector({self._profile_term(profile): 1.0}))

    def contrast(self, profile: str, baseline: str) -> Effect:
        """Hazard ratio of ``profile`` versus ``baseline`` (any two profiles)."""
        w: dict[str, float] = {}
        if profile != self.reference:
            w[self._profile_term(profile)] = w.get(self._profile_term(profile), 0) + 1.0
        if baseline != self.reference:
            w[self._profile_term(baseline)] = w.get(self._profile_term(baseline), 0) - 1.0
        return self._effect(f"{profile} vs {baseline}", self._vector(w))

    def covariate_effects(self, name: str) -> list[Effect]:
        """Hazard ratio of every level of covariate ``name`` vs its reference level.

        The reference level is included (HR 1, no CI), in the covariate's level order —
        e.g. the waning curve over time-since-exposure bands.
        """
        if name not in self.covariate_refs:
            raise KeyError(f"{name!r} is not a covariate of this fit")
        ref = self.covariate_refs[name]
        out = []
        for level in self.covariate_levels[name]:
            if level == ref:
                out.append(Effect(level, 0.0, np.nan))
            else:
                term = f"C({name}, Treatment({ref!r}))[T.{level}]"
                out.append(self._effect(level, self._vector({term: 1.0})))
        return out

    def table(self) -> pd.DataFrame:
        """One row per non-reference profile and covariate level."""
        rows = []
        for name in self._names:
            if name == "Intercept" or name.startswith("bs("):
                continue
            e = self._effect(_pretty(name), self._vector({name: 1.0}))
            lo, hi = e.ci
            rows.append(
                dict(
                    term=e.term,
                    log_hr=e.log_hr,
                    se=e.se,
                    hr=e.hr,
                    ci_low=lo,
                    ci_high=hi,
                    pe=e.pe,
                    p=e.p,
                )
            )
        return pd.DataFrame(rows)

    # --- calendar-time trend -------------------------------------------------
    def baseline_hazard(self, days: np.ndarray, **levels: str) -> np.ndarray:
        """Daily event probability over ``days`` for one covariate pattern.

        Unspecified covariates sit at their reference levels.
        """
        days = np.asarray(days)
        frame = pd.DataFrame({"day": days, "profile": levels.get("profile", self.reference)})
        for col, ref in self.covariate_refs.items():
            frame[col] = levels.get(col, ref)
        X = patsy.build_design_matrices([self.design_info], frame)[0]
        return -np.expm1(-np.exp(np.asarray(X) @ self.result.params))

    @property
    def n_events(self) -> int:
        return int(self.result.model.endog[:, 0].sum()) if self.result.model.endog.ndim == 2 else -1


#: Reference levels used when a covariate is named without one.
DEFAULT_REFERENCES: dict[str, str] = {"sex": "F", "age_band": "18-59"}


def _pretty(name: str) -> str:
    # "C(profile, Treatment('BB'))[T.BBB]" -> "BBB"; "C(sex, ...)[T.M]" -> "sex: M"
    level = name.split("[T.", 1)[-1].rstrip("]")
    if name.startswith("C(profile"):
        return level
    var = name[2:].split(",", 1)[0]
    return f"{var}: {level}"


def spline_knots(first_day: int, last_day: int, df: int) -> tuple[float, ...]:
    """Interior knots equally spaced in calendar time.

    Knots depend only on the calendar window — not on how rows are distributed —
    so the aggregated and the person-day fits share one design matrix.
    """
    if df < 4:
        raise ValueError("df must be >= 4 for a cubic B-spline")
    return tuple(np.linspace(first_day, last_day, df - 3 + 2)[1:-1].round(6))


def _levels(column: pd.Series) -> list[str]:
    if isinstance(column.dtype, pd.CategoricalDtype):
        present = set(column.astype(str).unique())
        return [str(c) for c in column.cat.categories if str(c) in present]
    return sorted(column.astype(str).unique())


def _formula(reference: str, df: int, refs: Mapping[str, str], bounds: tuple[int, int]) -> str:
    terms = [f"C(profile, Treatment({reference!r}))"]
    terms += [f"C({name}, Treatment({ref!r}))" for name, ref in refs.items()]
    lo, hi = bounds
    knots = spline_knots(lo, hi, df)
    terms.append(f"bs(day, knots={knots!r}, degree=3, lower_bound={lo}, upper_bound={hi})")
    return " + ".join(terms)


def fit(
    data: pd.DataFrame,
    *,
    reference: str,
    profiles: Sequence[str] | None = None,
    df: int = 7,
    covariates: Sequence[str] | Mapping[str, str] = ("sex", "age_band"),
) -> HybridImmunityFit:
    """Fit the cloglog calendar-time model.

    ``data`` is either the aggregated table from :func:`vvi.riskset.aggregate`
    (columns ``at_risk`` and ``events``) or a person-day table with a 0/1 ``event``
    column. Both give the same estimates; the aggregated one is orders of
    magnitude smaller.

    ``covariates`` are categorical adjustment variables. Pass a mapping
    ``{name: reference_level}`` to choose reference levels; otherwise ``sex`` and
    ``age_band`` use :data:`DEFAULT_REFERENCES` and any other covariate (e.g. the
    ``since`` band from :class:`vvi.riskset.Bands`) uses its first level.
    """
    if profiles is not None:
        data = data[data["profile"].isin(profiles)]
    profiles = sorted(data["profile"].unique())
    if reference not in profiles:
        raise ValueError(f"reference profile {reference!r} not in data")

    levels = {name: _levels(data[name]) for name in covariates}
    if isinstance(covariates, Mapping):
        refs = {str(k): str(v) for k, v in covariates.items()}
    else:
        refs = {n: DEFAULT_REFERENCES.get(n, levels[n][0]) for n in covariates}
    for name, ref in refs.items():
        if ref not in levels[name]:
            raise ValueError(f"reference level {ref!r} of {name!r} not in data")
    data = data.assign(**{n: data[n].astype(str) for n in refs})

    bounds = (int(data["day"].min()), int(data["day"].max()))
    rhs = _formula(reference, df, refs, bounds)
    count_col = "events" if "events" in data.columns else "event"
    empty = [p for p, d in data.groupby("profile")[count_col].sum().items() if d == 0]
    if empty:
        warnings.warn(
            f"profiles with zero events (estimates will diverge to HR=0): {empty}",
            RuntimeWarning,
            stacklevel=2,
        )
    X = patsy.dmatrix(rhs, data, return_type="dataframe")
    if {"at_risk", "events"} <= set(data.columns):
        d = data["events"].to_numpy(float)
        y = np.column_stack([d, data["at_risk"].to_numpy(float) - d])
    elif "event" in data.columns:
        y = data["event"].to_numpy(float)
    else:
        raise ValueError("need columns (at_risk, events) or event")

    family = sm.families.Binomial(link=sm.families.links.CLogLog())
    result = sm.GLM(y, X, family=family).fit(maxiter=200, tol=1e-10)  # IRLS
    return HybridImmunityFit(result, X.design_info, reference, profiles, refs, levels)
