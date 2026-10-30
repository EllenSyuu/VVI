"""How large a cohort does a contrast need? Wald power from a fitted model.

For a random sub-sample of a fixed design, the standard error of a log hazard ratio
scales as ``1 / sqrt(n)``. One fit on ``n_fit`` people therefore gives the power of
the same contrast at any cohort size::

    se(n) = se_fit * sqrt(n_fit / n)
    power(n) = Φ(|θ| / se(n) − z) + Φ(−|θ| / se(n) − z),   z = Φ⁻¹(1 − α/2)

This is why the sampling-free fit matters: analysing the whole population instead
of a 300k sub-sample multiplies the information by ``N / 300k``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

from vvi.model import Effect


def _z(alpha: float) -> float:
    return float(stats.norm.ppf(1 - alpha / 2))


def power(
    effect: Effect,
    n_fit: int,
    sizes: Sequence[int] | np.ndarray,
    *,
    log_hr: float | None = None,
    alpha: float = 0.05,
) -> np.ndarray:
    """Power of a two-sided Wald test of ``effect`` at each cohort size.

    ``log_hr`` is the true effect to detect (defaults to the estimate itself).
    """
    theta = abs(effect.log_hr if log_hr is None else log_hr)
    se = effect.se * np.sqrt(n_fit / np.asarray(sizes, dtype=float))
    z = _z(alpha)
    return stats.norm.cdf(theta / se - z) + stats.norm.cdf(-theta / se - z)


def required_size(
    effect: Effect,
    n_fit: int,
    *,
    target: float = 0.8,
    log_hr: float | None = None,
    alpha: float = 0.05,
) -> int:
    """Smallest cohort with at least ``target`` power (one-sided approximation)."""
    theta = abs(effect.log_hr if log_hr is None else log_hr)
    if theta == 0:
        raise ValueError("a zero effect cannot be detected")
    k = (_z(alpha) + stats.norm.ppf(target)) * effect.se / theta
    return int(np.ceil(n_fit * k**2))
