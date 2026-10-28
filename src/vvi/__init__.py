"""vvi — hybrid immunity and immune imprinting on calendar-time risk sets."""

from vvi.model import Effect, HybridImmunityFit, fit
from vvi.riskset import aggregate, at_risk_intervals, person_days
from vvi.sequence import Event, Profile, collapse, encode, generic, parse, select
from vvi.simulate import Cohort, Truth, epidemic_curve, simulate

__all__ = [
    "Cohort",
    "Effect",
    "Event",
    "HybridImmunityFit",
    "Profile",
    "Truth",
    "aggregate",
    "at_risk_intervals",
    "collapse",
    "encode",
    "epidemic_curve",
    "fit",
    "generic",
    "parse",
    "person_days",
    "select",
    "simulate",
]
__version__ = "0.1.0"
