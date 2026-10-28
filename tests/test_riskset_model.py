import numpy as np
import pandas as pd
import pytest

from vvi import aggregate, at_risk_intervals, fit, person_days, simulate


@pytest.fixture(scope="module")
def small():
    return simulate(4_000, seed=7)


@pytest.fixture(scope="module")
def large():
    return simulate(400_000, seed=11)


def test_intervals_respect_blanking():
    people = pd.DataFrame({"id": [0], "entry": [10], "exit": [100]})
    events = pd.DataFrame({"id": [0, 0], "day": [20, 60]})
    iv = at_risk_intervals(people, events, blank_days=28)
    assert iv[["start", "stop"]].values.tolist() == [[10, 21], [49, 61], [89, 100]]


def test_aggregate_matches_person_days(small):
    agg = aggregate(small.people, small.events, horizon=small.horizon)
    pdays = person_days(small.people, small.events)
    assert agg["at_risk"].sum() == len(pdays)
    assert agg["events"].sum() == pdays["event"].sum() == len(small.events)
    naive = pdays.groupby(["profile", "sex", "age_band", "day"], observed=True)["event"].agg(
        ["size", "sum"]
    )
    merged = agg.set_index(["profile", "sex", "age_band", "day"]).join(naive)
    assert (merged["size"] == merged["at_risk"]).all()
    assert (merged["sum"] == merged["events"]).all()


def test_every_event_is_at_risk(small):
    pdays = person_days(small.people, small.events)
    assert pdays["event"].sum() == len(small.events)


def test_aggregated_fit_equals_person_day_fit(small):
    agg = aggregate(small.people, small.events, horizon=small.horizon)
    pdays = person_days(small.people, small.events)
    a = fit(agg, reference="BB", df=6)
    b = fit(pdays, reference="BB", df=6)
    np.testing.assert_allclose(a.result.params, b.result.params, atol=1e-7)
    np.testing.assert_allclose(a.result.bse, b.result.bse, rtol=1e-6)


def test_recovers_truth(large):
    agg = aggregate(large.people, large.events, horizon=large.horizon)
    f = fit(agg, reference="BB")
    for profile, beta in large.truth.profile_effects.items():
        e = f.profile_effect(profile)
        if profile == "BB":
            continue
        assert abs(e.log_hr - beta) < 4 * e.se, profile
    t = f.table().set_index("term")
    assert abs(t.loc["sex: M", "log_hr"] - large.truth.sex_male) < 4 * t.loc["sex: M", "se"]


def test_imprinting_contrast_detects_planted_signal(large):
    f = fit(aggregate(large.people, large.events, horizon=large.horizon), reference="BB")
    c = f.contrast("BBBI", "BBI")
    lo, hi = c.ci
    true_hr = np.exp(large.truth.profile_effects["BBBI"] - large.truth.profile_effects["BBI"])
    assert lo < true_hr < hi
    assert lo > 1  # planted imprinting-like effect is detected
    # contrast is antisymmetric
    assert f.contrast("BBI", "BBBI").log_hr == pytest.approx(-c.log_hr)


def test_window_and_profile_subset(large):
    agg = aggregate(large.people, large.events, horizon=large.horizon, window=(180, 365))
    assert agg["day"].min() >= 180
    f = fit(agg, reference="SS", profiles=["SS", "SSS", "SSI", "SSSI"], df=5)
    assert set(f.profiles) == {"SS", "SSS", "SSI", "SSSI"}
    assert f.profile_effect("SS").hr == 1.0


def test_bad_reference_raises(small):
    agg = aggregate(small.people, small.events, horizon=small.horizon)
    with pytest.raises(ValueError):
        fit(agg, reference="XYZ")
