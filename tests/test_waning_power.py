import numpy as np
import pandas as pd
import pytest
from scipy import stats

import vvi
from vvi import WANING_EXAMPLE, Bands, Truth, aggregate, fit, person_days, simulate


@pytest.fixture(scope="module")
def waning_small():
    return simulate(4_000, seed=5, truth=Truth(waning=WANING_EXAMPLE))


@pytest.fixture(scope="module")
def waning_large():
    return simulate(300_000, seed=9, truth=Truth(waning=WANING_EXAMPLE))


def test_band_labels():
    assert Bands().labels == ["0-89d", "90-179d", "180-269d", "270d+"]
    assert Bands(edges=(0, 30)).labels == ["0-29d", "30d+"]


def test_split_cuts_at_band_edges():
    iv = pd.DataFrame({"id": [0], "start": [10], "stop": [200]})
    out = Bands(edges=(0, 90, 180)).split(iv, anchor=np.array([-20]))
    # anchor -20: edges fall on days 70 and 160
    assert out.sort_values("start")[["start", "stop", "band"]].values.tolist() == [
        [10, 70, 0],
        [70, 160, 1],
        [160, 200, 2],
    ]


def test_banded_aggregate_matches_person_days(waning_small):
    c, b = waning_small, Bands()
    agg = aggregate(c.people, c.events, horizon=c.horizon, bands=b)
    pdays = person_days(c.people, c.events, bands=b)
    assert agg["at_risk"].sum() == len(pdays)
    assert agg["events"].sum() == pdays["event"].sum() == len(c.events)
    keys = ["profile", "sex", "age_band", "since", "day"]
    naive = pdays.groupby(keys, observed=True)["event"].agg(["size", "sum"])
    merged = agg.set_index(keys).join(naive)
    assert (merged["size"] == merged["at_risk"]).all()
    assert (merged["sum"] == merged["events"]).all()


def test_banded_fit_is_exact(waning_small):
    c, b = waning_small, Bands()
    cov = ("sex", "age_band", "since")
    a = fit(
        aggregate(c.people, c.events, horizon=c.horizon, bands=b),
        reference="BB",
        df=6,
        covariates=cov,
    )
    p = fit(person_days(c.people, c.events, bands=b), reference="BB", df=6, covariates=cov)
    np.testing.assert_allclose(a.result.params, p.result.params, atol=1e-7)


def test_recovers_waning(waning_large):
    c = waning_large
    f = fit(
        aggregate(c.people, c.events, horizon=c.horizon, bands=Bands()),
        reference="BB",
        covariates=("sex", "age_band", "since"),
    )
    effects = f.covariate_effects("since")
    assert [e.term for e in effects] == Bands().labels
    assert effects[0].hr == 1.0
    for e in effects[1:]:
        assert abs(e.log_hr - WANING_EXAMPLE[e.term]) < 4 * e.se, e.term
    log_hrs = [e.log_hr for e in effects]
    assert log_hrs == sorted(log_hrs)  # protection wanes monotonically


def test_covariate_reference_mapping(waning_small):
    c = waning_small
    agg = aggregate(c.people, c.events, horizon=c.horizon)
    f = fit(agg, reference="BB", df=5, covariates={"sex": "M", "age_band": "60-79"})
    assert f.covariate_refs == {"sex": "M", "age_band": "60-79"}
    assert [e.term for e in f.covariate_effects("sex")] == ["F", "M"]
    with pytest.raises(ValueError):
        fit(agg, reference="BB", covariates={"sex": "X"})
    with pytest.raises(KeyError):
        f.covariate_effects("since")


def test_histories_round_trip():
    c = simulate(3_000, seed=2)
    h = vvi.histories(c, seed=4)
    assert (h.groupby("id")["day"].max() == c.people.set_index("id")["anchor"]).all()
    enc = vvi.encode_table(h)
    truth = c.people.set_index("id")["profile"]
    assert (enc.loc[truth.index] == truth).all()


def test_power_behaves():
    e = vvi.Effect("BBBI vs BBI", np.log(1.2), 0.1)
    sizes = np.array([1e4, 1e5, 1e6, 1e7])
    pw = vvi.power(e, 100_000, sizes)
    assert np.all(np.diff(pw) > 0) and pw[-1] > 0.999
    # at the fitted size, power is that of the observed z
    z = stats.norm.ppf(0.975)
    expected = stats.norm.cdf(e.z - z) + stats.norm.cdf(-e.z - z)
    assert vvi.power(e, 100_000, [100_000])[0] == pytest.approx(expected)
    n80 = vvi.required_size(e, 100_000, target=0.8)
    assert vvi.power(e, 100_000, [n80])[0] >= 0.8
    assert vvi.power(e, 100_000, [n80 * 0.9])[0] < 0.8
    with pytest.raises(ValueError):
        vvi.required_size(e, 100_000, log_hr=0.0)
