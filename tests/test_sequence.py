from datetime import date, timedelta

import pytest

from vvi.sequence import Event, collapse, encode, generic, parse, product_symbol, select

D0 = date(2022, 1, 1)


def ev(day: int, kind: str) -> Event:
    return Event(D0 + timedelta(days=day), kind)


def test_basic_order_and_products():
    events = [ev(40, "infection"), ev(0, "Comirnaty"), ev(21, "comirnaty"), ev(10, "BioNTech")]
    assert encode(events) == "BBBI"


def test_reinfection_is_outcome_not_exposure():
    events = [ev(0, "CoronaVac"), ev(30, "CoronaVac"), ev(60, "infection"), ev(200, "infection")]
    assert encode(events) == "SSI"


def test_post_infection_dose_is_follow_up():
    assert collapse("BIB") == "BI"
    assert collapse("BBB") == "BBB"


def test_same_episode_positives_are_merged():
    events = [ev(0, "Comirnaty"), ev(50, "infection"), ev(60, "infection")]
    assert encode(events, as_of=D0 + timedelta(days=100)) == "BI"


def test_same_day_dose_precedes_infection():
    assert encode([ev(5, "infection"), ev(5, "Comirnaty")]) == "BI"


def test_as_of_cutoff():
    events = [ev(0, "Comirnaty"), ev(30, "Comirnaty"), ev(90, "Comirnaty")]
    assert encode(events, as_of=D0 + timedelta(days=60)) == "BB"


def test_variant_coding():
    switch = date(2022, 10, 31)
    early = [ev(0, "CoronaVac"), ev(20, "CoronaVac"), ev(60, "infection")]
    late = [ev(0, "CoronaVac"), ev(20, "CoronaVac"), ev(320, "infection")]
    assert encode(early, variant_switch=switch) == "SSP"
    assert encode(late, variant_switch=switch) == "SSA"


def test_bivalent_booster():
    events = [ev(0, "Comirnaty"), ev(30, "Comirnaty"), ev(120, "Comirnaty"), ev(340, "BioNTech-BV")]
    assert encode(events) == "BBBO"


def test_unknown_product_raises():
    with pytest.raises(ValueError):
        product_symbol("Sputnik V")


@pytest.mark.parametrize(
    "code,doses,homologous,infected,variant",
    [
        ("BBBI", 3, True, True, None),
        ("SSA", 2, True, True, "BA.4/5"),
        ("SB", 2, False, False, None),
        ("I", 0, True, True, None),
        ("", 0, True, False, None),
    ],
)
def test_parse(code, doses, homologous, infected, variant):
    p = parse(code)
    assert (p.doses, p.homologous, p.infected, p.variant) == (doses, homologous, infected, variant)


@pytest.mark.parametrize("bad", ["IB", "BBII", "X", "bbi"])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        parse(bad)


def test_generic_and_select():
    assert generic("SSSI") == "VVVI"
    assert select(["BB", "SB", "BBBBB", "SSI"]) == [True, False, False, True]
