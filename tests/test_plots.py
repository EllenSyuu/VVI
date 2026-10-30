import io

import numpy as np
import pytest

pytest.importorskip("matplotlib")
from PIL import Image  # noqa: E402

import vvi  # noqa: E402
from vvi import plots  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    c = vvi.simulate(40_000, seed=3)
    agg = vvi.aggregate(c.people, c.events, horizon=c.horizon)
    return c, agg, vvi.fit(agg, reference="BB", df=6)


def _png(render) -> Image.Image:
    buf = io.BytesIO()
    render(buf)
    buf.seek(0)
    return Image.open(buf)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_every_figure_renders(fitted, theme):
    c, agg, f = fitted
    plots.set_theme(theme)
    assert plots.theme().name == theme
    w = vvi.simulate(30_000, seed=4, truth=vvi.Truth(waning=vvi.WANING_EXAMPLE))
    wf = vvi.fit(
        vvi.aggregate(w.people, w.events, horizon=365, bands=vvi.Bands()),
        reference="BB",
        df=6,
        covariates=("sex", "age_band", "since"),
    )
    sizes = np.geomspace(3e4, 3e6, 20)
    curves = {"HR 1.20": vvi.power(f.contrast("BBBI", "BBI"), 40_000, sizes, log_hr=np.log(1.2))}
    rows = __import__("pandas").DataFrame(
        {
            "pipeline": ["person-day", "aggregated", "aggregated"],
            "people": [1e3, 1e3, 1e6],
            "seconds": [20.0, 1.0, 13.0],
            "mib": [450.0, 60.0, 320.0],
        }
    )
    small = vvi.simulate(500, seed=1)
    renders = [
        lambda b: plots.hero(f, b),
        lambda b: plots.recovery(f, c.truth.profile_effects, b),
        lambda b: plots.calendar(f, agg, b),
        lambda b: plots.waning(wf, b, truth=vvi.WANING_EXAMPLE),
        lambda b: plots.power_curves(sizes, curves, b, markers={"300k": 3e5}),
        lambda b: plots.histories(small.people.head(6), vvi.histories(small), small.events, b),
        lambda b: plots.benchmark(rows, b),
    ]
    surface = tuple(int(plots.theme().surface[i : i + 2], 16) for i in (1, 3, 5))
    for render in renders:
        img = _png(render).convert("RGB")
        assert img.width > 1000
        assert img.getpixel((2, 2)) == surface  # background matches the theme
    plots.set_theme("light")


def test_animation_progress_frames(fitted, tmp_path):
    _, _, f = fitted
    out = plots.animate(
        lambda b, p: plots.hero(f, b, progress=p), tmp_path / "hero.gif", frames=4, dpi=40
    )
    gif = Image.open(out)
    assert gif.n_frames == 4
