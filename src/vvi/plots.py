"""Atelier figures: ink and graphite on warm paper, orange for the data that matters.

The visual language follows an engineer's drawing plate — ruled frames, Cormorant
Garamond labels in small caps, hand-lettered notes (La Belle Aurore) and hatched
ridges — so every chart reads as part of one set. Fonts (SIL OFL) ship with the
package; install the ``plot`` extra for matplotlib and Pillow.

Building blocks
---------------
:func:`plate`        figure with paper ground, title block and double-ruled frame
:func:`style_axes`   ink axes, serif tick labels, dotted faint grid
:func:`hand`         hand-lettered annotation with a paper halo
:func:`save`         write a PNG with the paper grain multiplied in

Charts
------
:func:`forest`, :func:`contrasts`, :func:`trend`, :func:`waning`,
:func:`power_curves`, :func:`histories`, :func:`benchmark`
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from vvi.model import HybridImmunityFit
from vvi.sequence import parse

# --- palette (validated: platform pair passes CVD and contrast checks on PAPER) ---
PAPER = "#f6f1e6"
INK = "#22314a"
GRAPH = "#6f7b8c"
FAINT = "#b4bcc7"
ORANGE = "#cf602b"
AMBER = "#e0a046"
RUST = "#a3432a"
BLUE = "#1d6fb8"
PLATFORM_COLORS: dict[str | None, str] = {"B": BLUE, "S": ORANGE, None: GRAPH}
PLATFORM_NAMES = {"B": "Comirnaty", "S": "CoronaVac"}

# Backwards-compatible aliases (v0.1)
SURFACE = PAPER
INK_2 = GRAPH
MUTED = GRAPH
GRID = FAINT

_HAIR = " "  # hair space, for letter-spaced small caps


@cache
def fonts() -> dict:
    """Register the bundled fonts and return ``FontProperties`` by role.

    Roles: ``label`` (Cormorant SemiBold), ``serif`` (Cormorant Medium),
    ``italic`` (Cormorant Italic), ``hand`` (La Belle Aurore), ``sans`` (Inter),
    ``mono`` (JetBrains Mono).
    """
    from matplotlib import font_manager as fm

    root = resources.files("vvi") / "fonts"
    files = {
        "label": "CormorantGaramond-SemiBold.ttf",
        "serif": "CormorantGaramond-Medium.ttf",
        "italic": "CormorantGaramond-Italic.ttf",
        "hand": "LaBelleAurore-Regular.ttf",
        "sans": "Inter-Regular.ttf",
        "mono": "JetBrainsMono-Regular.ttf",
    }
    out = {}
    for role, name in files.items():
        path = str(root / name)
        fm.fontManager.addfont(path)
        out[role] = fm.FontProperties(fname=path)
    return out


def _fp(role: str, size: float):
    fp = fonts()[role].copy()
    fp.set_size(size)
    return fp


def spaced(text: str) -> str:
    """Letter-space a small-caps label: ``"FIG"`` -> ``"F I G"`` with hair spaces."""
    return _HAIR.join(text.upper())


# ---------------------------------------------------------------- building blocks
def plate(
    figsize: tuple[float, float] = (8.0, 4.8),
    *,
    kicker: str = "",
    title: str = "",
    note: str | None = None,
    top: float = 0.80,
):
    """A figure laid out as a drawing plate.

    Returns ``fig``; the drawing area is ``[0.08, 0.12, 0.88, top - 0.14]`` in figure
    coordinates (use ``fig.add_axes`` or ``fig.add_gridspec(top=top - 0.06, ...)``).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=figsize, facecolor=PAPER)
    w, h = figsize
    mx, my = 0.18 / w, 0.18 / h  # outer frame inset (inches -> fraction)
    for inset, lw in ((0.0, 0.9), (0.05, 0.35)):
        ix, iy = mx + inset / w, my + inset / h
        fig.add_artist(
            Line2D(
                [ix, 1 - ix, 1 - ix, ix, ix],
                [iy, iy, 1 - iy, 1 - iy, iy],
                transform=fig.transFigure,
                color=INK,
                lw=lw,
                solid_joinstyle="miter",
            )
        )
    left = 0.42 / w
    if kicker:
        fig.text(
            left,
            1 - 0.46 / h,
            spaced(kicker),
            fontproperties=_fp("label", 8.6),
            color=GRAPH,
            va="baseline",
        )
    if title:
        fig.text(
            left, 1 - 0.80 / h, title, fontproperties=_fp("italic", 17), color=INK, va="baseline"
        )
    rule_y = 1 - 0.98 / h
    fig.add_artist(
        Line2D([left, 1 - left], [rule_y, rule_y], transform=fig.transFigure, color=INK, lw=0.6)
    )
    fig.add_artist(
        Line2D(
            [left, 1 - left], [rule_y - 0.035 / h] * 2, transform=fig.transFigure, color=INK, lw=0.3
        )
    )
    if note:
        fig.text(
            1 - left,
            1 - 0.80 / h,
            note,
            fontproperties=_fp("hand", 11),
            color=ORANGE,
            ha="right",
            va="baseline",
        )
    return fig


def style_axes(ax, *, grid: str = "x"):
    """Ink axes on paper, Cormorant tick labels, dotted faint grid."""
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK, width=0.6, length=4, direction="out", labelcolor=GRAPH)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(_fp("label", 10))
    if grid:
        ax.grid(axis=grid, color=FAINT, lw=0.55, ls=(0, (1, 3)))
    ax.set_axisbelow(True)
    return ax


style = style_axes  # v0.1 name


def _tick_fonts(ax, size: float = 10):
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(_fp("label", size))


def axis_label(ax, text: str, which: str = "x", size: float = 12):
    """Hand-lettered axis caption, as on the website charts."""
    if which == "x":
        ax.set_xlabel(text, fontproperties=_fp("hand", size), color=GRAPH, labelpad=6)
    else:
        ax.set_ylabel(text, fontproperties=_fp("hand", size), color=GRAPH, labelpad=8)


def hand(ax, x, y, text, *, color: str = INK, size: float = 12, transform=None, **kw):
    """Hand-lettered note with a paper halo so it reads over linework."""
    from matplotlib import patheffects as pe

    return ax.text(
        x,
        y,
        text,
        fontproperties=_fp("hand", size),
        color=color,
        transform=transform or ax.transData,
        path_effects=[pe.withStroke(linewidth=3.2, foreground=PAPER)],
        **kw,
    )


def caps(ax, x, y, text, *, color: str = GRAPH, size: float = 8.6, transform=None, **kw):
    """Letter-spaced small-caps label."""
    return ax.text(
        x,
        y,
        spaced(text),
        fontproperties=_fp("label", size),
        color=color,
        transform=transform or ax.transData,
        **kw,
    )


def hr_axis(ax, ticks=(0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3)):
    """Log-scaled hazard-ratio axis with plain decimal tick labels."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    ax.xaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    _tick_fonts(ax)
    return ax


def save(fig, path: str | Path, *, dpi: int = 220, grain: bool = True) -> Path:
    """Save a PNG, multiplying the paper grain over it (as on the website plates)."""
    import matplotlib.pyplot as plt
    from PIL import Image, ImageChops

    path = Path(path)
    fig.savefig(path, dpi=dpi, facecolor=PAPER)
    plt.close(fig)
    if grain:
        img = Image.open(path).convert("RGB")
        with resources.as_file(resources.files("vvi") / "assets" / "grain.png") as g:
            tile = Image.open(g).convert("RGBA")
        scale = max(1, round(dpi / 100))
        tile = tile.resize((tile.width * scale, tile.height * scale))
        layer = Image.new("RGB", img.size, "white")
        flat = Image.new("RGB", tile.size, "white")
        flat.paste(tile, mask=tile.split()[3])
        for x in range(0, img.width, flat.width):
            for y in range(0, img.height, flat.height):
                layer.paste(flat, (x, y))
        img = Image.blend(img, ImageChops.multiply(img, layer), 0.3)
        img.save(path, optimize=True)
    return path


# ---------------------------------------------------------------- charts
def _whisker(ax, lo, hi, y, color, cap: float = 0.13, lw: float = 1.6):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=3)
    for v in (lo, hi):
        ax.plot([v, v], [y - cap, y + cap], color=color, lw=0.9, zorder=3)


def forest(
    fit: HybridImmunityFit,
    ax=None,
    *,
    profiles: Sequence[str] | None = None,
    truth: Mapping[str, float] | None = None,
    title: str | None = None,
):
    """Hazard ratio of each profile vs the reference, grouped by platform.

    Estimate = dot with 95% CI and end caps; ``truth`` (log HRs) = ink tick.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=PAPER)
    style_axes(ax)
    profiles = [p for p in (profiles or fit.profiles) if p != fit.reference]
    groups: dict[str | None, list[str]] = {}
    for p in sorted(profiles, key=lambda p: (parse(p).infected, len(p))):
        groups.setdefault(parse(p).platform, []).append(p)
    order = [k for k in ("B", "S") if k in groups] + [k for k in groups if k not in ("B", "S")]
    y, ys, labels, heads = 0.0, [], [], []
    for g in order:
        heads.append((y, g))
        y -= 0.85
        for p in groups[g]:
            ys.append((y, p))
            labels.append(p)
            y -= 1.0
        y -= 0.35
    ref_lhr = (truth or {}).get(fit.reference, 0.0)
    for yi, p in ys:
        e = fit.profile_effect(p)
        lo, hi = e.ci
        color = PLATFORM_COLORS.get(parse(p).platform, GRAPH)
        _whisker(ax, lo, hi, yi, color)
        ax.plot(e.hr, yi, "o", ms=6.5, color=color, mec=PAPER, mew=1.4, zorder=4)
        if truth is not None and p in truth:
            ax.plot(
                [np.exp(truth[p] - ref_lhr)] * 2, [yi - 0.3, yi + 0.3], color=INK, lw=1.4, zorder=5
            )
        ax.text(
            1.015,
            yi,
            f"{e.hr:.2f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontproperties=_fp("serif", 10.5),
            color=INK,
        )
    ax.axvline(1, color=GRAPH, lw=0.7, ls=(0, (2, 3)))
    ax.set_yticks([v for v, _ in ys], labels)
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(_fp("label", 11))
        lab.set_color(INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y + 0.6, 0.6)
    hr_axis(ax)
    for yh, g in heads:
        caps(
            ax,
            0.0,
            yh - 0.25,
            f"{PLATFORM_NAMES.get(g or '', 'other')} · {g}",
            transform=ax.get_yaxis_transform(),
            color=PLATFORM_COLORS.get(g, GRAPH),
            ha="left",
            va="center",
        )
    axis_label(ax, f"hazard ratio vs {fit.reference} — log scale, 95% CI")
    if title:
        ax.set_title(title, loc="left", fontproperties=_fp("italic", 14), color=INK)
    return ax


def contrasts(
    fit: HybridImmunityFit,
    pairs: Sequence[tuple[str, str]],
    ax=None,
    *,
    truth: Mapping[str, float] | None = None,
    ticks=(0.7, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2),
):
    """Hazard ratios of profile pairs (e.g. the imprinting contrast BBBI vs BBI)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 2.4), facecolor=PAPER)
    style_axes(ax)
    for k, (a, b) in enumerate(pairs):
        yk = -k
        c = fit.contrast(a, b)
        lo, hi = c.ci
        color = PLATFORM_COLORS.get(parse(a).platform, GRAPH)
        if lo > 1:  # the data that matters: an interval clear of 1
            ax.axhspan(yk - 0.32, yk + 0.32, xmin=0, xmax=1, color=ORANGE, alpha=0.06, lw=0)
        _whisker(ax, lo, hi, yk, color, cap=0.16, lw=2)
        ax.plot(c.hr, yk, "o", ms=7.5, color=color, mec=PAPER, mew=1.5, zorder=4)
        if truth is not None:
            t = np.exp(truth[a] - truth[b])
            ax.plot([t, t], [yk - 0.3, yk + 0.3], color=INK, lw=1.4, zorder=5)
        ax.text(
            1.015,
            yk,
            f"{c.hr:.2f}  ({lo:.2f}–{hi:.2f})",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontproperties=_fp("serif", 10.5),
            color=INK,
        )
    ax.axvline(1, color=GRAPH, lw=0.7, ls=(0, (2, 3)))
    ax.set_yticks([-k for k in range(len(pairs))], [f"{a}  vs  {b}" for a, b in pairs])
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(_fp("label", 11))
        lab.set_color(INK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-len(pairs) + 0.45, 0.55)
    hr_axis(ax, ticks=ticks)
    return ax


def ridge(ax, x, y, *, offsets=(0.06, 0.12, 0.19, 0.27, 0.36), color=INK, lw=1.6):
    """The website's engraved ridge: a crest line with fading offset hatching below.

    ``offsets`` are fractions of the y-range.
    """
    span = np.nanmax(y) - np.nanmin(y)
    for i, o in enumerate(offsets):
        ax.plot(
            x,
            np.maximum(y - o * span, np.nanmin(y)),
            color=GRAPH,
            lw=0.45,
            alpha=max(0.12, 0.75 - i * 0.14),
            zorder=2,
        )
    ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", zorder=3)


def trend(fit: HybridImmunityFit, agg: pd.DataFrame, ax=None, *, title: str | None = None):
    """Observed daily incidence in the reference cell with the fitted calendar curve."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.2), facecolor=PAPER)
    style_axes(ax, grid="y")
    ref = agg[agg["profile"] == fit.reference]
    for col, lev in fit.covariate_refs.items():
        ref = ref[ref[col].astype(str) == lev]
    daily = ref.groupby("day")[["events", "at_risk"]].sum()
    rate = daily["events"] / daily["at_risk"] * 1e4
    ax.plot(rate.index, rate.values, "o", ms=2.1, color=GRAPH, alpha=0.55, mew=0, zorder=1)
    days = np.arange(agg["day"].min(), agg["day"].max() + 1)
    curve = fit.baseline_hazard(days) * 1e4
    ridge(ax, days, curve)
    ax.set_ylim(0, None)
    ax.set_xlim(days[0] - 4, days[-1] + 4)
    # peaks, marked in orange and hand-lettered
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(curve, prominence=float(np.max(curve)) * 0.2)
    for k, pk in enumerate(peaks[:2]):
        ax.plot(days[pk], curve[pk], "o", ms=5, color=ORANGE, zorder=5)
        hand(ax, days[pk] + 9, curve[pk] * 1.02, ["first wave", "second wave"][k], size=12.5)
    month0 = np.arange(0, 365, 30.42)
    ax.set_xticks(month0 + 15.2, list("JFMAMJJASOND")[: len(month0)])
    ax.tick_params(axis="x", length=0)
    _tick_fonts(ax)
    axis_label(ax, "daily infections per 10,000 at risk", which="y")
    if title:
        ax.set_title(title, loc="left", fontproperties=_fp("italic", 14), color=INK)
    return ax


def waning(
    fit: HybridImmunityFit,
    ax=None,
    *,
    name: str = "since",
    truth: Mapping[str, float] | None = None,
):
    """Hazard ratio by time since the last exposure (a step chart with hatching)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.2), facecolor=PAPER)
    style_axes(ax, grid="y")
    eff = fit.covariate_effects(name)
    x = np.arange(len(eff))
    hrs = np.array([e.hr for e in eff])
    for xi, e in zip(x, eff, strict=True):
        ax.fill_between(
            [xi - 0.36, xi + 0.36],
            1,
            e.hr,
            color="none",
            edgecolor=GRAPH,
            hatch="////",
            lw=0,
            alpha=0.55,
            zorder=1,
        )
        ax.plot([xi - 0.36, xi + 0.36], [e.hr, e.hr], color=INK, lw=1.6, zorder=3)
        if np.isfinite(e.se):
            lo, hi = e.ci
            ax.plot([xi, xi], [lo, hi], color=ORANGE, lw=1.6, zorder=4)
            for v in (lo, hi):
                ax.plot([xi - 0.06, xi + 0.06], [v, v], color=ORANGE, lw=0.9, zorder=4)
        ax.plot(
            xi,
            e.hr,
            "o",
            ms=5.5,
            color=ORANGE if np.isfinite(e.se) else INK,
            mec=PAPER,
            mew=1.2,
            zorder=5,
        )
        if truth is not None and e.term in truth:
            t = np.exp(truth[e.term] - truth.get(eff[0].term, 0.0))
            ax.plot([xi - 0.2, xi + 0.2], [t, t], color=INK, lw=1.3, ls=(0, (1, 1.5)), zorder=6)
    ax.axhline(1, color=GRAPH, lw=0.7, ls=(0, (2, 3)))
    ax.plot(x, hrs, color=INK, lw=0.6, ls=(0, (3, 3)), zorder=2)

    def _band(term: str) -> str:
        if term.endswith("d+"):
            return f"{term[:-2]}+ days"
        return term.replace("-", "–").replace("d", " days")

    ax.set_xticks(x, [_band(e.term) for e in eff])
    _tick_fonts(ax, 10.5)
    ax.set_xlim(-0.6, len(eff) - 0.4)
    ax.set_ylim(0.9, max(float(np.max(hrs)), 1.0) * 1.12)
    axis_label(ax, "hazard ratio vs the first band", which="y")
    axis_label(ax, "days since the last dose or infection")
    return ax


def power_curves(
    sizes: np.ndarray,
    curves: Mapping[str, np.ndarray],
    ax=None,
    *,
    simulated: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    markers: Mapping[str, float] | None = None,
):
    """Power against cohort size; one curve per effect size (sequential amber→rust).

    ``simulated`` maps a curve label to Monte-Carlo ``(sizes, power, half_width)``.
    ``markers`` draws labelled vertical rules at named cohort sizes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=PAPER)
    style_axes(ax, grid="y")
    ramp = [AMBER, ORANGE, RUST, INK]
    for (label, pw), color in zip(curves.items(), ramp, strict=False):
        ax.plot(sizes, pw, color=color, lw=1.8, zorder=3)
        j = int(np.argmin(np.abs(pw - 0.62)))
        hand(ax, sizes[j] / 1.1, pw[j] + 0.03, label, color=color, size=12, ha="right")
        if simulated and label in simulated:
            sx, sp, hw = simulated[label]
            ax.errorbar(
                sx,
                sp,
                yerr=hw,
                fmt="o",
                ms=5.5,
                color=color,
                mec=PAPER,
                mew=1.2,
                elinewidth=0.9,
                capsize=2.5,
                zorder=5,
            )
    ax.axhline(0.8, color=GRAPH, lw=0.6, ls=(0, (2, 3)))
    ax.text(
        sizes[0],
        0.815,
        spaced("80% power"),
        fontproperties=_fp("label", 8),
        color=GRAPH,
        va="bottom",
    )
    for name, n in (markers or {}).items():
        ax.axvline(n, color=INK, lw=0.7)
        hand(ax, n * 1.05, 0.06, name, size=11.5, color=INK)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator

    ax.xaxis.set_major_locator(FixedLocator([3e4, 1e5, 3e5, 1e6, 3e6]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k")
    )
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    _tick_fonts(ax)
    axis_label(ax, "people in the cohort")
    axis_label(ax, "power to detect the contrast", which="y")
    return ax


def histories(
    people: pd.DataFrame,
    exposures: pd.DataFrame,
    events: pd.DataFrame,
    ax=None,
    *,
    blank_days: int = 28,
    horizon: int = 365,
    curve: np.ndarray | None = None,
):
    """Swimmer lanes: dated doses and infections, then calendar follow-up.

    One lane per person: dose glyphs coloured by platform, the first infection as an
    ink dot, the at-risk follow-up as a ruled line, the 28-day episode window as a
    hatched gap, later infections in orange, censoring as a short bar.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5), facecolor=PAPER)
    style_axes(ax, grid="")
    ax.spines["left"].set_visible(False)
    ppl = people.reset_index(drop=True)
    n = len(ppl)
    lo_day = int(min(exposures["day"].min(), 0)) - 15
    ax.axvspan(0, horizon, color=INK, alpha=0.025, lw=0, zorder=0)
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    caps(ax, 4, n - 0.15, "follow-up", color=INK, va="bottom")
    caps(ax, -6, n - 0.15, "exposure history", ha="right", va="bottom")
    if curve is not None:  # epidemic ridge above the lanes
        t = np.arange(horizon)
        y = n + 0.55 + 2.6 * curve / float(np.max(curve))
        ridge(ax, t, y, offsets=(0.08, 0.16, 0.25), lw=1.2)
    for i, row in ppl.iterrows():
        yi = n - 1 - i
        pid = row["id"]
        ax.plot([lo_day + 10, horizon], [yi, yi], color=FAINT, lw=0.4, zorder=1)
        ex = exposures[exposures["id"] == pid]
        for _, e in ex.iterrows():
            if e["kind"] == "infection":
                ax.plot(e["day"], yi, "o", ms=6.2, color=INK, mec=PAPER, mew=1.0, zorder=4)
            else:
                sym = "B" if "omirn" in e["kind"] else "S"
                ax.plot(
                    [e["day"]] * 2,
                    [yi - 0.26, yi + 0.26],
                    color=PLATFORM_COLORS[sym],
                    lw=2.0,
                    solid_capstyle="butt",
                    zorder=3,
                )
        # follow-up: entry .. exit with blanked windows after each event
        a, end = int(row["entry"]), int(row["exit"])
        evs = sorted(events.loc[events["id"] == pid, "day"].tolist())
        for d in evs:
            ax.plot([a, d], [yi, yi], color=INK, lw=1.3, zorder=2)
            ax.add_patch(
                Rectangle(
                    (d, yi - 0.2),
                    blank_days,
                    0.4,
                    facecolor="none",
                    edgecolor=GRAPH,
                    hatch="//////",
                    lw=0,
                    zorder=2,
                )
            )
            ax.plot(d, yi, "o", ms=5.6, color=ORANGE, mec=PAPER, mew=1.0, zorder=5)
            a = d + 1 + blank_days
        if a < end:
            ax.plot([a, end], [yi, yi], color=INK, lw=1.3, zorder=2)
        if end < horizon:
            ax.plot([end, end], [yi - 0.22, yi + 0.22], color=INK, lw=1.2, zorder=3)
        ax.text(
            lo_day,
            yi,
            row["profile"],
            fontproperties=_fp("label", 10.5),
            color=INK,
            ha="right",
            va="center",
        )
    ax.set_yticks([])
    ax.set_xlim(lo_day - 8, horizon + 6)
    ax.set_ylim(-0.8, n + (4.0 if curve is not None else 0.6))
    ticks = [t for t in range(-720, 1, 90)] + [90, 180, 270, 365]
    ticks = [t for t in ticks if t >= lo_day]
    ax.set_xticks(ticks, [f"{t:+d}" if t else "day 0" for t in ticks])
    _tick_fonts(ax, 9.5)
    return ax


def benchmark(rows: pd.DataFrame, axes=None):
    """Build+fit time and peak memory against cohort size, naive vs aggregated.

    ``rows`` has columns ``pipeline, people, seconds, mib``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), facecolor=PAPER)
    fmt_n = FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k")
    for ax, col, label, unit in (
        (axes[0], "seconds", "build + fit time", "s"),
        (axes[1], "mib", "peak memory", "MiB"),
    ):
        style_axes(ax, grid="y")
        for name, color, lw in (("person-day", INK, 1.3), ("aggregated", ORANGE, 2.0)):
            r = rows[rows["pipeline"] == name].sort_values("people")
            ax.plot(
                r["people"], r[col], "-o", color=color, lw=lw, ms=4.5, mec=PAPER, mew=1.0, zorder=3
            )
            last = r.iloc[-1]
            if name == "person-day":
                ax.plot(last["people"], last[col], marker="x", ms=8, mew=1.4, color=INK, zorder=4)
                hand(
                    ax,
                    last["people"] * 1.25,
                    last[col],
                    "out of memory next",
                    size=11.5,
                    va="center",
                )
            else:
                hand(
                    ax,
                    last["people"] * 0.7,
                    last[col] * 1.12,
                    f"vvi · 1M people · {last[col]:,.0f} {unit}",
                    color=ORANGE,
                    size=11.5,
                    ha="right",
                    va="bottom",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(fmt_n)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
        )
        _tick_fonts(ax)
        caps(ax, 0.0, 1.04, f"{label} ({unit})", transform=ax.transAxes, color=INK)
        axis_label(ax, "people in the cohort")
    return axes
