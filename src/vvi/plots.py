"""Static figures, rendered in a light and a dark theme.

Figures are laid out for the width at which they are read: a README column on GitHub
is about 880 px wide, so figures are 10 inches wide with type sized so that the
smallest label still renders at roughly 11 px. Multi-panel figures stack vertically
rather than squeezing panels side by side.

Colour follows the job it does: the two vaccine platforms take two fixed categorical
slots (Comirnaty blue, CoronaVac orange), ordered quantities use one sequential hue,
and text always uses ink tokens, never a series colour. Install the ``plot`` extra.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from vvi.model import HybridImmunityFit  # noqa: E402
from vvi.sequence import parse  # noqa: E402

WIDTH = 10.0  # inches; displayed at ~880 px in a README column


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink_2: str
    ink_3: str
    grid: str
    series: tuple[str, str, str]  # categorical slots 1-3: blue, orange, aqua
    ramp: tuple[str, ...]  # sequential blue, faint -> strong
    soft: tuple[str, str, str]  # tints of the series, for bands and fills


LIGHT = Theme(
    name="light",
    surface="#ffffff",
    ink="#16202e",
    ink_2="#4b5768",
    ink_3="#7d8896",
    grid="#e6eaef",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    ramp=("#b9d3f0", "#7eaee6", "#3b7dd8", "#1b4f9c", "#0e3a73"),
    soft=("#dce9f8", "#fbe1d6", "#d4f1e6"),
)
DARK = Theme(
    name="dark",
    surface="#0d1117",  # GitHub dark canvas, so README images sit flush
    ink="#e8edf3",
    ink_2="#a3afbd",
    ink_3="#6f7c8c",
    grid="#222b36",
    series=("#4a94ea", "#f07a45", "#1fb98f"),
    ramp=("#1d3f6e", "#2f6ab2", "#5c9be6", "#9cc4f3", "#d3e5fa"),
    soft=("#16273d", "#36221a", "#13302a"),
)
THEMES = {"light": LIGHT, "dark": DARK}
_theme: Theme = LIGHT


def set_theme(name: str = "light") -> Theme:
    """Activate a theme for every figure rendered afterwards."""
    global _theme
    _theme = THEMES[name]
    t = _theme
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 12,
            "figure.facecolor": t.surface,
            "axes.facecolor": t.surface,
            "savefig.facecolor": t.surface,
            "savefig.dpi": 200,
            "axes.edgecolor": t.grid,
            "axes.linewidth": 1.0,
            "axes.labelcolor": t.ink_2,
            "axes.labelsize": 12,
            "axes.labelpad": 8,
            "axes.titlecolor": t.ink,
            "axes.titlesize": 13.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": t.grid,
            "grid.linewidth": 1.0,
            "xtick.color": t.ink_2,
            "ytick.color": t.ink_2,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "ytick.major.size": 0,
            "xtick.major.size": 4,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "text.color": t.ink,
            "legend.labelcolor": t.ink_2,
            "legend.fontsize": 11.5,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.columnspacing": 1.6,
            "lines.solid_capstyle": "round",
        }
    )
    return t


set_theme("light")


def theme() -> Theme:
    """The active theme."""
    return _theme


# --------------------------------------------------------------------------- helpers


def platform_color(profile: str) -> str:
    """Comirnaty profiles blue, CoronaVac orange, anything else muted ink."""
    p = parse(profile).platform
    return {"B": _theme.series[0], "S": _theme.series[1]}.get(p or "", _theme.ink_3)


def _figure(height: float, title: str, subtitle: str, legend: bool = False) -> Figure:
    """A constrained-layout figure with a title block laid out in inches."""
    fig = plt.figure(figsize=(WIDTH, height), layout="constrained")
    fig.text(
        0.012,
        1 - 0.2 / height,
        title,
        ha="left",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=_theme.ink,
    )
    fig.text(
        0.012,
        1 - 0.62 / height,
        subtitle,
        ha="left",
        va="top",
        fontsize=12.5,
        color=_theme.ink_2,
        wrap=True,
    )
    top = 1 - (1.42 if legend else 1.05) / height
    engine = fig.get_layout_engine()
    if engine is not None:
        cast(Any, engine).set(rect=(0, 0, 1, top), w_pad=0.08, h_pad=0.12, hspace=0.08, wspace=0.06)
    return fig


def _legend(fig: Figure, handles: Sequence[Any], ncols: int) -> None:
    fig.legend(
        handles=handles,
        loc="upper left",
        ncols=ncols,
        bbox_to_anchor=(0.006, 1 - 0.98 / fig.get_figheight()),
    )


def _save(fig: Figure, out: Path | BinaryIO, dpi: float | None = None) -> Path:
    if isinstance(out, Path):
        out.parent.mkdir(parents=True, exist_ok=True)
    if dpi is None:
        fig.savefig(out, format="png")
    else:
        fig.savefig(out, dpi=dpi, format="png")
    plt.close(fig)
    return out if isinstance(out, Path) else Path()


def _ease(x: float) -> float:
    """Ease-out cubic on [0, 1]."""
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def _stage(progress: float, k: int, n: int) -> float:
    """Progress of the k-th of n sequential stages, eased."""
    return _ease(progress * n - k)


def _log_axis(ax: Axes, ticks: Sequence[float], axis: str = "x") -> None:
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    a.set_major_locator(FixedLocator(list(ticks)))
    a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _people(v: float, _: object = None) -> str:
    return f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"


def _ci(ax: Axes, lo: float, hi: float, y: float, color: str, lw: float = 2.2) -> None:
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=3)


def _dot(ax: Axes, x: float, y: float, color: str, s: float = 60) -> None:
    ax.scatter([x], [y], s=s, color=color, edgecolor=_theme.surface, linewidth=1.8, zorder=4)


def _ring(ax: Axes, x: float, y: float, s: float = 120) -> None:
    ax.scatter([x], [y], s=s, facecolor="none", edgecolor=_theme.ink, linewidth=1.4, zorder=5)


# --------------------------------------------------------------------------- flourishes


def _gradient_fill(
    ax: Axes, x: np.ndarray, y: np.ndarray, color: str, alpha: float = 0.22, zorder: float = 1
) -> None:
    """Fill under a curve with a vertical fade, transparent at the baseline."""
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Polygon

    if len(x) < 2:
        return
    r, g, b = to_rgb(color)
    ramp = np.zeros((256, 1, 4))
    ramp[:, :, 0], ramp[:, :, 1], ramp[:, :, 2] = r, g, b
    ramp[:, :, 3] = np.linspace(0, alpha, 256)[:, None] ** 1.2 / alpha**0.2
    top = float(np.max(y)) or 1.0
    im = ax.imshow(
        ramp,
        aspect="auto",
        origin="lower",
        zorder=zorder,
        extent=(float(x[0]), float(x[-1]), 0.0, top),
        interpolation="bilinear",
    )
    clip = Polygon(
        np.column_stack([np.r_[x, x[::-1]], np.r_[y, np.zeros_like(y)]]),
        closed=True,
        facecolor="none",
        edgecolor="none",
    )
    ax.add_patch(clip)
    im.set_clip_path(clip)


def _pill(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    *,
    transform: Any = None,
    ha: str = "center",
    size: float = 10.5,
) -> None:
    """A small rounded label, filled with the colour and lettered in the surface colour."""
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=size,
        fontweight="bold",
        color=_theme.surface,
        transform=transform or ax.transData,
        zorder=8,
        bbox={"boxstyle": "round,pad=0.35,rounding_size=0.8", "fc": color, "ec": "none"},
    )


def _partial(xs: np.ndarray, ys: np.ndarray, frac: float, log: bool = True) -> tuple:
    """The part of a polyline left of a cut at ``frac`` of the (log) x-span."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if frac >= 1:
        return xs, ys
    lx = np.log(xs) if log else xs
    cut = lx[0] + frac * (lx[-1] - lx[0])
    keep = lx <= cut
    if keep.sum() == len(xs):
        return xs, ys
    i = int(keep.sum())
    if i == 0:
        return xs[:1], ys[:1]
    t = (cut - lx[i - 1]) / (lx[i] - lx[i - 1])
    ly = np.log(ys) if log else ys
    yc = ly[i - 1] + t * (ly[i] - ly[i - 1])
    xc = np.exp(cut) if log else cut
    return np.r_[xs[:i], xc], np.r_[ys[:i], np.exp(yc) if log else yc]


# --------------------------------------------------------------------------- hero


HERO_PROFILES = ("BB", "BBB", "BBI", "BBBI")
_MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def hero(fit: HybridImmunityFit, out: Path | BinaryIO, progress: float = 1.0) -> Path:
    """Fitted daily infection rate by exposure history, and the protection behind it.

    Animated, a cursor sweeps through the calendar year drawing the curves; then the
    protection bars fill.
    """
    t = _theme
    colors = dict(zip(HERO_PROFILES, (t.ink_3, t.series[0], t.series[2], t.series[1]), strict=True))
    names = {
        "BB": "BB · two doses (reference)",
        "BBB": "BBB · booster",
        "BBI": "BBI · infected after 2 doses",
        "BBBI": "BBBI · infected after booster",
    }
    fig = _figure(
        6.4,
        "Reinfection risk in calendar time, by exposure history",
        "600,000 synthetic people. Every history rides the same epidemic waves; a hazard "
        "ratio sets how high.",
        legend=True,
    )
    _legend(
        fig,
        [
            Line2D([], [], color=colors[p], lw=3, label=p if p != "BB" else "BB (reference)")
            for p in HERO_PROFILES
        ],
        4,
    )
    ax, bx = fig.subplots(1, 2, width_ratios=(2.05, 1))

    days = np.arange(0, 365)
    rates = {p: fit.baseline_hazard(days, profile=p) * 1e4 for p in HERO_PROFILES}
    peak = float(np.max(rates["BB"]))
    sweep = min(1.0, progress / 0.6) if progress < 1 else 1.0  # steady calendar cursor
    n = max(2, int(round(sweep * len(days))))
    for p in HERO_PROFILES:  # BB first, BBBI last (on top)
        x, y = days[:n].astype(float), rates[p][:n]
        _gradient_fill(
            ax, x, y, colors[p], alpha=0.30 if p == "BB" else 0.22, zorder=1.5 if p == "BB" else 1.6
        )
        ax.plot(
            x, y, color=colors[p], lw=3.2 if p == "BBBI" else 2.3, zorder=4 if p == "BBBI" else 3
        )
        if sweep < 1:
            ax.scatter(
                [x[-1]],
                [y[-1]],
                s=44,
                color=colors[p],
                edgecolor=t.surface,
                linewidth=1.6,
                zorder=6,
            )
    if sweep < 1:
        cut = float(days[n - 1])
        ax.axvline(cut, color=t.ink_3, lw=1.0, ls=(0, (2, 3)), zorder=2)
        m = min(11, int(cut // 30.42))
        _pill(ax, cut, peak * 1.13, f"{_MONTH_NAMES[m]} 2022", t.ink_2, size=10)
    else:
        bb = rates["BB"]
        w1 = int(np.argmax(np.where(days < 180, bb, 0)))
        w2 = int(np.argmax(np.where(days >= 180, bb, 0)))
        for k, d in enumerate((w1, w2), start=1):
            _pill(ax, d, bb[d] + peak * 0.085, f"wave {k}", t.ink_3, size=9.5)
        # the gap between the reference and BBBI at the first peak
        x0 = w1 + 16
        top, bot = rates["BB"][x0], rates["BBBI"][x0]
        ax.annotate(
            "",
            xy=(x0, bot * 1.06),
            xytext=(x0, top * 0.98),
            arrowprops={
                "arrowstyle": "-|>",
                "color": t.series[1],
                "lw": 1.6,
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": 12,
            },
            zorder=7,
        )
        drop = 1 - fit.profile_effect("BBBI").hr
        ax.plot(
            [x0 + 4, w1 + 44], [(top + bot) / 2, peak * 0.58], color=t.series[1], lw=0.9, zorder=6
        )
        ax.text(
            w1 + 47,
            peak * 0.58,
            f"BBBI: {drop:.0%} lower\nhazard than BB",
            color=t.series[1],
            fontsize=11.5,
            fontweight="bold",
            va="center",
            zorder=7,
        )
    ax.set_xlim(-2, 366)
    ax.set_ylim(0, peak * 1.22)
    month = np.arange(0, 365, 30.42)
    ax.set_xticks(month + 15.2, _MONTHS)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Daily infections per 10,000 at risk")

    # Protection bars: 1 - HR on a track, filled after the sweep.
    bx.set_title("Lower hazard than BB")
    bars = [p for p in HERO_PROFILES if p != fit.reference]
    grow = _ease((progress - 0.6) / 0.4) if progress < 1 else 1.0
    for k, p in enumerate(bars):
        y = len(bars) - 1 - k
        e = fit.profile_effect(p)
        pe = 1 - e.hr
        lo_pe, hi_pe = 1 - e.ci[1], 1 - e.ci[0]
        bx.plot([0, 1], [y, y], color=t.grid, lw=17, solid_capstyle="round", zorder=1)
        if grow > 0:
            bx.plot(
                [0, max(pe * grow, 0.001)],
                [y, y],
                color=colors[p],
                lw=17,
                solid_capstyle="round",
                zorder=2,
            )
        bx.text(
            -0.02,
            y + 0.3,
            names[p],
            ha="left",
            va="bottom",
            color=t.ink,
            fontsize=11.5,
            fontweight="bold",
        )
        bx.text(
            1.13,
            y + 0.06,
            f"{pe * grow:.0%}",
            ha="left",
            va="center",
            color=t.ink,
            fontsize=19,
            fontweight="bold",
        )
        if grow >= 1:
            bx.text(
                1.13,
                y - 0.34,
                f"CI {lo_pe:.0%}–{hi_pe:.0%}",
                ha="left",
                va="center",
                color=t.ink_3,
                fontsize=9.5,
            )
    bx.set_xlim(-0.06, 1.5)
    bx.set_ylim(-0.6, len(bars) - 0.2)
    bx.axis("off")
    return _save(fig, out)


# --------------------------------------------------------------------------- panels


def recovery(
    fit: HybridImmunityFit,
    truth: Mapping[str, float],
    out: Path | BinaryIO,
    *,
    contrasts: Sequence[tuple[str, str]] = (("BBBI", "BBI"), ("SSSI", "SSI")),
    progress: float = 1.0,
) -> Path:
    """Estimated hazard ratios (dot + 95% CI) against the planted truth (ring)."""
    t = _theme
    fig = _figure(
        7.4,
        "Checked against a planted ground truth",
        "One synthetic cohort of 600,000: estimates with 95% CIs (dots) against the values "
        "the simulator planted (rings).",
        legend=True,
    )
    _legend(
        fig,
        [
            Line2D([], [], color=t.series[0], lw=3, marker="o", ms=7, label="Comirnaty (B)"),
            Line2D([], [], color=t.series[1], lw=3, marker="o", ms=7, label="CoronaVac (S)"),
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                ms=10,
                mfc="none",
                mec=t.ink,
                mew=1.4,
                label="planted truth",
            ),
        ],
        3,
    )
    ax, bx = fig.subplots(2, 1, height_ratios=(3.1, 1))
    profiles = [p for p in fit.profiles if p != fit.reference]
    profiles.sort(key=lambda p: (parse(p).platform or "", parse(p).infected, len(p)))
    n = len(profiles)
    for k, p in enumerate(profiles):
        y = n - 1 - k
        frac = _stage(progress, k, n) if progress < 1 else 1.0
        if frac <= 0:
            continue
        e = fit.profile_effect(p)
        lo, hi = e.ci
        color = platform_color(p)
        mid = e.hr
        _ci(ax, mid - (mid - lo) * frac, mid + (hi - mid) * frac, y, color)
        _dot(ax, e.hr, y, color)
        if frac >= 1:
            _ring(ax, float(np.exp(truth[p] - truth.get(fit.reference, 0.0))), y)
            ax.text(
                1.01,
                y,
                f"{e.hr:.2f}",
                transform=ax.get_yaxis_transform(),
                va="center",
                color=t.ink,
                fontsize=11.5,
                fontweight="bold",
            )
    ax.axvline(1, color=t.ink_3, lw=1.2, ls=(0, (3, 3)))
    ax.set_yticks(range(n), profiles[::-1])
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0.15, 1.6)
    _log_axis(ax, (0.2, 0.3, 0.5, 0.7, 1, 1.5))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_title(f"Each history vs {fit.reference}")
    ax.set_xlabel("Hazard ratio (log scale)")

    last = _stage(progress, n - 1, n) if progress < 1 else 1.0
    for k, (a, b) in enumerate(contrasts):
        y = len(contrasts) - 1 - k
        if last < 1:
            continue
        c = fit.contrast(a, b)
        lo, hi = c.ci
        color = platform_color(a)
        _ci(bx, lo, hi, y, color, lw=2.6)
        _dot(bx, c.hr, y, color, s=70)
        _ring(bx, float(np.exp(truth[a] - truth[b])), y)
        bx.text(
            1.01,
            y,
            f"{c.hr:.2f}  ({lo:.2f}–{hi:.2f})",
            transform=bx.get_yaxis_transform(),
            va="center",
            color=t.ink,
            fontsize=11.5,
        )
    bx.axvline(1, color=t.ink_3, lw=1.2, ls=(0, (3, 3)))
    bx.set_yticks(range(len(contrasts)), [f"{a} vs {b}" for a, b in contrasts][::-1])
    bx.set_ylim(-0.6, len(contrasts) - 0.4)
    bx.set_xlim(0.75, 1.75)
    _log_axis(bx, (0.8, 0.9, 1, 1.1, 1.25, 1.5))
    bx.grid(axis="y", visible=False)
    bx.grid(axis="x", visible=True)
    bx.set_title("The imprinting contrast: one extra dose before infection")
    return _save(fig, out)


def calendar(fit: HybridImmunityFit, agg: pd.DataFrame, out: Path | BinaryIO) -> Path:
    """Observed daily incidence in the reference cell and the fitted calendar curve."""
    t = _theme
    fig = _figure(
        4.6,
        "Calendar time absorbs the epidemic curve",
        "Reference cell (BB, female, 18–59): observed daily rate and the fitted cubic "
        "B-spline baseline (7 degrees of freedom).",
        legend=True,
    )
    _legend(
        fig,
        [
            Line2D([], [], ls="", marker="o", ms=5, color=t.ink_3, label="observed"),
            Line2D([], [], color=t.series[0], lw=3, label="fitted baseline"),
        ],
        2,
    )
    ax = fig.subplots()
    ref = agg[agg["profile"] == fit.reference]
    for col, lev in fit.covariate_refs.items():
        ref = ref[ref[col].astype(str) == lev]
    daily = ref.groupby("day")[["events", "at_risk"]].sum()
    rate = daily["events"] / daily["at_risk"] * 1e4
    ax.scatter(rate.index, rate.values, s=9, color=t.ink_3, alpha=0.7, lw=0, zorder=2)
    days = np.arange(int(agg["day"].min()), int(agg["day"].max()) + 1)
    ax.plot(days, fit.baseline_hazard(days) * 1e4, color=t.series[0], lw=2.8, zorder=3)
    ax.set_ylim(0, None)
    ax.set_xlim(days[0] - 4, days[-1] + 4)
    month = np.arange(0, 365, 30.42)
    ax.set_xticks(month + 15.2, ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Infections per 10,000 at risk")
    return _save(fig, out)


def waning(
    fit: HybridImmunityFit,
    out: Path | BinaryIO,
    *,
    name: str = "since",
    truth: Mapping[str, float] | None = None,
) -> Path:
    """Hazard ratio by time since the last exposure, against the planted values."""
    t = _theme
    fig = _figure(
        4.8,
        "Protection wanes with time since the last exposure",
        "A covariate that changes during follow-up: people move between bands, and the "
        "aggregated fit stays exact.",
        legend=truth is not None,
    )
    if truth is not None:
        _legend(
            fig,
            [
                Patch(color=t.ramp[2], label="estimate, 95% CI"),
                Line2D(
                    [],
                    [],
                    ls="",
                    marker="o",
                    ms=10,
                    mfc="none",
                    mec=t.ink,
                    mew=1.4,
                    label="planted truth",
                ),
            ],
            2,
        )
    ax = fig.subplots()
    eff = fit.covariate_effects(name)
    colors = (
        list(t.ramp[1 : 1 + len(eff)]) if len(eff) <= len(t.ramp) - 1 else [t.ramp[2]] * len(eff)
    )
    for i, e in enumerate(eff):
        ax.bar(i, e.hr, color=colors[i], width=0.6)
        top = e.hr
        if np.isfinite(e.se):
            lo, hi = e.ci
            ax.plot([i, i], [lo, hi], color=t.ink, lw=1.4, zorder=4)
            top = hi
        ax.text(
            i,
            top + 0.04,
            f"{e.hr:.2f}",
            ha="center",
            va="bottom",
            color=t.ink,
            fontsize=12.5,
            fontweight="bold",
        )
        if truth is not None and e.term in truth and i > 0:
            _ring(ax, i, float(np.exp(truth[e.term] - truth.get(eff[0].term, 0.0))))
    ax.axhline(1, color=t.ink_3, lw=1.2, ls=(0, (3, 3)))

    def band(term: str) -> str:
        return (
            f"{term[:-2]}+ days" if term.endswith("d+") else term.replace("-", "–")[:-1] + " days"
        )

    ax.set_xticks(range(len(eff)), [band(e.term) for e in eff])
    ax.set_ylim(0, max(float(np.max([e.hr for e in eff])), 1.0) * 1.22)
    ax.set_ylabel("Hazard ratio vs the first band")
    ax.set_xlabel("Time since the last dose or infection")
    return _save(fig, out)


def power_curves(
    sizes: np.ndarray,
    curves: Mapping[str, np.ndarray],
    out: Path | BinaryIO,
    *,
    simulated: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    markers: Mapping[str, float] | None = None,
    progress: float = 1.0,
) -> Path:
    """Power of a contrast against cohort size, one curve per true effect."""
    t = _theme
    labels = list(curves)
    colors = dict(zip(labels, t.ramp[-len(labels) :], strict=True))
    fig = _figure(
        5.2,
        "A sub-sample cannot see small effects",
        "Imprinting test (BBBI vs BBI, two-sided 5%): analytic curves from one fit; dots are "
        "30 simulated cohorts.",
        legend=True,
    )
    _legend(fig, [Line2D([], [], color=colors[k], lw=3, label=f"true {k}") for k in labels], 3)
    ax = fig.subplots()
    for name, n in (markers or {}).items():
        ax.axvline(n, color=t.ink_3, lw=1.2, ls=(0, (3, 3)), zorder=1)
        ax.text(n * 1.04, 0.04, name, color=t.ink_2, fontsize=11, va="bottom")
    m = len(sizes)
    for k, label in enumerate(labels):
        frac = _stage(progress, k, len(labels)) if progress < 1 else 1.0
        if frac <= 0:
            continue
        upto = max(2, int(frac * m))
        ax.plot(sizes[:upto], curves[label][:upto], color=colors[label], lw=2.6, zorder=3)
        if simulated and label in simulated and frac >= 1:
            sx, sp, hw = simulated[label]
            ax.errorbar(
                sx,
                sp,
                yerr=hw,
                fmt="o",
                ms=7,
                color=colors[label],
                mec=t.surface,
                mew=1.6,
                elinewidth=1.4,
                capsize=0,
                zorder=5,
            )
    ax.axhline(0.8, color=t.ink_3, lw=1.0, zorder=1)
    ax.text(sizes[0], 0.81, "80% power", color=t.ink_3, fontsize=10.5, va="bottom")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([3e4, 1e5, 3e5, 1e6, 3e6]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(_people))
    ax.set_xlim(sizes[0], sizes[-1])
    ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("People in the cohort")
    return _save(fig, out)


def histories(
    people: pd.DataFrame,
    exposures: pd.DataFrame,
    events: pd.DataFrame,
    out: Path | BinaryIO,
    *,
    blank_days: int = 28,
    horizon: int = 365,
) -> Path:
    """Swimmer lanes: dated doses and infections, then calendar follow-up."""
    t = _theme
    fig = _figure(
        6.6,
        "Exposure histories become calendar-time risk sets",
        "Sixteen synthetic people: doses and a first infection before day 0, then follow-up "
        "with reinfections.",
        legend=True,
    )
    _legend(
        fig,
        [
            Line2D(
                [], [], ls="", marker="|", ms=12, mew=2.6, color=t.series[0], label="Comirnaty dose"
            ),
            Line2D(
                [], [], ls="", marker="|", ms=12, mew=2.6, color=t.series[1], label="CoronaVac dose"
            ),
            Line2D([], [], ls="", marker="o", ms=8, color=t.ink, label="first infection"),
            Line2D([], [], ls="", marker="D", ms=7, color=t.series[2], label="reinfection"),
            Patch(color=t.grid, label="episode window"),
        ],
        5,
    )
    ax = fig.subplots()
    ppl = people.reset_index(drop=True)
    n = len(ppl)
    lo_day = int(min(exposures["day"].min(), 0)) - 20
    ax.axvspan(0, horizon, color=t.soft[0], lw=0, zorder=0, alpha=0.5)
    for i, row in ppl.iterrows():
        y = n - 1 - int(cast(int, i))
        pid = row["id"]
        ax.plot([lo_day, horizon], [y, y], color=t.grid, lw=1.0, zorder=1)
        ex = exposures[exposures["id"] == pid]
        for _, e in ex.iterrows():
            if e["kind"] == "infection":
                ax.scatter(
                    [e["day"]], [y], s=46, color=t.ink, zorder=4, edgecolor=t.surface, linewidth=1.2
                )
            else:
                c = t.series[0] if "omirn" in str(e["kind"]) else t.series[1]
                ax.plot(
                    [e["day"]] * 2,
                    [y - 0.28, y + 0.28],
                    color=c,
                    lw=2.6,
                    zorder=3,
                    solid_capstyle="butt",
                )
        a, end = int(row["entry"]), int(row["exit"])
        for d in sorted(events.loc[events["id"] == pid, "day"].tolist()):
            ax.plot([a, d], [y, y], color=t.ink_2, lw=2.0, zorder=2)
            ax.add_patch(
                plt.Rectangle((d, y - 0.22), blank_days, 0.44, color=t.grid, lw=0, zorder=2)
            )
            ax.scatter(
                [d],
                [y],
                s=36,
                marker="D",
                color=t.series[2],
                zorder=5,
                edgecolor=t.surface,
                linewidth=1.0,
            )
            a = d + 1 + blank_days
        if a < end:
            ax.plot([a, end], [y, y], color=t.ink_2, lw=2.0, zorder=2)
    ax.axvline(0, color=t.ink_2, lw=1.2, zorder=1)
    ax.set_yticks(range(n), ppl["profile"].tolist()[::-1])
    ax.set_ylim(-0.7, n - 0.3)
    ax.grid(axis="y", visible=False)
    ticks = [d for d in range(-720, 1, 180)] + [180, 365]
    ax.set_xticks(
        [d for d in ticks if d >= lo_day],
        [f"{d:+d}" if d else "day 0" for d in ticks if d >= lo_day],
    )
    ax.set_xlim(lo_day, horizon + 6)
    ax.set_xlabel("Days relative to the start of follow-up")
    return _save(fig, out)


def _card(
    fig: Figure, rect: tuple[float, float, float, float], value: str, label: str, color: str
) -> None:
    from matplotlib.patches import FancyBboxPatch

    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.04),
            0.98,
            0.92,
            boxstyle="round,pad=0,rounding_size=0.09",
            transform=ax.transAxes,
            fc=_theme.surface,
            ec=_theme.grid,
            lw=1.2,
            mutation_aspect=rect[3] / rect[2] * fig.get_figheight() / fig.get_figwidth(),
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.06, 0.18),
            0.022,
            0.64,
            boxstyle="round,pad=0,rounding_size=0.01",
            transform=ax.transAxes,
            fc=color,
            ec="none",
        )
    )
    ax.text(
        0.14,
        0.62,
        value,
        transform=ax.transAxes,
        fontsize=26,
        fontweight="bold",
        color=color,
        va="center",
    )
    ax.text(
        0.14, 0.27, label, transform=ax.transAxes, fontsize=10.5, color=_theme.ink_2, va="center"
    )


def benchmark(rows: pd.DataFrame, out: Path | BinaryIO, progress: float = 1.0) -> Path:
    """Build + fit time and peak memory against cohort size, naive vs aggregated.

    Person-day cost beyond the largest measured size is extrapolated linearly (it scales
    with person-days) and drawn dashed; the zone where it no longer fits in memory is shaded.
    """
    t = _theme
    fig = plt.figure(figsize=(WIDTH, 9.0))
    H = 9.0
    fig.text(
        0.012,
        1 - 0.2 / H,
        "Same likelihood, a table that does not grow",
        ha="left",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=t.ink,
    )
    fig.text(
        0.012,
        1 - 0.62 / H,
        "Build + fit cost on synthetic cohorts (2 vCPU, 7 GB RAM). "
        "Dashed: person-day cost extrapolated linearly.",
        ha="left",
        va="top",
        fontsize=12.5,
        color=t.ink_2,
    )

    pd_ = rows[rows["pipeline"] == "person-day"].sort_values("people")
    ag = rows[rows["pipeline"] == "aggregated"].sort_values("people")
    common = int(pd_["people"].max())
    a_c = ag[ag["people"] == common].iloc[0]
    p_c = pd_[pd_["people"] == common].iloc[0]
    big = ag.iloc[-1]
    stage_cards = _stage(progress, 0, 3) if progress < 1 else 1.0
    cards = [
        (
            f"{p_c['seconds'] / a_c['seconds']:.0f}×",
            f"faster at {_people(common)} people",
            t.series[0],
        ),
        (
            f"{p_c['mib'] / a_c['mib']:.0f}×",
            f"less memory at {_people(common)} people",
            t.series[2],
        ),
        (
            f"{p_c['person_days'] / a_c['glm_rows']:.0f}×",
            f"fewer rows ({p_c['person_days'] / 1e6:.1f}M → {a_c['glm_rows']:,})",
            t.ramp[3],
        ),
        (
            f"{big['seconds']:.0f} s",
            f"for {_people(big['people'])} people, {big['mib']:.0f} MiB",
            t.series[1],
        ),
    ]
    if stage_cards > 0:
        w, gap, x0, y0, h = 0.232, 0.012, 0.012, 1 - 1.95 / H, 0.95 / H
        for k, (v, lab, col) in enumerate(cards):
            if _stage(stage_cards, k, len(cards)) > 0.05 or progress >= 1:
                _card(fig, (x0 + k * (w + gap), y0, w, h), v, lab, col)

    leg_y = 1 - 2.22 / H
    fig.legend(
        handles=[
            Line2D([], [], color=t.ink_3, lw=3, marker="o", ms=6, label="person-day GLM, measured"),
            Line2D(
                [], [], color=t.ink_3, lw=2.2, ls=(0, (3, 2.5)), label="person-day, extrapolated"
            ),
            Line2D([], [], color=t.series[0], lw=3, marker="o", ms=6, label="vvi, aggregated"),
        ],
        loc="upper left",
        ncols=3,
        bbox_to_anchor=(0.006, leg_y),
    )

    grid = fig.add_gridspec(
        2, 1, left=0.085, right=0.83, top=1 - 3.1 / H, bottom=0.075, hspace=0.42
    )
    lines = _stage(progress, 1, 3) if progress < 1 else 1.0
    notes = _stage(progress, 2, 3) if progress < 1 else 1.0
    xmax = 1.6e6
    for i, (col, title, unit, fmt) in enumerate(
        (
            (
                "seconds",
                "Build + fit time",
                "s",
                lambda v: f"{v:,.0f} s" if v < 3600 else f"{v / 3600:.0f} h",
            ),
            (
                "mib",
                "Peak memory",
                "MiB",
                lambda v: (
                    f"{v:,.0f} MiB"
                    if v < 10_000
                    else f"{v / 1024**2:.1f} TiB"
                    if v > 1024**2 * 0.9
                    else f"{v / 1024:,.0f} GiB"
                ),
            ),
        )
    ):
        ax = fig.add_subplot(grid[i])
        ax.set_xscale("log")
        ax.set_yscale("log")
        # extrapolation of the person-day fit: linear in person-days
        scale = float(big["people"]) / common
        ext_x = np.array([common, float(big["people"])])
        ext_y = np.array([p_c[col], p_c[col] * scale])
        lo_y = float(min(ag[col].min(), pd_[col].min())) / 1.8
        hi_y = float(ext_y[-1]) * 3.5
        ax.set_ylim(lo_y, hi_y)
        ax.set_xlim(700, xmax)
        ax.axvspan(
            common * 1.6,
            xmax,
            color=t.soft[1],
            alpha=0.4 if t.name == "light" else 0.75,
            lw=0,
            zorder=0,
        )
        ax.text(
            common * 1.75,
            hi_y / 1.6,
            "person-day panel no longer fits in 7 GB",
            color=t.ink_2,
            fontsize=10.5,
            va="top",
            zorder=2,
        )

        xs, ys = _partial(ag["people"].to_numpy(float), ag[col].to_numpy(float), lines)
        ax.fill_between(xs, lo_y, ys, color=t.series[0], alpha=0.07, lw=0, zorder=1)
        ax.plot(xs, ys, color=t.series[0], lw=3, zorder=4)
        px, py = _partial(
            pd_["people"].to_numpy(float), pd_[col].to_numpy(float), min(1.0, lines * 3)
        )
        ax.plot(px, py, color=t.ink_3, lw=3, zorder=3)
        shown_ag = ag[ag["people"] <= xs[-1] * 1.0001]
        ax.scatter(
            shown_ag["people"],
            shown_ag[col],
            s=40,
            color=t.series[0],
            edgecolor=t.surface,
            linewidth=1.6,
            zorder=5,
        )
        shown_pd = pd_[pd_["people"] <= px[-1] * 1.0001]
        ax.scatter(
            shown_pd["people"],
            shown_pd[col],
            s=40,
            color=t.ink_3,
            edgecolor=t.surface,
            linewidth=1.6,
            zorder=5,
        )
        if lines >= 1:
            ex, ey = _partial(ext_x, ext_y, notes)
            ax.plot(ex, ey, color=t.ink_3, lw=2.2, ls=(0, (3, 2.5)), zorder=3)
        if notes >= 1:
            ax.annotate(
                f"{fmt(ext_y[-1])}\n(extrapolated)",
                (ext_x[-1], ext_y[-1]),
                xytext=(10, 0),
                textcoords="offset points",
                va="center",
                color=t.ink_2,
                fontsize=10.5,
            )
            ax.annotate(
                f"{fmt(big[col])}\nfor {_people(big['people'])}",
                (big["people"], big[col]),
                xytext=(10, 0),
                textcoords="offset points",
                va="center",
                color=t.ink,
                fontsize=11,
                fontweight="bold",
            )
            ratio = ext_y[-1] / float(big[col])
            gx = float(big["people"]) * 0.62
            top = float(np.exp(np.interp(np.log(gx), np.log(ext_x), np.log(ext_y))))
            bot = float(np.exp(np.interp(np.log(gx), np.log(ag["people"]), np.log(ag[col]))))
            ax.annotate(
                "",
                xy=(gx, bot * 1.25),
                xytext=(gx, top / 1.25),
                arrowprops={
                    "arrowstyle": "<|-|>",
                    "color": t.series[0],
                    "lw": 1.4,
                    "mutation_scale": 10,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=6,
            )
            _pill(ax, gx, float(np.sqrt(top * bot)), f"{ratio:,.0f}×", t.series[0], size=11)
        ax.xaxis.set_major_locator(FixedLocator([1e3, 1e4, 1e5, 1e6]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(_people))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
        )
        ax.set_title(f"{title} ({unit})")
        if i == 1:
            ax.set_xlabel("People in the cohort")
    return _save(fig, out)


# --------------------------------------------------------------------------- social card


def social_preview(fit: HybridImmunityFit, out: Path) -> Path:
    """1280×640 repository social preview."""
    t = _theme
    fig = plt.figure(figsize=(12.8, 6.4))
    fig.text(0.06, 0.84, "VVI", fontsize=40, fontweight="bold", color=t.ink, va="top")
    fig.text(
        0.06,
        0.66,
        "Hybrid immunity and immune imprinting in calendar time",
        fontsize=21,
        color=t.ink_2,
        va="top",
    )
    fig.text(
        0.06,
        0.57,
        "Exact, sampling-free hazard models for vaccination–infection histories",
        fontsize=15,
        color=t.ink_3,
        va="top",
    )
    ax = fig.add_axes((0.06, 0.08, 0.88, 0.38))
    days = np.arange(365)
    colors = (t.ink_3, t.ramp[2], t.series[2], t.series[1])
    for p, c in zip(HERO_PROFILES, colors, strict=True):
        ax.plot(days, fit.baseline_hazard(days, profile=p), color=c, lw=4.5)
    ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- animation


def animate(
    render: Callable[[BinaryIO, float], object],
    out: Path,
    *,
    frames: int = 44,
    frame_ms: int = 55,
    hold_ms: int = 3200,
    dpi: float = 150,
) -> Path:
    """Render ``render(buffer, progress)`` for progress 0..1 and write an optimised GIF.

    The last frame is held so the finished chart stays on screen before the loop restarts.
    Frames are quantised to a shared adaptive palette to keep files small.
    """
    from PIL import Image

    images = []
    for i in range(frames):
        buffer = io.BytesIO()
        with plt.rc_context({"savefig.dpi": dpi}):
            render(buffer, (i + 1) / frames)
        buffer.seek(0)
        images.append(Image.open(buffer).convert("RGB"))
    palette = images[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    frames_p = [img.quantize(palette=palette, dither=Image.Dither.NONE) for img in images]
    durations = [frame_ms] * (frames - 1) + [hold_ms]
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_p[0].save(
        out,
        save_all=True,
        append_images=frames_p[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    return out
