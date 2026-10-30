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


# --------------------------------------------------------------------------- hero


HERO_PROFILES = ("BB", "BBB", "BBI", "BBBI")


def hero(fit: HybridImmunityFit, out: Path | BinaryIO, progress: float = 1.0) -> Path:
    """Fitted daily infection rate by exposure history, and the hazard ratios behind it."""
    t = _theme
    colors = dict(zip(HERO_PROFILES, (t.ink_3, t.ramp[2], t.series[2], t.series[1]), strict=True))
    labels = {"BB": "BB (reference)", "BBB": "BBB", "BBI": "BBI", "BBBI": "BBBI"}
    fig = _figure(
        5.8,
        "Reinfection risk in calendar time, by exposure history",
        "600,000 synthetic people (B = Comirnaty dose, I = infection): the same epidemic "
        "waves, different hazard ratios.",
        legend=True,
    )
    _legend(fig, [Line2D([], [], color=colors[p], lw=3, label=labels[p]) for p in HERO_PROFILES], 4)
    ax, bx = fig.subplots(1, 2, width_ratios=(2.1, 1))

    days = np.arange(0, 365)
    frac = _stage(progress, 0, 2) if progress < 1 else 1.0
    n = max(2, int(frac * len(days)))
    peak = 0.0
    for p in HERO_PROFILES:
        rate = fit.baseline_hazard(days, profile=p) * 1e4
        peak = max(peak, float(np.max(rate)))
        ax.plot(
            days[:n],
            rate[:n],
            color=colors[p],
            lw=3.0 if p == "BBBI" else 2.2,
            zorder=3 if p == "BBBI" else 2,
        )
    ax.set_xlim(-4, 368)
    ax.set_ylim(0, peak * 1.12)
    month = np.arange(0, 365, 30.42)
    ax.set_xticks(month + 15.2, ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.tick_params(axis="x", length=0)
    ax.set_title("Daily infections per 10,000 at risk")
    ax.set_xlabel("Calendar year")

    bars = [p for p in HERO_PROFILES if p != fit.reference]
    grow = _stage(progress, 1, 2) if progress < 1 else 1.0
    for i, p in enumerate(bars):
        e = fit.profile_effect(p)
        h = 1 - (1 - e.hr) * grow
        bx.bar(i, h, color=colors[p], width=0.62)
        if grow >= 1:
            lo, hi = e.ci
            bx.plot([i, i], [lo, hi], color=t.ink, lw=1.4, zorder=4)
            bx.text(
                i,
                hi + 0.04,
                f"{e.hr:.2f}",
                ha="center",
                va="bottom",
                color=t.ink,
                fontsize=12.5,
                fontweight="bold",
            )
    bx.axhline(1, color=t.ink_3, lw=1.2, ls=(0, (3, 3)))
    bx.text(len(bars) - 0.5, 1.02, "BB = 1", ha="right", va="bottom", color=t.ink_3, fontsize=10.5)
    bx.set_xticks(range(len(bars)), bars)
    bx.set_ylim(0, 1.18)
    bx.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    bx.set_title("Hazard ratio vs BB")
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


def benchmark(rows: pd.DataFrame, out: Path | BinaryIO) -> Path:
    """Build + fit time and peak memory against cohort size, naive vs aggregated."""
    t = _theme
    fig = _figure(
        7.2,
        "Same likelihood, a table that does not grow",
        "Synthetic cohorts on 2 vCPU / 7 GB RAM: the person-day panel runs out of memory "
        "past ~5,000 people.",
        legend=True,
    )
    _legend(
        fig,
        [
            Line2D([], [], color=t.ink_3, lw=3, marker="o", ms=6, label="person-day GLM"),
            Line2D([], [], color=t.series[0], lw=3, marker="o", ms=6, label="vvi, aggregated"),
        ],
        2,
    )
    axes = fig.subplots(2, 1, sharex=True)
    for ax, col, title, unit in (
        (axes[0], "seconds", "Build + fit time", "s"),
        (axes[1], "mib", "Peak memory", "MiB"),
    ):
        for name, color in (("person-day", t.ink_3), ("aggregated", t.series[0])):
            r = rows[rows["pipeline"] == name].sort_values("people")
            ax.plot(r["people"], r[col], color=color, lw=2.6, zorder=3)
            ax.scatter(
                r["people"], r[col], s=40, color=color, edgecolor=t.surface, linewidth=1.6, zorder=4
            )
            last = r.iloc[-1]
            text = (
                f"{last[col]:,.0f} {unit} for {_people(last['people'])}"
                if name == "aggregated"
                else f"{last[col]:,.0f} {unit} for 5k"
            )
            ax.annotate(
                text,
                (last["people"], last[col]),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                color=t.ink,
                fontsize=11,
                fontweight="bold",
            )
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
        )
        ax.set_title(f"{title} ({unit})")
    axes[1].set_xscale("log")
    axes[1].xaxis.set_major_locator(FixedLocator([1e3, 1e4, 1e5, 1e6]))
    axes[1].xaxis.set_minor_locator(NullLocator())
    axes[1].xaxis.set_major_formatter(FuncFormatter(_people))
    axes[1].set_xlim(700, 3.2e6)
    axes[1].set_xlabel("People in the cohort")
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
