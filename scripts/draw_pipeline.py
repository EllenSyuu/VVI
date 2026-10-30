"""Draw docs/figures/pipeline.svg — the four steps as an animated Atelier plate.

    python scripts/draw_pipeline.py

Fonts are subset from the package's bundled TTFs and embedded as WOFF2, so the SVG
renders identically wherever it is shown (including as a GitHub README image).
"""

from __future__ import annotations

import base64
import html
import io
import re
from importlib import resources
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
INK, GRAPH, FAINT, PAPER = "#22314a", "#6f7b8c", "#b4bcc7", "#f6f1e6"
ORANGE, BLUE, AMBER = "#cf602b", "#1d6fb8", "#e0a046"
SERIF = "'Cormorant Garamond', Garamond, Georgia, serif"
HAND = "'La Belle Aurore', cursive"
CODE = "'JetBrains Mono', ui-monospace, monospace"

W, H = 960, 330
BOX_W, BOX_H, GAP, X0, Y0 = 206, 214, 26, 34, 78


def t(
    x,
    y,
    s,
    *,
    font=SERIF,
    size=12,
    fill=INK,
    weight=None,
    style=None,
    ls=None,
    anchor=None,
    cls=None,
    delay=None,
    halo=False,
):
    attrs = [
        f'x="{x}"',
        f'y="{y}"',
        f'font-family="{font}"',
        f'font-size="{size}"',
        f'fill="{fill}"',
    ]
    if weight:
        attrs.append(f'font-weight="{weight}"')
    if style:
        attrs.append(f'font-style="{style}"')
    if ls:
        attrs.append(f'letter-spacing="{ls}"')
    if anchor:
        attrs.append(f'text-anchor="{anchor}"')
    if cls:
        attrs.append(f'class="{cls}"')
    if delay is not None:
        attrs.append(f'style="animation-delay:{delay}s"')
    if halo:
        attrs.append(f'stroke="{PAPER}" stroke-width="3.5" paint-order="stroke"')
    return f"<text {' '.join(attrs)}>{html.escape(s)}</text>"


def ink(d, *, w=0.9, stroke=INK, delay=0.0, fill="none", extra=""):
    return (
        f'<path class="ink" pathLength="1" style="animation-delay:{delay}s" d="{d}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{w}" {extra}/>'
    )


def box(i, numeral, kicker, title, fn):
    x = X0 + i * (BOX_W + GAP)
    d0 = 0.3 + i * 0.45
    out = [
        f'<rect class="fade" style="animation-delay:{d0}s" x="{x}" y="{Y0}" width="{BOX_W}" '
        f'height="{BOX_H}" fill="#fbf8f1"/>',
        ink(f"M{x} {Y0} H{x + BOX_W} V{Y0 + BOX_H} H{x} Z", delay=d0),
        ink(
            f"M{x + 4} {Y0 + 4} H{x + BOX_W - 4} V{Y0 + BOX_H - 4} H{x + 4} Z",
            w=0.35,
            delay=d0 + 0.1,
        ),
        f'<g class="fade" style="animation-delay:{d0 + 0.4}s">',
        t(
            x + 16,
            Y0 + 26,
            f"{numeral} · {kicker.upper()}",
            weight=600,
            size=10,
            fill=GRAPH,
            ls="0.18em",
        ),
        t(x + 16, Y0 + 50, title, style="italic", size=19),
        f'<line x1="{x + 16}" x2="{x + BOX_W - 16}" y1="{Y0 + 60}" y2="{Y0 + 60}" '
        f'stroke="{INK}" stroke-width="0.5"/>',
        t(x + 16, Y0 + BOX_H - 16, fn, font=CODE, size=10, fill=GRAPH),
        "</g>",
    ]
    return "\n".join(out), x, d0


def arrow(x, y, delay):
    return "\n".join(
        [
            ink(
                f"M{x + 3} {y} C {x + 10} {y - 4}, {x + 16} {y + 4}, {x + GAP - 4} {y}",
                w=0.9,
                stroke=GRAPH,
                delay=delay,
            ),
            ink(
                f"M{x + GAP - 9} {y - 4} L{x + GAP - 3} {y} L{x + GAP - 9} {y + 4}",
                w=0.9,
                stroke=GRAPH,
                delay=delay + 0.4,
            ),
        ]
    )


def body() -> str:
    parts = []
    mid = Y0 + BOX_H / 2
    # 1 · histories
    b, x, d = box(0, "I", "histories", "Exposure histories", "vvi.encode_table()")
    parts.append(b)
    rows = [
        ("BBBI", [BLUE, BLUE, BLUE], True),
        ("SSI", [ORANGE, ORANGE], True),
        ("BB", [BLUE, BLUE], False),
    ]
    for k, (code, doses, inf) in enumerate(rows):
        yy = Y0 + 92 + k * 34
        parts.append(f'<g class="fade" style="animation-delay:{d + 0.7 + k * 0.15}s">')
        parts.append(
            f'<line x1="{x + 16}" x2="{x + 126}" y1="{yy}" y2="{yy}" stroke="{FAINT}" '
            f'stroke-width="0.5"/>'
        )
        for j, col in enumerate(doses):
            xx = x + 22 + j * 22 + (14 if j == 2 else 0)
            parts.append(
                f'<line x1="{xx}" x2="{xx}" y1="{yy - 7}" y2="{yy + 7}" stroke="{col}" '
                f'stroke-width="2.4"/>'
            )
        if inf:
            parts.append(
                f'<circle cx="{x + 112}" cy="{yy}" r="4.6" fill="{INK}" '
                f'stroke="#fbf8f1" stroke-width="1"/>'
            )
        parts.append(t(x + 140, yy + 4, code, weight=600, size=13, ls="0.06em"))
        parts.append("</g>")
    parts.append(arrow(x + BOX_W, mid, d + 1.2))

    # 2 · risk set
    b, x, d = box(1, "II", "risk set", "Calendar follow-up", "vvi.at_risk_intervals()")
    parts.append(b)
    lanes = [(18, 92, 66, True), (40, 174, None, False), (18, 120, 100, True)]
    for k, (a, e, ev, _) in enumerate(lanes):
        yy = Y0 + 92 + k * 26
        x1, x2 = x + a, x + e
        if ev is None:
            parts.append(ink(f"M{x1} {yy} H{x2}", w=1.6, delay=d + 0.8 + k * 0.2))
        else:
            xe = x + ev
            parts.append(ink(f"M{x1} {yy} H{xe}", w=1.6, delay=d + 0.8 + k * 0.2))
            parts.append(f'<g class="fade" style="animation-delay:{d + 1.1 + k * 0.2}s">')
            for h in range(0, 26, 5):
                parts.append(
                    f'<line x1="{xe + 4 + h}" x2="{xe + 9 + h}" y1="{yy + 4}" '
                    f'y2="{yy - 4}" stroke="{GRAPH}" stroke-width="0.8"/>'
                )
            parts.append(
                f'<circle cx="{xe}" cy="{yy}" r="4" fill="{ORANGE}" stroke="#fbf8f1" '
                f'stroke-width="1"/>'
            )
            parts.append("</g>")
            parts.append(ink(f"M{xe + 33} {yy} H{x + 182}", w=1.6, delay=d + 1.3 + k * 0.2))
    parts.append(
        f'<g class="fade" style="animation-delay:{d + 0.6}s">'
        f'<line x1="{x + 16}" x2="{x + 190}" y1="{Y0 + 166}" y2="{Y0 + 166}" '
        f'stroke="{INK}" stroke-width="0.6"/>'
        + "".join(
            f'<line x1="{x + 16 + m * 14.5}" x2="{x + 16 + m * 14.5}" '
            f'y1="{Y0 + 163}" y2="{Y0 + 169}" stroke="{INK}" stroke-width="0.5"/>'
            for m in range(13)
        )
        + t(x + 16, Y0 + 182, "JAN", weight=600, size=8.5, fill=GRAPH, ls="0.16em")
        + t(x + 190, Y0 + 182, "DEC", weight=600, size=8.5, fill=GRAPH, ls="0.16em", anchor="end")
        + "</g>"
    )
    parts.append(arrow(x + BOX_W, mid, d + 1.6))

    # 3 · ledger (the point)
    b, x, d = box(2, "III", "sufficient statistics", "A ledger, not a panel", "vvi.aggregate()")
    parts.append(
        b.replace(f'stroke="{INK}" stroke-width="0.9"', f'stroke="{ORANGE}" stroke-width="1.2"', 1)
    )
    cols = [16, 92, 128, 176]
    heads = ["CELL", "DAY", "N", "D"]
    rows3 = [
        ("BBBI·F·18", "70", "4 812", "3"),
        ("BBBI·F·18", "71", "4 809", "5"),
        ("SSI·M·60", "70", "1 290", "1"),
    ]
    parts.append(f'<g class="fade" style="animation-delay:{d + 0.7}s">')
    for c, h in zip(cols, heads, strict=True):
        parts.append(t(x + c, Y0 + 84, h, weight=600, size=9, fill=GRAPH, ls="0.16em"))
    parts.append(
        f'<line x1="{x + 14}" x2="{x + BOX_W - 14}" y1="{Y0 + 90}" y2="{Y0 + 90}" '
        f'stroke="{INK}" stroke-width="0.5"/>'
    )
    for k, r in enumerate(rows3):
        yy = Y0 + 108 + k * 18
        for c, v in zip(cols, r, strict=True):
            parts.append(t(x + c, yy, v, weight=600, size=11.5, fill=INK))
        parts.append(
            f'<line x1="{x + 14}" x2="{x + BOX_W - 14}" y1="{yy + 5}" y2="{yy + 5}" '
            f'stroke="{FAINT}" stroke-width="0.4"/>'
        )
    parts.append("</g>")
    parts.append(
        t(
            x + 16,
            Y0 + 176,
            "10⁸ person-days → 10⁴ rows",
            font=HAND,
            size=14,
            fill=ORANGE,
            cls="write",
            delay=d + 1.2,
        )
    )
    parts.append(arrow(x + BOX_W, mid, d + 1.6))

    # 4 · model
    b, x, d = box(3, "IV", "model", "Exact cloglog GLM", "vvi.fit()")
    parts.append(b)
    parts.append(
        f'<g class="fade" style="animation-delay:{d + 0.6}s">'
        + t(x + 16, Y0 + 88, "cloglog p = η(t)", style="italic", size=15)
        + t(x + 34, Y0 + 108, "+ α′x + β", style="italic", size=15)
        + t(x + 86, Y0 + 114, "profile", style="italic", size=10, fill=GRAPH)
        + "</g>"
    )
    crest = (
        f"M{x + 18} {Y0 + 150} C {x + 40} {Y0 + 150}, {x + 44} {Y0 + 116}, {x + 62} "
        f"{Y0 + 116} C {x + 80} {Y0 + 116}, {x + 84} {Y0 + 146}, {x + 108} {Y0 + 144} "
        f"C {x + 130} {Y0 + 142}, {x + 140} {Y0 + 128}, {x + 160} {Y0 + 130} L{x + 188} "
        f"{Y0 + 140}"
    )
    for k, off in enumerate((4, 8, 13)):
        parts.append(
            ink(
                crest,
                w=0.45,
                stroke=GRAPH,
                delay=d + 1.0 + k * 0.1,
                extra=f'transform="translate(0 {off})" opacity="{0.7 - k * 0.2}"',
            )
        )
    parts.append(ink(crest, w=1.5, delay=d + 0.8))
    parts.append(
        t(
            x + 16,
            Y0 + 178,
            "HR · PE = 1 − HR · contrasts",
            font=HAND,
            size=12.5,
            fill=INK,
            cls="write",
            delay=d + 1.4,
        )
    )

    # plate furniture
    parts.insert(
        0,
        "\n".join(
            [
                f'<rect width="{W}" height="{H}" fill="{PAPER}"/>',
                ink(f"M8 8 H{W - 8} V{H - 8} H8 Z", w=0.9, delay=0),
                ink(f"M12 12 H{W - 12} V{H - 12} H12 Z", w=0.35, delay=0.1),
                '<g class="fade" style="animation-delay:0.2s">',
                t(X0, 44, "HOW VVI WORKS", weight=600, size=10.5, fill=GRAPH, ls="0.28em"),
                t(
                    W - X0,
                    46,
                    "four steps, no person-day row ever built",
                    font=HAND,
                    size=15,
                    fill=ORANGE,
                    anchor="end",
                ),
                f'<line x1="{X0}" x2="{W - X0}" y1="56" y2="56" stroke="{INK}" stroke-width="0.6"/>',
                f'<line x1="{X0}" x2="{W - X0}" y1="59" y2="59" stroke="{INK}" stroke-width="0.3"/>',
                "</g>",
            ]
        ),
    )
    return "\n".join(parts)


CSS = """
.ink{stroke-dasharray:1;stroke-dashoffset:1;animation:vv-draw 1.4s cubic-bezier(.55,.05,.25,1) forwards}
.fade{opacity:0;animation:vv-fade 1s ease forwards}
.write{clip-path:inset(-20% 100% -20% 0);animation:vv-write 1.2s cubic-bezier(.5,0,.3,1) forwards}
@keyframes vv-draw{to{stroke-dashoffset:0}}
@keyframes vv-fade{to{opacity:1}}
@keyframes vv-write{to{clip-path:inset(-20% -5% -20% 0)}}
@media (prefers-reduced-motion:reduce){*{animation:none!important}.ink{stroke-dashoffset:0}.fade{opacity:1}.write{clip-path:none}}
"""


def font_face(file: str, family: str, weight: int, style: str, text: str) -> str:
    with resources.as_file(resources.files("vvi") / "fonts" / file) as p:
        f = TTFont(p)
    opts = subset.Options()
    opts.flavor = "woff2"
    sub = subset.Subsetter(opts)
    sub.populate(text=text)
    sub.subset(f)
    buf = io.BytesIO()
    f.flavor = "woff2"
    f.save(buf)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        f"@font-face{{font-family:'{family}';font-weight:{weight};font-style:{style};"
        f"src:url(data:font/woff2;base64,{b64}) format('woff2')}}"
    )


def main() -> None:
    svg_body = body()
    text = "".join(html.unescape(s) for s in re.findall(r">([^<]+)<", svg_body)) + "0123456789"
    faces = "\n".join(
        [
            font_face("CormorantGaramond-SemiBold.ttf", "Cormorant Garamond", 600, "normal", text),
            font_face("CormorantGaramond-Italic.ttf", "Cormorant Garamond", 400, "italic", text),
            font_face("LaBelleAurore-Regular.ttf", "La Belle Aurore", 400, "normal", text),
            font_face("JetBrainsMono-Regular.ttf", "JetBrains Mono", 400, "normal", text),
        ]
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" role="img" aria-labelledby="pl-t"><title id="pl-t">How VVI works: '
        "exposure histories, calendar follow-up, a ledger of sufficient statistics, and an "
        f"exact cloglog GLM.</title><style>{faces}\n{CSS}</style>\n{svg_body}\n</svg>"
    )
    out = ROOT / "docs" / "figures" / "pipeline.svg"
    out.write_text(svg)
    print(out, len(svg) // 1024, "KiB")


if __name__ == "__main__":
    main()
