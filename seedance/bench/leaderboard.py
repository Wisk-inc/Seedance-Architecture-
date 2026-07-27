"""Benchmark chart rendering.

Produces the dark bar-chart layout used across this repo: value labels above
rounded bars, a row of logo tiles, diagonal model names. Output is SVG written
by hand — no matplotlib, no fonts to install, crisp at any size, and GitHub
renders it inline in a README.

**Provenance is part of the chart, not a footnote.** Every entry declares where
its number came from:

* ``measured``        — produced by this harness on this machine.
* ``published``       — from a paper or a public leaderboard, with a citation.
* ``vendor-reported`` — the model's own marketing/benchmark claim.
* ``unverified``      — third-party or secondary; treat with suspicion.

Anything that is not ``measured`` is drawn with a hatch overlay and listed in
the legend. An entry with no ``source`` is rejected. This exists because the
easiest thing in the world is to render a chart where your own bar is tallest,
and a chart that cannot distinguish "we measured this" from "they claimed this"
is worse than no chart.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .logos import logo_tile_svg

__all__ = [
    "Entry",
    "Benchmark",
    "Theme",
    "DARK",
    "render_svg",
    "render_png",
    "load_benchmark",
    "save_benchmark",
    "SOURCES",
]

SOURCES = ("measured", "published", "vendor-reported", "unverified")

# Palette lifted from the reference layout: a bright accent for the leader, two
# supporting accents, muted slate for the field.
ACCENTS = ["#C55A38", "#0E7C66", "#12B886", "#4C4FE0", "#7A3FD1", "#D14A4A"]
MUTED = "#171A20"


@dataclass
class Entry:
    """One bar."""

    name: str
    value: float
    logo: str = "generic"
    source: str = "measured"
    citation: str = ""
    color: Optional[str] = None
    highlight: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(
                f"entry {self.name!r}: source must be one of {SOURCES}, got {self.source!r}"
            )
        if self.source != "measured" and not self.citation:
            raise ValueError(
                f"entry {self.name!r}: a non-measured number needs a citation "
                "(paper, leaderboard URL, or vendor page)"
            )


@dataclass
class Theme:
    background: str = "#0B0C0E"
    panel: str = "#0B0C0E"
    grid: str = "#22252B"
    text: str = "#FFFFFF"
    text_dim: str = "#6B7079"
    font: str = "Inter, SF Pro Display, Helvetica Neue, Arial, sans-serif"
    bar_muted: str = MUTED
    bar_radius: int = 5


DARK = Theme()


@dataclass
class Benchmark:
    """A titled set of entries, rendered as one chart."""

    title: str
    entries: List[Entry] = field(default_factory=list)
    subtitle: str = ""
    unit: str = ""
    y_max: Optional[float] = None
    higher_is_better: bool = True
    footnote: str = ""

    def sorted_entries(self, by_value: bool = False) -> List[Entry]:
        if not by_value:
            return list(self.entries)
        return sorted(self.entries, key=lambda e: e.value, reverse=self.higher_is_better)

    def leader(self) -> Optional[Entry]:
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.value) if self.higher_is_better else min(
            self.entries, key=lambda e: e.value
        )

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "entries": [asdict(e) for e in self.entries]}


def load_benchmark(path: str) -> List[Benchmark]:
    """Load one or more benchmarks from a JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    blocks = data if isinstance(data, list) else [data]
    out = []
    for block in blocks:
        entries = [Entry(**e) for e in block.get("entries", [])]
        payload = {k: v for k, v in block.items() if k != "entries"}
        out.append(Benchmark(entries=entries, **payload))
    return out


def save_benchmark(benchmarks: Sequence[Benchmark], path: str) -> str:
    payload = [b.to_dict() for b in benchmarks]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload if len(payload) > 1 else payload[0], handle, indent=2)
    return path


# --------------------------------------------------------------------- render

def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _text_width(text: str, size: float, bold: bool = True) -> float:
    """Rough advance width for a sans font; good enough for layout."""
    return len(text) * size * (0.58 if bold else 0.52)


def _wrap(text: str, width: int) -> List[str]:
    """Greedy word wrap. Returns [] for empty text so callers can skip cleanly."""
    if not text:
        return []
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _rounded_top_bar(x: float, y: float, w: float, h: float, r: float, fill: str, extra: str = "") -> str:
    """A bar with rounded top corners and square feet, as in the reference."""
    r = max(0.0, min(r, w / 2, h))
    d = (
        f"M{x:.2f},{y + h:.2f} "
        f"L{x:.2f},{y + r:.2f} "
        f"Q{x:.2f},{y:.2f} {x + r:.2f},{y:.2f} "
        f"L{x + w - r:.2f},{y:.2f} "
        f"Q{x + w:.2f},{y:.2f} {x + w:.2f},{y + r:.2f} "
        f"L{x + w:.2f},{y + h:.2f} Z"
    )
    return f'<path d="{d}" fill="{fill}"{extra}/>'


def render_svg(
    benchmark: Benchmark,
    path: Optional[str] = None,
    *,
    theme: Optional[Theme] = None,
    bar_width: float = 46.0,
    bar_gap: float = 30.0,
    plot_height: float = 400.0,
    sort_by_value: bool = False,
    show_legend: bool = True,
) -> str:
    """Render one benchmark to SVG. Returns the markup, writing it if ``path``."""
    import math

    theme = theme or DARK
    entries = benchmark.sorted_entries(by_value=sort_by_value)
    if not entries:
        raise ValueError("benchmark has no entries")

    title_h = 130.0
    tile = 44.0
    tile_gap = 22.0
    label_gap = 26.0
    label_size = 21.0
    label_angle = 52.0
    footnote_size = 13.0
    subtitle_size = 17.0

    n = len(entries)
    plot_w = n * bar_width + (n - 1) * bar_gap
    # Always leave headroom above the tallest bar, or its value label collides
    # with the title block. An explicit y_max sets the axis, not the ceiling.
    peak = max(e.value for e in entries)
    y_max = max(benchmark.y_max or 0.0, peak * 1.12, 1e-9)

    # Rotated labels are anchored at their end, so they spill down and to the
    # LEFT of the first bar. Size the left margin from that spill instead of
    # guessing, or the longest name walks off the canvas.
    longest = max(_text_width(e.name, label_size) for e in entries)
    spill_x = longest * math.cos(math.radians(label_angle))
    label_block = longest * math.sin(math.radians(label_angle))
    # 1.15 covers the letter-spacing our width estimate does not model
    pad_l = max(90.0, spill_x * 1.15 - bar_width / 2 + 30.0)
    pad_r = 70.0

    # Prose (subtitle, footnote, legend) is wrapped to the plot width, and the
    # canvas widens if even the wrapped lines need more room.
    wrap_chars = max(48, int(plot_w / (footnote_size * 0.52)))
    footnote_lines = _wrap(benchmark.footnote, wrap_chars)
    subtitle_lines = _wrap(benchmark.subtitle, max(40, int(plot_w / (subtitle_size * 0.52))))
    legend_lines = [
        f"hatched = {e.source} ({e.citation})" for e in entries if e.source != "measured"
    ]
    text_w = max(
        [_text_width(benchmark.title, 40)]
        + [_text_width(line, subtitle_size, bold=False) for line in subtitle_lines]
        + [_text_width(line, footnote_size, bold=False) for line in footnote_lines]
        + [_text_width(line, 14, bold=False) + 22 for line in legend_lines]
        or [0.0]
    )
    width = max(pad_l + plot_w + pad_r, pad_l + text_w + pad_r)

    title_h += max(0.0, (len(subtitle_lines) - 1) * (subtitle_size + 6))
    baseline = title_h + plot_height

    legend_h = 0.0
    if show_legend:
        flagged = {e.source for e in entries if e.source != "measured"}
        legend_h = 46.0 + 20.0 * len(flagged) + (footnote_size + 9) * len(footnote_lines)
    height = baseline + tile_gap + tile + label_gap + label_block * 1.15 + 34 + legend_h

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" font-family="{theme.font}">',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="{theme.background}"/>',
        '<defs>'
        '<pattern id="unmeasured" width="6" height="6" patternTransform="rotate(45)" '
        'patternUnits="userSpaceOnUse">'
        '<rect width="6" height="6" fill="none"/>'
        '<line x1="0" y1="0" x2="0" y2="6" stroke="#000000" stroke-opacity="0.38" stroke-width="3"/>'
        '</pattern></defs>',
    ]

    # title
    parts.append(
        f'<text x="{pad_l:.0f}" y="{72:.0f}" fill="{theme.text}" font-size="40" '
        f'font-weight="600">{_esc(benchmark.title)}</text>'
    )
    for i, line in enumerate(subtitle_lines):
        parts.append(
            f'<text x="{pad_l:.0f}" y="{100 + i * (subtitle_size + 6):.0f}" fill="{theme.text_dim}" '
            f'font-size="{subtitle_size:.0f}" font-weight="500">{_esc(line)}</text>'
        )

    # dashed gridlines
    for i in range(1, 5):
        y = baseline - plot_height * i / 4
        parts.append(
            f'<line x1="{pad_l:.0f}" y1="{y:.1f}" x2="{pad_l + plot_w:.0f}" y2="{y:.1f}" '
            f'stroke="{theme.grid}" stroke-width="1" stroke-dasharray="6 7"/>'
        )

    # bars
    for i, entry in enumerate(entries):
        x = pad_l + i * (bar_width + bar_gap)
        h = max(2.0, plot_height * (entry.value / y_max))
        y = baseline - h
        color = entry.color or (
            ACCENTS[i] if entry.highlight and i < len(ACCENTS) else theme.bar_muted
        )
        parts.append(_rounded_top_bar(x, y, bar_width, h, theme.bar_radius, color))
        if entry.source != "measured":
            parts.append(
                _rounded_top_bar(x, y, bar_width, h, theme.bar_radius, "url(#unmeasured)")
            )

        value_text = f"{entry.value:g}{benchmark.unit}"
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 12:.1f}" fill="{theme.text}" '
            f'font-size="31" font-weight="700" text-anchor="middle" '
            f'fill-opacity="{1.0 if entry.highlight else 0.9:.2f}">{_esc(value_text)}</text>'
        )

        # logo tile
        tile_x = x + (bar_width - tile) / 2
        tile_y = baseline + tile_gap
        parts.append(logo_tile_svg(entry.logo, tile_x, tile_y, tile))

        # diagonal label
        lx = x + bar_width / 2 + 6
        ly = tile_y + tile + label_gap
        parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" fill="{theme.text if entry.highlight else theme.text_dim}" '
            f'font-size="{label_size:.0f}" font-weight="700" text-anchor="end" '
            f'transform="rotate(-{label_angle:.0f} {lx:.1f} {ly:.1f})">{_esc(entry.name)}</text>'
        )

    # legend
    if show_legend:
        ly = height - legend_h + 6
        flagged = sorted({e.source for e in entries if e.source != "measured"})
        parts.append(
            f'<line x1="{pad_l:.0f}" y1="{ly - 18:.1f}" x2="{width - pad_r:.0f}" y2="{ly - 18:.1f}" '
            f'stroke="{theme.grid}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l:.0f}" y="{ly:.1f}" fill="{theme.text_dim}" font-size="14" '
            f'font-weight="600">solid = measured by this harness</text>'
        )
        for j, source in enumerate(flagged):
            cite = next(e.citation for e in entries if e.source == source and e.citation)
            parts.append(
                f'<rect x="{pad_l:.0f}" y="{ly + 10 + 20 * j:.1f}" width="14" height="11" rx="2" '
                f'fill="{theme.bar_muted}"/>'
                f'<rect x="{pad_l:.0f}" y="{ly + 10 + 20 * j:.1f}" width="14" height="11" rx="2" '
                f'fill="url(#unmeasured)"/>'
                f'<text x="{pad_l + 22:.0f}" y="{ly + 19 + 20 * j:.1f}" fill="{theme.text_dim}" '
                f'font-size="14">hatched = {_esc(source)} ({_esc(cite)})</text>'
            )
        foot_y = ly + 26 + 20 * len(flagged)
        for i, line in enumerate(footnote_lines):
            parts.append(
                f'<text x="{pad_l:.0f}" y="{foot_y + i * (footnote_size + 9):.1f}" '
                f'fill="{theme.text_dim}" font-size="{footnote_size:.0f}">{_esc(line)}</text>'
            )

    parts.append("</svg>")
    svg = "\n".join(parts)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(svg)
    return svg


def render_png(benchmark: Benchmark, path: str, *, scale: float = 2.0, **kwargs: Any) -> Optional[str]:
    """Render to PNG via cairosvg if installed. Returns ``None`` when it is not.

    SVG is the primary format on purpose — it needs no dependency and GitHub
    renders it. PNG is here for slide decks.
    """
    svg = render_svg(benchmark, None, **kwargs)
    try:
        import cairosvg  # type: ignore
    except ImportError:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=path, scale=scale)
    return path


def render_grid(
    benchmarks: Sequence[Benchmark], path: str, *, columns: int = 1, **kwargs: Any
) -> str:
    """Stack several charts into one SVG (one per row per column)."""
    if not benchmarks:
        raise ValueError("no benchmarks to render")
    import re

    tiles = [render_svg(b, None, **kwargs) for b in benchmarks]
    sizes = []
    for svg in tiles:
        w = float(re.search(r'width="([\d.]+)"', svg).group(1))
        h = float(re.search(r'height="([\d.]+)"', svg).group(1))
        sizes.append((w, h))

    rows = (len(tiles) + columns - 1) // columns
    col_w = max(w for w, _ in sizes)
    row_h = [max(sizes[r * columns + c][1] for c in range(columns) if r * columns + c < len(tiles))
             for r in range(rows)]
    total_w = col_w * columns
    total_h = sum(row_h)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.0f}" height="{total_h:.0f}" '
        f'viewBox="0 0 {total_w:.0f} {total_h:.0f}">',
        f'<rect width="{total_w:.0f}" height="{total_h:.0f}" fill="{DARK.background}"/>',
    ]
    y = 0.0
    for r in range(rows):
        for c in range(columns):
            idx = r * columns + c
            if idx >= len(tiles):
                continue
            inner = tiles[idx].split("\n", 1)[1].rsplit("</svg>", 1)[0]
            parts.append(f'<g transform="translate({c * col_w:.0f},{y:.0f})">{inner}</g>')
        y += row_h[r]
    parts.append("</svg>")
    svg = "\n".join(parts)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg)
    return svg
