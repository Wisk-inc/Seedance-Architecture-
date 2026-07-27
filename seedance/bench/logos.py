"""Logo tiles for the benchmark chart.

Official vendor logos are trademarks and are not redistributed with this
package. Two options, in order of preference:

1. **Drop in the real ones.** Put ``<key>.svg`` in ``seedance/bench/logos/``
   (e.g. ``sora.svg``, ``veo.svg``) and it is embedded automatically, scaled
   into the tile. That is what the chart is designed for.
2. **Use the built-in monograms.** Without a file, a clean geometric letter
   tile is drawn instead — same size, same rounded-square language, no
   trademark issues.

Keys are stable identifiers (``seedance``, ``wan``, ``sora``, ...) so a results
file written today keeps working when you add real logo assets tomorrow.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Optional

__all__ = ["LogoStyle", "LOGOS", "logo_tile_svg", "logo_dir", "available_logo_files"]


@dataclass(frozen=True)
class LogoStyle:
    """A monogram tile: short label, tile fill, glyph colour."""

    label: str
    fill: str = "#1A1C21"
    fg: str = "#FFFFFF"
    radius: int = 10


# Muted tiles by default so the bars carry the colour, exactly like the
# reference chart: the logo row reads as a quiet index, not a second palette.
LOGOS: Dict[str, LogoStyle] = {
    "seedance": LogoStyle("SD", fill="#E9E9EC", fg="#101114"),
    "seedance-suite": LogoStyle("S", fill="#E9E9EC", fg="#101114"),
    "wan": LogoStyle("W", fill="#16202B", fg="#7FC4FF"),
    "sora": LogoStyle("O", fill="#151515", fg="#FFFFFF"),
    "veo": LogoStyle("V", fill="#101725", fg="#8AB4F8"),
    "kling": LogoStyle("K", fill="#1B1420", fg="#D7A6FF"),
    "runway": LogoStyle("R", fill="#151515", fg="#FFFFFF"),
    "pika": LogoStyle("P", fill="#1C1418", fg="#FF8FB1"),
    "luma": LogoStyle("L", fill="#141A1A", fg="#7FE3D0"),
    "hunyuan": LogoStyle("H", fill="#111A22", fg="#5FB3E4"),
    "cogvideo": LogoStyle("C", fill="#1A1620", fg="#B79BFF"),
    "ltx": LogoStyle("LT", fill="#141414", fg="#E0E0E0"),
    "mochi": LogoStyle("M", fill="#1B1712", fg="#F2C879"),
    "svd": LogoStyle("SV", fill="#12181C", fg="#9FD8E8"),
    "opensora": LogoStyle("OS", fill="#131B17", fg="#8BE0A8"),
    "videopoet": LogoStyle("VP", fill="#101725", fg="#8AB4F8"),
    "generic": LogoStyle("?", fill="#17181C", fg="#8A8F98"),
}


def logo_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "logos")


def available_logo_files() -> Dict[str, str]:
    """Map of ``key -> path`` for user-supplied SVG logos."""
    directory = logo_dir()
    if not os.path.isdir(directory):
        return {}
    out = {}
    for entry in sorted(os.listdir(directory)):
        if entry.lower().endswith(".svg"):
            out[entry[:-4].lower()] = os.path.join(directory, entry)
    return out


_SVG_OPEN = re.compile(r"<svg\b[^>]*>", re.IGNORECASE)
_SVG_CLOSE = re.compile(r"</svg>\s*$", re.IGNORECASE)
_XML_DECL = re.compile(r"<\?xml[^>]*\?>", re.IGNORECASE)
_DOCTYPE = re.compile(r"<!DOCTYPE[^>]*>", re.IGNORECASE)
_VIEWBOX = re.compile(r'viewBox\s*=\s*"([^"]+)"', re.IGNORECASE)
_WIDTH = re.compile(r'\bwidth\s*=\s*"([\d.]+)', re.IGNORECASE)
_HEIGHT = re.compile(r'\bheight\s*=\s*"([\d.]+)', re.IGNORECASE)


def _embed_file(path: str, x: float, y: float, size: float, radius: int) -> Optional[str]:
    """Inline a user-supplied SVG, scaled to fit the tile."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()
    except OSError:
        return None

    head = _SVG_OPEN.search(raw)
    if not head:
        return None
    viewbox = _VIEWBOX.search(head.group(0))
    if viewbox:
        parts = [float(v) for v in viewbox.group(1).replace(",", " ").split()]
        vw, vh = (parts[2], parts[3]) if len(parts) == 4 else (size, size)
    else:
        w = _WIDTH.search(head.group(0))
        h = _HEIGHT.search(head.group(0))
        vw = float(w.group(1)) if w else size
        vh = float(h.group(1)) if h else size
    vw, vh = max(vw, 1e-6), max(vh, 1e-6)

    inner = raw[head.end():]
    inner = _SVG_CLOSE.sub("", inner)
    inner = _XML_DECL.sub("", inner)
    inner = _DOCTYPE.sub("", inner)

    pad = size * 0.16
    scale = min((size - 2 * pad) / vw, (size - 2 * pad) / vh)
    dx = x + (size - vw * scale) / 2
    dy = y + (size - vh * scale) / 2
    return (
        f'<g><rect x="{x:.1f}" y="{y:.1f}" width="{size:.1f}" height="{size:.1f}" '
        f'rx="{radius}" fill="#1A1C21"/>'
        f'<g transform="translate({dx:.2f},{dy:.2f}) scale({scale:.4f})">{inner}</g></g>'
    )


def logo_tile_svg(key: str, x: float, y: float, size: float = 44.0) -> str:
    """Return an SVG fragment for one logo tile at ``(x, y)``."""
    key = (key or "generic").lower()
    files = available_logo_files()
    if key in files:
        embedded = _embed_file(files[key], x, y, size, LOGOS.get(key, LOGOS["generic"]).radius)
        if embedded:
            return embedded

    style = LOGOS.get(key, LOGOS["generic"])
    font_size = size * (0.44 if len(style.label) <= 1 else 0.34)
    return (
        f'<g><rect x="{x:.1f}" y="{y:.1f}" width="{size:.1f}" height="{size:.1f}" '
        f'rx="{style.radius}" fill="{style.fill}"/>'
        f'<text x="{x + size / 2:.1f}" y="{y + size / 2:.1f}" fill="{style.fg}" '
        f'font-family="Inter, Helvetica Neue, Arial, sans-serif" font-size="{font_size:.1f}" '
        f'font-weight="700" text-anchor="middle" dominant-baseline="central" '
        f'letter-spacing="0.5">{style.label}</text></g>'
    )
