"""Render every shipped results file into ``seedance/assets/``.

    python -m seedance.bench.make_charts

Regenerate after editing anything in ``seedance/bench/results/`` or after
dropping official logo SVGs into ``seedance/bench/logos/``.
"""

from __future__ import annotations

import os
from typing import List

from .leaderboard import Benchmark, load_benchmark, render_png, render_svg

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
)


def render_all(results_dir: str = RESULTS_DIR, assets_dir: str = ASSETS_DIR) -> List[str]:
    os.makedirs(assets_dir, exist_ok=True)
    written: List[str] = []
    for entry in sorted(os.listdir(results_dir)):
        if not entry.endswith(".json"):
            continue
        stem = entry[:-5]
        benchmarks = load_benchmark(os.path.join(results_dir, entry))
        for i, benchmark in enumerate(benchmarks):
            suffix = "" if len(benchmarks) == 1 else f"-{i + 1}"
            out = os.path.join(assets_dir, f"{stem}{suffix}.svg")
            render_svg(benchmark, out)
            written.append(out)
            png = render_png(benchmark, out[:-4] + ".png")
            if png:
                written.append(png)
    return written


def main() -> int:
    written = render_all()
    for path in written:
        print(path)
    if not any(p.endswith(".png") for p in written):
        print("(PNG export skipped: pip install cairosvg to also emit PNGs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
