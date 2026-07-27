"""Chart-harness tests. Torch-free: rendering is pure Python + SVG."""

from __future__ import annotations

import json
import os
import re

import pytest

from seedance.bench.leaderboard import (
    Benchmark,
    Entry,
    load_benchmark,
    render_grid,
    render_svg,
    save_benchmark,
)
from seedance.bench.logos import LOGOS, logo_tile_svg
from seedance.bench.make_charts import RESULTS_DIR, render_all


def _bench() -> Benchmark:
    return Benchmark(
        title="Coding",
        subtitle="a subtitle",
        entries=[
            Entry("Ours", 77, logo="seedance", source="measured", highlight=True),
            Entry("Theirs", 59, logo="sora", source="vendor-reported", citation="vendor docs"),
        ],
        footnote="a footnote",
    )


# --------------------------------------------------------------------- schema

def test_non_measured_entries_must_carry_a_citation():
    Entry("ok", 1.0, source="measured")  # no citation needed
    with pytest.raises(ValueError, match="citation"):
        Entry("claim", 99.0, source="vendor-reported")
    with pytest.raises(ValueError, match="source"):
        Entry("bad", 1.0, source="vibes")


def test_leader_respects_direction():
    bench = _bench()
    assert bench.leader().name == "Ours"
    bench.higher_is_better = False
    assert bench.leader().name == "Theirs"


def test_results_roundtrip(tmp_path):
    path = str(tmp_path / "r.json")
    save_benchmark([_bench()], path)
    restored = load_benchmark(path)[0]
    assert restored.title == "Coding"
    assert [e.value for e in restored.entries] == [77, 59]
    assert restored.entries[1].source == "vendor-reported"


# --------------------------------------------------------------------- render

def test_svg_contains_values_labels_and_tiles(tmp_path):
    out = str(tmp_path / "chart.svg")
    svg = render_svg(_bench(), out)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert ">77<" in svg and ">59<" in svg
    assert ">Ours<" in svg and ">Theirs<" in svg
    assert "rotate(-52" in svg, "labels should be diagonal like the reference"
    assert os.path.exists(out)


def test_unmeasured_bars_are_hatched_and_explained():
    svg = render_svg(_bench())
    assert 'fill="url(#unmeasured)"' in svg
    assert "vendor-reported" in svg
    assert "measured by this harness" in svg


def test_all_measured_chart_has_no_hatching():
    bench = Benchmark(
        title="Ablation",
        entries=[Entry("a", 1.0), Entry("b", 2.0)],
    )
    svg = render_svg(bench)
    assert 'fill="url(#unmeasured)"' not in svg


def test_canvas_is_wide_enough_for_the_longest_diagonal_label():
    """The first bar's label spills left; the canvas must contain it."""
    long_name = "A Very Long Model Name That Spills To The Left"
    svg = render_svg(
        Benchmark(title="t", entries=[Entry(long_name, 50), Entry("b", 40)])
    )
    width = float(re.search(r'width="([\d.]+)"', svg).group(1))
    # anchor x of the first label, minus its rotated horizontal extent
    anchor = float(re.search(r'transform="rotate\(-52 ([\d.]+)', svg).group(1))
    spill = len(long_name) * 21 * 0.58 * 0.6157  # cos(52 deg)
    assert anchor - spill > 0, "longest label runs off the left edge"
    assert width > anchor


def test_empty_benchmark_is_rejected():
    with pytest.raises(ValueError):
        render_svg(Benchmark(title="empty"))


def test_grid_stacks_charts(tmp_path):
    out = str(tmp_path / "grid.svg")
    svg = render_grid([_bench(), _bench()], out)
    assert svg.count("<g transform=") >= 2
    assert os.path.exists(out)


def test_sorting_reorders_bars():
    bench = Benchmark(title="t", entries=[Entry("low", 10), Entry("high", 90)])
    ordered = [e.name for e in bench.sorted_entries(by_value=True)]
    assert ordered == ["high", "low"]


# ---------------------------------------------------------------------- logos

def test_monogram_tile_is_drawn_for_a_known_key():
    svg = logo_tile_svg("wan", 0, 0, 44)
    assert "<rect" in svg and LOGOS["wan"].label in svg


def test_unknown_logo_key_falls_back_without_raising():
    assert "<rect" in logo_tile_svg("not-a-vendor", 0, 0, 44)


def test_user_supplied_logo_is_embedded(tmp_path, monkeypatch):
    from seedance.bench import logos

    directory = tmp_path / "logos"
    directory.mkdir()
    (directory / "sora.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<circle cx="5" cy="5" r="4" fill="#fff"/></svg>'
    )
    monkeypatch.setattr(logos, "logo_dir", lambda: str(directory))
    svg = logos.logo_tile_svg("sora", 0, 0, 44)
    assert "<circle" in svg, "the real logo should replace the monogram"
    assert "scale(" in svg, "and be scaled into the tile"


# ------------------------------------------------------------ shipped results

def test_every_shipped_results_file_loads_and_renders():
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
    assert files, "no shipped results files"
    for name in files:
        for benchmark in load_benchmark(os.path.join(RESULTS_DIR, name)):
            svg = render_svg(benchmark)
            assert svg.startswith("<svg")


def test_shipped_claims_about_other_vendors_are_never_marked_measured():
    """Only our own numbers may claim to be measured here."""
    for name in os.listdir(RESULTS_DIR):
        if not name.endswith(".json"):
            continue
        for benchmark in load_benchmark(os.path.join(RESULTS_DIR, name)):
            for entry in benchmark.entries:
                if entry.source == "measured":
                    assert "seedance-staged" in entry.name.lower(), (
                        f"{entry.name!r} is marked measured but is not ours"
                    )
                else:
                    assert entry.citation


def test_render_all_writes_into_assets(tmp_path):
    written = render_all(RESULTS_DIR, str(tmp_path))
    assert written
    assert all(os.path.exists(p) for p in written)
    assert any(p.endswith("capability_coverage.svg") for p in written)
