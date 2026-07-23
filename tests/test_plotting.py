"""Tests for foldr.plotting.make_figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from PIL import Image

from foldr.core import fold
from foldr.io import LightCurve
from foldr.plotting import make_figure


def _to_lc(synth, source_path="synthetic"):
    return LightCurve(
        time=synth.time,
        flux=synth.flux,
        flux_err=synth.flux_err,
        source_path=source_path,
        time_label="TIME",
        n_removed=0,
    )


def _png_height(path) -> int:
    with Image.open(path) as img:
        return img.size[1]


def test_periodogram_panel_only_added_when_search_ran(tmp_path, transit_lc_factory):
    synth = transit_lc_factory(
        period=3.21, t0=1.5, depth=0.005, duration=0.1, noise=1e-4,
        span=27.0, cadence=0.5 / 24, seed=99,
    )
    lc = _to_lc(synth)

    search_result = fold(lc, period_min=0.5, period_max=13.5)
    assert search_result.search_meta.get("power_spectrum")
    search_path = make_figure(search_result, tmp_path / "search.png")
    assert search_path.exists()

    user_result = fold(lc, period=3.21, t0=1.5)
    assert user_result.search_meta == {}
    user_path = make_figure(user_result, tmp_path / "user.png")
    assert user_path.exists()

    # The search result gets a 3rd (periodogram) panel, so its figure
    # (figsize (9,10)) is taller than the 2-panel user-ephemeris one
    # (figsize (9,7)).
    assert _png_height(search_path) > _png_height(user_path)


def test_make_figure_creates_missing_parent_directories(tmp_path, transit_lc_factory):
    # --plot-path into a directory that doesn't exist yet (e.g. a fresh
    # per-run output folder) should work, not raise a raw FileNotFoundError
    # from matplotlib's savefig.
    synth = transit_lc_factory(
        period=3.21, t0=1.5, depth=0.005, duration=0.1, noise=1e-4,
        span=10.0, cadence=0.5 / 24, seed=1,
    )
    lc = _to_lc(synth)
    result = fold(lc, period=3.21, t0=1.5)

    out_path = tmp_path / "nested" / "does" / "not" / "exist" / "plot.png"
    saved_path = make_figure(result, out_path)

    assert saved_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
