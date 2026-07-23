"""Tests for foldr.core.fold() — end-to-end orchestration."""

from __future__ import annotations

import numpy as np

from foldr.core import fold
from foldr.io import LightCurve


def _to_lc(synth, source_path="synthetic"):
    return LightCurve(
        time=synth.time,
        flux=synth.flux,
        flux_err=synth.flux_err,
        source_path=source_path,
        time_label="TIME",
        n_removed=0,
    )


def test_fold_auto_search_recovers_transit(transit_lc_factory):
    period = 3.21
    depth = 0.005

    synth = transit_lc_factory(
        period=period,
        t0=1.5,
        depth=depth,
        duration=0.1,
        noise=1e-4,
        span=27.0,
        cadence=0.5 / 24,
        seed=99,
    )
    lc = _to_lc(synth)

    result = fold(lc, period_min=0.5, period_max=13.5)

    assert result.engine in ("bls", "tls")
    assert abs(result.period - period) / period < 0.01
    assert abs(result.depth_ppm - depth * 1e6) / (depth * 1e6) < 0.30
    assert result.sde is not None and result.sde > 7


def test_fold_user_ephemeris_skips_search(transit_lc_factory):
    period = 3.21
    t0 = 1.5
    synth = transit_lc_factory(
        period=period,
        t0=t0,
        depth=0.005,
        duration=0.1,
        noise=1e-4,
        span=27.0,
        cadence=2 / 60 / 24,
        seed=3,
    )
    lc = _to_lc(synth)

    result = fold(lc, period=period, t0=t0)

    assert result.engine == "user"
    assert result.period == period
    assert result.t0 == t0
    assert result.sde is None


def test_fold_user_period_without_t0_estimates_epoch(transit_lc_factory):
    period = 3.21
    t0 = 1.5
    synth = transit_lc_factory(
        period=period,
        t0=t0,
        depth=0.005,
        duration=0.1,
        noise=1e-4,
        span=27.0,
        cadence=2 / 60 / 24,
        seed=4,
    )
    lc = _to_lc(synth)

    result = fold(lc, period=period)

    assert result.engine == "user"
    phase_offset = ((result.t0 - t0 + period / 2) % period) - period / 2
    assert abs(phase_offset) < 0.05


def test_fold_accepts_in_memory_lightcurve(transit_lc_factory):
    synth = transit_lc_factory(
        period=2.0,
        t0=0.3,
        depth=0.01,
        duration=0.05,
        noise=1e-4,
        span=10.0,
        cadence=1 / 24,
        seed=5,
    )
    lc = _to_lc(synth)
    result = fold(lc, period=2.0, t0=0.3)
    assert result.lc.time.size == lc.time.size


def test_fold_rejects_non_positive_period_directly(transit_lc_factory):
    # phase_fold() divides by period; period<=0 silently turns every phase
    # into NaN/inf instead of raising. This must be caught in fold() itself
    # (the public library entry point), not just cli.py, since fold() is
    # used as a library function, not only via the CLI.
    import pytest

    synth = transit_lc_factory(
        period=2.0, t0=0.3, depth=0.01, duration=0.05, noise=1e-4,
        span=10.0, cadence=1 / 24, seed=5,
    )
    lc = _to_lc(synth)

    for bad_period in (0, -3.0):
        with pytest.raises(ValueError, match="period must be positive"):
            fold(lc, period=bad_period, t0=0.3)
    with pytest.raises(ValueError, match="period_min must be positive"):
        fold(lc, period_min=-1.0)
    with pytest.raises(ValueError, match="period_max must be positive"):
        fold(lc, period=2.0, period_max=-1.0)


def test_fold_detrend_does_not_crash(transit_lc_factory):
    synth = transit_lc_factory(
        period=2.0,
        t0=0.3,
        depth=0.01,
        duration=0.05,
        noise=1e-4,
        span=10.0,
        cadence=1 / 24,
        seed=6,
    )
    lc = _to_lc(synth)
    result = fold(lc, period=2.0, t0=0.3, detrend_window_days=0.5)
    assert np.all(np.isfinite(result.folded_flux))


def test_fold_pure_noise_low_sde(noise_lc_factory):
    synth = noise_lc_factory(noise=3e-4, span=27.0, cadence=0.5 / 24, seed=0)
    lc = _to_lc(synth)
    result = fold(lc, period_min=0.5, period_max=13.5)
    assert result.sde is not None
    assert result.sde < 7


def test_fold_bin_count_matches_request(transit_lc_factory):
    synth = transit_lc_factory(
        period=2.0,
        t0=0.3,
        depth=0.01,
        duration=0.05,
        noise=1e-4,
        span=10.0,
        cadence=1 / 24,
        seed=9,
    )
    lc = _to_lc(synth)
    result = fold(lc, period=2.0, t0=0.3, bins=50)
    assert len(result.bin_centers) == 50
