"""Tests for foldr.search — BLS recovery (the test that matters), TLS optional."""

from __future__ import annotations

import pytest

import numpy as np

from foldr.io import LightCurve
from foldr.search import _sde_from_power, run_bls, run_tls


def _to_lc(synth):
    return LightCurve(
        time=synth.time,
        flux=synth.flux,
        flux_err=synth.flux_err,
        source_path="synthetic",
        time_label="TIME",
        n_removed=0,
    )


def test_run_bls_recovers_injected_transit(transit_lc_factory):
    period = 3.21
    depth = 0.005  # 5000 ppm

    synth = transit_lc_factory(
        period=period,
        t0=1.5,
        depth=depth,
        duration=0.1,
        noise=1e-4,
        span=27.0,
        cadence=0.5 / 24,
        seed=42,
    )
    lc = _to_lc(synth)

    result = run_bls(lc, period_min=0.5, period_max=13.5)

    assert abs(result["period"] - period) / period < 0.01
    assert abs(result["depth"] - depth) / depth < 0.30
    assert result["sde"] > 7


def test_run_tls_recovers_injected_transit(transit_lc_factory):
    pytest.importorskip("transitleastsquares")

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
        seed=42,
    )
    lc = _to_lc(synth)

    result = run_tls(lc, period_min=0.5, period_max=13.5)

    assert abs(result["period"] - period) / period < 0.01
    assert abs(result["depth"] - depth) / depth < 0.30


def test_sde_from_power_excludes_peak_window_from_baseline():
    rng = np.random.default_rng(0)
    periods = np.linspace(0.5, 13.5, 2000)
    power = rng.normal(0.0, 1.0, size=periods.size)
    peak_idx = 1000
    power[peak_idx] = 50.0  # a single very strong, narrow peak

    sde_excluding_peak = _sde_from_power(power, periods, peak_idx)
    naive_sde = (power[peak_idx] - np.mean(power)) / np.std(power)

    # Folding a single huge peak into the baseline std inflates it and
    # deflates the naive SDE relative to a baseline that excludes it.
    assert sde_excluding_peak > naive_sde


def test_sde_from_power_falls_back_to_full_power_with_few_trials():
    periods = np.linspace(1.0, 2.0, 5)
    power = np.array([1.0, 1.0, 1.0, 1.0, 5.0])
    sde = _sde_from_power(power, periods, best_idx=4)
    naive_sde = (power[4] - np.mean(power)) / np.std(power)
    assert sde == pytest.approx(naive_sde)


def test_run_tls_import_error_without_extra():
    try:
        import transitleastsquares  # noqa: F401

        pytest.skip("transitleastsquares is installed; nothing to test here")
    except ImportError:
        pass

    synth_lc = LightCurve(
        time=[0.0, 1.0],
        flux=[1.0, 1.0],
        flux_err=None,
        source_path="synthetic",
        time_label="TIME",
        n_removed=0,
    )

    with pytest.raises(ImportError, match=r"pip install \"foldr\[tls\]"):
        run_tls(synth_lc, period_min=0.5, period_max=1.0)
