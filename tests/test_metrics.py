"""Tests for foldr.metrics — pure numpy, NaN-aware."""

from __future__ import annotations

import warnings

import numpy as np

from foldr.metrics import estimate_transit_metrics
from foldr.phase import bin_folded, phase_fold


def test_estimate_metrics_recovers_known_transit(transit_lc_factory):
    period = 3.21
    duration = 0.1
    depth = 0.005  # 5000 ppm

    lc = transit_lc_factory(
        period=period,
        t0=1.5,
        depth=depth,
        duration=duration,
        noise=1e-4,
        span=27.0,
        cadence=2 / 60 / 24,
    )

    phase, folded_flux = phase_fold(lc.time, lc.flux, period, 1.5)
    bin_centers, bin_flux, bin_err = bin_folded(phase, folded_flux, bins=200)

    result = estimate_transit_metrics(bin_centers, bin_flux, bin_err, period)

    assert result["depth_ppm"] is not None
    assert abs(result["depth_ppm"] - depth * 1e6) / (depth * 1e6) < 0.30
    assert result["duration_hours"] is not None
    assert result["snr"] is not None
    assert result["snr"] > 5


def test_estimate_metrics_handles_all_nan_bins():
    bin_centers = np.linspace(-0.5, 0.5, 10)
    bin_flux = np.full(10, np.nan)
    bin_err = np.full(10, np.nan)

    result = estimate_transit_metrics(bin_centers, bin_flux, bin_err, period_days=1.0)

    assert result == {"depth_ppm": None, "duration_hours": None, "snr": None}


def test_estimate_metrics_pure_noise_no_runtime_warning(noise_lc_factory):
    lc = noise_lc_factory(noise=3e-4, span=27.0, cadence=2 / 60 / 24)
    phase, folded_flux = phase_fold(lc.time, lc.flux, period=3.21, t0=1.5)
    bin_centers, bin_flux, bin_err = bin_folded(phase, folded_flux, bins=200)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        estimate_transit_metrics(bin_centers, bin_flux, bin_err, period_days=3.21)


def test_estimate_metrics_too_few_finite_bins_returns_none():
    bin_centers = np.linspace(-0.5, 0.5, 10)
    bin_flux = np.array([1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    bin_err = np.full(10, np.nan)

    result = estimate_transit_metrics(bin_centers, bin_flux, bin_err, period_days=1.0)
    assert result == {"depth_ppm": None, "duration_hours": None, "snr": None}
