"""Tests for foldr.search — BLS recovery (the test that matters), TLS optional."""

from __future__ import annotations

import pytest

from foldr.io import LightCurve
from foldr.search import run_bls, run_tls


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
