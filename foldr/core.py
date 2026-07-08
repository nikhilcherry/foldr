"""fold() orchestrator and FoldResult. The public entry point of foldr."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .io import FoldrReadError, LightCurve, load_lightcurve
from .metrics import estimate_transit_metrics
from .phase import bin_folded, phase_fold
from .search import run_bls, run_tls

__all__ = ["LightCurve", "FoldResult", "FoldrReadError", "fold"]


@dataclass
class FoldResult:
    lc: LightCurve
    period: float
    t0: float
    duration_hours: float | None
    depth_ppm: float | None
    snr: float | None
    sde: float | None
    engine: str
    phase: np.ndarray
    folded_flux: np.ndarray
    bin_centers: np.ndarray
    bin_flux: np.ndarray
    bin_err: np.ndarray
    search_meta: dict = field(default_factory=dict)


def fold(
    path_or_lc: str | Path | LightCurve,
    *,
    period: float | None = None,
    t0: float | None = None,
    engine: str = "auto",
    period_min: float = 0.5,
    period_max: float | None = None,
    bins: int = 200,
    detrend_window_days: float | None = None,
) -> FoldResult:
    """If period is given, engine becomes 'user' (t0 defaults to the
    minimum-flux phase if not supplied). engine='auto' uses TLS when
    importable, else BLS.
    """
    if isinstance(path_or_lc, LightCurve):
        lc = path_or_lc
    else:
        lc = load_lightcurve(path_or_lc)

    if detrend_window_days is not None:
        detrended_flux = _running_median_detrend(lc.time, lc.flux, detrend_window_days)
        working_lc = LightCurve(
            time=lc.time,
            flux=detrended_flux,
            flux_err=lc.flux_err,
            source_path=lc.source_path,
            time_label=lc.time_label,
            n_removed=lc.n_removed,
        )
    else:
        working_lc = lc

    time = working_lc.time
    flux = working_lc.flux

    search_meta: dict = {}
    sde = None
    search_result = None

    if period is not None:
        used_engine = "user"
        used_period = float(period)
        used_t0 = (
            float(t0) if t0 is not None else _estimate_t0(time, flux, used_period, bins)
        )
    else:
        selected = _select_search_engine(engine)
        span = float(time.max() - time.min())
        eff_period_max = period_max if period_max is not None else span / 2

        if selected == "tls":
            search_result = run_tls(working_lc, period_min, eff_period_max)
            used_engine = "tls"
        else:
            search_result = run_bls(working_lc, period_min, eff_period_max)
            used_engine = "bls"

        used_period = search_result["period"]
        used_t0 = search_result["t0"]
        sde = search_result["sde"]
        search_meta = search_result.get("search_meta", {}) or {}

    phase, folded_flux = phase_fold(time, flux, used_period, used_t0)
    bin_centers, bin_flux, bin_err = bin_folded(phase, folded_flux, bins=bins)

    estimated = estimate_transit_metrics(bin_centers, bin_flux, bin_err, used_period)

    if used_engine == "user":
        depth_ppm = estimated["depth_ppm"]
        duration_hours = estimated["duration_hours"]
    else:
        depth_ppm = search_result["depth"] * 1e6
        duration_hours = search_result["duration_hours"]

    snr = estimated["snr"]

    return FoldResult(
        lc=working_lc,
        period=used_period,
        t0=used_t0,
        duration_hours=duration_hours,
        depth_ppm=depth_ppm,
        snr=snr,
        sde=sde,
        engine=used_engine,
        phase=phase,
        folded_flux=folded_flux,
        bin_centers=bin_centers,
        bin_flux=bin_flux,
        bin_err=bin_err,
        search_meta=search_meta,
    )


def _select_search_engine(engine: str) -> str:
    if engine in ("bls", "tls"):
        return engine
    if engine != "auto":
        raise ValueError(f"Unknown engine '{engine}'. Expected auto, bls, or tls.")
    try:
        importlib.import_module("transitleastsquares")
        return "tls"
    except Exception:
        # Covers both "not installed" and "installed but broken" (e.g. a
        # transitive dependency incompatibility) — auto mode should degrade
        # to BLS rather than crash either way.
        return "bls"


def _estimate_t0(
    time: np.ndarray, flux: np.ndarray, period: float, bins: int
) -> float:
    t0_ref = float(np.min(time))
    phase, sorted_flux = phase_fold(time, flux, period, t0_ref)
    bin_centers, bin_flux, _ = bin_folded(phase, sorted_flux, bins=bins)

    finite = np.isfinite(bin_flux)
    if not finite.any():
        return t0_ref

    masked = np.where(finite, bin_flux, np.inf)
    best_idx = int(np.argmin(masked))
    best_phase = float(bin_centers[best_idx])
    return t0_ref + best_phase * period


def _running_median_detrend(
    time: np.ndarray, flux: np.ndarray, window_days: float
) -> np.ndarray:
    """Running Tukey's-biweight-location detrend over a sliding time window.

    Uses the biweight location (astropy.stats.biweight_location) rather than
    a plain median. Hippke & Heller (2019, the wotan paper — see AJ 158, 143)
    show biweight recovers ~99%/94% of shallow Kepler/K2 transits vs. worse
    rates for median or Savitzky-Golay filters, because a plain median still
    lets a handful of in-window in-transit points pull the local trend down,
    partially "eating" the transit; biweight downweights outliers from the
    window's own robust center instead of treating every point equally. Falls
    back to the median automatically wherever a window's MAD is zero (see
    biweight_location docs). As a rule of thumb, prefer a window >= 3x the
    expected transit duration so a full transit can't dominate one window.
    """
    from astropy.stats import biweight_location

    order = np.argsort(time)
    t_sorted = time[order]
    f_sorted = flux[order]

    n = t_sorted.size
    half = window_days / 2.0
    trend = np.empty(n)

    left = 0
    right = 0
    for i in range(n):
        while left < n and t_sorted[left] < t_sorted[i] - half:
            left += 1
        while right < n and t_sorted[right] <= t_sorted[i] + half:
            right += 1
        window = f_sorted[left:right]
        trend[i] = biweight_location(window) if window.size > 1 else window[0]

    trend = np.where(trend != 0, trend, 1.0)
    detrended_sorted = f_sorted / trend

    detrended = np.empty(n)
    detrended[order] = detrended_sorted
    return detrended
