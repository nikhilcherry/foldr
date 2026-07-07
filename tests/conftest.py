"""Synthetic light curve fixtures. No binary fixtures are committed — every
light curve used by the test suite is generated on the fly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest


@dataclass
class SyntheticLC:
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray | None = None


def make_transit_lc(
    period: float,
    t0: float,
    depth: float,
    duration: float,
    noise: float,
    span: float,
    cadence: float,
    seed: int = 0,
) -> SyntheticLC:
    """Box-shaped injected transit on white noise.

    ``period``, ``t0``, ``duration``, ``span``, ``cadence`` are in days.
    ``depth`` is a fraction of baseline flux (e.g. 0.005 == 5000 ppm).
    ``noise`` is the white-noise sigma, also a fraction of baseline flux.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, span, cadence)
    flux = np.ones_like(time) + rng.normal(0.0, noise, time.size)

    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    in_transit = np.abs(phase) < (duration / 2.0)
    flux[in_transit] -= depth

    flux_err = np.full_like(time, noise) if noise > 0 else None
    return SyntheticLC(time=time, flux=flux, flux_err=flux_err)


def make_eclipsing_binary_lc(
    period: float,
    t0: float,
    primary_depth: float,
    secondary_depth: float,
    duration: float,
    noise: float,
    span: float,
    cadence: float,
    seed: int = 1,
) -> SyntheticLC:
    """Two different-depth boxes at phase 0 (primary) and phase 0.5
    (secondary eclipse) — for odd/even sanity checks.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, span, cadence)
    flux = np.ones_like(time) + rng.normal(0.0, noise, time.size)

    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    primary = np.abs(phase) < (duration / 2.0)
    flux[primary] -= primary_depth

    secondary_phase = ((phase - period / 2.0 + 0.5 * period) % period) - 0.5 * period
    secondary = np.abs(secondary_phase) < (duration / 2.0)
    flux[secondary] -= secondary_depth

    flux_err = np.full_like(time, noise) if noise > 0 else None
    return SyntheticLC(time=time, flux=flux, flux_err=flux_err)


def make_noise_lc(noise: float, span: float, cadence: float, seed: int = 2) -> SyntheticLC:
    """Pure-noise curve, no injected signal — for the exit-code-1 path."""
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, span, cadence)
    flux = np.ones_like(time) + rng.normal(0.0, noise, time.size)
    flux_err = np.full_like(time, noise) if noise > 0 else None
    return SyntheticLC(time=time, flux=flux, flux_err=flux_err)


def save_npz(lc: SyntheticLC, path: Path) -> Path:
    path = Path(path)
    if lc.flux_err is not None:
        np.savez(path, time=lc.time, flux=lc.flux, flux_err=lc.flux_err)
    else:
        np.savez(path, time=lc.time, flux=lc.flux)
    return path


def save_csv(lc: SyntheticLC, path: Path) -> Path:
    path = Path(path)
    if lc.flux_err is not None:
        header = "time,flux,flux_err"
        arr = np.column_stack([lc.time, lc.flux, lc.flux_err])
    else:
        header = "time,flux"
        arr = np.column_stack([lc.time, lc.flux])
    np.savetxt(path, arr, delimiter=",", header=header, comments="")
    return path


def save_fits(lc: SyntheticLC, path: Path) -> Path:
    from astropy.io import fits

    path = Path(path)
    cols = [
        fits.Column(name="TIME", format="D", array=lc.time),
        fits.Column(name="PDCSAP_FLUX", format="D", array=lc.flux),
    ]
    if lc.flux_err is not None:
        cols.append(fits.Column(name="PDCSAP_FLUX_ERR", format="D", array=lc.flux_err))
    hdu = fits.BinTableHDU.from_columns(cols, name="LIGHTCURVE")
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdul.writeto(path, overwrite=True)
    return path


@pytest.fixture
def transit_lc_factory():
    return make_transit_lc


@pytest.fixture
def eclipsing_binary_factory():
    return make_eclipsing_binary_lc


@pytest.fixture
def noise_lc_factory():
    return make_noise_lc
