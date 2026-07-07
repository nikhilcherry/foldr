"""Tests for foldr.phase — pure numpy, no I/O."""

from __future__ import annotations

import numpy as np

from foldr.phase import bin_folded, phase_fold


def test_phase_fold_transit_at_zero():
    period = 3.0
    t0 = 1.25
    time = t0 + period * np.arange(-5, 6)
    flux = np.ones_like(time)

    phase, folded_flux = phase_fold(time, flux, period, t0)

    assert np.allclose(phase, 0.0, atol=1e-9)
    assert len(phase) == len(time)
    assert len(folded_flux) == len(flux)


def test_phase_fold_preserves_length_and_range():
    rng = np.random.default_rng(0)
    time = np.sort(rng.uniform(0, 100, 500))
    flux = rng.normal(1.0, 0.001, time.size)

    phase, folded_flux = phase_fold(time, flux, period=2.7, t0=0.4)

    assert len(phase) == len(time)
    assert len(folded_flux) == len(flux)
    assert np.all(phase >= -0.5)
    assert np.all(phase < 0.5)
    assert np.all(np.diff(phase) >= 0)


def test_phase_fold_flux_travels_with_its_phase():
    time = np.array([0.1, 5.0, 2.0, 3.9])
    flux = np.array([10.0, 20.0, 30.0, 40.0])

    phase, folded_flux = phase_fold(time, flux, period=1.0, t0=0.0)

    expected_phase = ((time / 1.0) + 0.5) % 1.0 - 0.5
    order = np.argsort(expected_phase)
    assert np.array_equal(folded_flux, flux[order])
    assert np.all(np.diff(phase) >= 0)


def test_bin_folded_bin_count_and_nan_for_empty():
    phase = np.array([-0.49, -0.48, 0.0, 0.01, 0.49])
    flux = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    centers, mean_flux, sem = bin_folded(phase, flux, bins=100)

    assert len(centers) == 100
    assert len(mean_flux) == 100
    assert len(sem) == 100
    assert np.isnan(mean_flux).sum() > 90


def test_bin_folded_sem_is_sane():
    rng = np.random.default_rng(1)
    phase = rng.uniform(-0.5, 0.5, 2000)
    flux = rng.normal(1.0, 0.01, 2000)

    centers, mean_flux, sem = bin_folded(phase, flux, bins=20)

    finite = np.isfinite(mean_flux)
    assert finite.sum() > 0
    assert np.all(sem[finite] >= 0)
    assert np.all(np.isfinite(sem[finite]))


def test_bin_folded_single_point_bin_has_zero_sem():
    phase = np.array([-0.5 + 1e-6])
    flux = np.array([1.0])
    centers, mean_flux, sem = bin_folded(phase, flux, bins=4)
    finite = np.isfinite(mean_flux)
    assert finite.sum() == 1
    assert sem[finite][0] == 0.0
