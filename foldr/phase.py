"""Pure-numpy phase folding and binning. No I/O, no astropy, no matplotlib."""

from __future__ import annotations

import numpy as np


def phase_fold(
    time: np.ndarray, flux: np.ndarray, period: float, t0: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fold ``time``/``flux`` on ``period``/``t0``.

    Returns ``(phase, flux)`` where ``phase`` is in ``[-0.5, 0.5)`` with the
    transit centered at 0, and both arrays are sorted by phase.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5
    order = np.argsort(phase)
    return phase[order], flux[order]


def bin_folded(
    phase: np.ndarray, flux: np.ndarray, bins: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin a phase-folded curve into equal-width phase bins.

    Returns ``(bin_centers, mean_flux, sem)``. Bins with no points get NaN
    in both ``mean_flux`` and ``sem``.
    """
    phase = np.asarray(phase, dtype=float)
    flux = np.asarray(flux, dtype=float)

    edges = np.linspace(-0.5, 0.5, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bin_idx = np.clip(np.digitize(phase, edges) - 1, 0, bins - 1)

    mean_flux = np.full(bins, np.nan)
    sem = np.full(bins, np.nan)

    for i in range(bins):
        mask = bin_idx == i
        n = np.count_nonzero(mask & np.isfinite(flux))
        if n == 0:
            continue
        vals = flux[mask]
        vals = vals[np.isfinite(vals)]
        mean_flux[i] = np.mean(vals)
        sem[i] = np.std(vals, ddof=1) / np.sqrt(n) if n > 1 else 0.0

    return centers, mean_flux, sem
