"""Pure-numpy transit metric estimation from a folded/binned light curve.

No I/O, no astropy, no matplotlib. NaN-aware throughout; never raises
RuntimeWarning on empty or all-NaN input.
"""

from __future__ import annotations

import numpy as np


def estimate_transit_metrics(
    bin_centers: np.ndarray,
    bin_flux: np.ndarray,
    bin_err: np.ndarray,
    period_days: float,
) -> dict:
    """Estimate depth (ppm), duration (hours) and SNR from a binned, folded
    curve whose transit is centered at phase 0 (as produced by
    :func:`foldr.phase.phase_fold`).

    Returns a dict with keys ``depth_ppm``, ``duration_hours``, ``snr`` —
    any of which may be ``None`` if not estimable (e.g. too few finite bins).
    """
    bin_centers = np.asarray(bin_centers, dtype=float)
    bin_flux = np.asarray(bin_flux, dtype=float)

    finite = np.isfinite(bin_flux)
    n = bin_centers.size
    if np.count_nonzero(finite) < 3:
        return {"depth_ppm": None, "duration_hours": None, "snr": None}

    baseline = np.median(bin_flux[finite])
    mad = np.median(np.abs(bin_flux[finite] - baseline))
    robust_sigma = 1.4826 * mad if mad > 0 else np.std(bin_flux[finite], ddof=0)

    threshold = baseline - 3.0 * robust_sigma

    center_idx = int(np.argmin(np.abs(bin_centers)))

    in_transit = np.zeros(n, dtype=bool)
    if finite[center_idx] and bin_flux[center_idx] < baseline:
        in_transit[center_idx] = True
        i = center_idx - 1
        while i >= 0 and finite[i] and bin_flux[i] < threshold:
            in_transit[i] = True
            i -= 1
        i = center_idx + 1
        while i < n and finite[i] and bin_flux[i] < threshold:
            in_transit[i] = True
            i += 1

    if not in_transit.any():
        in_transit[center_idx] = bool(finite[center_idx])

    out_mask = finite & ~in_transit
    in_mask = finite & in_transit

    if not out_mask.any() or not in_mask.any():
        return {"depth_ppm": None, "duration_hours": None, "snr": None}

    out_level = float(np.mean(bin_flux[out_mask]))
    in_level = float(np.mean(bin_flux[in_mask]))
    depth = out_level - in_level
    depth_ppm = float(depth * 1e6)

    duration_fraction = np.count_nonzero(in_transit) / n
    duration_hours = float(duration_fraction * period_days * 24.0)

    n_out = np.count_nonzero(out_mask)
    if n_out > 1:
        scatter = np.std(bin_flux[out_mask], ddof=1)
    else:
        scatter = np.nan

    if np.isfinite(scatter) and scatter > 0:
        snr = float(depth / scatter)
    else:
        snr = None

    return {"depth_ppm": depth_ppm, "duration_hours": duration_hours, "snr": snr}
