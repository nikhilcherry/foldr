"""Two-panel figure: raw light curve + phase-folded curve. Agg only, never
opens an interactive window."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def make_figure(result, out_path: str | Path) -> Path:
    """Render and save the two-panel figure for a FoldResult. Returns the
    resolved output path.
    """
    out_path = Path(out_path).expanduser().resolve()
    lc = result.lc

    fig, (ax_raw, ax_fold) = plt.subplots(2, 1, figsize=(9, 7))

    # Raw panel: points only (never lines) so gaps > 0.5 d are never bridged.
    ax_raw.scatter(lc.time, lc.flux, s=3, c="0.5", alpha=0.6, linewidths=0)
    ax_raw.set_xlabel(lc.time_label)
    ax_raw.set_ylabel("Normalized flux")
    ax_raw.set_title("Raw light curve")

    if np.isfinite(result.period) and result.period > 0 and lc.time.size > 0:
        t_min, t_max = float(np.min(lc.time)), float(np.max(lc.time))
        n_start = int(np.floor((t_min - result.t0) / result.period)) - 1
        n_end = int(np.ceil((t_max - result.t0) / result.period)) + 1
        transit_times = [
            result.t0 + n * result.period for n in range(n_start, n_end + 1)
        ]
        transit_times = [t for t in transit_times if t_min <= t <= t_max]
        if transit_times:
            y_top = ax_raw.get_ylim()[1]
            ax_raw.plot(
                transit_times,
                [y_top] * len(transit_times),
                marker="v",
                linestyle="none",
                color="crimson",
                markersize=6,
                clip_on=False,
                label="predicted transit",
            )
            ax_raw.legend(loc="upper right", fontsize=8, frameon=False)

    # Folded panel: raw folded points (light gray) + binned curve with error bars.
    ax_fold.scatter(
        result.phase, result.folded_flux, s=3, c="0.75", alpha=0.4, linewidths=0
    )
    finite_bins = np.isfinite(result.bin_flux)
    ax_fold.errorbar(
        result.bin_centers[finite_bins],
        result.bin_flux[finite_bins],
        yerr=result.bin_err[finite_bins],
        fmt="o",
        color="tab:blue",
        markersize=3,
        elinewidth=1,
        capsize=0,
    )

    if result.duration_hours and result.period > 0:
        half_window = 2.5 * (result.duration_hours / 24.0) / result.period
        half_window = float(min(max(half_window, 1e-6), 0.5))
        ax_fold.set_xlim(-half_window, half_window)
    else:
        ax_fold.set_xlim(-0.5, 0.5)

    ax_fold.set_xlabel("Phase")
    ax_fold.set_ylabel("Normalized flux")
    ax_fold.set_title("Phase-folded")

    title = (
        f"{Path(lc.source_path).name}  —  "
        f"P = {result.period:.5f} d, t0 = {result.t0:.5f} ({result.engine})"
    )
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path
