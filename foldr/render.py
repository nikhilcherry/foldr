"""Terminal output (rich) and JSON serialization for FoldResult."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def _fmt(value, unit: str = "", precision: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{precision}f}{unit}"


def print_result(
    result, plot_path: str | Path | None, console: Console | None = None
) -> None:
    console = console or Console()

    n_used = int(result.lc.time.size)
    n_removed = int(result.lc.n_removed)

    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold")
    table.add_column()
    table.add_row("Engine:", result.engine)
    table.add_row("Period:", _fmt(result.period, " d", 6))
    table.add_row("T0:", _fmt(result.t0, "", 6))
    table.add_row("Duration:", _fmt(result.duration_hours, " h", 3))
    table.add_row("Depth:", _fmt(result.depth_ppm, " ppm", 1))
    table.add_row("SNR:", _fmt(result.snr, "", 2))
    table.add_row("SDE:", _fmt(result.sde, "", 2))
    table.add_row("Points used / removed:", f"{n_used} / {n_removed}")

    console.print(
        Panel(
            table,
            title=f"foldr — {Path(result.lc.source_path).name}",
            expand=False,
        )
    )

    if plot_path is not None:
        console.print(f"Plot saved to [bold]{plot_path}[/bold]")


def _sanitize(value):
    if isinstance(value, np.ndarray):
        return [_sanitize(v) for v in value.tolist()]
    if isinstance(value, np.floating):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


def to_json(result, plot_path: str | Path | None = None) -> str:
    """Serialize a FoldResult to a JSON string.

    Only ``bin_centers``/``bin_flux``/``bin_err`` are included as arrays —
    the full per-point ``phase``/``folded_flux`` arrays are intentionally
    omitted to keep output small.
    """
    payload = {
        "source_path": result.lc.source_path,
        "engine": result.engine,
        "period_days": result.period,
        "t0": result.t0,
        "duration_hours": result.duration_hours,
        "depth_ppm": result.depth_ppm,
        "snr": result.snr,
        "sde": result.sde,
        "n_points_used": int(result.lc.time.size),
        "n_points_removed": int(result.lc.n_removed),
        "bin_centers": result.bin_centers,
        "bin_flux": result.bin_flux,
        "bin_err": result.bin_err,
        "search_meta": result.search_meta,
        "plot_path": str(plot_path) if plot_path is not None else None,
    }
    return json.dumps(_sanitize(payload))
