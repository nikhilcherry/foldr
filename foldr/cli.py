"""Command-line interface for foldr."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from . import __version__
from .core import fold
from .io import FoldrReadError, load_lightcurve
from .plotting import make_figure
from .render import print_result, to_json

SDE_THRESHOLD = 7.0
LARGE_DATASET_WARNING_THRESHOLD = 200_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foldr",
        description="Phase-fold and search for periodic transits in a light curve.",
    )
    parser.add_argument(
        "file", help="Path to a light curve file (.fits/.fit/.npz/.csv/.txt/.dat)"
    )
    parser.add_argument("--version", action="version", version=f"foldr {__version__}")

    ephemeris = parser.add_argument_group("ephemeris")
    ephemeris.add_argument(
        "--period", type=float, default=None, help="Skip search, fold at this period (days)"
    )
    ephemeris.add_argument(
        "--t0",
        type=float,
        default=None,
        help="Transit epoch, same time units as the file. Estimated from the "
        "fold if --period is given without --t0.",
    )

    search = parser.add_argument_group("search")
    search.add_argument("--engine", choices=["auto", "bls", "tls"], default="auto")
    search.add_argument(
        "--period-min", type=float, default=0.5, dest="period_min", metavar="DAYS"
    )
    search.add_argument(
        "--period-max", type=float, default=None, dest="period_max", metavar="DAYS"
    )

    processing = parser.add_argument_group("processing")
    processing.add_argument(
        "--detrend",
        type=float,
        default=None,
        dest="detrend_window_days",
        metavar="DAYS",
        help="Running-median detrend window (days) applied before search/fold",
    )
    processing.add_argument("--bins", type=int, default=200)
    processing.add_argument(
        "--flux-col", type=str, default=None, dest="flux_col", metavar="NAME"
    )
    processing.add_argument(
        "--time-col", type=str, default=None, dest="time_col", metavar="NAME"
    )

    output = parser.add_argument_group("output")
    output.add_argument("--plot", dest="plot", action="store_true", default=True)
    output.add_argument("--no-plot", dest="plot", action="store_false")
    output.add_argument(
        "--plot-path", type=str, default=None, dest="plot_path", metavar="PATH"
    )
    output.add_argument("--json", action="store_true", dest="json_output")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    try:
        path = Path(args.file).expanduser().resolve()
        lc = load_lightcurve(path, time_col=args.time_col, flux_col=args.flux_col)
    except FoldrReadError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 2

    if args.period is None and lc.time.size > LARGE_DATASET_WARNING_THRESHOLD:
        console.print(
            f"Note: {lc.time.size:,} points with no --period given; "
            "period search may be slow."
        )

    try:
        result = fold(
            lc,
            period=args.period,
            t0=args.t0,
            engine=args.engine,
            period_min=args.period_min,
            period_max=args.period_max,
            bins=args.bins,
            detrend_window_days=args.detrend_window_days,
        )
    except ImportError as exc:
        if args.engine == "tls":
            print("TLS engine requires: pip install foldr[tls]")
            return 2
        console.print(f"[red]Error:[/red] {exc}")
        return 2
    except (FoldrReadError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 2

    plot_path = None
    if args.plot:
        out_path = (
            Path(args.plot_path)
            if args.plot_path
            else path.parent / f"{path.stem}.foldr.png"
        )
        plot_path = make_figure(result, out_path)

    if args.json_output:
        print(to_json(result, plot_path))
    else:
        print_result(result, plot_path, console=console)

    if result.sde is not None and result.sde < SDE_THRESHOLD:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
