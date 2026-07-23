"""CLI tests — run as a real subprocess from a foreign cwd."""

from __future__ import annotations

import json
import subprocess
import sys

from conftest import save_npz


def run_cli(args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "foldr", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_cli_strong_injection_exit_0(tmp_path, transit_lc_factory):
    synth = transit_lc_factory(
        period=3.21,
        t0=1.5,
        depth=0.005,
        duration=0.1,
        noise=1e-4,
        span=27.0,
        cadence=0.5 / 24,
        seed=42,
    )
    lc_path = save_npz(synth, tmp_path / "strong.npz")

    proc = run_cli([str(lc_path), "--no-plot"], cwd=tmp_path.parent)

    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_cli_pure_noise_exit_1(tmp_path, noise_lc_factory):
    synth = noise_lc_factory(noise=3e-4, span=27.0, cadence=0.5 / 24, seed=0)
    lc_path = save_npz(synth, tmp_path / "noise.npz")

    proc = run_cli([str(lc_path), "--no-plot"], cwd=tmp_path.parent)

    assert proc.returncode == 1, proc.stdout + proc.stderr


def test_cli_missing_file_exit_2(tmp_path):
    proc = run_cli([str(tmp_path / "nope.npz"), "--no-plot"], cwd=tmp_path.parent)
    assert proc.returncode == 2


def test_cli_rejects_zero_period(tmp_path, transit_lc_factory):
    # --period 0 divides by zero in phase_fold(), silently turning every
    # phase into NaN and printing a nonsense "Period: 0.000000 d" result
    # at exit 0 instead of a clear usage error.
    synth = transit_lc_factory(
        period=3.21, t0=1.5, depth=0.005, duration=0.1, noise=1e-4,
        span=27.0, cadence=0.5 / 24, seed=42,
    )
    lc_path = save_npz(synth, tmp_path / "lc.npz")

    proc = run_cli([str(lc_path), "--period", "0", "--t0", "1.5", "--no-plot"], cwd=tmp_path.parent)
    assert proc.returncode == 2
    assert "period must be positive" in proc.stdout + proc.stderr


def test_cli_rejects_negative_period_min(tmp_path, transit_lc_factory):
    synth = transit_lc_factory(
        period=3.21, t0=1.5, depth=0.005, duration=0.1, noise=1e-4,
        span=27.0, cadence=0.5 / 24, seed=42,
    )
    lc_path = save_npz(synth, tmp_path / "lc.npz")

    proc = run_cli([str(lc_path), "--period-min", "-1", "--no-plot"], cwd=tmp_path.parent)
    assert proc.returncode == 2
    assert "period-min must be positive" in proc.stdout + proc.stderr


def test_cli_rejects_non_positive_bins_and_detrend(tmp_path, transit_lc_factory):
    # bins<=0 and detrend_window_days<=0 crashed inside fold() itself
    # (bin_folded / _running_median_detrend) with a raw ValueError/
    # IndexError; confirm the CLI surfaces fold()'s new clean ValueError
    # instead (caught by the existing except (FoldrReadError, ValueError)
    # around the fold() call -- no separate CLI-side check needed).
    synth = transit_lc_factory(
        period=3.21, t0=1.5, depth=0.005, duration=0.1, noise=1e-4,
        span=27.0, cadence=0.5 / 24, seed=42,
    )
    lc_path = save_npz(synth, tmp_path / "lc.npz")

    proc = run_cli(
        [str(lc_path), "--period", "3.21", "--t0", "1.5", "--bins", "0", "--no-plot"],
        cwd=tmp_path.parent,
    )
    assert proc.returncode == 2
    assert "bins must be positive" in proc.stdout + proc.stderr

    proc = run_cli(
        [str(lc_path), "--period", "3.21", "--t0", "1.5", "--detrend", "-5", "--no-plot"],
        cwd=tmp_path.parent,
    )
    assert proc.returncode == 2
    assert "detrend_window_days must be positive" in proc.stdout + proc.stderr


def test_cli_json_roundtrips(tmp_path, transit_lc_factory):
    synth = transit_lc_factory(
        period=3.21,
        t0=1.5,
        depth=0.005,
        duration=0.1,
        noise=1e-4,
        span=27.0,
        cadence=2 / 60 / 24,
        seed=13,
    )
    lc_path = save_npz(synth, tmp_path / "json_test.npz")

    proc = run_cli([str(lc_path), "--no-plot", "--json"], cwd=tmp_path.parent)
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr

    payload = json.loads(proc.stdout)
    assert "period_days" in payload
    assert "bin_centers" in payload
    assert "phase" not in payload
    assert "folded_flux" not in payload


def test_cli_user_ephemeris_plot_saved(tmp_path, transit_lc_factory):
    synth = transit_lc_factory(
        period=2.0,
        t0=0.3,
        depth=0.01,
        duration=0.05,
        noise=1e-4,
        span=10.0,
        cadence=1 / 24,
        seed=14,
    )
    lc_path = save_npz(synth, tmp_path / "plot_test.npz")

    proc = run_cli(
        [str(lc_path), "--period", "2.0", "--t0", "0.3"], cwd=tmp_path.parent
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    plot_path = tmp_path / "plot_test.foldr.png"
    assert plot_path.exists()


def test_cli_version(tmp_path):
    proc = run_cli(["--version"], cwd=tmp_path)
    assert proc.returncode == 0
    assert "foldr" in proc.stdout.lower()


def test_cli_tls_missing_extra_message(tmp_path, transit_lc_factory):
    try:
        import transitleastsquares  # noqa: F401

        import pytest

        pytest.skip("transitleastsquares is installed; nothing to test here")
    except ImportError:
        pass

    synth = transit_lc_factory(
        period=2.0,
        t0=0.3,
        depth=0.01,
        duration=0.05,
        noise=1e-4,
        span=10.0,
        cadence=1 / 24,
        seed=15,
    )
    lc_path = save_npz(synth, tmp_path / "tls_test.npz")

    proc = run_cli([str(lc_path), "--engine", "tls", "--no-plot"], cwd=tmp_path.parent)
    assert proc.returncode == 2
    assert proc.stdout.strip() == (
        'TLS engine requires: pip install "foldr[tls] @ '
        'git+https://github.com/nikhilcherry/foldr"'
    )
