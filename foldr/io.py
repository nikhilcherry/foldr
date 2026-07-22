"""Light curve loading: FITS / npz / csv / txt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class FoldrReadError(Exception):
    """Raised when a light curve file cannot be read or parsed."""


@dataclass
class LightCurve:
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray | None
    source_path: str
    time_label: str
    n_removed: int


_TIME_KEYS_FITS = ["TIME"]
_FLUX_KEYS_FITS = ["PDCSAP_FLUX", "SAP_FLUX", "FLUX"]

_TIME_KEYS_NPZ = ["time", "t", "btjd", "bjd"]
_FLUX_KEYS_NPZ = ["flux", "f", "pdcsap_flux"]
_ERR_KEYS_NPZ = ["flux_err", "ferr", "err"]

_TIME_KEYS_DELIM = ["time", "t", "btjd", "bjd"]
_FLUX_KEYS_DELIM = ["flux", "f", "pdcsap_flux"]
_ERR_KEYS_DELIM = ["flux_err", "ferr", "err"]


def load_lightcurve(
    path: str | Path, *, time_col: str | None = None, flux_col: str | None = None
) -> LightCurve:
    """Load a light curve. Format is chosen by extension (case-insensitive):
    .fits/.fit, .npz, .csv, .txt/.dat. Raises FoldrReadError on failure.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FoldrReadError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".fits", ".fit"):
        return _load_fits(path, time_col, flux_col)
    if suffix == ".npz":
        return _load_npz(path, time_col, flux_col)
    if suffix == ".csv":
        return _load_delim(path, time_col, flux_col, delimiter=",")
    if suffix in (".txt", ".dat"):
        return _load_delim(path, time_col, flux_col, delimiter=None)

    raise FoldrReadError(
        f"Unsupported file extension '{path.suffix}' for {path}. "
        "Supported: .fits, .fit, .npz, .csv, .txt, .dat"
    )


def _load_fits(path: Path, time_col: str | None, flux_col: str | None) -> LightCurve:
    from astropy.io import fits

    try:
        with fits.open(path) as hdul:
            table_hdu = None
            colnames: list[str] = []
            tcol_upper = time_col.upper() if time_col else None

            for hdu in hdul:
                columns = getattr(hdu, "columns", None)
                if columns is None:
                    continue
                names = list(columns.names)
                names_upper = [n.upper() for n in names]
                if tcol_upper is not None:
                    if tcol_upper in names_upper:
                        table_hdu, colnames = hdu, names
                        break
                    continue
                if any(k in names_upper for k in _TIME_KEYS_FITS):
                    table_hdu, colnames = hdu, names
                    break

            if table_hdu is None:
                raise FoldrReadError(
                    f"No table HDU with a time-like column found in {path}. "
                    f"Tried columns: {_TIME_KEYS_FITS} (override with --time-col)"
                )

            names_upper = [n.upper() for n in colnames]

            def resolve(override: str | None, candidates: list[str], kind: str) -> str:
                if override:
                    up = override.upper()
                    if up in names_upper:
                        return colnames[names_upper.index(up)]
                    raise FoldrReadError(
                        f"{kind} column '{override}' not found in {path}. "
                        f"Available columns: {colnames}"
                    )
                for cand in candidates:
                    if cand in names_upper:
                        return colnames[names_upper.index(cand)]
                raise FoldrReadError(
                    f"No {kind} column found in {path}. Tried: {candidates}. "
                    f"Available columns: {colnames}"
                )

            time_name = resolve(time_col, _TIME_KEYS_FITS, "time")
            flux_name = resolve(flux_col, _FLUX_KEYS_FITS, "flux")

            data = table_hdu.data
            time = np.asarray(data[time_name], dtype=float)
            flux = np.asarray(data[flux_name], dtype=float)

            flux_err = None
            err_upper = f"{flux_name.upper()}_ERR"
            if err_upper in names_upper:
                flux_err = np.asarray(
                    data[colnames[names_upper.index(err_upper)]], dtype=float
                )

            time_label = "TIME"
            try:
                unit = table_hdu.columns[time_name].unit
                if unit:
                    time_label = f"TIME ({unit})"
            except Exception:
                pass
    except FoldrReadError:
        raise
    except Exception as exc:
        raise FoldrReadError(f"Failed to read FITS file {path}: {exc}") from exc

    return _finalize(time, flux, flux_err, str(path), time_label)


def _load_npz(path: Path, time_col: str | None, flux_col: str | None) -> LightCurve:
    try:
        with np.load(path) as data:
            keys = list(data.keys())
            time_name = _resolve_key(time_col, keys, _TIME_KEYS_NPZ, "time", path)
            flux_name = _resolve_key(flux_col, keys, _FLUX_KEYS_NPZ, "flux", path)

            time = np.asarray(data[time_name], dtype=float)
            flux = np.asarray(data[flux_name], dtype=float)

            flux_err = None
            for k in _ERR_KEYS_NPZ:
                if k in keys:
                    flux_err = np.asarray(data[k], dtype=float)
                    break
    except FoldrReadError:
        raise
    except Exception as exc:
        raise FoldrReadError(f"Failed to read npz file {path}: {exc}") from exc

    return _finalize(time, flux, flux_err, str(path), "TIME")


def _resolve_key(
    override: str | None, keys: list[str], candidates: list[str], kind: str, path: Path
) -> str:
    if override:
        if override in keys:
            return override
        raise FoldrReadError(
            f"{kind} key '{override}' not found in {path}. Available keys: {keys}"
        )
    for cand in candidates:
        if cand in keys:
            return cand
    raise FoldrReadError(
        f"No {kind} key found in {path}. Tried: {candidates}. Available keys: {keys}"
    )


def _sniff_has_header(path: Path, delimiter: str | None) -> bool:
    """Peek at the file to decide whether ``genfromtxt`` should read column names.

    A ``#``-prefixed first line is only treated as a header row if its
    tokens look like recognized column names (time/flux/...) -- matching
    what genfromtxt's own ``names=True`` does for a commented header line.
    Any other leading ``#`` line is ordinary metadata/provenance (e.g.
    "# generated by pipeline v2") and must not be mistaken for a header,
    or genfromtxt tries to use its tokens as field names and chokes on the
    resulting column-count mismatch against the real data rows -- so we
    sniff the first non-comment, non-blank line instead.
    """
    known = {k.lower() for k in _TIME_KEYS_DELIM + _FLUX_KEYS_DELIM + _ERR_KEYS_DELIM}

    def _tokens(line: str) -> list[str]:
        return [t.strip() for t in line.strip().split(delimiter if delimiter else None) if t.strip()]

    with open(path, "r") as f:
        first_line = f.readline()
        sniff_line = first_line
        if first_line.strip().startswith("#"):
            commented_tokens = [t.lower() for t in _tokens(first_line.lstrip("#"))]
            if any(t in known for t in commented_tokens):
                return True
            sniff_line = ""
            for line in f:
                if line.strip() == "" or line.strip().startswith("#"):
                    continue
                sniff_line = line
                break

    for tok in _tokens(sniff_line):
        try:
            float(tok)
        except ValueError:
            return True
    return False


def _load_delim(
    path: Path, time_col: str | None, flux_col: str | None, delimiter: str | None
) -> LightCurve:
    try:
        has_header = _sniff_has_header(path, delimiter)

        if has_header:
            data = np.genfromtxt(path, delimiter=delimiter, names=True, dtype=float)
            names = list(data.dtype.names)
            names_lower = [n.lower() for n in names]

            def resolve(override: str | None, candidates: list[str], kind: str) -> str:
                if override:
                    lo = override.lower()
                    if lo in names_lower:
                        return names[names_lower.index(lo)]
                    raise FoldrReadError(
                        f"{kind} column '{override}' not found in {path}. "
                        f"Available columns: {names}"
                    )
                for cand in candidates:
                    if cand in names_lower:
                        return names[names_lower.index(cand)]
                raise FoldrReadError(
                    f"No {kind} column found in {path}. Tried: {candidates}. "
                    f"Available columns: {names}"
                )

            time_name = resolve(time_col, _TIME_KEYS_DELIM, "time")
            flux_name = resolve(flux_col, _FLUX_KEYS_DELIM, "flux")

            time = np.asarray(data[time_name], dtype=float)
            flux = np.asarray(data[flux_name], dtype=float)

            flux_err = None
            for cand in _ERR_KEYS_DELIM:
                if cand in names_lower:
                    flux_err = np.asarray(
                        data[names[names_lower.index(cand)]], dtype=float
                    )
                    break
        else:
            raw = np.genfromtxt(path, delimiter=delimiter, dtype=float)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            if raw.ndim != 2 or raw.shape[1] < 2:
                raise FoldrReadError(
                    f"{path} must have at least 2 columns (time, flux); "
                    f"found shape {raw.shape}"
                )
            time = raw[:, 0]
            flux = raw[:, 1]
            flux_err = raw[:, 2] if raw.shape[1] > 2 else None
    except FoldrReadError:
        raise
    except Exception as exc:
        raise FoldrReadError(f"Failed to read {path}: {exc}") from exc

    return _finalize(time, flux, flux_err, str(path), "TIME")


def _finalize(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None,
    source_path: str,
    time_label: str,
) -> LightCurve:
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=float)

    finite_mask = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        finite_mask &= np.isfinite(flux_err)

    n_removed = int(time.size - np.count_nonzero(finite_mask))

    time = time[finite_mask]
    flux = flux[finite_mask]
    flux_err = flux_err[finite_mask] if flux_err is not None else None

    if flux.size == 0:
        raise FoldrReadError(f"No finite data points remain after cleaning {source_path}")

    median = np.median(flux)
    if median == 0 or not np.isfinite(median):
        raise FoldrReadError(
            f"Flux median is not usable (median={median}) for {source_path}"
        )

    flux = flux / median
    if flux_err is not None:
        flux_err = flux_err / median

    return LightCurve(
        time=time,
        flux=flux,
        flux_err=flux_err,
        source_path=source_path,
        time_label=time_label,
        n_removed=n_removed,
    )
