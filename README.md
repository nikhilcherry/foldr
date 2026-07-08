# foldr

Phase-folding and period-search CLI for astronomical light curves. Point it
at a `.fits`, `.npz`, `.csv`, or `.txt`/`.dat` file and it finds the best
period (via [Box Least Squares](https://docs.astropy.org/en/stable/timeseries/bls.html)
or, optionally, [Transit Least Squares](https://github.com/hippke/tls)),
phase-folds and bins the data, prints the key transit numbers, and saves a
two-panel diagnostic plot.

![Strong transit recovery](docs/example_strong.png)

## Install

```bash
pip install foldr
```

BLS (via astropy) is included in the base install and requires no extra
setup. TLS is optional and pulled in with an extra:

```bash
pip install "foldr[tls]"
```

If TLS isn't installed and you ask for it explicitly (`--engine tls`), foldr
exits with a clear message telling you to install the extra rather than
crashing with an import traceback. In `--engine auto` (the default), foldr
silently falls back to BLS whenever TLS isn't importable.

## Usage

Search for a period and fold on the best candidate:

```bash
foldr lc.fits
```

Already know the ephemeris? Skip the search and fold directly:

```bash
foldr lc.fits --period 3.524 --t0 2458520.1
```

Machine-readable output, no plot:

```bash
foldr lc.npz --json
```

## What it does

1. Loads the light curve, normalizing flux to its median and dropping
   non-finite rows.
2. If `--period` isn't given, runs a period search (BLS by default; TLS if
   installed and `--engine auto`, or forced via `--engine tls`) over
   `[--period-min, --period-max]` (default `period-max` is half the time
   baseline).
3. Phase-folds the light curve at the best (or user-supplied) period and
   epoch, bins it (`--bins`, default 200), and estimates depth, duration,
   and SNR from the binned curve.
4. Prints a summary table (or JSON with `--json`) and saves a two-panel
   PNG — raw light curve with predicted transit times marked, plus the
   phase-folded and binned curve — unless `--no-plot` is passed.

### Column resolution

foldr looks for time/flux columns under common names and lets you override
the guess:

| Format | Time candidates | Flux candidates |
|---|---|---|
| `.fits`/`.fit` | `TIME` | `PDCSAP_FLUX`, `SAP_FLUX`, `FLUX` |
| `.npz` | `time`, `t`, `btjd`, `bjd` | `flux`, `f`, `pdcsap_flux` |
| `.csv`/`.txt`/`.dat` | `time`, `t`, `btjd`, `bjd` | `flux`, `f`, `pdcsap_flux` |

Header-less `.csv`/`.txt`/`.dat` files are read positionally: column 0 is
time, column 1 is flux, an optional column 2 is flux error. Use `--time-col`
/ `--flux-col` to override the automatic guess by name. If no matching
column is found, foldr's error message lists the columns that *were*
available in the file, so you don't have to go re-inspect it separately.

### Detrending

`--detrend DAYS` removes slow stellar variability before search/fold with a
running window. Each point's local trend is the [biweight
location](https://docs.astropy.org/en/stable/api/astropy.stats.biweight_location.html)
of flux within `DAYS` of it, not a plain running median — a median still
lets a handful of in-window in-transit points drag the local trend down,
partially flattening the transit it's supposed to preserve; the biweight
estimator downweights those outliers from the window's own robust center
instead. Following [Hippke & Heller
2019](https://arxiv.org/abs/1906.00966) (the `wotan` detrending paper),
pick a window at least ~3x your expected transit duration so a full
transit can't dominate — and stay wary of even larger windows on noisy,
sparsely-sampled data, since long windows have fewer usable comparison
points per step and detrend less reliably regardless of estimator.

### SDE and the exit-code contract

`SDE` (Signal Detection Efficiency) measures how far the best period's
search-statistic power stands out from the noise floor of the periodogram:
`(power[best] - mean(power)) / std(power)`. foldr uses an SDE threshold of
**7.0** to decide whether a search-mode run found a credible signal.

| Exit code | Meaning |
|---|---|
| `0` | Success — either a user-supplied ephemeris was folded, or a search found `SDE >= 7.0` |
| `1` | Search completed but `SDE < 7.0` — no credible periodic signal found |
| `2` | Usage or read error — bad path, unparseable file, missing requested engine, etc. |

`--period` mode always exits `0` on success since no search (and therefore
no SDE) is involved.

## More examples

A fainter, noisier signal near the recovery threshold — still confidently
detected, but with visibly more scatter in the folded curve:

![Marginal-depth transit](docs/example_marginal.png)

Pure noise, no injected signal — SDE falls below 7.0 and foldr exits `1`:

![No signal](docs/example_noise.png)

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -q -W error::RuntimeWarning
```

See [ROADMAP.md](ROADMAP.md) for features considered and deliberately left
out of this release.

## License

MIT — see [LICENSE](LICENSE).
