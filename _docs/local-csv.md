Tags: backtester, local-data, csv
Created: 2026-10-31
Updated: 2026-11-01
ADRs: ADR-431
Repos: quantjourney-bt

# Run a strategy from local input

Status: source checkout. PyPI 0.12.5 does not include this feature.

## Local input contract

The `local_data` constructor argument accepts a CSV path, a Parquet path, or a pandas DataFrame.
Use `source="local"` with an explicit local input.
The existing `strategy.data` property continues to return prepared instrument data.
The existing `data_csv` argument and `--data-csv` option remain available.
The `--data-file PATH` option accepts CSV and Parquet files.
Its environment setting is `QJ_DATA_FILE`.

An explicit constructor input takes precedence over environment settings.
Keep `local_data=None` when you want to select a file through launcher options.
Two explicit input arguments, or two environment input paths without an explicit input, are an error.
The strategy keeps its `instruments` and `backtest_period` settings by default.
The existing `csv_config=True` shortcut replaces both settings and remains opt-in.
It does not configure a file format.

All local inputs use the same daily equity columns and validation rules below.
The launcher accepts an input configured inside strategy code without credentials.
The Cloud data path continues to require credentials.
The launcher keeps missing credentials fatal in optimization and validation workflows.
Direct Python calls retain their existing relaxed Cloud-fetch behavior.
Local input errors always stop a run.
Batch execution retains its credential check unless a local file is selected through options or the environment.

## 1. Prepare the environment

Use this source checkout, which includes the `--data-file` and `--data-csv` options.
Run the commands from its repository root.
The current public PyPI release does not include this feature.

The setup below uses Python 3.12.
Skip setup if this checkout already has a working `.venv` with the required dependencies.
For a checkout without `.venv`, run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[reports]"
```

On Windows, use Command Prompt:

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[reports]"
```

For Parquet, install the optional reader in the same environment:

```bash
python -m pip install -e ".[reports,parquet]"
```

In Windows Command Prompt, use `.venv\Scripts\python.exe` in place of `python`.
CSV and DataFrame input do not require this extra.
The minimum reader version is `pyarrow>=14.0.1`.

Installation needs network access to obtain dependencies.
After installation, CSV execution needs no account, credentials, Data SDK, or network connection.

## Download local observations

The [_local guide](../_local/README.md) documents the fixed company selection and direct Yahoo download.
Install the `data` and `parquet` extras. Then run:

```sh
python _local/download_yfinance.py --output-dir _local/data
```

The output folder must not exist. The script preserves existing snapshots.
Its default start is inclusive. Its `--end-exclusive` argument excludes the given date.
The script validates observations with the existing local-data validator.
It writes CSV, Parquet and per-download JSON together in the new output folder.
The tracked `_local/universe_50.json` contains only the fixed selection and its source.
Generated price files and download metadata remain ignored by Git.

## 2. Run the daily SMA example

Use an existing local CSV that follows the [CSV format](#csv-format).
The command below uses the local Yahoo research snapshot.
This snapshot is not included in a public clone or package.
Replace `_local/data/data_universe_50_yf_10years.csv` with your own CSV path when needed.

```bash
./strategy.sh example_weights_01_sma_daily \
  --data-csv _local/data/data_universe_50_yf_10years.csv \
  --csv-config \
  --output reports/local_csv
```

On Windows, use Command Prompt:

```bat
strategy.bat example_weights_01_sma_daily --data-csv _local\data\data_universe_50_yf_10years.csv --csv-config --output reports\local_csv
```

`--csv-config` selects the file's symbols and date span.
It preserves the strategy's trading rules.
For other strategies, see [Select the input configuration](#select-the-input-configuration).

Wait for the final summary before opening the report.
The run includes chart generation. Pressing Ctrl+C cancels it.
Add `--quiet` to reduce log messages.

## 3. Open the report

Open `reports/local_csv/ExampleWeights01_DailySMATrend/dashboard.html` in a browser.
The same directory contains `summary.json`, `metrics.csv`, `equity_curve.csv`, and `weights.csv`.
To create a separate report, change the `--output` directory.

The authenticated qj-api path and the existing sample mode remain available.

## Configure local input in strategy code

Insert this fragment in an existing daily equity strategy constructor.
Keep its indicators, trading rules, and report settings.

```python
source="local",
local_data="_local/data/data_universe_50_yf_10years.csv",
instruments=["AAPL", "AMZN"],
backtest_period={"start": "2016-10-31", "end": "2026-10-30"},
```

The strategy selects these two symbols from the file.
Other symbols in the file are not part of this run.
No automatic universe replacement is needed.

For Parquet, replace only the input:

```python
local_data="_local/data/data_universe_50_yf_10years.parquet",
```

For a DataFrame, load observations before you construct the strategy:

```python
import pandas as pd

observations = pd.read_csv(
    "_local/data/data_universe_50_yf_10years.csv",
    float_precision="round_trip",
)
```

Then pass `local_data=observations` in the same constructor.
The DataFrame uses the same flat columns as the file.
The constructor copies it, so later caller changes do not change the run's input.

The daily SMA example also accepts `main(local_data=..., csv_config=...)`.
To configure that example in its code, replace the final `asyncio.run(main())` call.
For the full local snapshot demo, use:

```python
asyncio.run(main(
    local_data="_local/data/data_universe_50_yf_10years.parquet",
    csv_config=True,
))
```

This demo shortcut takes all symbols and dates from the file.
For a fixed universe, keep `csv_config=False` and configure `instruments` and `backtest_period` in the constructor.

## From observations to signals

1. [local_data.py](../backtester/local_data.py) reads the input and validates the observations.
   `prepare_local_data` creates price, metric, and parameter tables.
   `prepare_csv_data` remains a compatibility alias.
2. [core.py](../backtester/core.py) uses `_process_market_data` to create `InstrumentData` and `PortfolioData`.
   Price columns identify an instrument and a field, such as close.
3. `_compute_indicators` calculates the configured indicators in parameter tables.
4. The strategy's `_compute_signals` and `_compute_weights` methods create signals and target weights.
   The engine stores them in the strategy tables.

The input contains market observations. The strategy calculates its signals after loading.
Local input validation stays at the file and DataFrame boundary.
The engine's existing table validation remains active.

## Select the input configuration

`--data-file PATH` selects a local daily equity CSV or Parquet file.
The reader detects `.csv` and `.parquet` suffixes without case sensitivity.
Other suffixes are rejected. The legacy `--data-csv` option keeps CSV-only parsing, including suffixless files.
`--csv-config` explicitly replaces the strategy's instrument list and period with the file's contents.
Trading rules, position caps, indicator warm-up, and execution behavior stay unchanged.
Only `example_weights_01_sma_daily` currently opts in to this replacement.
Other examples keep their own symbols and dates. The file must cover them.
The override is rejected for other strategies and for `--all`.
This prevents silent changes to pairs, futures, intraday, and walk-forward configurations.

Strategy authors can set `supports_csv_config = True` after checking that their logic supports an arbitrary equity universe.
Do not enable this declaration for strategies that depend on a specific pair or instrument list.

Python callers can pass `local_data=path_or_dataframe`.
The legacy `data_csv=path` argument remains available.
An opted-in strategy can also use `csv_config=True` with any local input.
The existing `backtest_period` constructor argument remains required.
An explicit keyword argument takes precedence over its environment setting.
The launcher uses `QJ_DATA_FILE`, the legacy `QJ_DATA_CSV`, and `QJ_CSV_CONFIG` in its child process.
These variables can also be set explicitly in the environment.
A CSV setting conflicts with sample mode, including `QJ_SAMPLE_DATA` from `.env`.
Local input errors always fail, even when Cloud fetch errors are configured as nonfatal.

Without `--csv-config`, the requested period must stay inside the file's observed date bounds.
The loader never silently shortens the requested history.
Dates are inclusive. Nontrading dates inside those bounds do not create extra bars.

## CSV format

CSV, Parquet, and DataFrame inputs require these flat columns: `date,symbol,open,high,low,close,adj_close,volume`.
Each row describes one daily session for one USD equity.
All price fields must use the same adjusted-price basis.
`adj_close` must equal `close`, within floating-point round-off.
Dates identify midnight UTC session dates. Symbols are normalized to uppercase.

Reject missing fields, duplicate symbol/date pairs, nonnumeric or nonfinite values,
nonpositive prices, negative volume, and inconsistent OHLC bars.
OHLC means open, high, low, and close.
The loader allows scale-aware floating-point round-off. It never repairs a price.

A complete panel has one row for each selected symbol on every date present in the selected input.
A missing row for one symbol is an error. The loader does not certify an exchange calendar.
It does not invent bars, backward-fill prices, or obtain missing data from a provider.
Daily CSV mode currently rejects other intervals, non-USD accounting, and known non-equity contract specifications.

## Input calculations and reports

Prices and volume come from the file.
Input returns use adjusted close divided by the previous close, minus one.
Input reference NAV uses adjusted close divided by the first close.
Input volatility uses a trailing 20-observation standard deviation, with at least two returns.
Input drawdown uses reference NAV divided by its running maximum, minus one.
Input daily P&L is the change in reference NAV. These are input calculations, not strategy results.

Initial units and transaction costs are zero because no trades have executed yet.
Initial portfolio NAV is a unit cash reference, scaled to initial capital by the existing engine.
Eligibility, activity, and session flags identify observed bars in the complete panel.
Unknown exchange, day type, and forecasts stay missing.
The existing execution path computes strategy NAV, costs, positions, and reports.

Run metadata records the file name, SHA-256 checksum, selected symbols, requested dates, actual dates, and override choice.
For files, the checksum identifies the same bytes that the loader parsed.
For a DataFrame, `sha256_basis=dataframe-selected-prices-csv-v1` identifies a snapshot of the selected validated prices.
The snapshot sorts columns, uses UTC dates, and writes 17 significant digits.
It is not a checksum of an original file. Its `file_name` is null.
A benchmark can use a matching series already in the local data.
If that series is absent, its comparison is unavailable. There is no provider fallback.

## Yahoo research snapshot

The local CSV has the owner-requested name `data_universe_50_yf_10years.csv`.
The company-description file is `data-universe_50_yf_10years.json`.
The same observations are available locally as `data_universe_50_yf_10years.parquet`.
`_local/data_universe_50_yf_10years.py` creates a DataFrame from CSV or Parquet.
The JSON records the Parquet checksum and DataFrame loader.
That JSON records company names, source, selection rule, exclusions, coverage, and checksum.
The different name stems are intentional.

The download uses Yahoo `auto_adjust=True` for all OHLC fields.
Its adjusted close is also saved as `adj_close`. Volume keeps the Yahoo value.
Adjusted history can change with the download date.
Adjusted-price execution is a research approximation, not a cash-dividend ledger.
Per-share commissions and volume-participation assumptions can differ from historical unadjusted execution.

The fixed selection uses current companies with sufficient history.
It has survivorship bias and does not represent historical index membership.
Public redistribution rights for the downloaded data are not verified.
Downloaded files under `_local/` are ignored by Git. The `_local/` examples are checkout files and are excluded from distributions.

## Verification status

CSV, Parquet, and DataFrame runs completed in the public sandbox on 2026-10-31.
Each run used the same explicit 50-company universe and dates, with `csv_config=False`.
Network access was denied, and QuantJourney credentials were absent.
All three runs completed at 300 DPI without warnings under standard Python warning settings.
Their equity and weight CSV files were byte-identical to the previous verified CSV run.
Each report retained 21 plot files and the separate equity-curve preview.
The Parquet observations matched the source CSV exactly.
The DataFrame had 125,700 rows and eight columns before preparation.
These checks establish format consistency. They do not validate an investment thesis.

The sandbox-only strategy preset uses local CSV by default.
It keeps its universe and dates explicit.
The public and private repository examples retain their existing defaults.

On 2026-10-31, the daily SMA example completed on the local 50-company Yahoo CSV.
The isolated run used strict reporting, no QuantJourney credentials, and denied network access.
The input checksum matched the dataset description. All 50 symbols had nonzero allocations during the run.
The report contains 2,514 equity observations and local HTML, CSV, JSON, and PNG files.
The configured SPY benchmark was absent. The chart did not substitute another instrument.
These checks establish execution and input consistency, not investment performance validity.

## Changelog

- 2026-11-01 — Prepare source publication with source-checkout instructions and a separate PyPI 0.12.5 limitation — owner request; candidate commit.

- 2026-11-01 — Add the `_local/` selection, Yahoo downloader and guide; preserve verified snapshot files — owner request; local changes.

- 2026-10-31 — Add local CSV, Parquet, and DataFrame input through one validator; preserve strategy selection and credential errors — commit: — (uncommitted).

- 2026-10-31 — Add setup steps, local CSV commands, and report instructions; move the README example near installation — commit: — (uncommitted).

- 2026-10-31 — Wrap allocation legends without removing labels; verify the complete 300 DPI report — commit: — (uncommitted).

- 2026-10-31 — Batch indicator columns to prevent fragmentation warnings; real equity and weights stay byte-identical — commit: — (uncommitted).

- 2026-10-31 — Add and verify local CSV input and the explicit daily SMA universe option — commit: — (uncommitted candidate).

## Roadmap

- PyPI release of local input support — not included in source publication; no release date is set.
