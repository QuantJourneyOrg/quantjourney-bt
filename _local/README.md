# Local data: CSV, Parquet and DataFrame

These instructions require this source checkout. PyPI 0.12.5 does not include local input support.

## Files

| Path | Purpose |
|---|---|
| `universe_50.json` | Fixed symbols, company names, selection rule and source. No prices or download checksums. |
| `download_yfinance.py` | Download adjusted daily observations directly from Yahoo Finance. |
| `data_universe_50_yf_10years.py` | Load CSV or Parquet as a pandas DataFrame in memory. |
| `data/` | Generated CSV, Parquet and per-download JSON metadata. Git ignores this directory. |

The fixed selection contains 50 companies from the original research snapshot.
It is not a historical index universe. It has survivorship bias.
Company metadata comes from the [constituent dataset](https://github.com/datasets/s-and-p-500-companies).
Its published data license is [ODC-PDDL-1.0](https://github.com/datasets/s-and-p-500-companies/blob/main/datapackage.json).
This license does not grant rights to Yahoo price data.

## 1. Install the checkout extras

Use the repository's virtual environment. Run commands from the repository root.

```sh
python -m pip install -e ".[data,parquet,reports]"
```

On Windows, use `.venv\Scripts\python.exe` in place of `python`.
The `data` extra supplies yfinance. The `parquet` extra supplies pyarrow.

## 2. Download a new snapshot

If the development copy already contains `data/`, continue with step 3 to use that snapshot.

```sh
python _local/download_yfinance.py --output-dir _local/data
```

The default requested period starts on 2016-10-31 and ends before 2026-10-31.
The script writes the actual first and last session dates in the generated JSON.
Use `--start YYYY-MM-DD` and `--end-exclusive YYYY-MM-DD` for a different period.
Different periods produce date-specific file names.

The output folder must not exist. The script refuses to replace an existing snapshot.
Choose a new output folder for another download. Use its file path when you run the strategy.
The script keeps the fixed symbols. It stops when a response fails validation; it does not replace companies or fill missing prices.

The download needs internet access. It needs no QuantJourney key, warehouse or Data SDK.
The [yfinance documentation](https://ranaroussi.github.io/yfinance/) directs users to Yahoo's terms and states that the API is for personal use.
Keep downloaded market data local. The Git repository contains the downloader and instructions, not Yahoo price snapshots.

## 3. Run CSV or Parquet

```sh
./strategy.sh example_weights_01_sma_daily \
  --data-file _local/data/data_universe_50_yf_10years.csv \
  --csv-config --output reports/local_csv
```

For Parquet, use `_local/data/data_universe_50_yf_10years.parquet` with the same option.

In Windows Command Prompt:

```bat
strategy.bat example_weights_01_sma_daily --data-file _local\data\data_universe_50_yf_10years.csv --csv-config --output reports\local_csv
```

`--csv-config` takes the symbols and dates from the input. It does not select a file format.
Without this option, the strategy keeps its configured universe and period.
The daily SMA example explicitly supports this shortcut. Other strategies must match their configured input requirements.
After the download, local execution needs no market-data network connection or QuantJourney credentials.

## 4. Pass a DataFrame in Python

```python
import asyncio

from _local.data_universe_50_yf_10years import load_dataframe
from strategies.example_weights_01_sma_daily import main

observations = load_dataframe()
asyncio.run(main(local_data=observations, csv_config=True))
```

The helper also accepts an explicit CSV or Parquet path.
A DataFrame is an object in memory, not a third downloaded file format.
Your own strategy constructor accepts `source="local", local_data=path_or_dataframe`.

All inputs use `date,symbol,open,high,low,close,adj_close,volume` columns.
The current local path supports daily USD equity data. All OHLC prices must use the adjusted-close basis.
The downloader sets `auto_adjust=True` and saves adjusted close in both `close` and `adj_close`.
Volume keeps the provider's value. This is not a separate cash-dividend ledger.

See the [local input contract](../_docs/local-csv.md) for validation, selection and report metadata.
