Tags: backtester, observations, local-data
Created: 2026-11-04
Updated: 2026-11-04
ADRs: ADR-431 (related execution correctness constraints)
Repos: quantjourney-bt

# Saved observations: usage and design

Status: implemented in the source checkout. This addition is not a tagged package release.
The optional `backtester.observations` module handles availability, freshness and provenance.
It supports one instrument and weight mode. Existing execution, costs and price loading stay unchanged.

## Module boundaries

| File or class | Responsibility |
|---|---|
| `ObservationData` | Validate one instrument's table, align availability and calculate input hashes. |
| `ObservationBacktester` | Connect aligned observations to native strategy execution and run metadata. |
| `backtester/observations/adanos.py` | Read the Adanos archive format and make an explicit capture request. |
| W26 | Define the sentiment rule, universe, dates and allocation. |

The strategy enables observation handling through inheritance from `ObservationBacktester`.
It passes an `ObservationData` as `observation_data`, then calls `observations_at(prices.index)`.
The shared module returns selected rows and records their provenance automatically.
`source="local"` selects price input. It does not enable sentiment handling by itself.

The generic package is named `observations`. The provider-specific adapter is named `adanos`.
Other sources can use the generic package without the Adanos adapter.
No SDK, warehouse connection, scheduler or extra dependency is added.

## Keep the trading rule in the strategy

[W26](../strategies/example_weights_26_adanos_observations.py) inherits `ObservationBacktester`.
Its signal method selects usable observations, then applies visible business conditions:

```python
def _compute_signals(self):
    prices = self.data.get_feature("adj_close")
    rows = self.observations_at(prices.index)
    active = (
        rows["usable"] & rows["found"].eq(True)
        & rows["mentions"].gt(0) & rows["sentiment_score"].gt(0)
    )
    return active.fillna(False).astype(float).to_frame(prices.columns[0])
```

The example's `main()` exposes the universe, period, input paths and age limit.
From the repository root, programmatic callers use the same constructor.
The `strategies` import below requires the source checkout.
With an installed wheel, copy W26 beside your script and import its class from that file:

```python
from strategies.example_weights_26_adanos_observations import AdanosSentiment
from backtester.observations.adanos import load_adanos_observations

strategy = AdanosSentiment(
    observation_data=load_adanos_observations("research/observations.json"),
    max_data_age_days=3,
    instruments=["AAPL"],
    backtest_period={"start": "2024-11-04", "end": "2026-11-03"},
    source="local", local_data="research/prices.csv",
)
```

Edit the dates for your inputs. Prices may be CSV, Parquet or a pandas DataFrame.
Install the existing `parquet` extra for Parquet. The explicit universe can select AAPL from a wider price file.

## Run locally

```bash
# Native sample prices and an explicitly empty sentiment archive; no key.
./strategy.sh example_weights_26_adanos_observations --output reports/observations-empty

# Labelled, invented two-year test inputs in the repository checkout.
ADANOS_SNAPSHOT_FILE=tests/fixtures/observations/adanos_two_years.json \
  ./strategy.sh example_weights_26_adanos_observations \
  --data-file tests/fixtures/observations/prices_two_years.csv \
  --output reports/observations-two-years
```

The empty sample holds cash and labels its synthetic inputs. It invents no provider observations.
The test fixture has 522 prices and 489 observations. Both inputs are synthetic.
Its dates match the configuration above. It demonstrates behavior, not investment performance or historical provider coverage.
Test fixtures are excluded from distributions. Supplied archives replay without network access.
Supply both local prices and `ADANOS_SNAPSHOT_FILE`, or neither for the empty sample.
Do not combine local files with `--sample-data`. W26 rejects `--csv-config`; edit its universe and dates explicitly.
Run W26 separately for local inputs. `--all --data-file` also reaches W26 and requires its observation archive.

## Inspect the report

The default W26 configuration saves text reports, CSV results and `run_metadata.json`.
It does not generate plots by default.
Set `save_portfolio_plots=True` in W26 to also generate PNG charts and `dashboard.html`.
Keep `show_portfolio_plots=False` for a run without interactive plot windows.

With the two-year command above, the output directory is:

```text
reports/observations-two-years/ExampleWeights26_AdanosObservations/
```

Open `performance_report.txt` for metrics, `weights.csv` for target weights,
and `equity_curve.csv` for the portfolio path.
Open `run_metadata.json` to inspect selected receipt times and input hashes.
When plots are enabled, open `dashboard.html` in a browser.
Synthetic report labels identify the demonstration. The results do not establish a profitable sentiment rule.

## Archive format and time fields

The Adanos archive has one `source`, one `ticker`, a boolean `synthetic` label and an `observations` list.
Each observation contains these fields:

| Field | Meaning |
|---|---|
| `observed_at` | The time the complete response was received, with a time zone. |
| `data_date` | The completed UTC day requested from Adanos. |
| `found` | The provider's boolean availability flag. |
| `mentions` | A nonnegative count, or null. |
| `sentiment_score` | A finite score from -1 to 1, or null. |

The shared module adds `decision_at`, `data_age_days` and `usable` during alignment.
`usable` means that an earlier receipt exists and passes the configured age limit.
The strategy still checks the business fields. A usable row can contain missing or negative sentiment.

The following excerpt comes from the repository's explicitly synthetic test fixture:

```json
{
  "source": "reddit_stocks",
  "ticker": "AAPL",
  "synthetic": true,
  "observations": [
    {
      "observed_at": "2024-11-14T06:00:00+00:00",
      "data_date": "2024-11-13",
      "found": true,
      "mentions": 17,
      "sentiment_score": 0.3863
    }
  ]
}
```

`data_date` identifies the period described by a value. `observed_at` identifies its availability.
A historical response received today retains today's receipt time.
An archived later revision must not replace what an earlier decision could see.

## Capture sentiment explicitly

Set `ADANOS_API_KEY` privately in the process environment. Run this from the project environment:

```python
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from backtester.observations.adanos import capture_adanos_snapshot

capture = capture_adanos_snapshot(
    ticker="AAPL", day=datetime.now(UTC).date() - timedelta(days=1),
)
folder = Path("research/adanos")
folder.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
with (folder / f"capture-{stamp}.json").open("x", encoding="utf-8") as stream:
    json.dump(capture, stream, indent=2)
```

This makes one request to Reddit Stocks v1 with explicit `from` and `to` for a completed UTC day.
Receipt time is recorded after the response body arrives. Redirects are not followed.
Files and successful response bodies are bounded to 1 MiB. Errors stop capture; they do not create neutral observations.
The key is never included in the saved result. No scheduler or automatic retry is included.

The repository ignores `research/` to keep saved inputs outside Git.
Keep the original captures. Combine their `observations` only when `source`, `ticker` and `synthetic` labels match.
Retain earlier entries unchanged and save each enlarged archive under a new name.
Validate it with `load_adanos_observations(path)` before replay. Duplicate receipts and oversized archives are rejected.
Use that new path as `ADANOS_SNAPSHOT_FILE`. Never backdate receipts or relabel real captures as synthetic.
A response downloaded today cannot supply an observation before today's receipt.
Collect prospectively or use independently verified historical vintages.

Adanos is an external commercial provider. Account plans determine access, quotas and history.
See its [reference](https://api.adanos.com/docs) and [OpenAPI contract](https://api.adanos.com/openapi.json).
The current v1 adapter requires `found`; `mentions` and `sentiment_score` can be null.
It follows the contribution proposed in [PR #6](https://github.com/QuantJourneyOrg/quantjourney-bt/pull/6).
Verification used the public contract and test transports, with no authenticated provider call.

## Use another source

Build an `ObservationData` from your own scalar DataFrame:

```python
import pandas as pd
from backtester.observations import ObservationData

observations = ObservationData(
    pd.read_parquet("research/earnings-observations.parquet"),
    instrument="AAPL", source="own_earnings", synthetic=False,
    provenance_columns=("eps_surprise",),
)
```

This is an input template, not supplied earnings data. Use `synthetic=True` for invented inputs.
Each row needs a time-zone-aware `observed_at` and the feature columns your rule reads.
Include `data_date` for an age limit. Document that date's meaning in your adapter.
Use the same base class and `observations_at()` call, then define your own rule, such as positive `eps_surprise`.
Identifier columns named `symbol`, `ticker`, `instrument` or `asset`, in any case, must match the declared instrument.
Mixed instruments are rejected. The extension does not expand the local price loader's asset or frequency support.

## Multiple instruments: planned extension

Status: design only. The current constructor accepts one observation stream for one instrument.
It rejects a multi-instrument configuration. The mapping below is a proposed interface, not runnable current code:

```python
observation_data={
    "AAPL": load_adanos_observations("research/AAPL.json"),
    "MSFT": load_adanos_observations("research/MSFT.json"),
}
```

The proposed design retains one validated archive per instrument.
The current Adanos loader and capture function can be reused for each archive.
The shared strategy base would accept the collection and align each stream independently.
Each instrument would retain its own hash, receipts, freshness checks and decision provenance.
This per-symbol approach is also used in the [QuantConnect custom-data interface](https://www.quantconnect.com/docs/v2/writing-algorithms/importing-data/streaming-data/custom-securities/key-concepts).

The strategy would return a dates-by-instruments signal matrix.
Allocation must remain explicit. Equal weights across active instruments are one proposed rule.
Position limits and execution timing would continue through the native engine.

The planned checks are:

- Match a receipt only to its own instrument.
- Allow equal receipt times for different instruments; reject duplicates within one stream.
- Keep a later missing revision selected for that instrument.
- Apply the age limit separately to each stream.
- Use a zero signal for an absent or stale stream, if that missing-stream policy is selected explicitly.
- Validate the configured universe against the price columns and declared streams.
- Record each selected instrument in run metadata.
- Preserve the existing single-instrument interface.

This extension requires implementation and focused tests before use. No delivery date is set.
The present example does not capture several instruments or run a multi-instrument sentiment portfolio.

## Availability and evidence contract

- Select the latest receipt strictly before the decision; exclude equal timestamps.
- Reject naive receipts, integer epochs, invalid times and duplicate receipts after UTC conversion.
- Require unique, increasing decision indices. Preserve their original index and time zone.
- Interpret naive midnight decision dates as UTC; reject naive intraday decisions.
- Measure freshness in UTC calendar days from `data_date`. A limit requires that column; `None` disables the age filter.
- Keep missing values distinct from zero. The latest missing revision stays selected.
- Keep latest-receipt semantics even when an older data day arrives last. A stale backfill can therefore select cash.

For the daily example, decision time is midnight UTC of the price-row date.
The native engine then shifts weights by one price bar. No extra shift is added here.
Alignment preserves declared availability; it does not prove historical correctness or remove price-history revisions.
This follows [backward time matching](https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html)
and the availability-time principle in [QuantConnect custom data](https://www.quantconnect.com/docs/v2/writing-algorithms/importing-data/streaming-data/custom-securities/key-concepts).

`run_metadata.json` records `external_observations`: source, instrument, synthetic label, normalized-table hash, archive hash,
age limit and per-call selected receipt/data/decision times with usability.
Hash bases are explicit; normalized hashes include labels and scalar/column types. Hashes do not authenticate availability.
Source, instrument and synthetic-observation labels are read-only after construction.
`sample_price_mode` records use of native sample prices. It does not classify a supplied CSV as real or synthetic.
Arbitrary provider text is excluded. Named `provenance_columns` may add numeric, boolean or null values.
Decimals use strings to retain precision. Synthetic inputs are labelled in report parameters and saved text-report headers.

The extension requires strict data fetching and strict reporting. Fetch errors retain their original cause. Explicit `strict_reporting=False` is rejected; environment relaxation does not disable it.
Metadata failures stop the run. A strategy must call `observations_at`; evidence resets on each new run.

## Changelog

- 2026-11-04 — Expand module boundaries, archive fields, report instructions and the planned multi-instrument design — owner documentation request.
- 2026-11-04 — Add shared observation replay, the Adanos adapter, explicit W26 settings and focused verification — PR #6 adaptation.

## Roadmap

- Multiple-instrument streams — design documented above; not implemented; no date is set.
