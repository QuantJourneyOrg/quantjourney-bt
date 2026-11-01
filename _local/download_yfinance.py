"""Download the fixed research universe directly from Yahoo Finance."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backtester.local_data import load_local_data, prepare_local_data

SELECTION_FILE = Path(__file__).with_name("universe_50.json")
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def read_selection(path: Path = SELECTION_FILE) -> dict[str, Any]:
    """Read the fixed selection without replacing unavailable companies."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("The selection must be a JSON object")
    selection: dict[str, Any] = value
    symbols = [company["symbol"] for company in selection["companies"]]
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("The selection must contain distinct symbols")
    return selection


def download_observations(symbols: list[str], start: str, end_exclusive: str) -> pd.DataFrame:
    """Keep adjusted OHLC and the provider's volume, without filling gaps."""
    yf = importlib.import_module("yfinance")

    raw = yf.download(
        symbols,
        start=start,
        end=end_exclusive,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        keepna=True,
        threads=False,
        progress=False,
        timeout=30,
    )
    if raw is None or raw.empty:
        raise ValueError("Yahoo returned no observations")
    frames = []
    for symbol in symbols:
        if isinstance(raw.columns, pd.MultiIndex):
            if symbol not in raw.columns.get_level_values(0):
                raise ValueError(f"Yahoo returned no observations for {symbol}")
            bars = raw[symbol]
        elif len(symbols) == 1:
            bars = raw
        else:
            raise ValueError("Yahoo returned an unexpected column layout")
        if not set(PRICE_COLUMNS).issubset(bars.columns):
            raise ValueError(f"Yahoo returned incomplete OHLCV columns for {symbol}")
        bars = bars[PRICE_COLUMNS].dropna(how="all").copy()
        bars.columns = ["open", "high", "low", "close", "volume"]
        bars["adj_close"] = bars["close"]
        bars.insert(0, "symbol", symbol)
        bars.insert(0, "date", pd.DatetimeIndex(bars.index).strftime("%Y-%m-%d"))
        frames.append(bars.reset_index(drop=True))
    observations: pd.DataFrame = pd.concat(frames, ignore_index=True)
    columns = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
    observations = observations[columns].sort_values(["date", "symbol"]).reset_index(drop=True)
    prepare_local_data(observations, instruments=symbols)
    return observations


def write_snapshot(
    observations: pd.DataFrame,
    selection: dict[str, Any],
    output_dir: Path,
    start: str,
    end_exclusive: str,
    provider_version: str,
) -> dict[str, Any]:
    """Publish a complete new directory; leave existing snapshots untouched."""
    if output_dir.exists():
        raise FileExistsError(f"Output folder already exists: {output_dir}. Use a new folder.")
    symbols = [company["symbol"] for company in selection["companies"]]
    prepared = prepare_local_data(observations, instruments=symbols)
    default_period = (selection["start"], selection["end_exclusive"])
    stem = (
        "data_universe_50_yf_10years"
        if (start, end_exclusive) == default_period
        else f"data_universe_{len(symbols)}_yf_{start}_{end_exclusive}"
    )
    csv_name, parquet_name = f"{stem}.csv", f"{stem}.parquet"
    metadata_name = f"{stem.replace('data_', 'data-', 1)}.json"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=output_dir.parent) as temporary:
        staged = Path(temporary)
        csv_file, parquet_file = staged / csv_name, staged / parquet_name
        observations.to_csv(csv_file, index=False, float_format="%.17g", lineterminator="\n")
        observations.to_parquet(parquet_file, engine="pyarrow", compression="zstd", index=False)
        csv_prices = load_local_data(csv_file, instruments=symbols)["prices"]
        parquet_prices = load_local_data(parquet_file, instruments=symbols)["prices"]
        try:
            for restored in (csv_prices, parquet_prices):
                # CSV can infer whole-number price columns as integers.
                pd.testing.assert_frame_equal(
                    restored, prepared["prices"], check_dtype=False, check_exact=True
                )
        except AssertionError as error:
            raise ValueError("Saved formats do not preserve the validated observations") from error
        metadata = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "provider": "Yahoo Finance via yfinance",
            "provider_version": provider_version,
            "requested_start": start,
            "requested_end_exclusive": end_exclusive,
            "actual_start": prepared["summary"]["start"],
            "actual_end": prepared["summary"]["end"],
            "interval": "1d",
            "currency": "USD",
            "company_count": len(symbols),
            "row_count": len(observations),
            "session_count": len(prepared["prices"]),
            "columns": list(observations.columns),
            "companies": selection["companies"],
            "selection": selection["selection"],
            "constituent_source": selection["constituent_source"],
            "point_in_time": False,
            "survivorship_bias": True,
            "price_basis": "All OHLC adjusted; adj_close equals close; volume as supplied.",
            "limitations": selection["limitations"],
            "files": {
                kind: {
                    "file": file.name,
                    "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    "bytes": file.stat().st_size,
                }
                for kind, file in [("csv", csv_file), ("parquet", parquet_file)]
            },
        }
        (staged / metadata_name).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            raise FileExistsError(f"Output folder already exists: {output_dir}")
        os.rename(staged, output_dir)
    return metadata


def main(argv: list[str] | None = None) -> None:
    selection = read_selection()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="New snapshot folder")
    parser.add_argument("--start", default=selection["start"], help="Inclusive YYYY-MM-DD")
    parser.add_argument(
        "--end-exclusive", default=selection["end_exclusive"], help="Exclusive YYYY-MM-DD"
    )
    args = parser.parse_args(argv)
    if date.fromisoformat(args.start) >= date.fromisoformat(args.end_exclusive):
        parser.error("--start must be before --end-exclusive")
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new folder to preserve its data")
    try:
        importlib.import_module("pyarrow")
        yf = importlib.import_module("yfinance")
    except ImportError as error:
        parser.error(
            f'Install the checkout extras first: pip install -e ".[data,parquet]": {error}'
        )
    symbols = [company["symbol"] for company in selection["companies"]]
    observations = download_observations(symbols, args.start, args.end_exclusive)
    metadata = write_snapshot(
        observations, selection, args.output_dir, args.start, args.end_exclusive, yf.__version__
    )
    print(
        f"Saved {metadata['row_count']} observations for {metadata['company_count']} companies "
        f"to {args.output_dir} ({metadata['actual_start']} to {metadata['actual_end']})."
    )


if __name__ == "__main__":
    main()
