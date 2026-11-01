"""Create a pandas DataFrame from a local CSV or Parquet snapshot."""

from pathlib import Path

import pandas as pd


def load_dataframe(path: str | Path | None = None) -> pd.DataFrame:
    """Read the default snapshot or an explicit local file."""
    file = Path(path) if path else Path(__file__).parent / "data/data_universe_50_yf_10years.csv"
    if file.suffix.lower() == ".parquet":
        return pd.read_parquet(file, engine="pyarrow")
    if file.suffix.lower() == ".csv":
        return pd.read_csv(file, float_precision="round_trip")
    raise ValueError("Choose a CSV or Parquet file")


if __name__ == "__main__":
    observations = load_dataframe()
    print(
        f"DataFrame: {len(observations)} observations; {observations['symbol'].nunique()} symbols"
    )
