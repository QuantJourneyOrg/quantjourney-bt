# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Example Weights 26 - Sentiment with shared observation replay.

Mode: weights, long/cash. Universe: one USD equity, AAPL in this configuration.
Idea: hold after a usable positive observation with positive mentions.
Default: native sample prices and an explicitly empty sentiment archive.
Supply both local prices and an archive for non-empty research replay.
"""

import asyncio
import os
from pathlib import Path

import pandas as pd

from backtester.observations import ObservationBacktester, ObservationData
from backtester.observations.adanos import load_adanos_observations
from backtester.portfolio.rebalance import RebalancePolicy


class AdanosSentiment(ObservationBacktester):
    def _compute_signals(self) -> pd.DataFrame:
        prices = self.data.get_feature("adj_close")
        observations = self.observations_at(prices.index)
        active = (
            observations["usable"]
            & observations["found"].eq(True)
            & observations["mentions"].gt(0)
            & observations["sentiment_score"].gt(0)
        )
        return active.fillna(False).astype(float).to_frame(prices.columns[0])

    def _compute_weights(self) -> pd.DataFrame:
        return self.signals.copy()


async def main() -> None:
    # Edit these research settings explicitly; file dates do not replace them.
    instruments = ["AAPL"]
    period = {"start": "2024-11-04", "end": "2026-11-03"}
    price_file, legacy_file = os.getenv("QJ_DATA_FILE"), os.getenv("QJ_DATA_CSV")
    archive_file = os.getenv("ADANOS_SNAPSHOT_FILE")
    if price_file and legacy_file:
        raise ValueError("Choose only one local price file")
    price_file = price_file or legacy_file
    if bool(price_file) != bool(archive_file):
        raise ValueError("Supply both local prices and ADANOS_SNAPSHOT_FILE")
    observations = (
        load_adanos_observations(Path(archive_file))
        if archive_file
        else ObservationData(
            pd.DataFrame(
                columns=["observed_at", "data_date", "found", "mentions", "sentiment_score"]
            ),
            instrument=instruments[0],
            source="reddit_stocks",
            synthetic=True,
        )
    )
    strategy = AdanosSentiment(
        observation_data=observations,
        max_data_age_days=3,
        instruments=instruments,
        backtest_period=period,
        source="local" if price_file else "sample",
        local_data=price_file,
        execution_mode="weights",
        strategy_name="ExampleWeights26_AdanosObservations",
        strategy_type="Own-observation research" if archive_file else "Empty observation sample",
        benchmark_symbol="AAPL",
        benchmark_name="Local price reference" if price_file else "Synthetic sample reference",
        rebalance_policy=RebalancePolicy(frequency="D"),
        show_text_reports=False,
        save_text_reports=True,
        save_portfolio_plots=False,
        show_portfolio_plots=False,
    )
    await strategy.run_strategy()
    strategy.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
