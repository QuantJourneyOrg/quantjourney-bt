# QuantJourney Backtester Public
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

from __future__ import annotations

from typing import Optional

import pandas as pd

from backtester.portfolio.instr_data import InstrumentData
from backtester.portfolio.config import CalcConfig, get_default_config

from backtester.portfolio.calc import rolling_stats as calc_roll
from backtester.portfolio.calc import metrics as calc_metrics


class InstrumentCalculations:
    """
    Facade class for instrument analytics. Performs alignment/orchestration and
    delegates numerical work to calc modules.
    """

    def __init__(self, instrument_data: InstrumentData, config: Optional[CalcConfig] = None):
        self._instrument_data = instrument_data
        self._config: CalcConfig = config or get_default_config()

    # Accessors --------------------------------------------------------
    @property
    def prices(self) -> pd.DataFrame:
        return self._instrument_data.prices

    @property
    def returns(self) -> pd.DataFrame:
        return self._instrument_data.get_feature("metrics", level="returns")

    @property
    def units(self) -> pd.DataFrame:
        """Backward-compatible alias for position_units.

        In QuantJourney data contracts ``units`` means executed position
        quantity/shares/contracts. It must not be used as portfolio weights.
        """
        return self.position_units

    @property
    def position_units(self) -> pd.DataFrame:
        """Executed position quantities per instrument."""
        return self._instrument_data.get_feature("parameters", level="units")

    @property
    def weights(self) -> pd.DataFrame:
        """Portfolio weights per instrument.

        Weights are distinct from position units. They may be stored as an
        optional ``parameters/weights`` frame by callers that want instrument
        analytics to compute weight-based attribution.
        """
        params = getattr(self._instrument_data, "parameters", pd.DataFrame())
        if (
            params is not None
            and isinstance(params, pd.DataFrame)
            and isinstance(params.columns, pd.MultiIndex)
            and "weights" in params.columns.get_level_values(-1)
        ):
            return params.xs("weights", axis=1, level=-1)
        raise ValueError(
            "weights are required for this calculation; pass weights explicitly "
            "or store them under InstrumentData.parameters level='weights'. "
            "Do not use units as weights."
        )

    @property
    def instruments(self) -> list[str]:
        return self._instrument_data.group_data.index.tolist()

    @property
    def data_index(self) -> pd.DatetimeIndex:
        return self.prices.index

    def compute_rolling_volatility(self, periods: int = 7) -> pd.DataFrame:
        return calc_roll.rolling_volatility(self.returns, window=periods)

    def compute_correlation_matrix(self) -> pd.DataFrame:
        return calc_metrics.correlation_matrix(self.returns)
