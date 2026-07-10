# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.


from dataclasses import dataclass
from enum import Enum
from functools import wraps

import numpy as np
import pandas as pd

from backtester.utils.decorators import error_logger
from backtester.utils.logger import logger


# MethodStatus class -----------------------------------------------------
class MethodStatus(Enum):
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    FAILED = "FAILED"
    ERROR = "ERROR"


def method_error(error_message: str):
    """
    Specific decorator for consistent error handling and logging
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{error_message}: {str(e)}")
                return {"status": MethodStatus.ERROR.value, "message": str(e), "data": None}

        return wrapper

    return decorator


# PortfolioUtils ----------------------------------------------
@dataclass
class PortfolioUtils:
    @error_logger("Error in make index")
    def make_index(self, ticker_weights: dict[str, float], rebalance: str = "1M") -> pd.Series:
        """
        Create weighted index with rebalancing

        Args:
                ticker_weights: Dictionary of weights
                rebalance: Rebalancing frequency
        """
        if self._data.prices is None:
            raise ValueError("Price data required for index creation")

        portfolio = pd.DataFrame(index=self._data.prices.index)

        # Handle no rebalancing case
        if rebalance is None:
            for ticker, weight in ticker_weights.items():
                portfolio[ticker] = weight * self._data.prices[ticker]
            return portfolio.sum(axis=1)

        # Rebalance at specified interval
        rbdf = portfolio.resample(rebalance).first()
        for ticker, weight in ticker_weights.items():
            portfolio[ticker] = np.where(
                portfolio.index.isin(rbdf.index),
                weight * self._data.prices[ticker],
                self._data.prices[ticker],
            )

        return portfolio.sum(axis=1)

    @error_logger("Error in make portfolio")
    def make_portfolio(
        self, start_balance: float = 1e5, mode: str = "comp", round_to: int | None = None
    ) -> pd.Series:
        """
        Calculate portfolio value

        Args:
                start_balance: Initial balance
                mode: Calculation mode
                round_to: Rounding precision
        """
        returns = self._data.returns

        if mode.lower() in ["cumsum", "sum"]:
            portfolio = start_balance + start_balance * returns.cumsum()
        elif mode.lower() in ["compsum", "comp"]:
            portfolio = self.to_prices(returns, start_balance)
        else:
            comp_rev = (start_balance + start_balance * returns.shift(1)).fillna(
                start_balance
            ) * returns
            portfolio = start_balance + comp_rev.cumsum()

        if round_to is not None:
            portfolio = np.round(portfolio, round_to)

        return portfolio

    @error_logger("Error in rebase calculation")
    def rebase(self, base: float = 100.0, data: pd.Series | None = None) -> pd.Series:
        """
        Rebase series to new base value

        Args:
                base: New base value
                data: Optional data series
        """
        if data is None:
            data = self._data.nav
        return data.dropna() / data.dropna().iloc[0] * base
