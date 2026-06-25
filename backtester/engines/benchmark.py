"""
Benchmark data fetching and comparison utilities.

Fetch priority:
  1. SDK warehouse API (api.quantjourney.cloud)
  2. DataConnector (internal DB)
  3. yfinance fallback

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from backtester.utils.logger import logger


# ── SDK warehouse fetch ────────────────────────────────────────────────

async def _fetch_benchmark_via_sdk(
	sdk_client: Any,
	symbol: str,
	start_date,
	end_date,
) -> Optional[pd.Series]:
	"""Fetch benchmark OHLCV from the QuantJourney warehouse API and return daily returns."""
	if sdk_client is None:
		return None
	try:
		start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else str(start_date)
		end_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, 'strftime') else str(end_date)

		data = await sdk_client.get(
			f"/data/v1/benchmarks/prices/{symbol}",
			params={"start": start_str, "end": end_str},
		)

		if data is None:
			return None

		if isinstance(data, list):
			df = pd.DataFrame(data)
		elif isinstance(data, dict) and "data" in data:
			df = pd.DataFrame(data["data"])
		else:
			df = pd.DataFrame(data)

		if df.empty:
			return None

		# Normalise column names (API may return mixed-case)
		df.columns = [c.lower() for c in df.columns]

		date_col = next((c for c in ("date", "datetime", "timestamp") if c in df.columns), None)
		close_col = next((c for c in ("close", "adj_close", "adjclose") if c in df.columns), None)
		if date_col is None or close_col is None:
			logger.warning(f"Warehouse benchmark response missing date/close columns: {list(df.columns)}")
			return None

		df[date_col] = pd.to_datetime(df[date_col])
		df = df.sort_values(date_col).set_index(date_col)
		close = df[close_col].astype(float)
		returns = close.pct_change().fillna(0)
		returns.index = returns.index.tz_localize(None)

		logger.info(f"Fetched {len(returns)} benchmark data points from warehouse API ({symbol})")
		return returns

	except Exception as e:
		message = str(e)
		if "Unauthorized" in message or "no refresh token" in message:
			logger.debug(f"SDK warehouse benchmark unavailable for {symbol}; trying fallback providers.")
		else:
			logger.warning(f"SDK warehouse benchmark fetch failed for {symbol}: {e}")
		return None


# ── Main entry point ───────────────────────────────────────────────────

async def get_benchmark_returns(
	data_connector,
	returns_index: pd.DatetimeIndex,
	symbol: str = '^GSPC',
	start_date=None,
	end_date=None,
	sdk_client: Any = None,
) -> Optional[pd.Series]:
	"""
	Fetch and process benchmark returns, aligning with the primary returns index.

	Priority: SDK warehouse → DataConnector → yfinance.
	"""
	try:
		# 1) SDK warehouse API
		benchmark_data = await _fetch_benchmark_via_sdk(sdk_client, symbol, start_date, end_date)

		# 2) DataConnector
		if benchmark_data is None:
			benchmark_data = await _get_returns_via_connector(data_connector, start_date, end_date)

		# 3) yfinance fallback
		if benchmark_data is None:
			benchmark_data = await _get_returns_via_yfinance(symbol, start_date, end_date)

		if benchmark_data is not None and len(benchmark_data) > 0:
			benchmark_data.index = benchmark_data.index.tz_localize(None)
			benchmark_data.index = benchmark_data.index.normalize()
			benchmark_data.index = benchmark_data.index.tz_localize('UTC')
			benchmark_returns = benchmark_data.reindex(returns_index).fillna(0)
			return benchmark_returns

		logger.warning("No benchmark data retrieved from any source.")
		return None
	except Exception as e:
		logger.error(f"Error processing benchmark data: {str(e)}")
		return None


# ── DataConnector fetch ────────────────────────────────────────────────

async def _get_returns_via_connector(
	data_connector,
	start_date=None,
	end_date=None,
) -> Optional[pd.Series]:
	"""Get SPY returns via DataConnector."""
	if data_connector is None:
		return None
	try:
		spy_data = await data_connector.indices.get_sp500_index_history(
			ticker='^GSPC',
			source='yfinance',
			period_start=start_date,
			period_end=end_date,
			read_from_db=True,
			write_to_db=True,
		)

		if spy_data is None or spy_data.empty:
			return None

		if spy_data.index.name == 'Date' or isinstance(spy_data.index, pd.DatetimeIndex):
			spy_data = spy_data.reset_index()
			spy_data = spy_data.rename(columns={'Date': 'datetime'})

		if pd.api.types.is_datetime64_any_dtype(spy_data['datetime']):
			spy_data['datetime'] = pd.to_datetime(spy_data['datetime']).dt.tz_convert('UTC').dt.tz_localize(None)

		close = spy_data.set_index('datetime')['Close'].pct_change().fillna(0)
		logger.info(f"Fetched {len(close)} benchmark data points from DataConnector.")
		return close

	except Exception as e:
		logger.warning(f"DataConnector benchmark fetch failed: {e}")
		return None


# ── yfinance fallback ──────────────────────────────────────────────────

async def _get_returns_via_yfinance(
	symbol: str,
	start_date=None,
	end_date=None,
) -> Optional[pd.Series]:
	"""Get benchmark returns via yfinance."""
	try:
		import yfinance as yf

		start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, 'strftime') else str(start_date)
		end_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, 'strftime') else str(end_date)

		ticker = yf.Ticker(symbol)
		hist = ticker.history(start=start_str, end=end_str, auto_adjust=True)

		if hist is None or hist.empty:
			logger.error(f"No data returned from yfinance for {symbol}.")
			return None

		close = hist['Close'].pct_change().fillna(0)
		close.index = close.index.tz_localize(None)
		logger.info(f"Fetched {len(close)} benchmark data points from yfinance ({symbol}).")
		return close

	except ImportError:
		logger.warning("yfinance not installed. Install with: pip install yfinance")
		return None
	except Exception as e:
		logger.warning(f"Benchmark fetch via yfinance failed: {e}")
		return None


def compute_benchmark_summary(
	returns: pd.Series,
	periods_per_year: int = 252,
) -> Dict[str, float]:
	"""
	Calculate benchmark summary: MTD, QTD, YTD, 1Y, 3Y ann., 5Y ann., ITD.
	Values in percent (e.g. 3.5 for 3.5%).
	"""
	if returns is None or returns.empty:
		return {}
	returns = returns.dropna().sort_index()
	if returns.empty:
		return {}
	periods_per_year = max(int(periods_per_year or 252), 1)

	def period_return(r: pd.Series, start=None, periods: int = None) -> float:
		if start is not None:
			r_ = r.loc[start:]
		elif periods is not None:
			r_ = r.iloc[-min(periods, len(r)):]
		else:
			r_ = r
		if len(r_) == 0:
			return np.nan
		return (1 + r_).prod() - 1

	def annualized_return(r: pd.Series, periods: int) -> float:
		if len(r) < periods:
			return np.nan
		r_ = r.iloc[-periods:]
		return (1 + r_).prod() ** (periods_per_year / periods) - 1

	latest_dt = returns.index[-1]
	current_month_start = latest_dt.replace(day=1)
	current_quarter_month = ((latest_dt.month - 1) // 3) * 3 + 1
	current_quarter_start = latest_dt.replace(month=current_quarter_month, day=1)
	current_year_start = latest_dt.replace(month=1, day=1)
	one_year_periods = periods_per_year
	three_year_periods = periods_per_year * 3
	five_year_periods = periods_per_year * 5

	return {
		"MTD":  period_return(returns, start=current_month_start) * 100,
		"QTD":  period_return(returns, start=current_quarter_start) * 100,
		"YTD":  period_return(returns, start=current_year_start) * 100,
		"1Y":   period_return(returns, periods=one_year_periods) * 100,
		"3Y":   annualized_return(returns, three_year_periods) * 100,
		"5Y":   annualized_return(returns, five_year_periods) * 100,
		"ITD":  period_return(returns) * 100,
	}


def excess_return(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
	"""
	Difference in final cumulative return (strategy - benchmark), in percent.
	"""
	if strategy_returns is None or benchmark_returns is None:
		return np.nan
	aligned = pd.DataFrame({"strategy": strategy_returns, "benchmark": benchmark_returns}).dropna()
	if aligned.empty:
		return np.nan

	strat_cum = (1 + aligned["strategy"]).prod() - 1
	bench_cum = (1 + aligned["benchmark"]).prod() - 1
	return (strat_cum - bench_cum) * 100


def active_return(
	strategy_returns: pd.Series,
	benchmark_returns: pd.Series,
	periods_per_year: int = 252,
) -> float:
	"""
	Annualized geometric active return (strategy annualized return minus benchmark).
	"""
	if strategy_returns is None or benchmark_returns is None:
		return np.nan
	aligned = pd.DataFrame({"strategy": strategy_returns, "benchmark": benchmark_returns}).dropna()
	if len(aligned) < 2:
		return np.nan
	periods_per_year = max(int(periods_per_year or 252), 1)
	years = len(aligned) / periods_per_year
	if years <= 0:
		return np.nan

	strat_total = (1 + aligned["strategy"]).prod() - 1
	bench_total = (1 + aligned["benchmark"]).prod() - 1
	strat_ann = (1 + strat_total) ** (1 / years) - 1
	bench_ann = (1 + bench_total) ** (1 / years) - 1
	return (strat_ann - bench_ann) * 100
