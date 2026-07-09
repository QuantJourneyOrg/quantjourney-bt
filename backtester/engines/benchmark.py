"""
Benchmark data fetching and comparison utilities.

Fetch priority:
  1. SDK warehouse API (api.quantjourney.cloud)
  2. yfinance fallback

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from backtester.utils.logger import logger


def _select_benchmark_price(
	df: pd.DataFrame,
	*,
	source: str,
	symbol: str,
	assume_close_adjusted: bool = False,
) -> Optional[pd.Series]:
	"""Prefer total-return/adjusted benchmark prices; fall back to raw close with warning."""
	if df is None or df.empty:
		return None
	columns = {str(c).lower().replace(" ", "_"): c for c in df.columns}
	for key in ("adj_close", "adjclose", "adjusted_close"):
		if key in columns:
			return df[columns[key]].astype(float)
	if "close" in columns:
		if not assume_close_adjusted:
			logger.warning(
				f"Benchmark {symbol} from {source} has no adjusted close; using raw close price-return series."
			)
		return df[columns["close"]].astype(float)
	return None


def _price_to_returns(price: pd.Series) -> pd.Series:
	return price.sort_index().astype(float).pct_change().replace([np.inf, -np.inf], np.nan)


def _align_benchmark_returns(benchmark_returns: pd.Series, returns_index: pd.DatetimeIndex) -> pd.Series:
	aligned = benchmark_returns.reindex(returns_index)
	missing = aligned.isna()
	if missing.any():
		missing_frac = float(missing.mean())
		logger.warning(
			f"Benchmark alignment has {missing.sum()} missing bars ({missing_frac:.1%}); "
			"leaving them as NaN instead of injecting 0% returns."
		)
	return aligned


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
		if date_col is None:
			logger.warning(f"Warehouse benchmark response missing date/close columns: {list(df.columns)}")
			return None

		df[date_col] = pd.to_datetime(df[date_col])
		df = df.sort_values(date_col).set_index(date_col)
		price = _select_benchmark_price(df, source="warehouse", symbol=symbol)
		if price is None:
			logger.warning(f"Warehouse benchmark response missing price columns: {list(df.columns)}")
			return None
		returns = _price_to_returns(price)
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
	returns_index: pd.DatetimeIndex,
	symbol: str = '^GSPC',
	start_date=None,
	end_date=None,
	sdk_client: Any = None,
) -> Optional[pd.Series]:
	"""
	Fetch and process benchmark returns, aligning with the primary returns index.

	Priority: SDK warehouse → yfinance.
	"""
	try:
		# 1) SDK warehouse API
		benchmark_data = await _fetch_benchmark_via_sdk(sdk_client, symbol, start_date, end_date)

		# 2) yfinance fallback
		if benchmark_data is None:
			benchmark_data = await _get_returns_via_yfinance(symbol, start_date, end_date)

		if benchmark_data is not None and len(benchmark_data) > 0:
			benchmark_data.index = benchmark_data.index.tz_localize(None)
			benchmark_data.index = benchmark_data.index.normalize()
			benchmark_data.index = benchmark_data.index.tz_localize('UTC')
			benchmark_returns = _align_benchmark_returns(benchmark_data, returns_index)
			return benchmark_returns

		logger.warning("No benchmark data retrieved from any source.")
		return None
	except Exception as e:
		logger.error(f"Error processing benchmark data: {str(e)}")
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

		price = _select_benchmark_price(
			hist,
			source="yfinance(auto_adjust=True)",
			symbol=symbol,
			assume_close_adjusted=True,
		)
		if price is None:
			return None
		close = _price_to_returns(price)
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
