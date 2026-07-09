"""
    Backtester — API-backed Strategy Backtester
    --------------------------------------------

    Fetches market data from the QuantJourney Cloud API via HTTP calls
    to the /bt/prepare endpoint. All strategy logic (signals, weights,
    positions, performance) runs locally.

    Strategies subclass Backtester and implement three hooks.

    Usage:
        class MySMAStrategy(Backtester):
            def _compute_signals(self): ...
            def _compute_weights(self): ...
            def _compute_positions(self): pass

        strategy = MySMAStrategy(
            api_url="https://api.quantjourney.cloud",
            email="...", password="...",
            strategy_name="SMA_Cloud",
            instruments=["AAPL", "MSFT"],
            backtest_period={"start": "2024-01-01", "end": "2024-12-31"},
        )
        asyncio.run(strategy.run_strategy())

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

import json
import asyncio
import logging
import os
import time
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from backtester.version import __version__ as BACKTESTER_VERSION
from backtester.mixins.reporting import ReportingMixin
from backtester.mixins.sdk_client import SDKClientMixin
from backtester.execution.contract_spec import ContractSpec, get_contract_spec
from backtester.portfolio._time import normalize_time_index_like

# ---------------------------------------------------------------------------
# Lightweight logger — avoid importing quantjourney.logger which may pull
# heavy deps.  If the full logger is available we reuse it.
# ---------------------------------------------------------------------------
try:
    from backtester.utils.logger import logger
except Exception:
    logger = logging.getLogger("backtester")
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# ---------------------------------------------------------------------------
# Dataclass stubs — BacktestPeriod & PortfolioState
# We define lightweight copies so the file can run without importing
# the full learning_backtester (which pulls in DataConnector, plots, etc.)
# ---------------------------------------------------------------------------

@dataclass
class BacktestPeriod:
    start: str
    end: str

@dataclass
class PortfolioState:
    nav: float = 0.0
    cash: float = 0.0


SUPPORTED_GRANULARITIES = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
}


def _normalize_granularity(granularity: Any) -> str:
    raw = str(granularity or "1d").strip().lower()
    if raw.isdigit():
        raw = f"{raw}m"
    aliases = {
        "1min": "1m",
        "2min": "2m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "60m": "1h",
        "60min": "1h",
        "1hour": "1h",
        "1hr": "1h",
        "d": "1d",
        "day": "1d",
        "daily": "1d",
        "w": "1wk",
        "week": "1wk",
        "weekly": "1wk",
        "m": "1mo",
        "month": "1mo",
        "monthly": "1mo",
    }
    value = aliases.get(raw, raw)
    if value not in SUPPORTED_GRANULARITIES:
        supported = ", ".join(sorted(SUPPORTED_GRANULARITIES))
        raise ValueError(f"Unsupported granularity={granularity!r}. Use one of: {supported}.")
    return value


# ─────────────────────────────────────────────────────────────────────
# Helpers: reconstruct DataFrames from FramePayload JSON
# ─────────────────────────────────────────────────────────────────────

def _payload_to_multiindex_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert a FramePayload JSON (columns, index, data) back into
    a MultiIndex-columned DataFrame:  (instrument, field).
    """
    columns_meta = payload.get("columns", [])
    index_iso = payload.get("index", [])
    rows = payload.get("data", [])

    if not columns_meta or not index_iso:
        return pd.DataFrame()

    # Build MultiIndex columns
    tuples = [(c["instrument"], c["field"]) for c in columns_meta]
    mi = pd.MultiIndex.from_tuples(tuples, names=["instrument", "field"])

    # Build DatetimeIndex
    idx = pd.DatetimeIndex(pd.to_datetime(index_iso), name="date")

    df = pd.DataFrame(rows, index=idx, columns=mi)

    # Coerce numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _payload_to_series(payload: Dict[str, Any], name: str = "nav") -> pd.Series:
    """Convert a {index, data} payload into a pd.Series."""
    index_iso = payload.get("index", [])
    values = payload.get("data", [])
    idx = pd.DatetimeIndex(pd.to_datetime(index_iso), name="date")
    return pd.Series(values, index=idx, name=name, dtype=float)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Parse bool-like config values from kwargs/env vars."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _env_flag(name: str, default: bool = False) -> bool:
    return _coerce_bool(os.environ.get(name), default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"[Backtester] Invalid {name}={raw!r}; using {default}")
        return default


# ─────────────────────────────────────────────────────────────────────
# Backtester
# ─────────────────────────────────────────────────────────────────────


class Backtester(SDKClientMixin, ReportingMixin):
    """
    Strategy backtester: fetches market data (QuantJourney API, yfinance, or a
    bundled sample dataset) and runs user-defined strategy logic locally.

    Write a strategy by subclassing and implementing the hooks:

    * ``_compute_signals() -> pd.DataFrame`` — raw alpha signals per instrument.
    * ``_compute_weights() -> pd.DataFrame`` — target portfolio weights (weights
      mode), or override ``_compute_orders()`` to emit explicit orders (orders mode).
    * ``_compute_positions()`` — optional; usually a no-op.

    Example
    -------
    A long/cash SMA(50/200) trend strategy, rebalanced daily::

        import asyncio, pandas as pd
        from backtester import Backtester
        from backtester.portfolio.rebalance import RebalancePolicy

        class SMATrend(Backtester):
            def _compute_signals(self) -> pd.DataFrame:
                fast = self.instruments_data.get_feature("SMA_50_close")
                slow = self.instruments_data.get_feature("SMA_200_close")
                return (fast > slow).astype(float)

            def _compute_weights(self) -> pd.DataFrame:
                active = self.signals == 1.0
                return active.div(active.sum(axis=1), axis=0).fillna(0.0)

        async def main():
            bt = SMATrend(
                strategy_name="sma_trend",
                instruments=["AAPL", "MSFT", "NVDA"],
                backtest_period={"start": "2015-01-01", "end": "2025-01-01"},
                source="sample",                      # bundled data, no API key
                execution_mode="weights",
                rebalance_policy=RebalancePolicy(frequency="D"),
                indicators_config=[
                    {"function": "SMA", "price_cols": ["close"],
                     "params": {"periods": [50, 200]}},
                ],
            )
            await bt.run_strategy()

        asyncio.run(main())
    """

    def __init__(
        self,
        *,
        # ── API Auth ──
        api_url: str = "https://api.quantjourney.cloud",
        email: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        # ── Provider ──
        source: str = "yfinance",
        granularity: str = "1d",
        # ── Strategy Config ──
        strategy_name: str = "cloud_strategy",
        strategy_type: str = "Long-Short",
        base_currency: str = "USD",
        initial_capital: float = 100_000.0,
        instruments: Optional[List[str]] = None,
        backtest_period: Optional[Dict[str, str]] = None,
        target_volatility: Optional[float] = None,
        max_position_size: Optional[float] = None,
        indicators_config: Optional[List[Dict[str, Any]]] = None,
        # ── Reporting ──
        reports_directory: str = "./reports",
        plots_directory: str = "./plots",
        theme_plots: str = "quantjourney",
        show_text_reports: bool = True,
        save_text_reports: bool = False,
        save_portfolio_plots: bool = False,
        show_portfolio_plots: bool = False,
        save_instrument_plots: bool = False,
        show_instrument_plots: bool = False,
        save_pdf_report: bool = False,
        reporting_frequency: str = "daily",
        # ── BT API options ──
        persist: bool = True,
        dedupe: bool = True,
        force_refresh: bool = False,
        # ── Benchmark ──
        benchmark_symbol: str = "^GSPC",
        benchmark_name: str = "S&P 500 Index",
        # ── Execution mode ──
        execution_mode: str = "weights",   # "weights" (default) or "orders"
        slippage_model=None,               # SlippageModel instance
        commission_scheme=None,            # CommissionScheme instance
        weight_cost_model=None,            # WeightCostModel instance for weight-mode implied trades
        contract_specs: Optional[Dict[str, ContractSpec]] = None,
        fill_at: str = "open",             # order-mode market fill convention
        max_volume_participation: Optional[float] = None,  # order-mode volume cap
        # ── Rebalancing ──
        rebalance_policy=None,             # RebalancePolicy instance (default: daily)
        # ── Benchmark returns (for tracking-error rebalance trigger) ──
        benchmark_returns: Optional[pd.Series] = None,
        # ── Risk model ──
        risk_model=None,                   # RiskModel instance (applied between weights and rebalance)
        # ── Performance flags ──
        skip_analysis: bool = False,       # skip StrategyPerformanceAnalysis (saves ~800ms)
        lite_init: bool = False,           # skip throwaway validation/metrics in data init (saves ~600ms)
        strict_reporting: Optional[bool] = None,
        strict_data_fetch: Optional[bool] = None,
        **kwargs,
    ):
        # ── Reject unknown kwargs — silent swallowing hides typos like
        # `rebalance_polcy=` and runs a materially different backtest. ──
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Backtester.__init__() got unexpected keyword argument(s): {unknown}. "
                "Check for typos — unknown kwargs are not silently ignored."
            )

        # ── Validate + normalize execution_mode / fill_at (case-insensitive).
        # Previously execution_mode='Orders' silently ran the WEIGHTS path and
        # fill_at='Open' silently meant close fills. ──
        execution_mode = str(execution_mode).strip().lower()
        if execution_mode not in {"weights", "orders"}:
            raise ValueError(
                f"execution_mode must be 'weights' or 'orders' (case-insensitive), "
                f"got {execution_mode!r}"
            )
        fill_at = str(fill_at).strip().lower()
        if fill_at not in {"open", "close"}:
            raise ValueError(
                f"fill_at must be 'open' or 'close' (case-insensitive), got {fill_at!r}"
            )

        # ── API credentials ──
        self.api_url = api_url.rstrip("/")
        self._email = email
        self._password = password
        self._api_key = api_key
        self._token: Optional[str] = None

        # ── SDK async client (lazy-initialized) ──
        self._sdk_client = None

        # ── Provider ──
        self._source = source
        self._granularity = _normalize_granularity(granularity)
        self._persist = persist
        self._dedupe = dedupe
        self._force_refresh = force_refresh

        # ── Strategy config ──
        self.strategy_name = strategy_name
        self.strategy_type = strategy_type
        self.base_currency = base_currency
        self.initial_capital = float(initial_capital)
        self.instruments = [s.strip().upper() for s in (instruments or [])]
        if backtest_period is None:
            raise ValueError(
                "backtest_period is required: {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}"
            )
        if "start" not in backtest_period or "end" not in backtest_period:
            raise ValueError(
                "backtest_period must contain both 'start' and 'end' keys"
            )
        self.backtest_period = BacktestPeriod(
            start=backtest_period["start"],
            end=backtest_period["end"],
        )
        # target_volatility / max_position_size default to None so an EXPLICIT
        # user setting is distinguishable from the report-metadata defaults.
        self.target_volatility = 0.15 if target_volatility is None else float(target_volatility)
        self.max_position_size = 0.10 if max_position_size is None else float(max_position_size)
        self.indicators_config = indicators_config or []

        # ── Reporting ──
        self._reports_directory = os.environ.get("QJ_OUTPUT_DIR", reports_directory)
        self._plots_directory = plots_directory
        self._theme_plots = os.environ.get("QJ_PLOT_THEME", theme_plots)
        self._plot_dpi = _env_int("QJ_PLOT_DPI", 300)
        self._benchmark = {
            "symbol": benchmark_symbol,
            "name": benchmark_name,
        }
        self._quiet = _env_flag("QJ_QUIET", False)
        self._no_reports = _env_flag(
            "QJ_NO_REPORTS",
            _env_flag("QJ_SKIP_ANALYSIS", bool(skip_analysis)),
        )
        if self._quiet or self._no_reports:
            show_text_reports = False
            show_portfolio_plots = False
            show_instrument_plots = False
        if self._no_reports:
            save_text_reports = False
            save_portfolio_plots = False
            save_instrument_plots = False
            save_pdf_report = False
            skip_analysis = True
        self._show_text_reports = show_text_reports
        self._save_text_reports = save_text_reports
        self._save_portfolio_plots = save_portfolio_plots
        self._show_portfolio_plots = show_portfolio_plots
        self._save_instrument_plots = save_instrument_plots
        self._show_instrument_plots = show_instrument_plots
        self._save_pdf_report = save_pdf_report
        self._reporting_frequency = os.environ.get(
            "QJ_REPORTING_FREQUENCY",
            reporting_frequency,
        )
        strict_reporting_default = bool(strict_reporting) if strict_reporting is not None else False
        self._strict_reporting = _env_flag("QJ_STRICT_REPORTING", strict_reporting_default)
        strict_data_default = bool(strict_data_fetch) if strict_data_fetch is not None else False
        self._strict_data_fetch = _env_flag(
            "QJ_STRICT_BACKTEST",
            _env_flag("QJ_STRICT_DATA_FETCH", strict_data_default),
        )

        # ── Portfolio state ──
        self.portfolio = PortfolioState(
            nav=self.initial_capital,
            cash=self.initial_capital,
        )

        # Transaction cost as a FRACTION of trade value — single source of
        # truth. NOTE: despite the legacy name this is NOT expressed in basis
        # points: 0.0001 is the fraction equal to 1 bp. Name kept for API
        # compatibility; do not multiply by 1e-4 again.
        self.TRANSACTION_COST_BPS = 0.0001  # fraction = 1 bp
        weight_cost_model_explicit = weight_cost_model is not None
        if weight_cost_model is None:
            from backtester.portfolio.weight_cost import FixedBpsWeightCostModel
            weight_cost_model = FixedBpsWeightCostModel(
                total_bps=self.TRANSACTION_COST_BPS * 10_000.0
            )
        self.weight_cost_model = weight_cost_model

        # ── Execution mode ──
        self.execution_mode = execution_mode  # "weights" or "orders"
        self.contract_specs = {
            str(symbol).upper(): spec
            for symbol, spec in (contract_specs or {}).items()
        }
        self.fill_engine = None
        if execution_mode == "orders":
            from backtester.execution import FillEngine
            self.fill_engine = FillEngine(
                slippage=slippage_model,
                commission=commission_scheme,
                fill_at=fill_at,
                max_volume_participation=max_volume_participation,
                notional_fn=lambda instrument, price, quantity: self._trade_notional(
                    instrument, quantity, price
                ),
            )
        self._order_context: Dict[str, Any] = {}
        # instrument → {"stop_loss": order_id, "take_profit": order_id} for
        # auto OCO-linking of protective exits (see _oco_link_protective_exit).
        self._protective_exit_ids: Dict[str, Dict[str, str]] = {}
        self.average_entry_price: Dict[str, Optional[float]] = {}
        self.current_positions_meta: Dict[str, Dict[str, Optional[float]]] = {}

        # ── Performance flags ──
        self._skip_analysis = skip_analysis
        self._lite_init = lite_init

        # ── Rebalancing ──
        from backtester.portfolio.rebalance import RebalancePolicy
        self._rebalance_policy = rebalance_policy if rebalance_policy is not None else RebalancePolicy()

        # ── Benchmark returns (used by the tracking-error rebalance trigger) ──
        self._benchmark_returns = benchmark_returns

        # ── Risk model ──
        self._risk_model = risk_model  # None = no adjustment
        if max_position_size is not None:
            if self.execution_mode == "weights" and risk_model is None:
                # Route the explicit cap into the existing weight-cap path so
                # the knob is actually enforced instead of being a report label.
                from backtester.risk.position_limit import PositionLimitModel
                self._risk_model = PositionLimitModel(
                    max_weight=self.max_position_size,
                    min_weight=0.0,
                    max_total_leverage=float("inf"),
                )
                logger.info(
                    f"[Backtester] max_position_size={self.max_position_size:.2%} "
                    "enforced via PositionLimitModel weight cap."
                )
            else:
                logger.warning(
                    "[Backtester] max_position_size accepted for report metadata "
                    "only — not enforced (orders mode or custom risk_model supplied)."
                )
        if target_volatility is not None:
            logger.warning(
                "[Backtester] target_volatility accepted for report metadata "
                "only — volatility targeting is not enforced."
            )

        # ── Per-mode ignored-knob warning (only knobs EXPLICITLY set) ──
        ignored_knobs: List[str] = []
        if self.execution_mode == "weights":
            if slippage_model is not None:
                ignored_knobs.append("slippage_model")
            if commission_scheme is not None:
                ignored_knobs.append("commission_scheme")
            if fill_at != "open":
                ignored_knobs.append("fill_at")
            if max_volume_participation is not None:
                ignored_knobs.append("max_volume_participation")
            if contract_specs:
                ignored_knobs.append("contract_specs")
        else:
            if weight_cost_model_explicit:
                ignored_knobs.append("weight_cost_model")
            if rebalance_policy is not None:
                ignored_knobs.append("rebalance_policy")
            if risk_model is not None:
                ignored_knobs.append(
                    "risk_model (weights are rewritten but the order path never reads them)"
                )
        if ignored_knobs:
            logger.warning(
                f"[Backtester] execution_mode={self.execution_mode!r} ignores "
                f"configured knob(s): {', '.join(ignored_knobs)}"
            )

        # ── Universe (lazy-initialized on first access) ──
        self._universe = None

        # ── Components (lazy-loaded to avoid heavy import chains) ──
        self.ti = None   # initialized on first use
        self.blotter = None  # initialized on first use

        # Data placeholders
        self.portfolio_data: Optional[PortfolioData] = None
        self.instruments_data: Optional[InstrumentData] = None
        self.instrument_calculations = None
        self.portfolio_calculations = None
        self.session_id: Optional[str] = None
        self.dataset_id: Optional[str] = None
        self._run_started_at: Optional[str] = None
        self._timings: Dict[str, float] = {}

        logger.info(
            f"[Backtester] Initialized: strategy={strategy_name}, "
            f"instruments={self.instruments}, "
            f"version={BACKTESTER_VERSION}, api=QuantJourney Cloud"
        )

    # ─────────────────────────────────────────────────────────────────
    # Convenience properties — short aliases for strategy authors
    # ─────────────────────────────────────────────────────────────────

    @property
    def data(self) -> "InstrumentData":
        """Shortcut for self.instruments_data (e.g. self.data.close, self.data.SMA_50_close)."""
        return self.instruments_data

    @property
    def signals(self) -> "pd.DataFrame":
        """Current strategy's signals: self.signals == self.data.get_feature('strategies', name, 'signals')."""
        return self.instruments_data.get_feature("strategies", self.strategy_name, "signals")

    @property
    def weights(self) -> "pd.DataFrame":
        """Current strategy's weights: self.weights == self.data.get_feature('strategies', name, 'weights')."""
        return self.instruments_data.get_feature("strategies", self.strategy_name, "weights")

    @property
    def universe(self) -> "Universe":
        """Tradeable universe grid — dates × instruments.

        Provides pre-shaped factories and cached derivatives::

            signals = self.universe.zeros()
            months  = self.universe.periods("M")
            rets    = self.universe.returns
        """
        if self._universe is None:
            from backtester.universe import Universe
            close = self.instruments_data.get_feature("adj_close")
            self._universe = Universe(
                _close=close,
                _sectors=getattr(self, "_sector_map", {}),
            )
        return self._universe



    # ─────────────────────────────────────────────────────────────────
    # Data Processing — reconstruct PortfolioData from API response
    # ─────────────────────────────────────────────────────────────────

    async def _process_market_data(self) -> None:
        """
        Reconstruct PortfolioData + InstrumentData from the /bt/prepare
        JSON response, matching what MarketDataProcessor.prepare_data() produces.
        """
        data = self._api_response

        # Lazy imports to avoid heavy dependency chains
        from backtester.portfolio.portf_data import PortfolioData
        from backtester.portfolio.instr_data import InstrumentData
        from backtester.portfolio import PortfolioCalculations, InstrumentCalculations

        # 1) Rebuild prices DataFrame (MultiIndex: instrument, field)
        prices_df = _payload_to_multiindex_df(data["prices"])

        # 2) Rebuild metrics DataFrame
        metrics_df = _payload_to_multiindex_df(data["metrics"])

        # 3) Rebuild parameters DataFrame
        parameters_df = _payload_to_multiindex_df(data["parameters"])

        # 4) NAV series
        nav_series = _payload_to_series(data["nav"], name="nav")

        # Build instrument list from actual response
        instrument_names = data.get("instrument_names", list(
            dict.fromkeys(prices_df.columns.get_level_values(0))
        ))

        # 5) Group data
        group_data = pd.Series(
            ["equity"] * len(instrument_names),
            index=instrument_names,
            name="group",
        )

        # 6) Empty strategies DataFrame
        strategies_df = pd.DataFrame()

        # 7) Scale NAV to initial capital
        if len(nav_series) > 0:
            scale = self.initial_capital / nav_series.iloc[0] if nav_series.iloc[0] != 0 else 1.0
            nav_series = nav_series * scale

        # 8) Build InstrumentData
        self.instruments_data = InstrumentData(
            group_data=group_data,
            group_order=instrument_names,
            strategies=strategies_df,
            prices=prices_df,
            metrics=metrics_df,
            parameters=parameters_df,
            _skip_validation=self._lite_init,
        )

        # 9) Build PortfolioData
        self.portfolio_data = PortfolioData(
            instruments=self.instruments_data,
            net_asset_value=nav_series,
            _skip_initial_metrics=self._lite_init,
        )

        # 10) Process technical indicators (locally)
        if self.indicators_config:
            await self._compute_indicators()

        # 11) Initialize calculations
        self.instrument_calculations = InstrumentCalculations(self.instruments_data)
        self.portfolio_calculations = PortfolioCalculations(self.portfolio_data)

        logger.info(
            f"[Backtester] Data reconstructed: "
            f"{len(instrument_names)} instruments, {len(prices_df)} dates, "
            f"prices cols={list(prices_df.columns.get_level_values(1).unique())}"
        )

    # ─────────────────────────────────────────────────────────────────
    # Technical Indicators (computed locally on cloud-fetched data)
    # ─────────────────────────────────────────────────────────────────

    async def _compute_indicators(self) -> None:
        """
        Compute technical indicators for each instrument (locally).

        Adapts the indicators_config format::

            {"function": "SMA", "price_cols": ["close"], "params": {"periods": [50, 200]}}

        to the quantjourney_ti per-period API::

            ti.SMA(data=series, period=50)  ->  pd.Series named "SMA_50"
        """
        if self.ti is None:
            from quantjourney_ti import TechnicalIndicators
            self.ti = TechnicalIndicators()

        instruments = self.instruments_data.prices.columns.get_level_values(0).unique()

        for instrument in instruments:
            for ind_dict in self.indicators_config:
                try:
                    func_name = ind_dict.get("function")
                    price_cols = ind_dict.get("price_cols", ["close"])
                    params = ind_dict.get("params", {})

                    if not func_name:
                        logger.error(f"Missing 'function' in indicator config: {ind_dict}")
                        continue

                    indicator_func = getattr(self.ti, func_name, None)
                    if indicator_func is None:
                        logger.error(f"Unknown indicator function: {func_name}")
                        continue

                    # Resolve periods — single int or list
                    periods = params.get("periods", [])
                    if isinstance(periods, int):
                        periods = [periods]

                    for col in price_cols:
                        col_name = "adj_close" if col == "close" else col
                        try:
                            series = self.instruments_data.prices[instrument][col_name]
                        except KeyError:
                            logger.error(f"Missing {col_name} for {instrument}")
                            continue

                        if periods:
                            # Period-based indicators (SMA, EMA, RSI, etc.)
                            for period in periods:
                                result = indicator_func(data=series, period=period)
                                out_col = f"{func_name}_{period}_{col}"
                                if isinstance(result, pd.Series):
                                    self.instruments_data.parameters[(instrument, out_col)] = result
                                elif isinstance(result, pd.DataFrame):
                                    for rc in result.columns:
                                        self.instruments_data.parameters[(instrument, f"{rc}_{col}")] = result[rc]
                        else:
                            # Non-period indicators — pass remaining params as kwargs
                            extra = {k: v for k, v in params.items() if k != "periods"}
                            result = indicator_func(data=series, **extra)
                            if isinstance(result, pd.Series):
                                out_col = f"{func_name}_{col}"
                                self.instruments_data.parameters[(instrument, out_col)] = result
                            elif isinstance(result, pd.DataFrame):
                                for rc in result.columns:
                                    self.instruments_data.parameters[(instrument, f"{rc}_{col}")] = result[rc]

                    logger.info(
                        f"Stored features for {instrument}: "
                        f"{[k[1] for k in self.instruments_data.parameters.keys() if k[0] == instrument]}"
                    )

                except Exception as e:
                    logger.error(f"Error calculating {ind_dict.get('function', '?')} for {instrument}: {e}")
                    continue

    # ─────────────────────────────────────────────────────────────────
    # Abstract Strategy Methods
    # ─────────────────────────────────────────────────────────────────

    def _compute_signals(self) -> pd.DataFrame:
        """
        Compute strategy signals — override in child class.

        Return a DataFrame of signals (1=long, 0=flat, -1=short).
        For order-based strategies (execution_mode='orders'), return None
        to skip signal generation.
        """
        if self.execution_mode == "orders":
            return None
        raise NotImplementedError("Subclass must implement _compute_signals()")

    def _compute_weights(self) -> pd.DataFrame:
        """
        Compute strategy weights — override in child class.

        For order-based strategies, return None to skip.
        """
        if self.execution_mode == "orders":
            return None
        raise NotImplementedError("Subclass must implement _compute_weights()")

    def _compute_positions(self) -> None:
        """Compute strategy positions — override in child class."""
        pass

    def _compute_orders(
        self,
        date,
        bars: Dict[str, Any],
        current_positions: Dict[str, float],
        nav: float,
    ) -> None:
        """
        Submit orders to self.fill_engine after the current bar closes.

        NOTE next-bar semantics: pending orders are processed against a bar
        BEFORE _compute_orders() is called for that bar, so an order
        submitted here is filled on the NEXT bar at the earliest (market
        orders at the next bar's open/close per ``fill_at``).

        Override this in your strategy when using execution_mode="orders".
        Called once per bar with OHLCV data and current position state.

        Args:
            date:              current bar date (pd.Timestamp)
            bars:              dict of instrument → BarData
            current_positions: dict of instrument → current share quantity
            nav:               current net asset value
        """
        pass

    # ─────────────────────────────────────────────────────────────────
    # Order-mode convenience API
    # ─────────────────────────────────────────────────────────────────

    def _require_order_context(self) -> Dict[str, Any]:
        if self.execution_mode != "orders" or self.fill_engine is None:
            raise RuntimeError("Order helpers require execution_mode='orders'")
        if not self._order_context:
            raise RuntimeError("Order helpers can only be used inside _compute_orders(...)")
        return self._order_context

    def _order_bar(self, instrument: str):
        ctx = self._require_order_context()
        try:
            return ctx["bars"][instrument]
        except KeyError as exc:
            raise KeyError(f"No current bar for instrument {instrument!r}") from exc

    def bar(self, instrument: str):
        """Return the current BarData for an instrument inside _compute_orders."""
        return self._order_bar(instrument)

    def position(self, instrument: str) -> float:
        """Return current share/unit position inside _compute_orders."""
        ctx = self._require_order_context()
        return float(ctx["positions"].get(instrument, 0.0))

    def has_open_orders(self, instrument: Optional[str] = None) -> bool:
        """Return True when active orders exist, optionally for one instrument."""
        self._require_order_context()
        return any(
            order.is_active and (instrument is None or order.instrument == instrument)
            for order in self.fill_engine.pending_orders
        )

    def cancel_orders(self, instrument: Optional[str] = None) -> int:
        """Cancel active orders, optionally for one instrument."""
        self._require_order_context()
        return self.fill_engine.cancel_all(instrument=instrument)

    def feature(self, instrument: str, name: str, date=None):
        """Return one feature value for an instrument/date.

        The helper first looks for a regular InstrumentData feature such as
        ``adj_close`` or ``realized_vol_20d``. For common strategy outputs it
        also supports singular/plural aliases:
        ``signal``/``signals``, ``weight``/``weights`` and
        ``position``/``positions`` for the current strategy.
        """
        if date is None:
            date = self._require_order_context()["date"]

        feature_aliases = {
            "signal": "signals",
            "signals": "signals",
            "weight": "weights",
            "weights": "weights",
            "position": "positions",
            "positions": "positions",
        }

        errors = []
        for feature_name in (name, feature_aliases.get(name, name)):
            try:
                if feature_name in {"signals", "weights", "positions"}:
                    data = self.instruments_data.get_feature(
                        "strategies", self.strategy_name, feature_name
                    )
                else:
                    data = self.instruments_data.get_feature(feature_name)
                return data.loc[date, instrument]
            except Exception as exc:
                errors.append(exc)
                continue

        raise KeyError(
            f"Feature {name!r} for instrument {instrument!r} at {date!r} was not found"
        ) from errors[-1] if errors else None

    @staticmethod
    def _order_side_from_delta(quantity_delta: float):
        from backtester.execution import OrderSide

        return OrderSide.BUY if quantity_delta > 0 else OrderSide.SELL

    @staticmethod
    def _normalize_order_type(order_type):
        if order_type is None:
            return None
        from backtester.execution import OrderType

        if isinstance(order_type, OrderType):
            return order_type
        raw = str(order_type).strip()
        try:
            return OrderType(raw.lower())
        except ValueError:
            return OrderType[raw.upper()]

    def _submit_quantity_delta(
        self,
        instrument: str,
        quantity_delta: float,
        *,
        order_type=None,
        **order_kwargs,
    ) -> Optional[str]:
        from backtester.execution import Order, OrderType

        self._require_order_context()
        # Non-finite deltas (NaN/inf sizing inputs) must never become orders:
        # NaN quantity passes `qty <= 0` validation and poisons cash/NAV.
        if not np.isfinite(quantity_delta) or abs(quantity_delta) <= 1e-12:
            return None
        resolved_order_type = self._normalize_order_type(order_type) or OrderType.MARKET
        order = Order(
            instrument=instrument,
            side=self._order_side_from_delta(quantity_delta),
            quantity=abs(float(quantity_delta)),
            order_type=resolved_order_type,
            **order_kwargs,
        )
        return self.fill_engine.submit(order)

    def order_value(self, instrument: str, value: float, **order_kwargs) -> Optional[str]:
        """Submit a market order for a signed notional value."""
        bar = self._order_bar(instrument)
        if not np.isfinite(bar.close) or bar.close <= 0:
            return None
        quantity_delta = float(value) / float(bar.close)
        return self._submit_quantity_delta(instrument, quantity_delta, **order_kwargs)

    def order_percent(
        self,
        instrument: str,
        percent: Optional[float] = None,
        *,
        weight: Optional[float] = None,
        **order_kwargs,
    ) -> Optional[str]:
        """Submit a market order sized as signed percent of current NAV."""
        if percent is None:
            percent = weight
        if percent is None:
            raise ValueError("order_percent requires percent or weight")
        ctx = self._require_order_context()
        return self.order_value(instrument, float(ctx["nav"]) * float(percent), **order_kwargs)

    def order_target_value(self, instrument: str, target_value: float, **order_kwargs) -> Optional[str]:
        """Submit an order to move current exposure to target notional value."""
        ctx = self._require_order_context()
        bar = self._order_bar(instrument)
        current_qty = float(ctx["positions"].get(instrument, 0.0))
        current_value = current_qty * float(bar.close)
        return self.order_value(instrument, float(target_value) - current_value, **order_kwargs)

    def order_target_percent(
        self,
        instrument: str,
        target_percent: Optional[float] = None,
        *,
        target_weight: Optional[float] = None,
        **order_kwargs,
    ) -> Optional[str]:
        """Submit an order to move current exposure to target percent of NAV."""
        if target_percent is None:
            target_percent = target_weight
        if target_percent is None:
            raise ValueError("order_target_percent requires target_percent or target_weight")
        ctx = self._require_order_context()
        target_value = float(ctx["nav"]) * float(target_percent)
        return self.order_target_value(instrument, target_value, **order_kwargs)

    def target_percent(
        self,
        instrument: str,
        weight: Optional[float] = None,
        *,
        target_weight: Optional[float] = None,
        target_percent: Optional[float] = None,
        **order_kwargs,
    ) -> Optional[str]:
        """Alias for order_target_percent using portfolio-weight language."""
        if target_percent is None:
            target_percent = target_weight
        if target_percent is None:
            target_percent = weight
        if target_percent is None:
            raise ValueError("target_percent requires weight, target_weight or target_percent")
        return self.order_target_percent(
            instrument,
            target_percent=target_percent,
            **order_kwargs,
        )

    def close_position(self, instrument: str, **order_kwargs) -> Optional[str]:
        """Submit an order that flattens the current position."""
        return self.order_target_value(instrument, 0.0, **order_kwargs)

    def order_market(self, instrument: str, quantity: float, **order_kwargs) -> Optional[str]:
        """Submit a signed market quantity. Positive buys, negative sells."""
        from backtester.execution import OrderType

        return self._submit_quantity_delta(
            instrument,
            float(quantity),
            order_type=OrderType.MARKET,
            **order_kwargs,
        )

    def limit_percent(
        self,
        instrument: str,
        *,
        limit_price: float,
        weight: Optional[float] = None,
        percent: Optional[float] = None,
        **order_kwargs,
    ) -> Optional[str]:
        """Submit a limit order sized as signed percent of current NAV."""
        from backtester.execution import OrderType

        if percent is None:
            percent = weight
        if percent is None:
            raise ValueError("limit_percent requires percent or weight")
        bar = self._order_bar(instrument)
        if not np.isfinite(bar.close) or bar.close <= 0:
            return None
        value = float(self._require_order_context()["nav"]) * float(percent)
        quantity_delta = value / float(bar.close)
        return self._submit_quantity_delta(
            instrument,
            quantity_delta,
            order_type=OrderType.LIMIT,
            limit_price=float(limit_price),
            **order_kwargs,
        )

    def _active_order(self, order_id: Optional[str]):
        """Return the still-active pending Order with this id, or None."""
        if not order_id or self.fill_engine is None:
            return None
        for order in self.fill_engine.pending_orders:
            if order.order_id == order_id:
                return order
        return None

    def _oco_link_protective_exit(self, instrument: str, kind: str, order_kwargs: Dict[str, Any]) -> None:
        """
        Auto-link stop_loss()/take_profit() exits on the same instrument as an
        OCO pair, so a single wide bar (low <= stop and high >= limit) fills
        only one of them instead of both flipping the position to an
        unintended short. Skipped when the caller passes oco_pair_id itself.
        Mutates order_kwargs in place; single-helper usage is unchanged.
        """
        if "oco_pair_id" in order_kwargs:
            return
        other_kind = "take_profit" if kind == "stop_loss" else "stop_loss"
        sibling_id = self._protective_exit_ids.get(instrument, {}).get(other_kind)
        sibling = self._active_order(sibling_id)
        if sibling is None:
            return
        if sibling.oco_pair_id is None:
            sibling.oco_pair_id = str(uuid.uuid4())
            # Retroactive registration: FillEngine.submit() only records
            # oco_pair_ids present at submit time.
            self.fill_engine._oco_pairs.setdefault(
                sibling.oco_pair_id, []
            ).append(sibling.order_id)
        order_kwargs["oco_pair_id"] = sibling.oco_pair_id

    def _track_protective_exit(self, instrument: str, kind: str, order_id: Optional[str]) -> None:
        if order_id is not None:
            self._protective_exit_ids.setdefault(instrument, {})[kind] = order_id

    def stop_loss(
        self,
        instrument: str,
        *,
        stop_loss: Optional[float] = None,
        sl: Optional[float] = None,
        stop_price: Optional[float] = None,
        quantity: Optional[float] = None,
        **order_kwargs,
    ) -> Optional[str]:
        """Attach a stop-loss order to the current position."""
        from backtester.execution import OrderType

        pos = self.position(instrument)
        if abs(pos) <= 1e-12:
            if self.has_open_orders(instrument):
                logger.warning(
                    f"[Backtester] stop_loss({instrument!r}) called with no open "
                    "position but a pending order for that instrument — the stop "
                    "was NOT placed. Use bracket_percent() to attach protective "
                    "exits to an entry order."
                )
            return None
        if quantity is None:
            quantity = abs(pos)
        if stop_price is None:
            if stop_loss is None:
                stop_loss = sl
            if stop_loss is None:
                raise ValueError("stop_loss requires stop_loss/sl or stop_price")
            entry = self.get_average_entry_price(instrument)
            if entry is None:
                raise ValueError(f"No average entry price available for {instrument}")
            stop_price = entry * (1.0 - float(stop_loss) if pos > 0 else 1.0 + float(stop_loss))
        quantity_delta = -float(quantity) if pos > 0 else float(quantity)
        order_kwargs = dict(order_kwargs)
        self._oco_link_protective_exit(instrument, "stop_loss", order_kwargs)
        order_id = self._submit_quantity_delta(
            instrument,
            quantity_delta,
            order_type=OrderType.STOP,
            stop_price=float(stop_price),
            **order_kwargs,
        )
        self._track_protective_exit(instrument, "stop_loss", order_id)
        return order_id

    def take_profit(
        self,
        instrument: str,
        *,
        take_profit: Optional[float] = None,
        tp: Optional[float] = None,
        limit_price: Optional[float] = None,
        quantity: Optional[float] = None,
        **order_kwargs,
    ) -> Optional[str]:
        """Attach a take-profit limit order to the current position."""
        from backtester.execution import OrderType

        pos = self.position(instrument)
        if abs(pos) <= 1e-12:
            if self.has_open_orders(instrument):
                logger.warning(
                    f"[Backtester] take_profit({instrument!r}) called with no open "
                    "position but a pending order for that instrument — the "
                    "take-profit was NOT placed. Use bracket_percent() to attach "
                    "protective exits to an entry order."
                )
            return None
        if quantity is None:
            quantity = abs(pos)
        if limit_price is None:
            if take_profit is None:
                take_profit = tp
            if take_profit is None:
                raise ValueError("take_profit requires take_profit/tp or limit_price")
            entry = self.get_average_entry_price(instrument)
            if entry is None:
                raise ValueError(f"No average entry price available for {instrument}")
            limit_price = entry * (1.0 + float(take_profit) if pos > 0 else 1.0 - float(take_profit))
        quantity_delta = -float(quantity) if pos > 0 else float(quantity)
        order_kwargs = dict(order_kwargs)
        self._oco_link_protective_exit(instrument, "take_profit", order_kwargs)
        order_id = self._submit_quantity_delta(
            instrument,
            quantity_delta,
            order_type=OrderType.LIMIT,
            limit_price=float(limit_price),
            **order_kwargs,
        )
        self._track_protective_exit(instrument, "take_profit", order_id)
        return order_id

    def bracket_percent(
        self,
        instrument: str,
        *,
        weight: float,
        tp: Optional[float] = None,
        sl: Optional[float] = None,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        **order_kwargs,
    ) -> Optional[str]:
        """Submit a bracket order sized as percent of NAV.

        ``tp`` and ``sl`` are relative percentages from the actual entry fill
        price. For example, ``tp=0.10`` and ``sl=0.05`` create +10%
        take-profit and -5% stop-loss levels for a long entry.
        """
        from backtester.execution import BracketSpec, OrderType

        if tp is None:
            tp = take_profit
        if sl is None:
            sl = stop_loss

        bar = self._order_bar(instrument)
        qty_value = float(self._require_order_context()["nav"]) * float(weight)
        if not np.isfinite(bar.close) or bar.close <= 0 or abs(qty_value) <= 1e-12:
            return None

        if (take_profit_price is None and tp is None) or (stop_loss_price is None and sl is None):
            raise ValueError(
                "bracket_percent requires take_profit/stop_loss, tp/sl, "
                "or explicit take_profit_price/stop_loss_price"
            )

        quantity_delta = qty_value / float(bar.close)
        return self._submit_quantity_delta(
            instrument,
            quantity_delta,
            order_type=OrderType.BRACKET,
            bracket=BracketSpec(
                take_profit_price=None if take_profit_price is None else round(float(take_profit_price), 8),
                stop_loss_price=None if stop_loss_price is None else round(float(stop_loss_price), 8),
                take_profit_pct=None if tp is None else float(tp),
                stop_loss_pct=None if sl is None else float(sl),
            ),
            **order_kwargs,
        )

    def get_average_entry_price(self, instrument: str) -> Optional[float]:
        """Return current average entry price for an order-mode position."""
        value = self.average_entry_price.get(instrument)
        return None if value is None else float(value)

    def avg_entry_price(self, instrument: str) -> Optional[float]:
        """Alias for get_average_entry_price(...)."""
        return self.get_average_entry_price(instrument)

    # ─────────────────────────────────────────────────────────────────
    # Signal / Weight / Position Pipeline
    # Standalone cloud-backed operation.
    # ─────────────────────────────────────────────────────────────────

    def _generate_signals(self) -> None:
        """Generate and validate strategy signals."""
        signals = self._compute_signals()
        if signals is None:
            return  # order-based strategies may skip signals
        if not signals.ne(0).any().any():
            if self.execution_mode == "orders":
                logger.info("[orders mode] Signals are zero — orders drive execution.")
                return
            logger.warning("All signals are zero — no trades to execute.")
            raise ValueError("No valid trading signals generated.")

        self._validate_signals(signals)
        self.instruments_data.add_strategy_data(self.strategy_name, "signals", signals)

    def _validate_signals(self, signals: pd.DataFrame) -> None:
        if not isinstance(signals, pd.DataFrame):
            raise ValueError("Signals must be a DataFrame")
        if signals.isna().any().any():
            raise ValueError("Signals contain NaN values")

    def _generate_weights(self) -> None:
        """Generate and validate strategy weights."""
        weights = self._compute_weights()
        if weights is not None and not weights.empty:
            self.instruments_data.add_strategy_data(self.strategy_name, "weights", weights)

    def _apply_risk_model(self) -> None:
        """Apply pluggable risk model to adjust weights before rebalance.

        Pipeline: signals → weights → **risk_model.adjust()** → rebalance → execution

        If no risk_model is set, this is a no-op.
        """
        if self._risk_model is None:
            return

        weights = self.instruments_data.get_feature(
            "strategies", self.strategy_name, "weights"
        )
        if weights is None or weights.empty:
            return

        close = self.instruments_data.get_feature("adj_close")
        returns = self._compute_returns_preserve_gaps(close)

        # Pass the sector map so sector_limits in PositionLimitModel (which
        # require metadata={"sectors": {instrument: sector}}) can actually fire.
        sector_map = getattr(self, "_sector_map", None)
        metadata = {"sectors": sector_map} if sector_map else None
        adjusted = self._risk_model.adjust(weights, returns, metadata=metadata)

        # Overwrite weights with risk-adjusted version
        # Drop existing weight columns first (add_strategy_data appends, not replaces)
        strats = self.instruments_data.strategies
        if not strats.empty:
            keep_mask = ~(
                (strats.columns.get_level_values(0) == self.strategy_name)
                & (strats.columns.get_level_values(1) == "weights")
            )
            self.instruments_data.strategies = strats.loc[:, keep_mask]

        self.instruments_data.add_strategy_data(
            self.strategy_name, "weights", adjusted
        )
        logger.info(f"[Risk] Applied {self._risk_model} to weights")

    def _generate_positions(self) -> None:
        """Generate and validate strategy positions."""
        positions = self._compute_positions()
        if positions is not None:
            self.instruments_data.add_strategy_data(self.strategy_name, "positions", positions)

    # ─────────────────────────────────────────────────────────────────
    # Performance Computation
    # Weight-based and order-based backtest accounting paths.
    # ─────────────────────────────────────────────────────────────────

    def _compute_strategy_performance(self) -> None:
        """
        Day-by-day performance: NAV, returns, positions, trades.
        Dispatches to order-based or weight-based engine.
        """
        # Make the benchmark_returns kwarg visible to the rebalance engine's
        # tracking-error trigger (portfolio_data.benchmark_returns).
        if self._benchmark_returns is not None and self.portfolio_data is not None:
            self.portfolio_data.benchmark_returns = self._benchmark_returns
        if self.execution_mode == "orders" and self.fill_engine is not None:
            return self._compute_performance_order_based()
        return self._compute_performance_weight_based()

    def _compute_performance_order_based(self) -> None:
        """
        Day-by-day performance using FillEngine for order execution.
        Strategy submits orders via _compute_orders(); FillEngine processes
        them against OHLCV bars with slippage + commission.
        """
        from backtester.execution.order_types import BarData, OrderSide

        # Re-running on the same instance must not inherit pending orders,
        # fill history or commission state from a previous run.
        self.fill_engine.reset()
        if self.blotter is not None:
            self.blotter.reset()
        self._order_context = {}
        self.average_entry_price = {}
        self.current_positions_meta = {}
        self._protective_exit_ids = {}
        if self.portfolio_data is not None:
            self.portfolio_data.average_entry_price = {}
            self.portfolio_data.current_positions_meta = {}
            if hasattr(self.portfolio_data, "_protective_exit_ids"):
                self.portfolio_data._protective_exit_ids = {}

        def _is_valid_price(value: Any) -> bool:
            try:
                return bool(pd.notna(value) and np.isfinite(float(value)) and float(value) > 0.0)
            except Exception:
                return False

        def _is_valid_bar(bar: BarData) -> bool:
            return all(
                _is_valid_price(value)
                for value in (bar.open, bar.high, bar.low, bar.close)
            )

        close_prices = self._instrument_price_frame("close", fallback="adj_close")
        instruments = close_prices.columns.tolist()
        all_dates = close_prices.index

        # Try to get full OHLCV; fall back to close ONLY for genuinely missing
        # fields (KeyError/AttributeError). Any other failure re-raises: a
        # blanket fallback silently disables intrabar stop/limit detection.
        degraded_fields: List[str] = []

        def _feature_or_close(field: str) -> pd.DataFrame:
            try:
                return self.instruments_data.get_feature(field)
            except (KeyError, AttributeError):
                degraded_fields.append(field)
                return close_prices

        open_prices = _feature_or_close("open")
        high_prices = _feature_or_close("high")
        low_prices = _feature_or_close("low")
        if degraded_fields and not getattr(self, "_warned_ohlc_degraded", False):
            self._warned_ohlc_degraded = True
            logger.warning(
                "[Backtester] OHLC unavailable (missing: "
                f"{', '.join(degraded_fields)}) — bars degraded to close-only; "
                "stop/limit intrabar triggers disabled."
            )
        try:
            volume_data = self.instruments_data.get_feature("volume")
        except Exception:
            # No volume feature: mark volume as unknown (NaN) instead of
            # fabricating 0.0, so a participation cap fails loudly downstream
            # rather than silently producing zero fills forever.
            volume_data = pd.DataFrame(np.nan, index=all_dates, columns=instruments)
            if getattr(self.fill_engine, "max_volume_participation", None):
                logger.warning(
                    "[Backtester] max_volume_participation is enabled but the "
                    "dataset has no 'volume' feature; bar volume is unknown "
                    "(NaN) and volume-capped orders cannot fill."
                )

        nav_series = pd.Series(index=all_dates, dtype=float)
        positions_df = pd.DataFrame(0.0, index=all_dates, columns=instruments)
        avg_entry_df = pd.DataFrame(np.nan, index=all_dates, columns=instruments)
        cash = float(self.initial_capital)
        current_pos = {inst: 0.0 for inst in instruments}
        avg_entry_price: Dict[str, Optional[float]] = {inst: None for inst in instruments}
        last_valid_price: Dict[str, Optional[float]] = {inst: None for inst in instruments}

        for i, date in enumerate(all_dates):
            # 1) Build BarData for each instrument
            bars: Dict[str, BarData] = {}
            for inst in instruments:
                close_value = close_prices.loc[date, inst]
                if _is_valid_price(close_value):
                    last_valid_price[inst] = float(close_value)
                bars[inst] = BarData(
                    timestamp=date,
                    open=float(open_prices.loc[date, inst]),
                    high=float(high_prices.loc[date, inst]),
                    low=float(low_prices.loc[date, inst]),
                    close=float(close_prices.loc[date, inst]),
                    volume=float(volume_data.loc[date, inst]) if inst in volume_data.columns else float("nan"),
                )

            # 2) Process pending orders against this bar
            for inst in instruments:
                if not _is_valid_bar(bars[inst]):
                    continue
                fills = self.fill_engine.process_bar(inst, bars[inst])
                if fills and self.blotter is None:
                    from backtester.engines.blotter import Blotter
                    self.blotter = Blotter()
                for fill in fills:
                    previous_pos = current_pos.get(fill.instrument, 0.0)
                    qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
                    current_pos[fill.instrument] += qty
                    avg_entry_price[fill.instrument] = self._updated_average_entry_price(
                        previous_position=previous_pos,
                        previous_avg_price=avg_entry_price.get(fill.instrument),
                        signed_fill_qty=qty,
                        fill_price=float(fill.fill_price),
                    )
                    cost = self._trade_notional(fill.instrument, fill.quantity, fill.fill_price)
                    # Inverse contracts: mirror the trade cash flow so that,
                    # together with the negated position value in
                    # _signed_position_value, the daily NAV delta equals
                    # ContractSpec.pnl = qty*mult*(1/p_entry - 1/p_exit).
                    signed_cost = -cost if self._contract_spec(fill.instrument).inverse else cost
                    if fill.side == OrderSide.BUY:
                        cash -= signed_cost + fill.commission
                    else:
                        cash += signed_cost - fill.commission
                    self.blotter.record_trade(
                        order_id=fill.order_id,
                        instrument=fill.instrument,
                        side=fill.side.value,
                        quantity=float(fill.quantity),
                        price=float(fill.fill_price),
                        trade_value=float(cost),
                        timestamp=fill.timestamp,
                        transaction_cost=float(fill.commission),
                        slippage=float(fill.slippage),
                        theoretical_price=(
                            None
                            if fill.theoretical_price is None
                            else float(fill.theoretical_price)
                        ),
                        fill_status=fill.order_status.value,
                    )

            # 3) Compute NAV
            position_value = 0.0
            position_values = {}
            for inst in instruments:
                qty = float(current_pos.get(inst, 0.0))
                if abs(qty) <= 1e-12:
                    position_values[inst] = 0.0
                    continue
                price = close_prices.loc[date, inst]
                if not _is_valid_price(price):
                    price = last_valid_price.get(inst) or avg_entry_price.get(inst)
                if _is_valid_price(price):
                    value = self._signed_position_value(inst, qty, float(price))
                    position_values[inst] = value
                    position_value += value
                else:
                    position_values[inst] = 0.0
            nav_series.iloc[i] = cash + position_value

            # 4) Let strategy submit new orders for next bar
            self.average_entry_price = dict(avg_entry_price)
            self.current_positions_meta = {
                inst: {
                    "quantity": float(current_pos.get(inst, 0.0)),
                    "avg_price": avg_entry_price.get(inst),
                }
                for inst in instruments
            }
            if self.portfolio_data is not None:
                self.portfolio_data.average_entry_price = dict(avg_entry_price)
                self.portfolio_data.current_positions_meta = dict(self.current_positions_meta)
            self._order_context = {
                "date": date,
                "bars": bars,
                "positions": dict(current_pos),
                "nav": float(nav_series.iloc[i]),
            }
            self._compute_orders(date, bars, dict(current_pos), nav_series.iloc[i])

            # 5) Record positions
            for inst in instruments:
                positions_df.loc[date, inst] = current_pos[inst]
                avg_entry_df.loc[date, inst] = (
                    np.nan if avg_entry_price.get(inst) is None else avg_entry_price[inst]
                )

        # Store results
        self._order_context = {}
        self.portfolio_data.update_net_asset_value(nav_series)
        self.portfolio_data.update_positions(positions_df)
        position_values_df = self._position_values_from_units(positions_df, close_prices)
        self.portfolio_data.position_values = position_values_df
        self.portfolio_data.update_weights(
            self._weights_from_position_values(position_values_df, nav_series)
        )
        raw_returns = nav_series.pct_change()
        daily_returns = raw_returns.fillna(0.0)
        self.portfolio_data.returns = daily_returns
        self.portfolio_data.returns_for_metrics = raw_returns.dropna()
        self.portfolio_data.average_entry_price = dict(avg_entry_price)
        self.portfolio_data.average_entry_price_history = avg_entry_df

    def _contract_spec(self, instrument: str) -> ContractSpec:
        key = str(instrument).upper()
        return self.contract_specs.get(key, get_contract_spec(key))

    def _trade_notional(self, instrument: str, quantity: float, price: float) -> float:
        return self._contract_spec(instrument).notional(quantity, price)

    @staticmethod
    def _compute_returns_preserve_gaps(prices: pd.DataFrame) -> pd.DataFrame:
        returns = prices.pct_change(fill_method=None)
        if len(returns) > 0:
            first_idx = returns.index[0]
            first_available = prices.loc[first_idx].notna()
            returns.loc[first_idx, first_available] = 0.0
        return returns

    @staticmethod
    def _compute_accounting_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Returns for NAV/PnL accounting: bridge price moves across NaN gaps.

        Bars INSIDE a gap (price is NaN) stay NaN, so availability masking
        keeps the instrument un-rebalanceable while the market is dark. The
        first bar AFTER the gap carries the bridged return
        ``price[resume] / last_valid_price - 1`` so the whole gap move books
        on the resume bar — matching the orders-mode convention of marking a
        frozen position at its last valid price. Leading NaNs (instrument not
        yet trading) remain NaN.
        """
        filled = prices.ffill()
        returns = filled.pct_change(fill_method=None).mask(prices.isna())
        if len(returns) > 0:
            first_idx = returns.index[0]
            first_available = prices.loc[first_idx].notna()
            returns.loc[first_idx, first_available] = 0.0
        return returns

    @staticmethod
    def _weights_from_position_values(
        position_values: pd.DataFrame, nav: pd.Series
    ) -> pd.DataFrame:
        """Weights = position value / NAV, with ±inf (NAV ~ 0) cleaned to 0."""
        return (
            position_values.divide(nav, axis=0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    def _signed_position_value(self, instrument: str, quantity: float, price: float) -> float:
        spec = self._contract_spec(instrument)
        if spec.inverse:
            # Convention: an inverse position is marked at the NEGATIVE coin
            # notional (-qty*mult/price). A long inverse contract gains when
            # price rises (1/price falls), so combined with the mirrored trade
            # cash flow in the fill loop the per-bar NAV delta equals
            # qty*mult*(1/p_prev - 1/p_curr) — same sign and magnitude as
            # ContractSpec.pnl and calc/pnl_multi_asset.compute_position_pnl.
            return float(-np.sign(quantity) * spec.notional(quantity, price))
        return float(quantity) * float(price) * float(spec.multiplier) * float(spec.lot_size)

    def _position_values_from_units(self, positions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        # Mark at the last valid price through NaN gaps (ffill) so reported
        # exposure matches the NAV valuation convention: the order-mode NAV
        # loop carries a gapped position at last_valid_price, so weights /
        # position_values must not collapse to 0 while NAV stays invested.
        marked_prices = prices.ffill()
        values = pd.DataFrame(0.0, index=positions.index, columns=positions.columns)
        for inst in positions.columns:
            spec = self._contract_spec(inst)
            if spec.inverse:
                # Same negative-coin-notional convention as _signed_position_value.
                values[inst] = -positions[inst].multiply(
                    spec.multiplier / marked_prices[inst].replace(0.0, np.nan)
                )
            else:
                values[inst] = positions[inst].multiply(marked_prices[inst]).multiply(
                    float(spec.multiplier) * float(spec.lot_size)
                )
        return values.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _instrument_price_frame(self, field: str, *, fallback: str) -> pd.DataFrame:
        try:
            return self.instruments_data.get_feature(field)
        except Exception:
            return self.instruments_data.get_feature(fallback)

    @staticmethod
    def _updated_average_entry_price(
        *,
        previous_position: float,
        previous_avg_price: Optional[float],
        signed_fill_qty: float,
        fill_price: float,
    ) -> Optional[float]:
        """Update average entry price after a signed fill."""
        new_position = previous_position + signed_fill_qty
        if abs(new_position) <= 1e-12:
            return None

        if abs(previous_position) <= 1e-12:
            return fill_price

        same_direction = (previous_position > 0 and signed_fill_qty > 0) or (
            previous_position < 0 and signed_fill_qty < 0
        )
        if same_direction:
            previous_abs = abs(previous_position)
            fill_abs = abs(signed_fill_qty)
            previous_avg = fill_price if previous_avg_price is None else previous_avg_price
            return ((previous_avg * previous_abs) + (fill_price * fill_abs)) / (
                previous_abs + fill_abs
            )

        # Reducing an existing position keeps the original average price.
        if (previous_position > 0 and new_position > 0) or (previous_position < 0 and new_position < 0):
            return previous_avg_price

        # Reversal: residual position starts at the reversal fill price.
        return fill_price

    def _compute_performance_weight_based(self) -> None:
        """
        Weight-based performance WITH conditional rebalancing.

        Between rebalance dates, positions are held constant and weights
        drift naturally with prices.  On rebalance dates, weights snap
        back to targets and trades are recorded for the delta.

        The RebalanceEngine evaluates the RebalancePolicy (calendar,
        drift, signal-change, circuit-breaker, cost-gate) to decide
        which days to actually trade.

        Default policy (frequency="D") reproduces the legacy daily-
        rebalance behaviour exactly.
        """
        from backtester.portfolio.rebalance import RebalanceEngine

        target_weights = self.instruments_data.get_feature(
            "strategies", self.strategy_name, "weights"
        )
        close = self.instruments_data.get_feature("adj_close")

        # Shift weights forward by 1 day to prevent look-ahead bias
        target_weights = target_weights.shift(1).fillna(0.0)

        all_dates = target_weights.index

        raw_cash_buffer = self.portfolio_data.cash_buffer
        # Accept float, int (cash_buffer=0 means "no buffer") and pd.Series
        # (per-date buffer, matching the documented Union[float, pd.Series]).
        # bool is an int subclass but True/False are never a valid buffer
        # fraction (True would mean a nonsensical 100% buffer), so bools fall
        # through to the 5% default (long-standing documented behaviour).
        # Any other type raises instead of silently running with 5%.
        if isinstance(raw_cash_buffer, pd.Series):
            # Per-date buffer: align to the run dates, forward-fill between
            # provided dates, default 5% before the series starts.
            cash_buffer = (
                normalize_time_index_like(
                    pd.to_numeric(raw_cash_buffer, errors="coerce"),
                    all_dates,
                )
                .reindex(all_dates)
                .ffill()
                .fillna(0.05)
            )
        elif raw_cash_buffer is None or isinstance(raw_cash_buffer, bool):
            cash_buffer = 0.05
        elif isinstance(raw_cash_buffer, (int, float, np.integer, np.floating)):
            cash_buffer = float(raw_cash_buffer)
        else:
            raise TypeError(
                "cash_buffer must be a float fraction or pd.Series indexed by "
                f"date, got {type(raw_cash_buffer).__name__}"
            )

        # Apply cash buffer to target weights (row-wise when per-date Series)
        if isinstance(cash_buffer, pd.Series):
            target_weights_inv = target_weights.multiply(1.0 - cash_buffer, axis=0)
        else:
            target_weights_inv = target_weights * (1.0 - cash_buffer)

        # Asset returns for NAV/PnL accounting: NaN only INSIDE a gap, with
        # the resume-bar return bridged across it. The rebalance engine treats
        # in-gap NaN bars as untradeable and keeps existing positions frozen.
        accounting_returns = self._compute_accounting_returns(close)

        # Benchmark returns for tracking-error trigger (if available)
        benchmark_returns = getattr(self.portfolio_data, "benchmark_returns", None)
        if (
            benchmark_returns is None
            and getattr(self._rebalance_policy, "tracking_error_threshold", None) is not None
        ):
            logger.warning(
                "[Backtester] tracking_error_threshold is configured on the "
                "rebalance policy but no benchmark_returns were provided "
                "(Backtester(benchmark_returns=...)) — the tracking-error "
                "trigger is INACTIVE for this run."
            )

        # ── Run RebalanceEngine ──
        # NaNs in the returns frame mark instruments as unavailable, so
        # passing accounting_returns keeps rebalancing INTO an asset blocked
        # during its gap while making the bridged resume-bar return visible.
        engine = RebalanceEngine(self._rebalance_policy)
        actual_weights, rebal_flags = engine.run(
            target_weights_inv, accounting_returns,
            benchmark_returns=benchmark_returns,
        )

        # Log rebalance stats
        stats = engine.stats
        extra_parts = []
        if stats.get('tracking_error_count', 0):
            extra_parts.append(f"te={stats['tracking_error_count']}")
        if stats.get('partial_positions_saved', 0):
            extra_parts.append(f"partial_saved={stats['partial_positions_saved']}")
        extra = f" | {', '.join(extra_parts)}" if extra_parts else ""
        logger.info(
            f"[Rebalance] {stats['rebalance_count']} rebalances "
            f"(cal={stats['calendar_count']}, drift={stats['drift_count']}, "
            f"sig={stats['signal_count']}, cb={stats['circuit_breaker_count']})"
            f"{extra} "
            f"| avg {stats['avg_days_between']:.1f} days between | "
            f"policy={self._rebalance_policy}"
        )

        # ── Portfolio daily returns from weights that earned each bar's return ──
        # RebalanceEngine.actual_weights is the realised/end-of-bar exposure
        # state. return_weights is the beginning-of-bar exposure used for P&L.
        return_weights = getattr(engine, "return_weights", actual_weights)
        portfolio_returns = getattr(engine, "portfolio_returns", None)
        if portfolio_returns is None or len(portfolio_returns) == 0:
            portfolio_returns = (
                return_weights * accounting_returns.fillna(0.0)
            ).sum(axis=1)

        # ── Transaction costs: implied trades, only on rebalance days ──
        # Weight mode has no explicit orders, so the accounting layer infers
        # shares from target weights, pre-cost NAV and prices, then charges
        # costs on the implied share deltas.
        gross_nav_series = self.initial_capital * (1.0 + portfolio_returns).cumprod()
        gross_nav_series.iloc[0] = self.initial_capital
        cost_breakdown = self.weight_cost_model.compute(
            actual_weights=actual_weights,
            prices=close,
            nav=gross_nav_series,
            rebalance_flags=rebal_flags,
        )
        txn_cost_pct = cost_breakdown.total_cost_pct

        # ── NAV via cumprod ──
        net_returns = portfolio_returns - txn_cost_pct
        nav_series = self.initial_capital * (1.0 + net_returns).cumprod()
        nav_series.iloc[0] = self.initial_capital

        # ── Positions (shares) = weight × NAV / price ──
        marked_close = close.ffill()
        positions = actual_weights.multiply(nav_series, axis=0).divide(
            marked_close.replace(0.0, np.nan), axis=1
        ).fillna(0.0)
        position_changes = positions.diff().fillna(positions)
        # Zero out position changes on non-rebalance days for trade recording
        position_changes_trades = position_changes.copy()
        position_changes_trades[~rebal_flags] = 0.0

        # Store results
        self.portfolio_data.update_net_asset_value(nav_series)
        self.portfolio_data.update_positions(positions)
        self.portfolio_data.update_weights(
            self._weights_from_position_values(
                positions.multiply(marked_close), nav_series
            )
        )

        daily_returns = nav_series.pct_change().fillna(0.0)
        self.portfolio_data.returns = daily_returns
        self.portfolio_data.total_transaction_costs = cost_breakdown.total_cost
        self._weight_cost_breakdown = cost_breakdown

        # Store rebalance flags + stats on portfolio data
        self.portfolio_data.rebalance_flags = rebal_flags
        self._rebalance_stats = stats

        # Record trades (only rebalance-day changes)
        self._record_trades(
            position_changes_trades,
            close,
            all_dates,
            transaction_costs=cost_breakdown.transaction_costs,
        )

    def _record_trades(
        self,
        position_changes: pd.DataFrame,
        prices: pd.DataFrame,
        dates: pd.DatetimeIndex,
        transaction_costs: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Vectorized trade recording."""
        trade_mask = position_changes != 0
        trades_list = []

        for instrument in position_changes.columns:
            instrument_trades = position_changes[instrument][trade_mask[instrument]]
            if len(instrument_trades) == 0:
                continue

            trade_dates = instrument_trades.index
            trade_prices = prices.loc[trade_dates, instrument]

            trades_df = pd.DataFrame({
                "Timestamp": trade_dates,
                "Instrument": instrument,
                "Side": np.where(instrument_trades > 0, "buy", "sell"),
                "Quantity": abs(instrument_trades),
                "Price": trade_prices,
                "TradeValue": abs(instrument_trades * trade_prices),
            })
            if transaction_costs is not None:
                costs = transaction_costs.reindex(index=trade_dates, columns=[instrument])[instrument]
                trades_df["TransactionCost"] = costs.fillna(0.0).to_numpy()
            trades_list.append(trades_df)

        if not trades_list:
            return pd.DataFrame(columns=["Timestamp", "Instrument", "Side", "Quantity", "Price", "TradeValue"])

        all_trades = pd.concat(trades_list, axis=0).reset_index(drop=True)
        all_trades["OrderID"] = [f"order_{i}" for i in range(len(all_trades))]
        all_trades["TradeID"] = [f"trade_{i}" for i in range(len(all_trades))]
        if "TransactionCost" not in all_trades:
            all_trades["TransactionCost"] = all_trades["TradeValue"] * self.TRANSACTION_COST_BPS
        all_trades = all_trades.sort_values("Timestamp").reset_index(drop=True)
        if self.blotter is None:
            from backtester.engines.blotter import Blotter
            self.blotter = Blotter()
        self.blotter.record_trades_bulk(all_trades)
        return all_trades

    # ─────────────────────────────────────────────────────────────────
    # Main Entry Point
    # ─────────────────────────────────────────────────────────────────

    async def run_strategy(self) -> None:
        """
        Run the full cloud backtesting pipeline:
          1. Authenticate + fetch data from API
          2. Reconstruct PortfolioData
          3. Compute indicators, signals, weights
          4. Compute performance
          5. Generate analysis + archive
        """
        total_started = time.perf_counter()
        self._timings = {}
        logger.info(f"═══ Running Cloud Strategy: {self.strategy_name} ═══")
        self._run_started_at = datetime.now(timezone.utc).isoformat()

        # Data Operations (API-backed)
        stage_started = time.perf_counter()
        try:
            await self._fetch_market_data()
        except Exception as e:
            self._timings["data_fetch_seconds"] = time.perf_counter() - stage_started
            self._timings["total_seconds"] = time.perf_counter() - total_started
            logger.error(
                f"[Backtester] Could not fetch market data — skipping strategy.\n"
                f"  Error: {e}"
            )
            if self._strict_data_fetch:
                raise
            return
        self._timings["data_fetch_seconds"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        await self._process_market_data()
        self._timings["data_processing_seconds"] = time.perf_counter() - stage_started

        # Strategy Computations (local)
        stage_started = time.perf_counter()
        self._generate_signals()
        self._generate_weights()
        self._apply_risk_model()
        self._generate_positions()

        # Performance (dispatches to order-based or weight-based)
        self._compute_strategy_performance()
        self._timings["calculation_seconds"] = time.perf_counter() - stage_started

        # Reports & Archive
        stage_started = time.perf_counter()
        if not self._skip_analysis:
            await self._generate_strategy_analysis()
        self._timings["reporting_seconds"] = time.perf_counter() - stage_started

        self._timings["total_before_archive_seconds"] = time.perf_counter() - total_started
        stage_started = time.perf_counter()
        await self._archive_strategy_data()
        self._timings["archive_seconds"] = time.perf_counter() - stage_started
        self._timings["total_seconds"] = time.perf_counter() - total_started

        logger.info(
            "[Backtester] Runtime: "
            f"fetch={self._timings.get('data_fetch_seconds', 0.0):.2f}s | "
            f"data={self._timings.get('data_processing_seconds', 0.0):.2f}s | "
            f"calc={self._timings.get('calculation_seconds', 0.0):.2f}s | "
            f"reports={self._timings.get('reporting_seconds', 0.0):.2f}s | "
            f"archive={self._timings.get('archive_seconds', 0.0):.2f}s | "
            f"total={self._timings.get('total_seconds', 0.0):.2f}s"
        )
        logger.info(f"═══ Cloud Strategy {self.strategy_name} completed ═══")
