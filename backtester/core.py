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

import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from backtester.execution.contract_spec import (
    ContractSpec,
    contract_spec_from_mapping,
    get_contract_spec,
)
from backtester.mixins.reporting import ReportingMixin
from backtester.mixins.sdk_client import SDKClientMixin
from backtester.portfolio._time import normalize_time_index_like
from backtester.version import __version__ as BACKTESTER_VERSION

if TYPE_CHECKING:
    from backtester.portfolio.instr_data import InstrumentData
    from backtester.portfolio.portf_data import PortfolioData
    from backtester.portfolio.weight_cost import WeightCostBreakdown
    from backtester.universe import Universe

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


def _payload_to_multiindex_df(payload: dict[str, Any]) -> pd.DataFrame:
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


def _payload_to_series(payload: dict[str, Any], name: str = "nav") -> pd.Series:
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
        email: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
        # ── Provider ──
        source: str = "yfinance",
        granularity: str = "1d",
        # ── Strategy Config ──
        strategy_name: str = "cloud_strategy",
        strategy_type: str = "Long-Short",
        base_currency: str = "USD",
        initial_capital: float = 100_000.0,
        instruments: list[str] | None = None,
        backtest_period: dict[str, str] | None = None,
        target_volatility: float | None = None,
        max_position_size: float | None = None,
        indicators_config: list[dict[str, Any]] | None = None,
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
        execution_mode: str = "weights",  # "weights" (default) or "orders"
        weight_execution: str = "fast",  # weights only: "fast" or "orders"
        slippage_model: Any = None,  # SlippageModel instance
        commission_scheme: Any = None,  # CommissionScheme instance
        weight_cost_model: Any = None,  # WeightCostModel instance for weight-mode implied trades
        contract_specs: dict[str, ContractSpec] | None = None,
        fill_at: str | None = None,  # defaults to open; policy timing for weight-orders
        max_volume_participation: float | None = None,  # order-mode volume cap
        volume_lookback: int = 20,  # lagged ADV window for opening fills
        expected_open_volume_fraction: float = 1.0,  # forecast share available at open
        # ── Rebalancing ──
        rebalance_policy: Any = None,  # RebalancePolicy instance (default: daily)
        # ── Benchmark returns (for tracking-error rebalance trigger) ──
        benchmark_returns: pd.Series | None = None,
        # ── Risk model ──
        risk_model: Any = None,  # RiskModel instance (applied between weights and rebalance)
        pre_trade_risk: Any = None,  # PreTradeRisk for concrete orders (opt-in limits)
        # ── Performance flags ──
        skip_analysis: bool = False,  # skip StrategyPerformanceAnalysis (saves ~800ms)
        lite_init: bool = False,  # skip throwaway validation/metrics in data init (saves ~600ms)
        strict_reporting: bool | None = None,
        strict_data_fetch: bool | None = None,
        allow_partial_data: bool = False,
        **kwargs: Any,
    ) -> None:
        # ── Reject unknown kwargs — silent swallowing hides typos like
        # `rebalance_polcy=` and runs a materially different backtest. ──
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Backtester.__init__() got unexpected keyword argument(s): {unknown}. "
                "Check for typos — unknown kwargs are not silently ignored."
            )

        # ── Validate + normalize execution modes / fill timing.
        # Previously execution_mode='Orders' silently ran the WEIGHTS path and
        # fill_at='Open' silently meant close fills. ──
        execution_mode = str(execution_mode).strip().lower()
        if execution_mode not in {"weights", "orders"}:
            raise ValueError(
                f"execution_mode must be 'weights' or 'orders' (case-insensitive), "
                f"got {execution_mode!r}"
            )
        weight_execution = str(weight_execution).strip().lower()
        if weight_execution not in {"fast", "orders"}:
            raise ValueError(
                "weight_execution must be 'fast' or 'orders' "
                f"(case-insensitive), got {weight_execution!r}"
            )
        from backtester.portfolio.rebalance import RebalanceAt, RebalancePolicy

        resolved_rebalance_policy = (
            rebalance_policy if rebalance_policy is not None else RebalancePolicy()
        )
        fill_at_explicit = fill_at is not None
        if (
            execution_mode == "weights"
            and weight_execution == "orders"
            and resolved_rebalance_policy.rebalance_at == RebalanceAt.VWAP_WINDOW
        ):
            raise NotImplementedError(
                "weight_execution='orders' cannot honor VWAP_WINDOW without "
                "intraday VWAP data; use RebalanceAt.OPEN/CLOSE explicitly"
            )
        if fill_at is None:
            if execution_mode == "weights" and weight_execution == "orders":
                fill_at = resolved_rebalance_policy.rebalance_at.value
            else:
                fill_at = "open"
        fill_at = str(fill_at).strip().lower()
        if fill_at not in {"open", "close"}:
            raise ValueError(
                f"fill_at must be 'open' or 'close' (case-insensitive), got {fill_at!r}"
            )
        if (
            execution_mode == "weights"
            and weight_execution == "orders"
            and fill_at_explicit
            and fill_at != resolved_rebalance_policy.rebalance_at.value
        ):
            from dataclasses import replace

            resolved_rebalance_policy = replace(
                resolved_rebalance_policy,
                rebalance_at=RebalanceAt(fill_at),
            )

        # ── API credentials ──
        self.api_url = api_url.rstrip("/")
        self._email = email
        self._password = password
        self._api_key = api_key
        self._token: str | None = None

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
            raise ValueError("backtest_period must contain both 'start' and 'end' keys")
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
        strict_data_default = bool(strict_data_fetch) if strict_data_fetch is not None else True
        self._strict_data_fetch = _env_flag(
            "QJ_STRICT_BACKTEST",
            _env_flag("QJ_STRICT_DATA_FETCH", strict_data_default),
        )
        self._allow_partial_data = bool(allow_partial_data)

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
        self.weight_execution = weight_execution
        self.contract_specs = {
            str(symbol).upper(): spec for symbol, spec in (contract_specs or {}).items()
        }
        self.instrument_specs: dict[str, dict[str, Any]] = {}
        self.fill_engine = None
        self.execution_simulator = None
        if execution_mode == "orders" or weight_execution == "orders":
            from backtester.execution import FillEngine

            self.fill_engine = FillEngine(
                slippage=slippage_model,
                commission=commission_scheme,
                fill_at=fill_at,
                max_volume_participation=max_volume_participation,
                volume_lookback=volume_lookback,
                expected_open_volume_fraction=expected_open_volume_fraction,
                notional_fn=lambda instrument, price, quantity: self._trade_notional(
                    instrument, quantity, price
                ),
            )
        self._order_context: dict[str, Any] = {}
        # instrument → {"stop_loss": order_id, "take_profit": order_id} for
        # auto OCO-linking of protective exits (see _oco_link_protective_exit).
        self._protective_exit_ids: dict[str, dict[str, str]] = {}
        self.average_entry_price: dict[str, float | None] = {}
        self.current_positions_meta: dict[str, dict[str, float | None]] = {}

        # ── Performance flags ──
        self._skip_analysis = skip_analysis
        self._lite_init = lite_init

        # ── Rebalancing ──
        self._rebalance_policy = resolved_rebalance_policy

        # ── Benchmark returns (used by the tracking-error rebalance trigger) ──
        self._benchmark_returns = benchmark_returns

        # ── Risk model ──
        self._risk_model = risk_model  # None = no adjustment
        if pre_trade_risk is None:
            from backtester.risk.pre_trade import PreTradeRisk

            pre_trade_risk = PreTradeRisk(
                max_margin_utilization=(
                    1.0 if execution_mode == "weights" and weight_execution == "orders" else None
                )
            )
        self.pre_trade_risk = pre_trade_risk
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
        ignored_knobs: list[str] = []
        if self.execution_mode == "weights" and self.weight_execution == "fast":
            if slippage_model is not None:
                ignored_knobs.append("slippage_model")
            if commission_scheme is not None:
                ignored_knobs.append("commission_scheme")
            if fill_at_explicit and fill_at != "open":
                ignored_knobs.append("fill_at")
            if max_volume_participation is not None:
                ignored_knobs.append("max_volume_participation")
            if volume_lookback != 20:
                ignored_knobs.append("volume_lookback")
            if expected_open_volume_fraction != 1.0:
                ignored_knobs.append("expected_open_volume_fraction")
            if not getattr(self.pre_trade_risk, "is_passthrough", False):
                ignored_knobs.append("pre_trade_risk")
        elif self.execution_mode == "weights":
            if weight_cost_model_explicit:
                ignored_knobs.append("weight_cost_model")
        else:
            if self.weight_execution != "fast":
                ignored_knobs.append("weight_execution")
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
                f"[Backtester] execution_mode={self.execution_mode!r}, "
                f"weight_execution={self.weight_execution!r} ignores "
                f"configured knob(s): {', '.join(ignored_knobs)}"
            )

        # ── Universe (lazy-initialized on first access) ──
        self._universe = None

        # ── Components (lazy-loaded to avoid heavy import chains) ──
        self.ti = None  # initialized on first use
        self.blotter = None  # initialized on first use

        # Data placeholders
        self.portfolio_data: PortfolioData | None = None
        self.instruments_data: InstrumentData | None = None
        self.instrument_calculations = None
        self.portfolio_calculations = None
        self.session_id: str | None = None
        self.dataset_id: str | None = None
        self._run_started_at: str | None = None
        self._timings: dict[str, float] = {}

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
        """Return the current strategy's signals from the strategy-data schema."""
        return self.instruments_data.get_feature("strategies", self.strategy_name, "signals")

    @property
    def weights(self) -> "pd.DataFrame":
        """Return the current strategy's weights from the strategy-data schema."""
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
        from backtester.portfolio import InstrumentCalculations, PortfolioCalculations
        from backtester.portfolio.instr_data import InstrumentData
        from backtester.portfolio.portf_data import PortfolioData
        from backtester.reporting_frequency import infer_periods_per_year

        # 1) Rebuild prices DataFrame (MultiIndex: instrument, field)
        prices_df = _payload_to_multiindex_df(data["prices"])
        if prices_df.empty:
            raise ValueError("Market-data response contains no price observations")
        actual_instruments = {
            str(symbol).strip().upper() for symbol in prices_df.columns.get_level_values(0)
        }
        missing_instruments = sorted(set(self.instruments) - actual_instruments)
        if missing_instruments:
            message = "Market-data response is missing requested instruments: " + ", ".join(
                missing_instruments
            )
            if not self._allow_partial_data:
                raise ValueError(message + ". Pass allow_partial_data=True to opt in.")
            logger.warning(f"[Backtester] {message}; continuing by explicit opt-in")
        price_fields = {str(field).lower() for field in prices_df.columns.get_level_values(1)}
        if not {"close", "adj_close"}.intersection(price_fields):
            raise ValueError("Market-data response must contain close or adj_close prices")

        # 2) Rebuild metrics DataFrame
        metrics_df = _payload_to_multiindex_df(data["metrics"])

        # 3) Rebuild parameters DataFrame
        parameters_df = _payload_to_multiindex_df(data["parameters"])

        # 4) NAV series
        nav_series = _payload_to_series(data["nav"], name="nav")

        # Build instrument list from actual response
        instrument_names = data.get(
            "instrument_names", list(dict.fromkeys(prices_df.columns.get_level_values(0)))
        )

        # API metadata augments execution sizing but never overrides an
        # explicit ContractSpec supplied by the strategy author.
        raw_instrument_specs = data.get("instrument_specs") or {}
        if not isinstance(raw_instrument_specs, dict):
            raise ValueError("Invalid instrument_specs payload: expected an object keyed by symbol")
        self.instrument_specs = {}
        for symbol, spec_payload in raw_instrument_specs.items():
            normalized_symbol = str(symbol).strip().upper()
            if not isinstance(spec_payload, dict):
                raise ValueError(
                    f"Invalid instrument spec received for {normalized_symbol}: "
                    f"expected an object, got {type(spec_payload).__name__}"
                )
            self.instrument_specs[normalized_symbol] = dict(spec_payload)
        for symbol, spec_payload in self.instrument_specs.items():
            if symbol in self.contract_specs:
                continue
            try:
                self.contract_specs[symbol] = contract_spec_from_mapping(symbol, spec_payload)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid instrument spec received for {symbol}: {exc}") from exc

        # 5) Group data
        group_values = [
            self._contract_spec(str(symbol)).asset_class.value for symbol in instrument_names
        ]
        group_data = pd.Series(
            group_values,
            index=instrument_names,
            name="group",
        )
        group_order = list(dict.fromkeys(group_values))

        # 6) Empty strategies DataFrame
        strategies_df = pd.DataFrame()

        # 7) Scale NAV to initial capital
        if len(nav_series) > 0:
            scale = self.initial_capital / nav_series.iloc[0] if nav_series.iloc[0] != 0 else 1.0
            nav_series = nav_series * scale

        # 8) Build InstrumentData
        self.instruments_data = InstrumentData(
            group_data=group_data,
            group_order=group_order,
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
            periods_per_year=infer_periods_per_year(prices_df.index),
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
                                        self.instruments_data.parameters[
                                            (instrument, f"{rc}_{col}")
                                        ] = result[rc]
                        else:
                            # Non-period indicators — pass remaining params as kwargs
                            extra = {k: v for k, v in params.items() if k != "periods"}
                            result = indicator_func(data=series, **extra)
                            if isinstance(result, pd.Series):
                                out_col = f"{func_name}_{col}"
                                self.instruments_data.parameters[(instrument, out_col)] = result
                            elif isinstance(result, pd.DataFrame):
                                for rc in result.columns:
                                    self.instruments_data.parameters[
                                        (instrument, f"{rc}_{col}")
                                    ] = result[rc]

                    stored_features = [
                        key[1] for key in self.instruments_data.parameters if key[0] == instrument
                    ]
                    logger.info(f"Stored features for {instrument}: {stored_features}")

                except Exception as e:
                    logger.error(
                        f"Error calculating {ind_dict.get('function', '?')} for {instrument}: {e}"
                    )
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
        bars: dict[str, Any],
        current_positions: dict[str, float],
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

    def _require_order_context(self) -> dict[str, Any]:
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

    def has_open_orders(self, instrument: str | None = None) -> bool:
        """Return True when active orders exist, optionally for one instrument."""
        self._require_order_context()
        return any(
            order.is_active and (instrument is None or order.instrument == instrument)
            for order in self.fill_engine.pending_orders
        )

    def cancel_orders(self, instrument: str | None = None) -> int:
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
    ) -> str | None:
        from backtester.execution import Order, OrderType

        ctx = self._require_order_context()
        # Non-finite deltas (NaN/inf sizing inputs) must never become orders:
        # NaN quantity passes `qty <= 0` validation and poisons cash/NAV.
        if not np.isfinite(quantity_delta) or abs(quantity_delta) <= 1e-12:
            return None
        resolved_order_type = self._normalize_order_type(order_type) or OrderType.MARKET
        order_kwargs = dict(order_kwargs)
        order_kwargs.setdefault("created_at", ctx.get("date"))
        order = Order(
            instrument=instrument,
            side=self._order_side_from_delta(quantity_delta),
            quantity=abs(float(quantity_delta)),
            order_type=resolved_order_type,
            **order_kwargs,
        )
        return self.fill_engine.submit(order)

    def order_value(self, instrument: str, value: float, **order_kwargs) -> str | None:
        """Submit a market order for a signed notional value."""
        bar = self._order_bar(instrument)
        if not np.isfinite(bar.close) or bar.close <= 0:
            return None
        quantity_delta = self._quantity_for_notional(instrument, float(value), float(bar.close))
        return self._submit_quantity_delta(instrument, quantity_delta, **order_kwargs)

    def order_percent(
        self,
        instrument: str,
        percent: float | None = None,
        *,
        weight: float | None = None,
        **order_kwargs,
    ) -> str | None:
        """Submit a market order sized as signed percent of current NAV."""
        if percent is None:
            percent = weight
        if percent is None:
            raise ValueError("order_percent requires percent or weight")
        ctx = self._require_order_context()
        return self.order_value(instrument, float(ctx["nav"]) * float(percent), **order_kwargs)

    def order_target_value(
        self, instrument: str, target_value: float, **order_kwargs
    ) -> str | None:
        """Submit an order to move current exposure to target notional value."""
        ctx = self._require_order_context()
        bar = self._order_bar(instrument)
        current_qty = float(ctx["positions"].get(instrument, 0.0))
        current_value = self._signed_contract_notional(instrument, current_qty, float(bar.close))
        return self.order_value(instrument, float(target_value) - current_value, **order_kwargs)

    def order_target_percent(
        self,
        instrument: str,
        target_percent: float | None = None,
        *,
        target_weight: float | None = None,
        **order_kwargs,
    ) -> str | None:
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
        weight: float | None = None,
        *,
        target_weight: float | None = None,
        target_percent: float | None = None,
        **order_kwargs,
    ) -> str | None:
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

    def close_position(self, instrument: str, **order_kwargs) -> str | None:
        """Submit an order that flattens the current position."""
        return self.order_target_value(instrument, 0.0, **order_kwargs)

    def order_market(self, instrument: str, quantity: float, **order_kwargs) -> str | None:
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
        weight: float | None = None,
        percent: float | None = None,
        **order_kwargs,
    ) -> str | None:
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
        quantity_delta = self._quantity_for_notional(instrument, value, float(bar.close))
        return self._submit_quantity_delta(
            instrument,
            quantity_delta,
            order_type=OrderType.LIMIT,
            limit_price=float(limit_price),
            **order_kwargs,
        )

    def _active_order(self, order_id: str | None):
        """Return the still-active pending Order with this id, or None."""
        if not order_id or self.fill_engine is None:
            return None
        for order in self.fill_engine.pending_orders:
            if order.order_id == order_id:
                return order
        return None

    def _oco_link_protective_exit(
        self, instrument: str, kind: str, order_kwargs: dict[str, Any]
    ) -> None:
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
            self.fill_engine._oco_pairs.setdefault(sibling.oco_pair_id, []).append(sibling.order_id)
        order_kwargs["oco_pair_id"] = sibling.oco_pair_id

    def _track_protective_exit(self, instrument: str, kind: str, order_id: str | None) -> None:
        if order_id is not None:
            self._protective_exit_ids.setdefault(instrument, {})[kind] = order_id

    def stop_loss(
        self,
        instrument: str,
        *,
        stop_loss: float | None = None,
        sl: float | None = None,
        stop_price: float | None = None,
        quantity: float | None = None,
        **order_kwargs,
    ) -> str | None:
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
        take_profit: float | None = None,
        tp: float | None = None,
        limit_price: float | None = None,
        quantity: float | None = None,
        **order_kwargs,
    ) -> str | None:
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
            limit_price = entry * (
                1.0 + float(take_profit) if pos > 0 else 1.0 - float(take_profit)
            )
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
        tp: float | None = None,
        sl: float | None = None,
        take_profit: float | None = None,
        stop_loss: float | None = None,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
        **order_kwargs,
    ) -> str | None:
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

        quantity_delta = self._quantity_for_notional(instrument, qty_value, float(bar.close))
        return self._submit_quantity_delta(
            instrument,
            quantity_delta,
            order_type=OrderType.BRACKET,
            bracket=BracketSpec(
                take_profit_price=None
                if take_profit_price is None
                else round(float(take_profit_price), 8),
                stop_loss_price=None
                if stop_loss_price is None
                else round(float(stop_loss_price), 8),
                take_profit_pct=None if tp is None else float(tp),
                stop_loss_pct=None if sl is None else float(sl),
            ),
            **order_kwargs,
        )

    def get_average_entry_price(self, instrument: str) -> float | None:
        """Return current average entry price for an order-mode position."""
        value = self.average_entry_price.get(instrument)
        return None if value is None else float(value)

    def avg_entry_price(self, instrument: str) -> float | None:
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
        signals = self._validate_signals(signals)
        if not signals.ne(0).any().any():
            logger.info("All strategy signals are zero — portfolio remains flat.")
        self.instruments_data.add_strategy_data(self.strategy_name, "signals", signals)

    def _validate_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize signal output without mutating the caller."""
        return self._validate_strategy_output(signals, output_name="signals")

    def _validate_weights(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize target-weight output."""
        return self._validate_strategy_output(weights, output_name="weights")

    def _validate_positions(self, positions: pd.DataFrame) -> pd.DataFrame:
        """Validate complete stateful positions; sparse/NaN state is rejected."""
        return self._validate_strategy_output(positions, output_name="positions")

    def _validate_strategy_output(self, data: pd.DataFrame, *, output_name: str) -> pd.DataFrame:
        """Enforce the strategy DataFrame contract used by the engine.

        Outputs must cover the complete market-data grid and universe.  Column
        order is normalized to the universe order; missing dates or instruments
        are rejected rather than silently aligned by pandas.
        """
        label = output_name.capitalize()
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"{label} must be a pandas DataFrame")
        if data.empty:
            raise ValueError(f"{label} must not be empty")
        if isinstance(data.columns, pd.MultiIndex):
            raise ValueError(f"{label} columns must be a flat instrument index, not MultiIndex")
        if data.index.has_duplicates:
            raise ValueError(f"{label} index contains duplicate timestamps")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(f"{label} index must be a DatetimeIndex")
        if not data.index.is_monotonic_increasing:
            raise ValueError(f"{label} index must be monotonic increasing")
        if data.columns.has_duplicates:
            duplicates = data.columns[data.columns.duplicated()].tolist()
            raise ValueError(f"{label} columns contain duplicate instruments: {duplicates}")

        get_dates = getattr(self.instruments_data, "get_dates", None)
        if callable(get_dates):
            expected_index = get_dates()
        elif isinstance(getattr(self.instruments_data, "prices", None), pd.DataFrame):
            expected_index = self.instruments_data.prices.index
        else:
            # Lightweight strategy-store test doubles have no market frame.
            expected_index = data.index
        if not data.index.equals(expected_index):
            raise ValueError(
                f"{label} index must exactly match the market-data index "
                f"({len(expected_index)} rows expected, got {len(data.index)})"
            )

        get_instruments = getattr(self.instruments_data, "get_instruments", None)
        if callable(get_instruments):
            expected_columns = list(get_instruments())
        else:
            expected_columns = list(getattr(self, "instruments", data.columns))
        actual_columns = list(data.columns)
        missing = [column for column in expected_columns if column not in actual_columns]
        unexpected = [column for column in actual_columns if column not in expected_columns]
        if missing or unexpected:
            raise ValueError(
                f"{label} columns must exactly match the strategy universe; "
                f"missing={missing}, unexpected={unexpected}"
            )
        non_numeric = [
            column
            for column in data.columns
            if not pd.api.types.is_numeric_dtype(data[column].dtype)
        ]
        if non_numeric:
            raise ValueError(f"{label} columns must be numeric: {non_numeric}")

        result = data.reindex(columns=expected_columns).copy(deep=True)
        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError(f"{label} must contain only finite numeric values")
        return result

    def _generate_weights(self) -> None:
        """Generate and validate strategy weights."""
        weights = self._compute_weights()
        if weights is None:
            return
        weights = self._validate_weights(weights)
        self.instruments_data.add_strategy_data(self.strategy_name, "weights", weights)

    def _apply_risk_model(self) -> None:
        """Apply pluggable risk model to adjust weights before rebalance.

        Pipeline: signals → weights → **risk_model.adjust()** → rebalance → execution

        If no risk_model is set, this is a no-op.
        """
        if self._risk_model is None:
            return

        weights = self.instruments_data.get_feature("strategies", self.strategy_name, "weights")
        if weights is None or weights.empty:
            return

        close = self.instruments_data.get_feature("adj_close")
        returns = self._compute_returns_preserve_gaps(close)

        # Pass the sector map so sector_limits in PositionLimitModel (which
        # require metadata={"sectors": {instrument: sector}}) can actually fire.
        sector_map = getattr(self, "_sector_map", None)
        metadata = {"periods_per_year": getattr(self.portfolio_data, "periods_per_year", 252)}
        if sector_map:
            metadata["sectors"] = sector_map
        adjusted = self._risk_model.adjust(weights, returns, metadata=metadata)
        adjusted = self._validate_weights(adjusted)

        # Overwrite weights with risk-adjusted version
        # Drop existing weight columns first (add_strategy_data appends, not replaces)
        strats = self.instruments_data.strategies
        if not strats.empty:
            keep_mask = ~(
                (strats.columns.get_level_values(0) == self.strategy_name)
                & (strats.columns.get_level_values(1) == "weights")
            )
            self.instruments_data.strategies = strats.loc[:, keep_mask]

        self.instruments_data.add_strategy_data(self.strategy_name, "weights", adjusted)
        logger.info(f"[Risk] Applied {self._risk_model} to weights")

    def _generate_positions(self) -> None:
        """Generate and validate strategy positions."""
        positions = self._compute_positions()
        if positions is None:
            return
        positions = self._validate_positions(positions)
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
        if self.weight_execution == "orders" and self.fill_engine is not None:
            return self._compute_performance_weight_orders()
        return self._compute_performance_weight_based()

    def _compute_performance_order_based(self) -> None:
        """
        Day-by-day performance using FillEngine for order execution.
        Strategy submits orders via _compute_orders(); FillEngine processes
        them against OHLCV bars with slippage + commission.
        """
        from backtester.execution.simulator import ExecutionSimulator
        from backtester.portfolio.accounting.ledger import PortfolioLedger

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

        close_prices = self._instrument_price_frame("close", fallback="adj_close")
        instruments = close_prices.columns.tolist()
        all_dates = close_prices.index

        # Try to get full OHLCV; fall back to close ONLY for genuinely missing
        # fields (KeyError/AttributeError). Any other failure re-raises: a
        # blanket fallback silently disables intrabar stop/limit detection.
        degraded_fields: list[str] = []

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

        def _on_fill(fill, trade_notional: float) -> None:
            if self.blotter is None:
                from backtester.engines.blotter import Blotter

                self.blotter = Blotter()
            self.blotter.record_trade(
                order_id=fill.order_id,
                instrument=fill.instrument,
                side=fill.side.value,
                quantity=float(fill.quantity),
                price=float(fill.fill_price),
                trade_value=float(trade_notional),
                timestamp=fill.timestamp,
                transaction_cost=float(fill.commission),
                slippage=float(fill.slippage),
                theoretical_price=(
                    None if fill.theoretical_price is None else float(fill.theoretical_price)
                ),
                fill_status=fill.order_status.value,
            )

        def _on_bar(date, bars, current_positions, nav, average_entry_price) -> None:
            self.average_entry_price = dict(average_entry_price)
            self.current_positions_meta = {
                inst: {
                    "quantity": float(current_positions.get(inst, 0.0)),
                    "avg_price": average_entry_price.get(inst),
                }
                for inst in instruments
            }
            if self.portfolio_data is not None:
                self.portfolio_data.average_entry_price = dict(average_entry_price)
                self.portfolio_data.current_positions_meta = dict(self.current_positions_meta)
            self._order_context = {
                "date": date,
                "bars": dict(bars),
                "positions": dict(current_positions),
                "nav": float(nav),
            }
            self._compute_orders(
                date,
                dict(bars),
                dict(current_positions),
                float(nav),
            )

        ledger = PortfolioLedger(
            initial_cash=self.initial_capital,
            instruments=instruments,
            contract_spec_resolver=self._contract_spec,
            settlement_currency=self.base_currency,
        )
        self.execution_simulator = ExecutionSimulator(
            fill_engine=self.fill_engine,
            ledger=ledger,
            pre_trade_risk=self.pre_trade_risk,
        )
        try:
            result = self.execution_simulator.run(
                close=close_prices,
                open_=open_prices,
                high=high_prices,
                low=low_prices,
                volume=volume_data,
                on_bar=_on_bar,
                on_fill=_on_fill,
            )
        finally:
            self._order_context = {}

        # Store results
        self.portfolio_data.update_net_asset_value(result.nav)
        self.portfolio_data.update_positions(result.positions)
        self._store_accounting_ledger(result.position_values, result.cash)
        self.portfolio_data.update_weights(result.exposure_weights)
        raw_returns = result.nav.pct_change()
        self.portfolio_data.returns = result.returns
        self.portfolio_data.returns_for_metrics = raw_returns.dropna()
        self.portfolio_data.average_entry_price = dict(result.average_entry_price)
        self.portfolio_data.average_entry_price_history = result.average_entry_price_history
        self._store_accounting_risk(result)
        self.portfolio_data.total_transaction_costs = self._fill_cost_history(result.nav.index)
        if self.fill_engine.order_history:
            if self.blotter is None:
                from backtester.engines.blotter import Blotter

                self.blotter = Blotter()
            self.blotter.record_engine_orders(self.fill_engine.order_history)

    def _compute_performance_weight_orders(self) -> None:
        """Execute target weights through real orders, fills and the ledger."""
        from backtester.execution.simulator import (
            ExecutionSimulator,
            TargetWeightOrderExecutor,
        )
        from backtester.portfolio.accounting.ledger import PortfolioLedger
        from backtester.portfolio.rebalance import ExecutionRebalanceEngine

        if self.blotter is not None:
            self.blotter.reset()
        self._order_context = {}
        self.average_entry_price = {}
        self.current_positions_meta = {}
        self._protective_exit_ids = {}
        if self.portfolio_data is not None:
            self.portfolio_data.average_entry_price = {}
            self.portfolio_data.current_positions_meta = {}

        close_prices = self._instrument_price_frame("close", fallback="adj_close")
        instruments = close_prices.columns.tolist()
        all_dates = close_prices.index
        target_weights = self.instruments_data.get_feature(
            "strategies", self.strategy_name, "weights"
        ).reindex(index=all_dates, columns=instruments)

        raw_cash_buffer = self.portfolio_data.cash_buffer
        if isinstance(raw_cash_buffer, pd.Series):
            cash_buffer = (
                normalize_time_index_like(
                    pd.to_numeric(raw_cash_buffer, errors="coerce"), all_dates
                )
                .reindex(all_dates)
                .ffill()
                .fillna(0.05)
            )
            target_weights = target_weights.multiply(1.0 - cash_buffer, axis=0)
        elif raw_cash_buffer is None or isinstance(raw_cash_buffer, bool):
            target_weights = target_weights * 0.95
        elif isinstance(raw_cash_buffer, (int, float, np.integer, np.floating)):
            target_weights = target_weights * (1.0 - float(raw_cash_buffer))
        else:
            raise TypeError(
                "cash_buffer must be a float fraction or pd.Series indexed by "
                f"date, got {type(raw_cash_buffer).__name__}"
            )

        degraded_fields: list[str] = []

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
            volume_data = pd.DataFrame(np.nan, index=all_dates, columns=instruments)
            if getattr(self.fill_engine, "max_volume_participation", None):
                logger.warning(
                    "[Backtester] max_volume_participation is enabled but the "
                    "dataset has no 'volume' feature; no volume-capped fills "
                    "can occur while volume is unknown."
                )

        benchmark_returns = getattr(self.portfolio_data, "benchmark_returns", None)
        if (
            benchmark_returns is None
            and self._rebalance_policy.tracking_error_threshold is not None
        ):
            logger.warning(
                "[Backtester] tracking_error_threshold is configured but no "
                "benchmark_returns were provided; the trigger is inactive."
            )

        ledger = PortfolioLedger(
            initial_cash=self.initial_capital,
            instruments=instruments,
            contract_spec_resolver=self._contract_spec,
            settlement_currency=self.base_currency,
        )
        simulator = ExecutionSimulator(
            fill_engine=self.fill_engine,
            ledger=ledger,
            pre_trade_risk=self.pre_trade_risk,
        )
        planner = ExecutionRebalanceEngine(
            self._rebalance_policy,
            dates=all_dates,
            instruments=instruments,
            periods_per_year=getattr(self.portfolio_data, "periods_per_year", 252),
            benchmark_returns=benchmark_returns,
        )
        executor = TargetWeightOrderExecutor(
            target_weights=target_weights,
            simulator=simulator,
            rebalance_engine=planner,
        )
        self.execution_simulator = simulator
        self._weight_order_executor = executor

        def _on_fill(fill, trade_notional: float) -> None:
            executor.on_fill(fill, trade_notional)
            if self.blotter is None:
                from backtester.engines.blotter import Blotter

                self.blotter = Blotter()
            self.blotter.record_trade(
                order_id=fill.order_id,
                instrument=fill.instrument,
                side=fill.side.value,
                quantity=float(fill.quantity),
                price=float(fill.fill_price),
                trade_value=float(trade_notional),
                timestamp=fill.timestamp,
                transaction_cost=float(fill.commission),
                slippage=float(fill.slippage),
                theoretical_price=(
                    None if fill.theoretical_price is None else float(fill.theoretical_price)
                ),
                fill_status=fill.order_status.value,
            )

        def _on_bar(date, bars, current_positions, nav, average_entry_price) -> None:
            self.average_entry_price = dict(average_entry_price)
            self.current_positions_meta = {
                instrument: {
                    "quantity": float(current_positions.get(instrument, 0.0)),
                    "avg_price": average_entry_price.get(instrument),
                }
                for instrument in instruments
            }
            if self.portfolio_data is not None:
                self.portfolio_data.average_entry_price = dict(average_entry_price)
                self.portfolio_data.current_positions_meta = dict(self.current_positions_meta)
            executor.on_bar(
                date,
                bars,
                current_positions,
                float(nav),
            )

        result = simulator.run(
            close=close_prices,
            open_=open_prices,
            high=high_prices,
            low=low_prices,
            volume=volume_data,
            on_bar=_on_bar,
            on_fill=_on_fill,
        )

        self.portfolio_data.update_net_asset_value(result.nav)
        self.portfolio_data.update_positions(result.positions)
        self._store_accounting_ledger(result.position_values, result.cash)
        self.portfolio_data.update_weights(result.exposure_weights)
        self.portfolio_data.returns = result.returns
        self.portfolio_data.returns_for_metrics = result.nav.pct_change().dropna()
        self.portfolio_data.average_entry_price = dict(result.average_entry_price)
        self.portfolio_data.average_entry_price_history = result.average_entry_price_history
        self._store_accounting_risk(result)

        self.portfolio_data.rebalance_decision_flags = planner.decision_flags
        self.portfolio_data.rebalance_submission_flags = planner.submission_flags
        self.portfolio_data.rebalance_fill_flags = planner.fill_flags
        self.portfolio_data.rebalance_flags = planner.planned_execution_flags
        self._rebalance_stats = dict(planner.stats)

        self.portfolio_data.total_transaction_costs = self._fill_cost_history(result.nav.index)
        self._weight_cost_breakdown: WeightCostBreakdown | None = None

        if self.fill_engine.order_history:
            if self.blotter is None:
                from backtester.engines.blotter import Blotter

                self.blotter = Blotter()
            self.blotter.record_engine_orders(self.fill_engine.order_history)

    def _contract_spec(self, instrument: str) -> ContractSpec:
        key = str(instrument).upper()
        spec = self.contract_specs.get(key)
        if spec is None:
            spec = get_contract_spec(key)
            self.contract_specs[key] = spec
        spec.validate_settlement_currency(self.base_currency)
        return spec

    def _fill_cost_history(self, index: pd.Index) -> pd.Series:
        """Return commission plus modeled execution-price slippage by bar."""
        costs = pd.Series(0.0, index=index, name="transaction_cost")
        if self.fill_engine is None:
            return costs
        for fill in self.fill_engine.fill_history:
            if fill.timestamp not in costs.index:
                continue
            slippage_cost = 0.0
            if fill.theoretical_price is not None:
                spec = self._contract_spec(fill.instrument)
                slippage_cost = abs(
                    float(
                        spec.pnl(
                            float(fill.quantity),
                            float(fill.theoretical_price),
                            float(fill.fill_price),
                        )
                    )
                )
            costs.loc[fill.timestamp] += float(fill.commission) + slippage_cost
        return costs

    def _trade_notional(self, instrument: str, quantity: float, price: float) -> float:
        return self._contract_spec(instrument).notional(quantity, price)

    def _trade_cash_value(self, instrument: str, quantity: float, price: float) -> float:
        """Unsigned-quantity transaction value with the market price sign."""
        spec = self._contract_spec(instrument)
        if spec.inverse:
            return spec.notional(quantity, price)
        return abs(float(quantity)) * float(price) * float(spec.multiplier) * float(spec.lot_size)

    def _store_accounting_ledger(self, position_values: pd.DataFrame, cash: pd.Series) -> None:
        """Store and validate NAV components, including lightweight test doubles."""
        self.portfolio_data.position_values = position_values.copy()
        if hasattr(self.portfolio_data, "update_cash"):
            self.portfolio_data.update_cash(cash)
        else:
            self.portfolio_data.cash = cash.copy()
        if hasattr(self.portfolio_data, "assert_accounting_identity"):
            self.portfolio_data.assert_accounting_identity()

    def _store_accounting_risk(self, result) -> None:
        """Store margin and buying power, including on lightweight test doubles."""
        if hasattr(self.portfolio_data, "update_exposures"):
            self.portfolio_data.update_exposures(
                exposure_values=result.exposure_values,
                exposure_weights=result.exposure_weights,
            )
        else:
            self.portfolio_data.exposure_values = result.exposure_values.copy()
            self.portfolio_data.exposure_weights = result.exposure_weights.copy()
        if hasattr(self.portfolio_data, "update_accounting_risk"):
            self.portfolio_data.update_accounting_risk(
                margin_by_instrument=result.margin_by_instrument,
                margin_used=result.margin_used,
                buying_power=result.buying_power,
            )
            return
        self.portfolio_data.margin_by_instrument = result.margin_by_instrument.copy()
        self.portfolio_data.margin_used = result.margin_used.copy()
        self.portfolio_data.buying_power = result.buying_power.copy()

    def _quantity_for_notional(self, instrument: str, value: float, price: float) -> float:
        """Convert signed exposure notional to signed units/contracts/lots."""
        unit_notional = self._contract_spec(instrument).notional(1.0, price)
        if not np.isfinite(unit_notional) or unit_notional <= 0:
            raise ValueError(f"Cannot size {instrument!r}: invalid unit notional {unit_notional!r}")
        return float(value) / float(unit_notional)

    def _signed_contract_notional(self, instrument: str, quantity: float, price: float) -> float:
        """Return exposure notional with the position direction preserved."""
        if abs(quantity) <= 1e-12:
            return 0.0
        return float(np.sign(quantity)) * self._trade_notional(instrument, quantity, price)

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
        return position_values.divide(nav, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

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

    def _position_values_from_units(
        self, positions: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
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
                values[inst] = (
                    positions[inst]
                    .multiply(marked_prices[inst])
                    .multiply(float(spec.multiplier) * float(spec.lot_size))
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
        previous_avg_price: float | None,
        signed_fill_qty: float,
        fill_price: float,
    ) -> float | None:
        """Update average entry price after a signed fill."""
        from backtester.portfolio.accounting.ledger import PortfolioLedger

        return PortfolioLedger.updated_average_entry_price(
            previous_position=previous_position,
            previous_avg_price=previous_avg_price,
            signed_fill_qty=signed_fill_qty,
            fill_price=fill_price,
        )

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
        engine = RebalanceEngine(
            self._rebalance_policy,
            periods_per_year=getattr(self.portfolio_data, "periods_per_year", 252),
        )
        actual_weights, rebal_flags = engine.run(
            target_weights_inv,
            accounting_returns,
            benchmark_returns=benchmark_returns,
        )

        # Log rebalance stats
        stats = engine.stats
        extra_parts = []
        if stats.get("tracking_error_count", 0):
            extra_parts.append(f"te={stats['tracking_error_count']}")
        if stats.get("partial_positions_saved", 0):
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
            portfolio_returns = (return_weights * accounting_returns.fillna(0.0)).sum(axis=1)

        # Weight mode remains the fast close-to-close simulator. The common
        # ledger builder publishes the same accounting/risk result schema as
        # event-driven orders without pretending implied trades are fills.
        from backtester.portfolio.accounting.ledger import build_weight_ledger

        result, cost_breakdown, position_changes_trades = build_weight_ledger(
            actual_weights=actual_weights,
            portfolio_returns=portfolio_returns,
            prices=close,
            initial_capital=self.initial_capital,
            rebalance_flags=rebal_flags,
            cost_model=self.weight_cost_model,
            contract_spec_resolver=self._contract_spec,
            settlement_currency=self.base_currency,
        )

        # Store results
        self.portfolio_data.update_net_asset_value(result.nav)
        self.portfolio_data.update_positions(result.positions)
        self._store_accounting_ledger(result.position_values, result.cash)
        self.portfolio_data.update_weights(result.weights)
        self.portfolio_data.returns = result.returns
        self.portfolio_data.total_transaction_costs = cost_breakdown.total_cost
        self._weight_cost_breakdown = cost_breakdown
        self._store_accounting_risk(result)

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
        transaction_costs: pd.DataFrame | None = None,
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

            trades_df = pd.DataFrame(
                {
                    "Timestamp": trade_dates,
                    "Instrument": instrument,
                    "Side": np.where(instrument_trades > 0, "buy", "sell"),
                    "Quantity": abs(instrument_trades),
                    "Price": trade_prices,
                    "TradeValue": abs(instrument_trades * trade_prices),
                }
            )
            if transaction_costs is not None:
                costs = transaction_costs.reindex(index=trade_dates, columns=[instrument])[
                    instrument
                ]
                trades_df["TransactionCost"] = costs.fillna(0.0).to_numpy()
            trades_list.append(trades_df)

        if not trades_list:
            return pd.DataFrame(
                columns=["Timestamp", "Instrument", "Side", "Quantity", "Price", "TradeValue"]
            )

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
        self._run_started_at = datetime.now(UTC).isoformat()

        # Data Operations (API-backed)
        stage_started = time.perf_counter()
        try:
            await self._fetch_market_data()
        except Exception as e:
            self._timings["data_fetch_seconds"] = time.perf_counter() - stage_started
            self._timings["total_seconds"] = time.perf_counter() - total_started
            from backtester.sdk.client import PrepareValidationError

            friendly_prepare_error = (
                isinstance(e, PrepareValidationError)
                and _env_flag("QJ_FRIENDLY_ERRORS", False)
                and os.getenv("QJ_LOG_LEVEL", "").strip().upper() != "DEBUG"
            )
            if not friendly_prepare_error:
                logger.error(
                    f"[Backtester] Could not fetch market data — skipping strategy.\n  Error: {e}"
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
