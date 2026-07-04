# QuantJourney Backtester Public
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

from __future__ import annotations

import importlib.util
import asyncio
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ROOT / "strategies"
sys.path.insert(0, str(ROOT))


def test_python_files_have_quantjourney_public_header() -> None:
    missing = []
    skipped_roots = {".git", ".venv", "reports", "plots"}

    for path in ROOT.rglob("*.py"):
        parts = path.relative_to(ROOT).parts
        if parts[0] in skipped_roots or "__pycache__" in parts:
            continue
        text = path.read_text(encoding="utf-8")
        if (
            "QuantJourney" not in text
            or "Copyright (c) 2026 QuantJourney." not in text
            or "Licensed under the Apache License 2.0." not in text
        ):
            missing.append(str(path.relative_to(ROOT)))

    assert missing == []


def test_public_package_ships_only_quantjourney_plot_theme() -> None:
    theme_files = sorted(
        path.name
        for path in (ROOT / "backtester" / "plots" / "theme" / "configs").glob("*.py")
    )

    assert theme_files == ["__init__.py", "quantjourney.py"]


def test_public_strategy_suite_has_21_strategies() -> None:
    files = sorted(STRATEGIES.glob("example_*.py"))

    assert len(files) == 21
    assert len(list(STRATEGIES.glob("example_orders_*.py"))) == 14
    assert len(list(STRATEGIES.glob("example_weights_*.py"))) == 7


def test_public_strategy_modules_import() -> None:
    for path in sorted(STRATEGIES.glob("example_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def test_backtester_public_imports() -> None:
    from backtester import Backtester
    from backtester.engines import StrategyPerformanceAnalysis
    from backtester.sdk.client import APIClient, AsyncAPIClient
    from backtester.execution.order_types import Order, OrderSide, OrderType
    from backtester.portfolio.rebalance import RebalancePolicy

    assert Backtester is not None
    assert StrategyPerformanceAnalysis is not None
    assert APIClient("https://api.quantjourney.cloud").auth_url == "https://auth.quantjourney.cloud"
    async_client = AsyncAPIClient("https://api.quantjourney.cloud")
    try:
        assert async_client.auth_url == "https://auth.quantjourney.cloud"
        assert async_client.client.follow_redirects is True
    finally:
        asyncio.run(async_client.client.aclose())
    assert Order is not None
    assert OrderSide.BUY is not None
    assert OrderType.MARKET is not None
    assert RebalancePolicy(frequency="D").frequency == "D"


def test_backtester_granularity_config_supports_intraday_aliases() -> None:
    from backtester.core import _normalize_granularity

    assert _normalize_granularity("1d") == "1d"
    assert _normalize_granularity("5m") == "5m"
    assert _normalize_granularity(5) == "5m"
    assert _normalize_granularity("15min") == "15m"
    assert _normalize_granularity("60m") == "1h"


def test_sdk_prepare_payload_uses_normalized_granularity() -> None:
    from types import SimpleNamespace

    from backtester.core import _normalize_granularity
    from backtester.mixins.sdk_client import SDKClientMixin

    captured: dict = {}

    class DummyClient:
        async def _request(self, path: str, payload: dict) -> dict:
            captured["path"] = path
            captured["payload"] = payload
            return {"session_id": "s1", "dataset_id": "d1", "summary": {}}

    class DummyBacktester(SDKClientMixin):
        async def _get_sdk_client(self):
            return DummyClient()

    dummy = DummyBacktester()
    dummy._source = "yfinance"
    dummy._granularity = _normalize_granularity(5)
    dummy.backtest_period = SimpleNamespace(start="2026-06-01", end="2026-06-05")
    dummy.instruments = ["AAPL"]
    dummy._persist = True
    dummy._dedupe = True
    dummy._force_refresh = False

    asyncio.run(dummy._fetch_market_data())

    assert captured["path"] == "/bt/prepare"
    assert captured["payload"]["provider"]["granularity"] == "5m"


def test_smart_date_axis_uses_time_labels_for_intraday_data() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from backtester.plots.plot_compat import smart_date_axis

    index = pd.date_range("2026-06-01 13:30:00", periods=12, freq="15min")
    data = pd.Series(range(len(index)), index=index)
    fig, ax = plt.subplots()
    ax.plot(data.index, data.values)

    smart_date_axis(ax, data)
    fig.canvas.draw()
    labels = [tick.get_text() for tick in ax.get_xticklabels() if tick.get_text()]

    try:
        assert ax.get_xlabel() == "Date / time"
        assert any(":" in label for label in labels)
    finally:
        plt.close(fig)


def test_auth_active_session_conflict_is_detected(monkeypatch) -> None:
    from backtester.mixins.sdk_client import SDKClientMixin

    class Response:
        status_code = 409

        @staticmethod
        def json() -> dict:
            return {"detail": {"code": "active_session_exists"}}

    monkeypatch.delenv("QJ_REPLACE_EXISTING_SESSION", raising=False)
    assert SDKClientMixin._replace_existing_session_enabled() is True
    assert SDKClientMixin._is_active_session_conflict(Response()) is True

    monkeypatch.setenv("QJ_REPLACE_EXISTING_SESSION", "0")
    assert SDKClientMixin._replace_existing_session_enabled() is False


def test_public_metadata_archive_writes_json_only(tmp_path) -> None:
    import json
    import re
    from types import SimpleNamespace

    from backtester.mixins.reporting import ReportingMixin

    class DummyBacktester(ReportingMixin):
        pass

    strategy_dir = tmp_path / "SmokeLight"
    strategy_dir.mkdir()

    dummy = DummyBacktester()
    dummy.strategy_name = "SmokeLight"
    dummy.strategy_type = "Smoke"
    dummy.base_currency = "USD"
    dummy.initial_capital = 100_000.0
    dummy.instruments = ["AAPL"]
    dummy.backtest_period = SimpleNamespace(start="2024-01-01", end="2024-12-31")
    dummy._reports_directory = str(tmp_path)
    dummy._source = "yfinance"
    dummy._granularity = "1d"
    dummy.execution_mode = "weights"
    dummy._rebalance_policy = "D"
    dummy._reporting_frequency = "daily"
    dummy._theme_plots = "quantjourney"
    dummy._plot_dpi = 300
    dummy._benchmark = {"symbol": "^GSPC", "name": "S&P 500 Index"}
    dummy._strict_reporting = False
    dummy._strict_data_fetch = False
    dummy._quiet = False
    dummy._no_reports = False
    dummy._timings = {
        "compute_seconds": float("nan"),
        "fetch_seconds": float("inf"),
        "ok_seconds": 1.25,
    }
    dummy._run_started_at = None
    dummy.session_id = "session"
    dummy.dataset_id = "dataset"
    dummy.portfolio_data = SimpleNamespace()
    dummy.instruments_data = SimpleNamespace()
    dummy.blotter = SimpleNamespace(trades=[{"symbol": "AAPL"}])

    asyncio.run(dummy._archive_strategy_data())

    assert (strategy_dir / "run_metadata.json").exists()
    metadata_text = (strategy_dir / "run_metadata.json").read_text(encoding="utf-8")
    metadata = json.loads(metadata_text)
    assert metadata["timings_seconds"]["compute_seconds"] is None
    assert metadata["timings_seconds"]["fetch_seconds"] is None
    assert metadata["timings_seconds"]["ok_seconds"] == 1.25
    assert ("QJ_SAVE_" + "PICKLE_ARCHIVE") not in metadata["runtime_options"]
    assert not re.search(r":\s*NaN\b", metadata_text)
    assert not re.search(r":\s*-?Infinity\b", metadata_text)


def test_strategy_launcher_check_mode() -> None:
    completed = subprocess.run(
        ["./strategy.sh", "example_weights_01_sma_daily", "--check"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Import check passed" in completed.stdout


def test_public_light_excludes_pro_report_modules() -> None:
    forbidden_paths = [
        "backtester/engines/pdf_creation.py",
        "backtester/engines/factsheet_pdf.py",
        "backtester/engines/narrative.py",
        "backtester/engines/plot_orchestrator.py",
        "backtester/portfolio/" + "portfolio_plots_extra.py",
        "backtester/portfolio/blotter_plots.py",
        "backtester/portfolio/strategy_trace_plots.py",
        "backtester/portfolio/crisis_analysis.py",
        "backtester/metrics/configs/crisis_periods.json",
        "backtester/portfolio/calc/montecarlo.py",
        "backtester/engines/archive.py",
        "backtester/risk",
        "backtester/portfolio/calc/attribution.py",
        "backtester/portfolio/calc/exposures.py",
        "backtester/portfolio/calc/liquidity.py",
        "backtester/portfolio/calc/outliers.py",
        "backtester/portfolio/calc/pnl.py",
        "backtester/portfolio/calc/pnl_multi_asset.py",
        "backtester/portfolio/calc/rolling.py",
        "backtester/portfolio/calc/round_trips.py",
        "backtester/portfolio/calc/sampling.py",
        "backtester/portfolio/calc/scenario.py",
        "backtester/utils/reproducibility.py",
    ]

    for relative_path in forbidden_paths:
        assert not (ROOT / relative_path).exists(), relative_path


def test_public_light_excludes_object_archive_hooks() -> None:
    forbidden_tokens = [
        "QJ_SAVE_" + "PICKLE_ARCHIVE",
        "portfolio_data" + ".pkl",
        "instruments_data" + ".pkl",
        "blotter" + ".pkl",
        "Strategy" + "Archive",
    ]

    checked_suffixes = {".py", ".md", ".sh", ".toml", ".txt"}
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        parts = path.relative_to(ROOT).parts
        if parts[0] in {".git", ".venv", "reports"} or "__pycache__" in parts:
            continue
        if not path.is_file() or path.suffix not in checked_suffixes:
            continue
        if path.relative_to(ROOT) == Path("tests/test_public_repo.py"):
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)}: {token}")

    assert offenders == []


def test_public_light_report_generates_dashboard_metrics_and_plots(tmp_path) -> None:
    import re
    import pandas as pd
    from types import SimpleNamespace

    from backtester.engines import StrategyPerformanceAnalysis

    dates = pd.date_range("2024-01-01", periods=260, freq="B", tz="UTC")
    nav = pd.Series([100_000 * (1.0004 ** i) for i in range(len(dates))], index=dates)
    weights = pd.DataFrame(
        {
            "AAPL": [0.25 if i % 20 < 10 else 0.0 for i in range(len(dates))],
            "MSFT": [0.25 for _ in range(len(dates))],
            "NVDA": [0.25 if i % 30 < 15 else 0.0 for i in range(len(dates))],
            "CASH": [0.25 for _ in range(len(dates))],
        },
        index=dates,
    )
    returns = nav.pct_change().fillna(0.0)
    prices = pd.DataFrame(
        {
            "AAPL": [190 * (1.0005 ** i) for i in range(len(dates))],
            "MSFT": [410 * (1.0003 ** i) for i in range(len(dates))],
            "NVDA": [120 * (1.0008 ** i) for i in range(len(dates))],
            "CASH": [1.0 for _ in range(len(dates))],
        },
        index=dates,
    )
    positions = (weights * nav.values.reshape(-1, 1)).div(prices).fillna(0.0)
    portfolio = SimpleNamespace(
        net_asset_value=nav,
        returns=returns,
        weights=weights,
        positions=positions,
        position_values=weights * nav.values.reshape(-1, 1),
        cash=(weights["CASH"] * nav).rename("cash"),
        total_transaction_costs=pd.Series(0.0, index=dates),
        rebalance_flags=pd.Series(False, index=dates),
    )

    class FakeInstruments:
        def __init__(self, price_frame: pd.DataFrame) -> None:
            self.group_data = pd.Series(
                {"AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "CASH": "Cash"}
            )
            self.prices = price_frame

        def get_prices(self, field: str = "close") -> pd.DataFrame:
            return self.prices

        def get_feature(self, section: str, level: str | None = None) -> pd.DataFrame:
            if section == "metrics" and level == "returns":
                return self.prices.pct_change().fillna(0.0)
            raise KeyError(f"{section}.{level}")

    instruments = FakeInstruments(prices)
    spa = StrategyPerformanceAnalysis(
        config={
            "reports_directory": str(tmp_path),
            "save_text_reports": True,
            "show_text_reports": False,
            "save_portfolio_plots": True,
            "dpi": 80,
        },
        portfolio_data=portfolio,
        instruments_data=instruments,
        strategy_name="SmokeLight",
        strategy_type="Smoke",
    )

    asyncio.run(
        spa.generate_strategy_performance_analysis(
            portfolio_data=portfolio,
            instruments_data=instruments,
            blotter=None,
            strategy_parameters={"param": "value"},
        )
    )

    output = tmp_path / "SmokeLight"
    assert (output / "dashboard.html").exists()
    assert (output / "summary.json").exists()
    assert (output / "summary.txt").exists()
    assert (output / "metrics.csv").exists()
    assert (output / "equity_curve.csv").exists()
    assert (output / "equity_curve.png").exists()
    plot_names = {path.name for path in (output / "plots").glob("*.png")}
    assert len(plot_names) >= 10
    assert "monthly_returns_heatmap.png" in plot_names
    assert "portfolio_weights.png" in plot_names
    assert "percentage_weights.png" in plot_names
    assert "latest_holdings.png" in plot_names
    assert "instrument_cumulative_returns.png" in plot_names
    assert "instrument_rolling_volatility.png" in plot_names

    dashboard = (output / "dashboard.html").read_text(encoding="utf-8")
    metrics_text = (output / "metrics.csv").read_text(encoding="utf-8")
    summary_text = (output / "summary.json").read_text(encoding="utf-8")
    metrics = pd.read_csv(output / "metrics.csv")

    assert "monthly_returns_heatmap.png" in dashboard
    assert "instrument_cumulative_returns.png" in dashboard
    assert "Metric" in dashboard
    assert "Sharpe" in dashboard
    assert list(metrics.columns) == ["section", "metric", "value"]
    assert "Annualized return" in set(metrics["metric"])
    assert metrics.loc[metrics["metric"] == "Annualized return", "value"].iloc[0].endswith("%")
    assert "compute_annualized_return" not in dashboard
    assert "compute_annualized_return" not in metrics_text
    assert not re.search(r"\d+\.\d{7,}", dashboard)
    assert not re.search(r"\d+\.\d{7,}", metrics_text)
    assert not re.search(r":\s*NaN\b", summary_text)
    assert not re.search(r":\s*-?Infinity\b", summary_text)
