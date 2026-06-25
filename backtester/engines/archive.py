"""
	StrategyArchive - Institutional Backtest Persistence
	----------------------------------------------------

	Module to archive and load strategy data. This engine provides functionality to persist and retrieve the data generated during backtests, including PortfolioData, InstrumentData, and Blotter objects.

	Main Features:
		- Save and load backtesting results to/from disk.
		- Saves data in pickle format for efficient storage and retrieval.
		- Manages file paths and ensures data integrity.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from typing import Any, Optional, Union
from pathlib import Path
from datetime import datetime, timezone
import json
import math
import numbers
import pickle
import time

from backtester.version import __version__ as BACKTESTER_VERSION
from backtester.utils.decorators import error_logger
from backtester.utils.logger import logger


def save_pickle(path: Path, obj: Any) -> None:
	"""Save an object to a pickle file."""
	with open(path, 'wb') as f:
		pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Any:
	"""Load an object from a pickle file."""
	if not path.exists():
		logger.warning(f"Pickle file not found: {path}")
		return None
	with open(path, 'rb') as f:
		return pickle.load(f)


def json_safe(value: Any) -> Any:
	"""Convert run metadata to strict JSON-compatible values."""
	if isinstance(value, dict):
		return {str(key): json_safe(item) for key, item in value.items()}
	if isinstance(value, (list, tuple, set)):
		return [json_safe(item) for item in value]
	if isinstance(value, numbers.Integral) and not isinstance(value, bool):
		return int(value)
	if isinstance(value, numbers.Real) and not isinstance(value, bool):
		number = float(value)
		if math.isnan(number) or math.isinf(number):
			return None
		return number
	return value

RESULTS_PATH = Path("quantjourney/backtestings/reports")

# StrategyArchive class ---------------------------------------------------------
class StrategyArchive:
	"""
	Engine to save and load backtesting results.

	Saving files as:
		- portfolio_data.pkl (PortfolioData object)
		- instruments_data.pkl (InstrumentData object)
		- blotter.pkl (Blotter object)

	"""
	@error_logger("Error initializing SaveLoadEngine")
	def __init__(
		self,
		strategy_name: str,
		save_folder: Optional[Union[str, Path]] = None,
		save_blotter: bool = False,
		save_portfolio_data: bool = False,
		save_instruments_data: bool = False
	):
		# If save_folder is not provided, it defaults to: quantjourney/backtestings/reports/<strategy_name>
		self.strategy_name = strategy_name
		
		if save_folder is None:
			# Use our new default path if user didn't specify
			self.save_folder = RESULTS_PATH / strategy_name
		else:
			# Otherwise, place everything under the user’s chosen path
			self.save_folder = Path(save_folder) / strategy_name

		logger.info(f"Archive folder set to: {self.save_folder}")
		
		self.save_blotter = save_blotter
		self.save_portfolio_data = save_portfolio_data
		self.save_instruments_data = save_instruments_data

	@error_logger("Error saving backtesting results")
	async def archive_strategy_data(
		self,
		portfolio_data: Any,
		instruments_data: Any,
		blotter: Any,
		save_dir: Optional[Union[str, Path]] = None,
		metadata: Optional[dict[str, Any]] = None,
	) -> None:
		"""
		Archive the backtesting results to disk.
		"""
		archive_started = time.perf_counter()
		save_path = (Path(save_dir) / self.strategy_name) if save_dir else self.save_folder
		save_path.mkdir(parents=True, exist_ok=True)

		run_metadata = {
			"backtester_version": BACKTESTER_VERSION,
			"strategy_name": self.strategy_name,
			"archived_at": datetime.now(timezone.utc).isoformat(),
		}
		if metadata:
			run_metadata.update(metadata)

		if self.save_portfolio_data and portfolio_data is not None:
			portfolio_data_path = save_path / "portfolio_data.pkl"
			save_pickle(portfolio_data_path, portfolio_data)
			logger.info(f"Portfolio data saved to {portfolio_data_path}")

		if self.save_instruments_data and instruments_data is not None:
			instruments_data_path = save_path / "instruments_data.pkl"
			save_pickle(instruments_data_path, instruments_data)
			logger.info(f"Instruments data saved to {instruments_data_path}")

		if self.save_blotter and blotter is not None:
			blotter_path = save_path / "blotter.pkl"
			save_pickle(blotter_path, blotter)
			logger.info(f"Blotter data saved to {blotter_path}")

		archive_seconds = time.perf_counter() - archive_started
		timings = dict(run_metadata.get("timings_seconds") or {})
		timings["archive_seconds"] = archive_seconds
		if "total_before_archive_seconds" in timings:
			timings["total_seconds"] = timings["total_before_archive_seconds"] + archive_seconds
		run_metadata["timings_seconds"] = timings

		metadata_path = save_path / "run_metadata.json"
		with open(metadata_path, "w", encoding="utf-8") as f:
			json.dump(json_safe(run_metadata), f, indent=2, sort_keys=True, default=str, allow_nan=False)
		logger.info(f"Run metadata saved to {metadata_path}")

		logger.info(f"Backtesting results saved for strategy: {self.strategy_name}")

	@error_logger("Error loading backtesting results")
	def load_strategy_data(self):
		"""
		Load the backtesting results from disk.
		"""
		# Load PortfolioData
		portfolio_data_path = self.save_folder / 'portfolio_data.pkl'
		portfolio_data = load_pickle(portfolio_data_path)

		# Load InstrumentData
		instruments_data_path = self.save_folder / 'instruments_data.pkl'
		instruments_data = load_pickle(instruments_data_path)

		# Load Blotter
		blotter_path = self.save_folder / 'blotter.pkl'
		blotter = load_pickle(blotter_path)

		logger.info(f"Backtesting results loaded for strategy: {self.strategy_name}")

		return portfolio_data, instruments_data, blotter
