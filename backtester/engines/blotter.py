"""
	Blotter - Institutional Trade and Fill Ledger
	---------------------------------------------

	This module provides a deterministic blotter designed to record all trades executed
	during a QuantJourney Backtester run. It logs fills, normalizes trade records,
	and persists trade history for replay, audit, and reporting workflows.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any
import uuid
from enum import Enum
import os

from backtester.utils.decorators import error_logger
from backtester.utils.logger import logger


# Blotter class ---------------------------------------------------------
@dataclass
class Blotter:
	"""
	Blotter class to record all orders and trades executed during the backtest.
	"""

	trades: List[Dict[str, Any]] = field(default_factory=list)
	orders: List[Dict[str, Any]] = field(default_factory=list)

	@error_logger("Error recording order")
	def record_order(
		self,
		instrument: str,
		order_type: str,
		quantity: float,
		price: float,
		status: str = "Pending",
		order_id: str = None,
		timestamp: pd.Timestamp = None,
	) -> str:
		"""
		Record an order in the blotter.

		Args:
			instrument (str): The instrument symbol.
			order_type (str): Type of order (e.g., 'buy', 'sell', 'limit', 'market').
			quantity (float): Quantity of the instrument.
			price (float): Price at which the order is placed.
			status (str): Status of the order (e.g., 'Pending', 'Filled', 'Canceled').
			order_id (str): Unique identifier for the order.
			timestamp (pd.Timestamp): Time when the order was placed.

		Returns:
			order_id (str): The unique identifier for the order.
		"""
		if order_id is None:
			order_id = str(uuid.uuid4())
		if timestamp is None:
			timestamp = pd.Timestamp.now()

		order_record = {
			"OrderID": order_id,
			"Timestamp": timestamp,
			"Instrument": instrument,
			"OrderType": order_type,
			"Quantity": quantity,
			"OrderPrice": price,
			"Status": status,
		}
		self.orders.append(order_record)
		return order_id

	@error_logger("Error recording trade")
	def record_trade(
		self,
		order_id: str,
		instrument: str,
		side: str,
		quantity: float,
		price: float,
		trade_value: float,
		timestamp: pd.Timestamp,
		transaction_cost: float = 0.0,
		trade_id: str = None,
	) -> None:
		"""
		Record a trade in the blotter.

		Args:
			order_id (str): The unique identifier for the associated order.
			instrument (str): The instrument symbol.
			side (str): 'buy' or 'sell'.
			quantity (float): Quantity of the instrument traded.
			price (float): Execution price.
			trade_value (float): Total value of the trade.
			timestamp (pd.Timestamp): Time when the trade was executed.
			transaction_cost (float): Transaction cost associated with the trade.
			trade_id (str): Unique identifier for the trade.
		"""
		if trade_id is None:
			trade_id = str(uuid.uuid4())

		trade_record = {
			"TradeID": trade_id,
			"OrderID": order_id,
			"Timestamp": timestamp,
			"Instrument": instrument,
			"Side": side,
			"Quantity": quantity,
			"Price": price,
			"TradeValue": trade_value,
			"TransactionCost": transaction_cost,
		}
		self.trades.append(trade_record)

	@error_logger("Error recording bulk orders")
	def record_orders_bulk(
		self,
		instruments: List[str],
		order_types: List[str],
		quantities: List[float],
		prices: List[float],
		statuses: List[str] = None,
		order_ids: List[str] = None,
		timestamps: List[pd.Timestamp] = None
	) -> List[str]:
		"""
		Record multiple orders in bulk.

		Args:
			instruments: List of instrument symbols
			order_types: List of order types
			quantities: List of quantities
			prices: List of prices
			statuses: Optional list of statuses (defaults to 'Pending')
			order_ids: Optional list of order IDs (generated if None)
			timestamps: Optional list of timestamps (current time if None)

		Returns:
			List[str]: List of order IDs
		"""
		n_orders = len(instruments)
		if not all(len(x) == n_orders for x in [order_types, quantities, prices]):
			raise ValueError("All input lists must have the same length")

		# Set default values
		if statuses is None:
			statuses = ['Pending'] * n_orders
		if order_ids is None:
			order_ids = [str(uuid.uuid4()) for _ in range(n_orders)]
		if timestamps is None:
			timestamps = [pd.Timestamp.now()] * n_orders

		# Create order records
		order_records = [
			{
				"OrderID": oid,
				"Timestamp": ts,
				"Instrument": inst,
				"OrderType": ot,
				"Quantity": qty,
				"OrderPrice": price,
				"Status": status
			}
			for oid, ts, inst, ot, qty, price, status in zip(
				order_ids, timestamps, instruments, order_types, 
				quantities, prices, statuses
			)
		]

		# Extend orders list
		self.orders.extend(order_records)
		return order_ids

	@error_logger("Error recording bulk trades")
	def record_trades_bulk(
		self,
		trades_df: pd.DataFrame
	) -> None:
		"""
		Record multiple trades in bulk from a DataFrame.

		Args:
			trades_df: DataFrame with columns:
				- Timestamp
				- Instrument
				- Side
				- Quantity
				- Price
				- TradeValue
				Optional columns:
				- OrderID (generated if missing)
				- TradeID (generated if missing)
				- TransactionCost (defaults to 0.0)
		"""
		required_columns = ['Timestamp', 'Instrument', 'Side', 'Quantity', 'Price', 'TradeValue']
		if not all(col in trades_df.columns for col in required_columns):
			raise ValueError(f"DataFrame must contain columns: {required_columns}")

		# Generate IDs if missing
		if 'OrderID' not in trades_df.columns:
			trades_df['OrderID'] = [str(uuid.uuid4()) for _ in range(len(trades_df))]
			# Record corresponding orders
			self.record_orders_bulk(
				instruments=trades_df['Instrument'].tolist(),
				order_types=['market'] * len(trades_df),
				quantities=trades_df['Quantity'].tolist(),
				prices=trades_df['Price'].tolist(),
				statuses=['Filled'] * len(trades_df),
				order_ids=trades_df['OrderID'].tolist(),
				timestamps=trades_df['Timestamp'].tolist()
			)

		if 'TradeID' not in trades_df.columns:
			trades_df['TradeID'] = [str(uuid.uuid4()) for _ in range(len(trades_df))]

		if 'TransactionCost' not in trades_df.columns:
			trades_df['TransactionCost'] = 0.0

		# Convert DataFrame to list of dictionaries
		trade_records = trades_df.to_dict('records')

		# Standardize record format
		formatted_records = [
			{
				"TradeID": record['TradeID'],
				"OrderID": record['OrderID'],
				"Timestamp": record['Timestamp'],
				"Instrument": record['Instrument'],
				"Side": record['Side'],
				"Quantity": record['Quantity'],
				"Price": record['Price'],
				"TradeValue": record['TradeValue'],
				"TransactionCost": record['TransactionCost']
			}
			for record in trade_records
		]

		# Extend trades list
		self.trades.extend(formatted_records)

	def get_trades(self) -> List[Dict[str, Any]]:
		"""
		Get all recorded trades.

		Returns:
			List[Dict[str, Any]]: List of trade records
		"""
		return self.trades

	def get_trades_dataframe(self) -> pd.DataFrame:
		"""
		Get all recorded trades as a DataFrame.

		Returns:
			pd.DataFrame: DataFrame containing all trades.
		"""
		return pd.DataFrame(self.trades)

	def get_orders_dataframe(self) -> pd.DataFrame:
		"""
		Get all recorded orders as a DataFrame.

		Returns:
			pd.DataFrame: DataFrame containing all orders.
		"""
		return pd.DataFrame(self.orders)

	def save_trades_to_csv(self, file_path: str) -> None:
		"""
		Save trades to a CSV file. Ensures the directory exists.

		Args:
			file_path (str): The path to save the CSV file.
		"""
		# Ensure the parent directory exists
		directory = os.path.dirname(file_path)
		if not os.path.exists(directory):
			os.makedirs(directory)  # Create the directory if it doesn't exist

		df = self.get_trades_dataframe()
		df.to_csv(file_path, index=False)
		logger.info(f"Trades saved to '{file_path}'.")

	def save_orders_to_csv(self, file_path: str) -> None:
		"""
		Save orders to a CSV file. Ensures the directory exists.

		Args:
			file_path (str): The path to save the CSV file.
		"""
		# Ensure the parent directory exists
		directory = os.path.dirname(file_path)
		if not os.path.exists(directory):
			os.makedirs(directory)  # Create the directory if it doesn't exist

		df = self.get_orders_dataframe()
		df.to_csv(file_path, index=False)
		logger.info(f"Orders saved to '{file_path}'.")

	def load_trades_from_csv(self, file_path: str) -> None:
		"""
		Load trades from a CSV file into the blotter.

		Args:
			file_path (str): The path to the CSV file.

		If the file does not exist, it will display an error message.
		"""
		if not os.path.exists(file_path):
			logger.error(f"Error: The file '{file_path}' does not exist.")
			return

		df = pd.read_csv(file_path, parse_dates=["Timestamp"])
		self.trades = df.to_dict("records")
		logger.info(f"Successfully loaded {len(self.trades)} trades from '{file_path}'.")

	def load_orders_from_csv(self, file_path: str) -> None:
		"""
		Load orders from a CSV file into the blotter.

		Args:
			file_path (str): The path to the CSV file.

		If the file does not exist, it will display an error message.
		"""
		if not os.path.exists(file_path):
			logger.error(f"Error: The file '{file_path}' does not exist.")
			return

		df = pd.read_csv(file_path, parse_dates=["Timestamp"])
		self.orders = df.to_dict("records")
		logger.info(f"Successfully loaded {len(self.orders)} orders from '{file_path}'.")


# Unit tests ---------------------------------------------------------
class UnitTests(Enum):
	LOAD_TRADES = 1
	SAVE_TRADES = 2
	RECORD_TRADE = 3
	RECORD_ORDER = 4
	GET_TRADES_DF = 5
	SAVE_ORDERS = 6
	GET_ORDERS_DF = 7
	LOAD_ORDERS = 8


def run_unit_test(unit_test: UnitTests):

	blotter = Blotter()

	if unit_test == UnitTests.LOAD_TRADES:
		blotter.load_trades_from_csv("_output/trades.csv")
		print(blotter.trades)

	elif unit_test == UnitTests.SAVE_TRADES:
		blotter.record_trade(
			order_id=str(uuid.uuid4()),
			instrument="AAPL",
			side="buy",
			quantity=100,
			price=150.0,
			trade_value=15000.0,
			timestamp=pd.Timestamp("2024-01-01")
		)
		blotter.save_trades_to_csv("_output/trades.csv")
		print("Trades saved to '_output/trades.csv'.")

	elif unit_test == UnitTests.RECORD_TRADE:
		order_id = blotter.record_order(
			instrument="AAPL",
			order_type="market",
			quantity=100,
			price=150.0,
			status="Filled",
			timestamp=pd.Timestamp("2024-01-01")
		)
		blotter.record_trade(
			order_id=order_id,
			instrument="AAPL",
			side="buy",
			quantity=100,
			price=150.0,
			trade_value=15000.0,
			timestamp=pd.Timestamp("2024-01-01")
		)
		print(blotter.trades)

	elif unit_test == UnitTests.RECORD_ORDER:
		order_id = blotter.record_order(
			instrument="AAPL",
			order_type="market",
			quantity=100,
			price=150.0,
			status="Pending",
			timestamp=pd.Timestamp("2024-01-01")
		)
		print(blotter.orders)

	elif unit_test == UnitTests.GET_TRADES_DF:
		blotter.record_trade(
			order_id=str(uuid.uuid4()),
			instrument="AAPL",
			side="buy",
			quantity=100,
			price=150.0,
			trade_value=15000.0,
			timestamp=pd.Timestamp("2024-01-01")
		)
		print(blotter.get_trades_dataframe())

	elif unit_test == UnitTests.SAVE_ORDERS:
		order_id = blotter.record_order(
			instrument="AAPL",
			order_type="market",
			quantity=100,
			price=150.0,
			status="Filled",
			timestamp=pd.Timestamp("2024-01-01")
		)
		blotter.save_orders_to_csv("_output/orders.csv")
		print("Orders saved to '_output/orders.csv'.")

	elif unit_test == UnitTests.GET_ORDERS_DF:
		order_id = blotter.record_order(
			instrument="AAPL",
			order_type="market",
			quantity=100,
			price=150.0,
			status="Pending",
			timestamp=pd.Timestamp("2024-01-01")
		)
		print(blotter.get_orders_dataframe())

	elif unit_test == UnitTests.LOAD_ORDERS:
		blotter.load_orders_from_csv("_output/orders.csv")
		print(blotter.orders)

	else:
		print("Invalid unit test")

if __name__ == "__main__":
	unit_test = UnitTests.LOAD_TRADES
	start_from = UnitTests.LOAD_TRADES
	is_run_all_tests = True
	if is_run_all_tests:
		for unit_test in UnitTests:
			if unit_test.value >= start_from.value:
				print(f"\n--- Running Unit Test: {unit_test.name} ---")
				run_unit_test(unit_test=unit_test)
	else:
		print(f"\n--- Running Unit Test: {unit_test.name} ---")
		run_unit_test(unit_test=unit_test)
