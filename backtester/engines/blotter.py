"""
        Blotter - Institutional Trade and Fill Ledger
        ---------------------------------------------

        This module provides a deterministic blotter designed to record all trades executed
        during a QuantJourney Backtester run. It logs fills, normalizes trade records,
        and persists trade history for replay, audit, and reporting workflows.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from backtester.utils.decorators import error_logger
from backtester.utils.logger import logger


# Blotter class ---------------------------------------------------------
@dataclass
class Blotter:
    """
    Blotter class to record all orders and trades executed during the backtest.
    """

    trades: list[dict[str, Any]] = field(default_factory=list)
    orders: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        """
        Clear all recorded orders and trades (per-run state).

        Runners that reuse a Backtester / Blotter instance across runs
        must call this at run start so trade ledgers do not accumulate
        across runs.
        """
        self.trades = []
        self.orders = []

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
        slippage: float = 0.0,
        theoretical_price: float | None = None,
        fill_status: str | None = None,
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
                slippage (float): Absolute slippage per unit/share/contract.
                theoretical_price (float): Pre-slippage fill price, when available.
                fill_status (str): Fill status from the execution engine.
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
            "Slippage": slippage,
            "TheoreticalPrice": theoretical_price,
            "FillStatus": fill_status,
        }
        self.trades.append(trade_record)

    @error_logger("Error recording bulk orders")
    def record_orders_bulk(
        self,
        instruments: list[str],
        order_types: list[str],
        quantities: list[float],
        prices: list[float],
        statuses: list[str] = None,
        order_ids: list[str] = None,
        timestamps: list[pd.Timestamp] = None,
    ) -> list[str]:
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
            statuses = ["Pending"] * n_orders
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
                "Status": status,
            }
            for oid, ts, inst, ot, qty, price, status in zip(
                order_ids,
                timestamps,
                instruments,
                order_types,
                quantities,
                prices,
                statuses,
                strict=True,
            )
        ]

        # Extend orders list
        self.orders.extend(order_records)
        return order_ids

    def record_engine_orders(self, orders: list[Any]) -> None:
        """Upsert final FillEngine order state into the audit ledger."""
        existing = {
            record.get("OrderID"): record
            for record in self.orders
            if record.get("OrderID") is not None
        }
        for order in orders:
            order_id = getattr(order, "order_id", None)
            if order_id is None:
                continue
            order_type = getattr(getattr(order, "order_type", None), "value", None)
            side = getattr(getattr(order, "side", None), "value", None)
            status = getattr(getattr(order, "status", None), "value", None)
            time_in_force = getattr(getattr(order, "time_in_force", None), "value", None)
            if time_in_force is None:
                time_in_force = str(getattr(order, "time_in_force", ""))
            order_price = getattr(order, "limit_price", None)
            if order_price is None:
                order_price = getattr(order, "stop_price", None)
            record = {
                "OrderID": order_id,
                "Timestamp": getattr(order, "created_at", None),
                "Instrument": getattr(order, "instrument", None),
                "OrderType": order_type,
                "Side": side,
                "Quantity": getattr(order, "quantity", None),
                "OrderPrice": order_price,
                "Status": status,
                "FilledQuantity": getattr(order, "filled_qty", 0.0),
                "AverageFillPrice": getattr(order, "avg_fill_price", None),
                "TimeInForce": time_in_force,
                "Tag": getattr(order, "tag", ""),
                "RejectionReason": getattr(order, "rejection_reason", None),
            }
            if order_id in existing:
                existing[order_id].update(record)
            else:
                self.orders.append(record)
                existing[order_id] = record

    @error_logger("Error recording bulk trades")
    def record_trades_bulk(self, trades_df: pd.DataFrame) -> None:
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
                        - Slippage (defaults to 0.0)
                        - TheoreticalPrice (defaults to Price)
                        - FillStatus (defaults to None)
        """
        trades_df = trades_df.copy()
        required_columns = ["Timestamp", "Instrument", "Side", "Quantity", "Price", "TradeValue"]
        if not all(col in trades_df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        # Generate IDs if missing
        if "OrderID" not in trades_df.columns:
            trades_df["OrderID"] = [str(uuid.uuid4()) for _ in range(len(trades_df))]

        # Every trade must have a corresponding order record, including callers
        # that supplied their own OrderID (the weight engine does this).
        existing_order_ids = {
            record.get("OrderID") for record in self.orders if record.get("OrderID") is not None
        }
        missing_orders = trades_df.loc[
            ~trades_df["OrderID"].isin(existing_order_ids)
        ].drop_duplicates(subset=["OrderID"])
        if not missing_orders.empty:
            self.record_orders_bulk(
                instruments=missing_orders["Instrument"].tolist(),
                order_types=["market"] * len(missing_orders),
                quantities=missing_orders["Quantity"].tolist(),
                prices=missing_orders["Price"].tolist(),
                statuses=["Filled"] * len(missing_orders),
                order_ids=missing_orders["OrderID"].tolist(),
                timestamps=missing_orders["Timestamp"].tolist(),
            )

        if "TradeID" not in trades_df.columns:
            trades_df["TradeID"] = [str(uuid.uuid4()) for _ in range(len(trades_df))]

        if "TransactionCost" not in trades_df.columns:
            trades_df["TransactionCost"] = 0.0
        if "Slippage" not in trades_df.columns:
            trades_df["Slippage"] = 0.0
        if "TheoreticalPrice" not in trades_df.columns:
            trades_df["TheoreticalPrice"] = trades_df["Price"]
        if "FillStatus" not in trades_df.columns:
            trades_df["FillStatus"] = None

        # Convert DataFrame to list of dictionaries
        trade_records = trades_df.to_dict("records")

        # Standardize record format
        formatted_records = [
            {
                "TradeID": record["TradeID"],
                "OrderID": record["OrderID"],
                "Timestamp": record["Timestamp"],
                "Instrument": record["Instrument"],
                "Side": record["Side"],
                "Quantity": record["Quantity"],
                "Price": record["Price"],
                "TradeValue": record["TradeValue"],
                "TransactionCost": record["TransactionCost"],
                "Slippage": record.get("Slippage", 0.0),
                "TheoreticalPrice": record.get("TheoreticalPrice", record["Price"]),
                "FillStatus": record.get("FillStatus"),
            }
            for record in trade_records
        ]

        # Extend trades list
        self.trades.extend(formatted_records)

    def get_trades(self) -> list[dict[str, Any]]:
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
            timestamp=pd.Timestamp("2024-01-01"),
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
            timestamp=pd.Timestamp("2024-01-01"),
        )
        blotter.record_trade(
            order_id=order_id,
            instrument="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
            trade_value=15000.0,
            timestamp=pd.Timestamp("2024-01-01"),
        )
        print(blotter.trades)

    elif unit_test == UnitTests.RECORD_ORDER:
        order_id = blotter.record_order(
            instrument="AAPL",
            order_type="market",
            quantity=100,
            price=150.0,
            status="Pending",
            timestamp=pd.Timestamp("2024-01-01"),
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
            timestamp=pd.Timestamp("2024-01-01"),
        )
        print(blotter.get_trades_dataframe())

    elif unit_test == UnitTests.SAVE_ORDERS:
        order_id = blotter.record_order(
            instrument="AAPL",
            order_type="market",
            quantity=100,
            price=150.0,
            status="Filled",
            timestamp=pd.Timestamp("2024-01-01"),
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
            timestamp=pd.Timestamp("2024-01-01"),
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
