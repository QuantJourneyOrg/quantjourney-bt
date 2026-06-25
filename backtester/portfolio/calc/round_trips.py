"""
Round-Trip Analyzer — canonical FIFO trade matching engine.

This is the single source of truth for:
  - Round-trip P&L (per trade, per instrument)
  - Holding periods (per round-trip)
  - Turnover (dollar-based, from actual trades)
  - Win/loss statistics at the trade level
  - Cross-validation checks (trades ↔ volume ↔ turnover consistency)

Supports both long and short positions via signed quantities:
  - BUY  opens/adds to long, closes/reduces short
  - SELL opens/adds to short, closes/reduces long

All metrics flow from the same FIFO matching, ensuring mutual consistency.

Usage:
    from backtester.portfolio.calc.round_trips import RoundTripAnalyzer

    analyzer = RoundTripAnalyzer(trades_df, returns, initial_capital=100_000)
    summary = analyzer.summary()       # full dict for report
    rt_df   = analyzer.round_trips     # DataFrame of individual round-trips

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ── Round-trip record ──────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class RoundTrip:
    """Single completed round-trip trade."""
    instrument: str
    direction: str          # "long" or "short"
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: object      # pd.Timestamp
    exit_time: object       # pd.Timestamp
    entry_cost: float       # commission on entry leg
    exit_cost: float        # commission on exit leg
    pnl_gross: float        # qty × (exit - entry) × direction_sign
    pnl_net: float          # gross - total costs
    holding_days: float     # calendar days
    return_pct: float       # pnl_net / entry_notional


# ── FIFO Matching Engine ──────────────────────────────────────────────

def _fifo_match(trades_df: pd.DataFrame) -> List[RoundTrip]:
    """
    Match trades into round-trips using FIFO.

    Position tracking uses signed quantity:
      positive = long, negative = short.
    A trade that crosses zero creates two round-trips
    (close old + open new).
    """
    df = trades_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    round_trips: List[RoundTrip] = []

    for instrument in df["Instrument"].unique():
        inst = df[df["Instrument"] == instrument]
        # Queue of open lots: (signed_qty, price, timestamp, cost)
        open_lots: List[list] = []

        for _, trade in inst.iterrows():
            raw_qty = float(trade["Quantity"])
            price = float(trade["Price"])
            cost = float(trade.get("TransactionCost", 0.0))
            ts = trade["Timestamp"]
            side = str(trade["Side"]).lower()

            # Signed quantity: buy = +, sell = -
            signed_qty = raw_qty if side == "buy" else -raw_qty

            # Determine if this trade closes existing position
            remaining = signed_qty
            trade_cost_remaining = cost

            while remaining != 0 and open_lots:
                lot = open_lots[0]  # FIFO: oldest first
                lot_qty, lot_price, lot_ts, lot_cost = lot

                # Same sign = adding to position, not closing
                if (lot_qty > 0 and remaining > 0) or (lot_qty < 0 and remaining < 0):
                    break

                # How much can we close?
                close_qty = min(abs(remaining), abs(lot_qty))

                if lot_qty > 0:
                    # Closing long position (we're selling)
                    direction = "long"
                    entry_price = lot_price
                    exit_price = price
                    direction_sign = 1.0
                else:
                    # Closing short position (we're buying)
                    direction = "short"
                    entry_price = lot_price
                    exit_price = price
                    direction_sign = -1.0

                pnl_gross = close_qty * (exit_price - entry_price) * direction_sign
                # Allocate costs proportionally
                entry_cost_alloc = lot_cost * (close_qty / abs(lot_qty)) if lot_qty != 0 else 0
                exit_cost_alloc = trade_cost_remaining * (close_qty / abs(remaining)) if remaining != 0 else 0
                pnl_net = pnl_gross - entry_cost_alloc - exit_cost_alloc

                entry_notional = close_qty * entry_price
                ret_pct = pnl_net / entry_notional if entry_notional > 0 else 0.0
                holding = (ts - lot_ts).days if hasattr(ts - lot_ts, 'days') else 0

                round_trips.append(RoundTrip(
                    instrument=instrument,
                    direction=direction,
                    quantity=close_qty,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_time=lot_ts,
                    exit_time=ts,
                    entry_cost=entry_cost_alloc,
                    exit_cost=exit_cost_alloc,
                    pnl_gross=pnl_gross,
                    pnl_net=pnl_net,
                    holding_days=holding,
                    return_pct=ret_pct,
                ))

                # Update lot and remaining
                lot_cost -= entry_cost_alloc
                trade_cost_remaining -= exit_cost_alloc

                if close_qty == abs(lot_qty):
                    open_lots.pop(0)
                else:
                    lot[0] = lot_qty - (close_qty * (1 if lot_qty > 0 else -1))
                    lot[3] = lot_cost

                if abs(remaining) > 0:
                    remaining = remaining + (close_qty if remaining < 0 else -close_qty)

            # Any remaining qty opens a new position
            if remaining != 0:
                open_lots.append([remaining, price, ts, trade_cost_remaining])

    return round_trips


# ── Analyzer ──────────────────────────────────────────────────────────

class RoundTripAnalyzer:
    """
    Canonical trade-level analytics from blotter data.

    All metrics (round-trips, holding periods, turnover, win/loss stats)
    come from the same FIFO matching, ensuring mutual consistency.
    """

    def __init__(
        self,
        trades_df: Optional[pd.DataFrame],
        returns: pd.Series,
        initial_capital: float = 100_000.0,
    ):
        self._returns = returns
        self._initial_capital = initial_capital
        self._trades_df = trades_df

        # NAV series
        self._nav = (1 + returns).cumprod() * initial_capital

        # FIFO matching
        if trades_df is not None and not trades_df.empty:
            self._raw_trades = trades_df.copy()
            self._raw_trades["Timestamp"] = pd.to_datetime(self._raw_trades["Timestamp"])
            self._raw_trades = self._raw_trades.sort_values("Timestamp")
            self._round_trip_list = _fifo_match(trades_df)
        else:
            self._raw_trades = pd.DataFrame()
            self._round_trip_list = []

        self._rt_df: Optional[pd.DataFrame] = None

    # ── Round-trip DataFrame ──────────────────────────────────────────

    @property
    def round_trips(self) -> pd.DataFrame:
        """DataFrame of all completed round-trips."""
        if self._rt_df is None:
            if not self._round_trip_list:
                self._rt_df = pd.DataFrame(columns=[
                    "instrument", "direction", "quantity", "entry_price",
                    "exit_price", "entry_time", "exit_time", "entry_cost",
                    "exit_cost", "pnl_gross", "pnl_net", "holding_days",
                    "return_pct",
                ])
            else:
                self._rt_df = pd.DataFrame([
                    {
                        "instrument": rt.instrument,
                        "direction": rt.direction,
                        "quantity": rt.quantity,
                        "entry_price": rt.entry_price,
                        "exit_price": rt.exit_price,
                        "entry_time": rt.entry_time,
                        "exit_time": rt.exit_time,
                        "entry_cost": rt.entry_cost,
                        "exit_cost": rt.exit_cost,
                        "pnl_gross": rt.pnl_gross,
                        "pnl_net": rt.pnl_net,
                        "holding_days": rt.holding_days,
                        "return_pct": rt.return_pct,
                    }
                    for rt in self._round_trip_list
                ])
        return self._rt_df

    # ── NAV-based metrics ─────────────────────────────────────────────

    def _nav_metrics(self) -> Dict[str, Any]:
        nav = self._nav
        final = float(nav.iloc[-1]) if len(nav) > 0 else self._initial_capital
        net_profit = final - self._initial_capital
        peak = nav.cummax()
        dd_dollar = nav - peak
        return {
            "net_profit": net_profit,
            "net_profit_pct": net_profit / self._initial_capital if self._initial_capital else 0,
            "max_dd_dollar": float(dd_dollar.min()),
        }

    # ── Trade counts ──────────────────────────────────────────────────

    def _trade_counts(self) -> Dict[str, Any]:
        df = self._raw_trades
        if df.empty:
            return {
                "total_trades": 0, "total_buy_trades": 0,
                "total_sell_trades": 0, "instruments": 0,
            }
        return {
            "total_trades": len(df),
            "total_buy_trades": int((df["Side"].str.lower() == "buy").sum()),
            "total_sell_trades": int((df["Side"].str.lower() == "sell").sum()),
            "instruments": int(df["Instrument"].nunique()),
        }

    # ── Volume & Turnover (dollar-based, from actual trades) ──────────

    def _volume_and_turnover(self) -> Dict[str, Any]:
        df = self._raw_trades
        if df.empty:
            return {
                "total_volume": 0.0, "total_lots": 0.0,
                "avg_trade_size": 0.0,
                "daily_turnover_pct": 0.0,
                "annualized_turnover_pct": 0.0,
            }

        # Dollar volume from actual fills
        if "TradeValue" in df.columns:
            total_volume = float(df["TradeValue"].abs().sum())
        else:
            total_volume = float((df["Quantity"] * df["Price"]).abs().sum())

        total_lots = float(df["Quantity"].abs().sum())

        # Average NAV over the period
        avg_nav = float(self._nav.mean())

        # Number of trading days in the period
        n_days = len(self._returns)

        # One-way turnover: total_bought / avg_nav
        # (we use total_volume / 2 because buy+sell are symmetric for round-trips)
        half_volume = total_volume / 2.0

        # Daily turnover rate
        daily_turnover = (half_volume / avg_nav / n_days) if (avg_nav > 0 and n_days > 0) else 0.0

        if isinstance(self._returns.index, pd.DatetimeIndex) and len(self._returns.index) >= 2:
            years = max((self._returns.index[-1] - self._returns.index[0]).days / 365.25, 1e-9)
        else:
            years = n_days / 252.0 if n_days > 0 else 0.0
        annual_turnover = (half_volume / avg_nav / years) if (avg_nav > 0 and years > 0) else 0.0

        return {
            "total_volume": total_volume,
            "total_lots": total_lots,
            "avg_trade_size": total_volume / len(df) if len(df) > 0 else 0.0,
            "daily_turnover_pct": daily_turnover * 100,
            "annualized_turnover_pct": annual_turnover * 100,
        }

    # ── Commission ────────────────────────────────────────────────────

    def _commission_stats(self) -> Dict[str, Any]:
        df = self._raw_trades
        if df.empty or "TransactionCost" not in df.columns:
            return {"total_commission": 0.0, "commission_pct": 0.0}

        total_comm = float(df["TransactionCost"].sum())
        total_vol = float(df["TradeValue"].abs().sum()) if "TradeValue" in df.columns else 1.0
        return {
            "total_commission": total_comm,
            "commission_pct": (total_comm / total_vol * 100) if total_vol > 0 else 0.0,
        }

    # ── Round-trip statistics ─────────────────────────────────────────

    def _round_trip_stats(self) -> Dict[str, Any]:
        rt = self.round_trips
        n_rt = len(rt)
        if n_rt == 0:
            return {
                "total_round_trips": 0,
                "win_rate_trade": 0.0,
                "expectancy": 0.0,
                "largest_win": 0.0, "largest_loss": 0.0,
                "avg_win_dollar": 0.0, "avg_loss_dollar": 0.0,
                "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
                "profit_factor_trade": 0.0,
                "avg_gain_loss_ratio_trade": 0.0,
                "avg_pnl_per_trade": 0.0,
                "max_consecutive_wins_trade": 0,
                "max_consecutive_losses_trade": 0,
                "long_round_trips": 0, "short_round_trips": 0,
                "long_win_rate": 0.0, "short_win_rate": 0.0,
            }

        pnl = rt["pnl_net"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]

        long_rt = rt[rt["direction"] == "long"]
        short_rt = rt[rt["direction"] == "short"]

        gross_wins = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_losses = float(losses.sum()) if len(losses) > 0 else 0.0
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        if avg_loss < 0:
            avg_gain_loss = avg_win / abs(avg_loss)
        elif avg_win > 0:
            avg_gain_loss = float("inf")
        else:
            avg_gain_loss = 0.0

        ordered = rt.sort_values("exit_time") if "exit_time" in rt.columns else rt
        outcomes = (ordered["pnl_net"] > 0).tolist()

        def _max_consecutive(target: bool) -> int:
            current = 0
            best = 0
            for is_win in outcomes:
                if is_win == target:
                    current += 1
                    best = max(best, current)
                else:
                    current = 0
            return best

        return {
            "total_round_trips": n_rt,
            "win_rate_trade": len(wins) / n_rt if n_rt > 0 else 0.0,
            "expectancy": float(pnl.mean()),
            "largest_win": float(wins.max()) if len(wins) > 0 else 0.0,
            "largest_loss": float(losses.min()) if len(losses) > 0 else 0.0,
            "avg_win_dollar": avg_win,
            "avg_loss_dollar": avg_loss,
            "avg_win_pct": float(rt.loc[pnl > 0, "return_pct"].mean()) if len(wins) > 0 else 0.0,
            "avg_loss_pct": float(rt.loc[pnl <= 0, "return_pct"].mean()) if len(losses) > 0 else 0.0,
            "profit_factor_trade": (gross_wins / abs(gross_losses)) if gross_losses != 0 else float("inf"),
            "avg_gain_loss_ratio_trade": avg_gain_loss,
            "avg_pnl_per_trade": float(pnl.mean()),
            "max_consecutive_wins_trade": _max_consecutive(True),
            "max_consecutive_losses_trade": _max_consecutive(False),
            "long_round_trips": len(long_rt),
            "short_round_trips": len(short_rt),
            "long_win_rate": float((long_rt["pnl_net"] > 0).mean()) if len(long_rt) > 0 else 0.0,
            "short_win_rate": float((short_rt["pnl_net"] > 0).mean()) if len(short_rt) > 0 else 0.0,
        }

    # ── Holding period statistics ─────────────────────────────────────

    def _holding_period_stats(self) -> Dict[str, Any]:
        rt = self.round_trips
        if len(rt) == 0:
            return {
                "avg_holding_days": 0.0,
                "median_holding_days": 0.0,
                "max_holding_days": 0.0,
                "min_holding_days": 0.0,
                "avg_holding_winners": 0.0,
                "avg_holding_losers": 0.0,
            }

        days = rt["holding_days"]
        wins = rt[rt["pnl_net"] > 0]["holding_days"]
        losses = rt[rt["pnl_net"] <= 0]["holding_days"]

        return {
            "avg_holding_days": float(days.mean()),
            "median_holding_days": float(days.median()),
            "max_holding_days": float(days.max()),
            "min_holding_days": float(days.min()),
            "avg_holding_winners": float(wins.mean()) if len(wins) > 0 else 0.0,
            "avg_holding_losers": float(losses.mean()) if len(losses) > 0 else 0.0,
        }

    # ── Consistency cross-checks ──────────────────────────────────────

    def _cross_checks(self) -> Dict[str, Any]:
        """
        Cross-validation that the numbers tell a coherent story.

        Checks:
          1. trade coverage vs FIFO lot-level round-trips
          2. sum(round_trip quantities × avg_price) ≈ total_volume / 2
          3. avg_holding_days × round_trips / track record estimates lot overlap
        """
        tc = self._trade_counts()
        rt_stats = self._round_trip_stats()
        vol = self._volume_and_turnover()
        hp = self._holding_period_stats()

        checks = {}

        # Check 1: trade count vs round-trips ratio.
        #
        # FIFO matching is lot-level.  A single rebalance trade can close many
        # historical lots, so this diagnostic is intentionally reported as a
        # strict sanity check rather than a broad "green" pass.
        n_trades = tc["total_trades"]
        n_rt = rt_stats["total_round_trips"]
        expected_trades = n_rt * 2
        if n_trades > 0 and n_rt > 0:
            ratio = n_trades / expected_trades
            trades_per_round_trip = n_trades / n_rt
            checks["expected_trade_count_2x_rt"] = int(expected_trades)
            checks["trade_to_rt_ratio"] = round(ratio, 2)
            checks["trades_per_round_trip"] = round(trades_per_round_trip, 2)
            checks["trade_rt_consistent"] = 0.85 <= ratio <= 1.15
            checks["round_trip_granularity"] = "fifo_lot_level"
            checks["round_trip_interpretation"] = (
                "FIFO lot-level matching; weight-based rebalances can create "
                "many partial lot matches from one fill."
            )
        else:
            checks["expected_trade_count_2x_rt"] = 0
            checks["trade_to_rt_ratio"] = 0.0
            checks["trades_per_round_trip"] = 0.0
            checks["trade_rt_consistent"] = n_trades == 0
            checks["round_trip_granularity"] = "fifo_lot_level"
            checks["round_trip_interpretation"] = "No completed FIFO lot round-trips."

        # Check 2: RT notional vs blotter volume
        rt = self.round_trips
        if len(rt) > 0 and vol["total_volume"] > 0:
            rt_notional = float((rt["quantity"] * (rt["entry_price"] + rt["exit_price"]) / 2).sum() * 2)
            vol_ratio = rt_notional / vol["total_volume"]
            checks["volume_consistency_ratio"] = round(vol_ratio, 2)
            checks["volume_consistent"] = 0.5 <= vol_ratio <= 2.0
        else:
            checks["volume_consistency_ratio"] = 0.0
            checks["volume_consistent"] = True

        # Check 3: holding period vs track record
        n_days = len(self._returns)
        if n_rt > 0 and n_days > 0 and hp["avg_holding_days"] > 0:
            # With avg_holding × n_rt positions, total position-days = avg_hold × n_rt
            # vs track record = n_days. Ratio > 1 means overlapping positions (ok for multi-instrument)
            overlap = (hp["avg_holding_days"] * n_rt) / n_days
            checks["rt_position_day_load"] = round(overlap, 2)
            checks["position_overlap_ratio"] = checks["rt_position_day_load"]
        else:
            checks["rt_position_day_load"] = 0.0
            checks["position_overlap_ratio"] = 0.0

        return checks

    # ── Full summary ──────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """
        Complete trade-level analytics dict.

        Keys are usable via dot-path in PORTFOLIO_PERF_METRICS, e.g.:
          'compute_trade_analytics.net_profit'
          'compute_trade_analytics.avg_holding_days'
          'compute_trade_analytics.annualized_turnover_pct'
        """
        result: Dict[str, Any] = {}
        result.update(self._nav_metrics())
        result.update(self._trade_counts())
        result.update(self._volume_and_turnover())
        result.update(self._commission_stats())
        result.update(self._round_trip_stats())
        result.update(self._holding_period_stats())
        result.update(self._cross_checks())
        return result

    def holding_periods_list(self) -> List[float]:
        """List of holding periods in days (for histogram plots)."""
        rt = self.round_trips
        if len(rt) == 0:
            return []
        return rt["holding_days"].tolist()

    def pnl_series(self) -> pd.Series:
        """P&L per round-trip as a Series (for distribution plots)."""
        rt = self.round_trips
        if len(rt) == 0:
            return pd.Series(dtype=float)
        return rt["pnl_net"].reset_index(drop=True)

    def pnl_with_timestamps(self) -> pd.DataFrame:
        """P&L per round-trip with exit timestamps (for time-series plots)."""
        rt = self.round_trips
        if len(rt) == 0:
            return pd.DataFrame(columns=["Timestamp", "Instrument", "PnL"])
        return rt[["exit_time", "instrument", "pnl_net"]].rename(columns={
            "exit_time": "Timestamp",
            "instrument": "Instrument",
            "pnl_net": "PnL",
        }).reset_index(drop=True)
