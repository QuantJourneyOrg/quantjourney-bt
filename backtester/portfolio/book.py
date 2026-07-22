"""Portfolio-of-strategies accounting for capital-allocation books.

``StrategyBook`` treats each strategy NAV as a tradeable sleeve.  It is a
capital allocator, not a security-level portfolio merger: orders, positions,
cash and costs inside a sleeve remain owned by that sleeve.  Consequently the
book does not net offsetting underlying-security positions across strategies.

Allocation timestamps are point-in-time inputs.  Dynamic allocations are
forward-filled onto sleeve valuation timestamps and are never backfilled.
The rebalance engine applies a target at the close of its timestamp, so that
target can affect returns only from the following period.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.portfolio.rebalance import RebalanceEngine, RebalancePolicy
from backtester.portfolio.weight_cost import (
    FixedBpsWeightCostModel,
    WeightCostBreakdown,
    WeightCostModel,
    solve_recursive_weight_costs,
)


@dataclass(frozen=True, slots=True)
class StrategyBookResult:
    """Auditable output of one :class:`StrategyBook` run.

    ``actual_allocations`` are end-of-period sleeve allocations.  The
    ``returns`` series includes transaction-cost drag, including any initial
    allocation cost on its first row. ``cost_breakdown`` and ``costs`` share
    the same recursive post-cost capital path as sleeve units and book NAV.
    """

    nav: pd.Series
    returns: pd.Series
    target_allocations: pd.DataFrame
    actual_allocations: pd.DataFrame
    sleeve_units: pd.DataFrame
    sleeve_values: pd.DataFrame
    cash: pd.Series
    rebalance_flags: pd.Series
    turnover: pd.Series
    costs: pd.Series
    sleeve_nav: pd.DataFrame
    gross_returns: pd.Series
    cost_breakdown: WeightCostBreakdown
    stats: Mapping[str, object]

    @property
    def transaction_costs(self) -> pd.Series:
        """Alias using the terminology of :class:`PortfolioData`."""
        return self.costs


class StrategyBook:
    """Allocate capital across independently simulated strategy sleeves.

    Parameters
    ----------
    sleeves:
        Mapping of stable sleeve name to either a ``pd.Series`` or an object
        exposing a ``pd.Series`` named ``net_asset_value`` (for example
        ``PortfolioData``).  Series interpretation is controlled by
        ``series_kind``.
    allocations:
        Static sleeve weights or a point-in-time allocation DataFrame.  A
        dynamic table may be sparse through time; each column is forward-
        filled independently.  It must provide a known allocation for every
        sleeve on or before the first book timestamp.  Missing static sleeves
        are treated as zero and unknown columns are rejected.
    rebalance_policy:
        Existing weight-mode rebalance policy.  Its close-of-period timing is
        preserved: a target observed at ``t`` earns no return at ``t``.
    weight_cost_model:
        Cost model applied to implied sleeve-unit trades.  Defaults to zero
        cost.  Costs are funded pro rata at book level; sleeve-internal costs
        must already be reflected in each sleeve NAV.
    initial_capital:
        Initial book NAV before any first-period allocation cost.
    series_kind:
        ``"nav"`` or ``"returns"`` for all Series, or a mapping by sleeve.
        Objects with ``net_asset_value`` are always interpreted as NAV.
        The explicit setting prevents an unsafe numeric heuristic from
        guessing whether a small-valued Series contains returns or a NAV.
    periods_per_year:
        Annualisation basis passed to ``RebalanceEngine``.

    Notes
    -----
    Sleeve dates are aligned over their common observable lifetime.  Missing
    observations inside that interval are carried forward as a zero sleeve
    return.  No value is carried before inception or after termination.
    """

    _VALID_SERIES_KINDS = frozenset({"nav", "returns"})

    def __init__(
        self,
        sleeves: Mapping[str, object],
        allocations: Mapping[str, float] | pd.DataFrame,
        *,
        rebalance_policy: RebalancePolicy | None = None,
        weight_cost_model: WeightCostModel | None = None,
        initial_capital: float = 1_000_000.0,
        series_kind: str | Mapping[str, str] = "nav",
        periods_per_year: int = 252,
    ) -> None:
        if not isinstance(sleeves, Mapping) or not sleeves:
            raise ValueError("sleeves must be a non-empty mapping")
        names = list(sleeves)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("every sleeve name must be a non-empty string")
        if len(set(names)) != len(names):
            raise ValueError("sleeve names must be unique")

        capital = float(initial_capital)
        if not np.isfinite(capital) or capital <= 0.0:
            raise ValueError("initial_capital must be finite and positive")
        if not isinstance(periods_per_year, (int, np.integer)) or periods_per_year <= 0:
            raise ValueError("periods_per_year must be a positive integer")
        if rebalance_policy is not None and not isinstance(rebalance_policy, RebalancePolicy):
            raise TypeError("rebalance_policy must be a RebalancePolicy")

        cost_model = weight_cost_model or FixedBpsWeightCostModel(total_bps=0.0)
        if not callable(getattr(cost_model, "compute", None)):
            raise TypeError("weight_cost_model must implement compute(...)")

        if not isinstance(allocations, (Mapping, pd.DataFrame)):
            raise TypeError("allocations must be a mapping or DataFrame")

        self._names = names
        self._sleeves = dict(sleeves)
        self._allocations = (
            allocations.copy() if isinstance(allocations, pd.DataFrame) else dict(allocations)
        )
        self._policy = rebalance_policy or RebalancePolicy()
        self._cost_model = cost_model
        self._initial_capital = capital
        self._series_kind = self._validate_series_kind(series_kind)
        self._periods_per_year = int(periods_per_year)

    def run(self) -> StrategyBookResult:
        """Run the book and return aligned allocation and accounting output."""
        sleeve_nav = self._prepare_sleeve_nav()
        target_allocations = self._prepare_allocations(sleeve_nav.index)
        sleeve_returns = sleeve_nav.pct_change(fill_method=None).fillna(0.0)

        engine = RebalanceEngine(
            self._policy,
            periods_per_year=self._periods_per_year,
        )
        actual_allocations, rebalance_flags = engine.run(
            target_allocations,
            sleeve_returns,
        )
        gross_returns = engine.portfolio_returns.reindex(sleeve_nav.index).astype(float)
        gross_returns.name = "gross_book_return"

        nav, net_returns, cost_breakdown = solve_recursive_weight_costs(
            actual_weights=actual_allocations,
            prices=sleeve_nav,
            gross_returns=gross_returns,
            initial_capital=self._initial_capital,
            rebalance_flags=rebalance_flags,
            cost_model=self._cost_model,
        )
        net_returns.name = "book_return"
        nav.name = "book_nav"
        costs = cost_breakdown.total_cost.rename("book_transaction_cost")

        actual_allocations = actual_allocations.astype(float)
        sleeve_values = actual_allocations.multiply(nav, axis=0)
        sleeve_units = sleeve_values.divide(sleeve_nav)
        cash = (nav - sleeve_values.sum(axis=1)).rename("book_cash")
        self._assert_accounting_identity(nav, cash, sleeve_values)

        trade_values = cost_breakdown.trade_values.reindex(
            index=sleeve_nav.index,
            columns=self._names,
        )
        turnover = trade_values.sum(axis=1).divide(nav.replace(0.0, np.nan))
        turnover = turnover.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        turnover.name = "book_turnover"

        stats = dict(engine.stats)
        stats.update(
            {
                "sleeve_count": len(self._names),
                "period_count": len(nav),
                "start": nav.index[0],
                "end": nav.index[-1],
                "initial_capital": self._initial_capital,
                "final_nav": float(nav.iloc[-1]),
                "total_return": float(nav.iloc[-1] / self._initial_capital - 1.0),
                "total_turnover": float(turnover.sum()),
                "total_costs": float(costs.sum()),
                "total_cost_pct_initial_capital": float(costs.sum() / self._initial_capital),
                "average_gross_allocation": float(actual_allocations.abs().sum(axis=1).mean()),
                "underlying_security_netting": False,
            }
        )

        return StrategyBookResult(
            nav=nav,
            returns=net_returns,
            target_allocations=target_allocations,
            actual_allocations=actual_allocations,
            sleeve_units=sleeve_units,
            sleeve_values=sleeve_values,
            cash=cash,
            rebalance_flags=rebalance_flags,
            turnover=turnover,
            costs=costs,
            sleeve_nav=sleeve_nav,
            gross_returns=gross_returns,
            cost_breakdown=cost_breakdown,
            stats=stats,
        )

    def _validate_series_kind(self, series_kind: str | Mapping[str, str]) -> str | dict[str, str]:
        if isinstance(series_kind, str):
            kind = series_kind.lower().strip()
            if kind not in self._VALID_SERIES_KINDS:
                raise ValueError("series_kind must be 'nav' or 'returns'")
            return kind
        if not isinstance(series_kind, Mapping):
            raise TypeError("series_kind must be a string or mapping by sleeve")
        unknown = set(series_kind) - set(self._names)
        missing = set(self._names) - set(series_kind)
        if unknown or missing:
            raise ValueError(
                "series_kind mapping must contain exactly the sleeve names; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        result: dict[str, str] = {}
        for name, raw_kind in series_kind.items():
            kind = str(raw_kind).lower().strip()
            if kind not in self._VALID_SERIES_KINDS:
                raise ValueError(f"series_kind[{name!r}] must be 'nav' or 'returns'")
            result[name] = kind
        return result

    def _kind_for(self, name: str) -> str:
        if isinstance(self._series_kind, str):
            return self._series_kind
        return self._series_kind[name]

    @staticmethod
    def _utc_index(index: pd.Index, *, label: str) -> pd.DatetimeIndex:
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError(f"{label} index must be a DatetimeIndex")
        if index.has_duplicates:
            raise ValueError(f"{label} index must not contain duplicate timestamps")
        if not index.is_monotonic_increasing:
            raise ValueError(f"{label} index must be strictly increasing")
        return index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")

    def _coerce_sleeve(self, name: str, source: object) -> pd.Series:
        if isinstance(source, pd.Series):
            raw = source.copy()
            kind = self._kind_for(name)
        else:
            raw = getattr(source, "net_asset_value", None)
            if not isinstance(raw, pd.Series):
                raise TypeError(
                    f"sleeve {name!r} must be a Series or expose a Series attribute net_asset_value"
                )
            raw = raw.copy()
            kind = "nav"

        if raw.empty:
            raise ValueError(f"sleeve {name!r} is empty")
        raw.index = self._utc_index(raw.index, label=f"sleeve {name!r}")
        try:
            values = raw.astype(float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"sleeve {name!r} values must be numeric") from exc
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"sleeve {name!r} contains missing or non-finite values")

        if kind == "returns":
            if (values <= -1.0).any():
                raise ValueError(f"sleeve {name!r} contains a return <= -100%")
            nav = (1.0 + values).cumprod()
            nav = nav / float(nav.iloc[0])
        else:
            if (values <= 0.0).any():
                raise ValueError(f"sleeve {name!r} NAV must stay positive")
            nav = values
        nav.name = name
        return nav

    def _prepare_sleeve_nav(self) -> pd.DataFrame:
        series = {name: self._coerce_sleeve(name, self._sleeves[name]) for name in self._names}
        common_start = max(item.index[0] for item in series.values())
        common_end = min(item.index[-1] for item in series.values())
        if common_start > common_end:
            raise ValueError("sleeves have no overlapping observable lifetime")

        index = pd.DatetimeIndex([], tz="UTC")
        for item in series.values():
            observed = item.index[(item.index >= common_start) & (item.index <= common_end)]
            index = index.union(observed)
        index = index.sort_values()
        if index.empty:
            raise ValueError("sleeves have no common valuation timestamps")

        sleeve_nav = pd.DataFrame(
            {name: item.reindex(index).ffill() for name, item in series.items()},
            index=index,
            columns=self._names,
            dtype=float,
        )
        if sleeve_nav.isna().any().any():
            raise ValueError("sleeve alignment would require look-ahead backfilling")
        return sleeve_nav

    def _prepare_allocations(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        if isinstance(self._allocations, pd.DataFrame):
            raw = self._allocations.copy()
            if raw.columns.has_duplicates:
                raise ValueError("allocation columns must be unique")
            unknown = set(raw.columns) - set(self._names)
            if unknown:
                raise ValueError(f"allocations contain unknown sleeves: {sorted(unknown)}")
            raw.index = self._utc_index(raw.index, label="allocations")
            for name in self._names:
                if name not in raw:
                    raw[name] = 0.0
            raw = raw.loc[:, self._names]
            try:
                raw = raw.astype(float)
            except (TypeError, ValueError) as exc:
                raise TypeError("allocation values must be numeric") from exc
            finite_values = raw.to_numpy(dtype=float)
            if np.isinf(finite_values).any():
                raise ValueError("allocations contain infinite values")

            combined_index = raw.index.union(index).sort_values()
            aligned = raw.reindex(combined_index).ffill().reindex(index)
            if aligned.isna().any().any():
                missing = aligned.columns[aligned.iloc[0].isna()].tolist()
                raise ValueError(
                    "dynamic allocations require an initial point-in-time value "
                    f"on or before the first book timestamp; missing={missing}"
                )
            return aligned.astype(float)

        if not isinstance(self._allocations, Mapping) or not self._allocations:
            raise ValueError("allocations must be a non-empty mapping or DataFrame")
        unknown = set(self._allocations) - set(self._names)
        if unknown:
            raise ValueError(f"allocations contain unknown sleeves: {sorted(unknown)}")
        row = {}
        for name in self._names:
            try:
                value = float(self._allocations.get(name, 0.0))
            except (TypeError, ValueError) as exc:
                raise TypeError(f"allocation for {name!r} must be numeric") from exc
            if not np.isfinite(value):
                raise ValueError(f"allocation for {name!r} must be finite")
            row[name] = value
        return pd.DataFrame(row, index=index, dtype=float)

    @staticmethod
    def _validated_cost_rate(rate: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        if not isinstance(rate, pd.Series):
            raise TypeError("weight cost model total_cost_pct must be a Series")
        aligned = rate.reindex(index).astype(float)
        values = aligned.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("weight cost model returned missing or non-finite costs")
        if (values < -1e-15).any():
            raise ValueError("weight cost model returned a negative transaction cost")
        return aligned.clip(lower=0.0).rename("book_transaction_cost_rate")

    @staticmethod
    def _assert_accounting_identity(
        nav: pd.Series,
        cash: pd.Series,
        sleeve_values: pd.DataFrame,
    ) -> None:
        reconciled = cash + sleeve_values.sum(axis=1)
        if not np.allclose(
            nav.to_numpy(dtype=float),
            reconciled.to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-8,
        ):
            raise AssertionError("StrategyBook accounting identity failed")


__all__ = ["StrategyBook", "StrategyBookResult"]
