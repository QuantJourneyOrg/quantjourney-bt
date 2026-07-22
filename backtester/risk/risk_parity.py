"""
Risk-Parity (Equal Risk Contribution) Model
============================================

Adjusts weights so that each instrument contributes equally to
total portfolio risk.  Uses a fast iterative approximation
(no optimiser dependency) suitable for backtesting loops.

For *n* instruments with covariance matrix Σ, the risk-parity
solution satisfies::

    w_i * (Σ w)_i  =  constant  ∀ i

We use the standard iterative algorithm:

1. Start with inverse-vol weights.
2. Compute marginal risk contribution.
3. Scale each weight inversely to its marginal contribution.
4. Renormalise.
5. Repeat until convergence.

This converges in 5–15 iterations for typical 5–20 asset portfolios.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.risk.base import RiskModel


@dataclass
class RiskParityModel(RiskModel):
    """
    Equal Risk Contribution weighting.

    Parameters
    ----------
    lookback : int
        Rolling window for covariance estimation.
    max_iter : int
        Maximum iterations for the ERC solver.
    tol : float
        Convergence tolerance on weight change.
    rebalance_freq : str
        How often to recompute (default: monthly).
    """

    lookback: int = 63
    max_iter: int = 50
    tol: float = 1e-8
    rebalance_freq: str = "BMS"

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if self.tol <= 0:
            raise ValueError("tol must be positive")

    def adjust(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        *,
        metadata: dict | None = None,
    ) -> pd.DataFrame:
        n = len(weights)
        if n == 0:
            return weights

        out = weights.copy()

        # Detect rescoring dates
        if self.rebalance_freq == "D":
            is_rescore = pd.Series(True, index=weights.index)
        else:
            periods = pd.Series(
                weights.index.to_period(
                    {"BMS": "M", "W-MON": "W", "MS": "M", "QS": "Q"}.get(self.rebalance_freq, "M")
                ),
                index=weights.index,
            )
            is_rescore = periods != periods.shift(1)

        prev_rp_weights = None
        prev_signature = None

        weight_values = weights.to_numpy(dtype=float, copy=True)
        output_values = weight_values.copy()
        rescore_values = is_rescore.to_numpy(dtype=bool)

        for i in range(self.lookback, n):
            row_w = weight_values[i]
            active_mask = np.abs(row_w) > 1e-10
            n_active = int(active_mask.sum())
            signature = tuple(np.sign(row_w))

            if n_active < 2:
                # Single or no asset — pass through
                output_values[i] = row_w
                prev_rp_weights = None
                prev_signature = signature
                continue

            if rescore_values[i] or prev_rp_weights is None or signature != prev_signature:
                # Compute new risk-parity weights for active instruments
                window = returns.iloc[max(0, i - self.lookback) : i]
                active_cols = weights.columns[active_mask]
                sub_returns = window[active_cols]

                # ERC solves for positive magnitudes. Transform each return
                # series by the strategy's sign, then restore those signs to
                # the solution. This preserves long/short intent while making
                # risk contributions consistent with the signed portfolio.
                signs = np.sign(row_w[active_mask])
                cov = sub_returns.mul(signs, axis=1).cov().values
                rp_w = self._solve_erc(cov, n_active)

                # Build full weight vector
                full_rp = np.zeros(len(weights.columns), dtype=float)
                full_rp[active_mask] = signs * rp_w

                # Preserve original total exposure
                original_exposure = float(np.nansum(np.abs(row_w)))
                full_rp = full_rp * original_exposure

                prev_rp_weights = full_rp
                prev_signature = signature

            output_values[i] = prev_rp_weights

        out.iloc[:] = output_values

        return out

    def _solve_erc(self, cov: np.ndarray, n: int) -> np.ndarray:
        """
        Solve for Equal Risk Contribution weights via iterative method.

        Returns weight vector (sums to 1, all positive).
        """
        # Start with inverse-vol
        diag = np.diag(cov)
        diag = np.maximum(diag, 1e-12)
        w = 1.0 / np.sqrt(diag)
        w = w / w.sum()

        for _ in range(self.max_iter):
            sigma_w = cov @ w
            # Marginal risk contribution
            mrc = w * sigma_w
            total_risk = mrc.sum()

            if total_risk < 1e-16:
                return np.ones(n) / n

            # Target: each contributes 1/n of total risk
            target_rc = total_risk / n

            # Scale inversely to deviation from target
            adj = np.where(mrc > 1e-16, target_rc / mrc, 1.0)
            w_new = w * adj
            w_new = np.maximum(w_new, 1e-10)
            w_new = w_new / w_new.sum()

            if np.max(np.abs(w_new - w)) < self.tol:
                return w_new
            w = w_new

        return w

    def __repr__(self) -> str:
        return f"RiskParityModel(lookback={self.lookback}, freq={self.rebalance_freq})"
