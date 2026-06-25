"""
Attribution Analytics - Factor OLS, Alpha, Performance Attribution

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_factor_exposures_ols(
    returns: pd.DataFrame, factor_returns: pd.DataFrame, *, add_intercept: bool = True,
) -> pd.DataFrame:
    ret_al, fac_al = returns.align(factor_returns, join="inner", axis=0)
    X = fac_al.copy()
    if add_intercept:
        X = pd.concat([pd.Series(1.0, index=X.index, name="const"), X], axis=1)
    XtX = X.T.dot(X)
    XtX_inv = np.linalg.pinv(XtX.values)
    Xt = X.T.values
    coefs = {}
    for inst in ret_al.columns:
        y = ret_al[inst].values
        beta = XtX_inv.dot(Xt.dot(y))
        coefs[inst] = beta
    coef_df = pd.DataFrame(coefs, index=X.columns).T
    if add_intercept:
        coef_df = coef_df.rename(columns={"const": "alpha"})
    return coef_df


def compute_factor_alpha(
    returns: pd.DataFrame, factor_returns: pd.DataFrame, *, add_intercept: bool = True,
) -> pd.Series:
    coefs = compute_factor_exposures_ols(returns=returns, factor_returns=factor_returns, add_intercept=add_intercept)
    if "alpha" in coefs.columns:
        return coefs["alpha"]
    return pd.Series(0.0, index=coefs.index)


def compute_factor_attribution(factor_returns: pd.DataFrame, factor_exposures: pd.DataFrame) -> pd.DataFrame:
    common = factor_returns.columns.intersection(factor_exposures.columns)
    if common.empty:
        raise ValueError("No overlapping factor names between returns and exposures.")
    fr = factor_returns[common]
    fe_T = factor_exposures[common].T
    fe_T = fe_T.loc[common]
    return fr.dot(fe_T)


def compute_performance_attribution(excess_returns: pd.DataFrame) -> pd.DataFrame:
    total = excess_returns.sum(axis=1)
    return pd.concat([excess_returns, total.rename("Total")], axis=1)
