"""
Portfolio performance metric configuration.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

PORTFOLIO_PERF_METRICS = {
    "Return": {
        "Annualized Return": ("compute_annualized_return", "percentage"),
        "Total Return": ("cumulative_returns.total_return", "percentage"),
        "MTD": ("compute_periodic_returns.statistics.MTD", "percentage"),
        "QTD": ("compute_periodic_returns.statistics.QTD", "percentage"),
        "YTD": ("compute_periodic_returns.statistics.YTD", "percentage"),
        "1Y": ("compute_periodic_returns.statistics.1Y", "percentage"),
        "3Y Annualized": ("compute_periodic_returns.statistics.3Y", "percentage"),
        "5Y Annualized": ("compute_periodic_returns.statistics.5Y", "percentage"),
    },
    "Risk": {
        "Sharpe Ratio": ("compute_sharpe_ratio", "ratio"),
        "Sortino Ratio": ("compute_sortino_ratio", "ratio"),
        "Max Drawdown": ("compute_max_drawdown", "percentage"),
        "Calmar Ratio": ("compute_advanced_calmar_ratio.base_calmar", "ratio"),
        "Recovery Factor": ("compute_recovery_factor", "ratio"),
        "Annualized Volatility": (
            "compute_advanced_annualized_volatility.standard",
            "percentage_raw",
        ),
    },
    "Consistency": {
        "Winning Days": ("compute_period_stats.win_days", "percentage_raw"),
        "Winning Months": ("compute_period_stats.win_month", "percentage_raw"),
        "Winning Quarters": ("compute_period_stats.win_quarter", "percentage_raw"),
        "Winning Years": ("compute_period_stats.win_year", "percentage_raw"),
    },
    "Trading": {
        "Average Daily Turnover": ("compute_advanced_turnover.average_turnover", "percentage_raw"),
        "Annualized Turnover": ("compute_advanced_turnover.annualized_turnover", "percentage_raw"),
        "Total Turnover": ("compute_advanced_turnover.total_turnover", "percentage_raw"),
    },
    "Execution Context": {
        "Backtester Version": ("execution_context.backtester_version", "text"),
        "Data Start": ("execution_context.data_start", "text"),
        "Data End": ("execution_context.data_end", "text"),
        "Initial Capital": ("execution_context.initial_capital", "currency0"),
    },
}
