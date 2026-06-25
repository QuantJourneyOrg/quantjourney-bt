"""
	Portfolio Perf Metrics Configuration
	---------------------------------------------------------

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

PORTFOLIO_PERF_METRICS = {
	# Section 1: Executive Summary
	"Executive Summary": {
		"CAGR": ('compute_annualized_return', 'percentage'),
		"Net Profit": ('compute_trade_analytics.net_profit', 'currency0'),
		"Sharpe Ratio": ('compute_sharpe_ratio', 'ratio'),
		"Max Drawdown": ('compute_max_drawdown', 'percentage'),
	},

	# Section 2: Performance Breakdown
	"Performance Breakdown": {
		"MTD": ('compute_periodic_returns.statistics.MTD', 'percentage'),
		"QTD": ('compute_periodic_returns.statistics.QTD', 'percentage'),
		"YTD": ('compute_periodic_returns.statistics.YTD', 'percentage'),
		"1Y": ('compute_periodic_returns.statistics.1Y', 'percentage'),
		"3Y (ann.)": ('compute_periodic_returns.statistics.3Y', 'percentage'),
		"5Y (ann.)": ('compute_periodic_returns.statistics.5Y', 'percentage'),
		"ITD": ('cumulative_returns.total_return', 'percentage'),
		"Cumulative Return": ('cumulative_returns.total_return', 'percentage'),
		"ATH Value": ('compute_periodic_returns.statistics.ATH Value', 'ratio4'),
		"Drawdown from ATH": ('compute_periodic_returns.statistics.Drawdown from ATH (%)', 'percentage'),
	},

	# Section 3: Risk-Adjusted Metrics
	"Risk-Adjusted Metrics": {
		"Sharpe Ratio": ('compute_sharpe_ratio', 'ratio'),
		"Smart Sharpe Ratio": ('compute_advanced_sharpe_ratio.smart_sharpe', 'ratio'),
		"Sortino Ratio": ('compute_sortino_ratio', 'ratio'),
		"Smart Sortino Ratio": ('compute_advanced_sortino_ratio.smart_sortino', 'ratio'),
		"Calmar Ratio": ('compute_advanced_calmar_ratio.base_calmar', 'ratio'),
		"Omega Ratio": ('compute_advanced_omega_ratio.base_omega', 'ratio'),
		"Information Ratio": ('information_ratio', 'ratio'),
	},

	# Section 4: Benchmark Comparison
	"Benchmark Comparison": {
		"Benchmark Name": ('benchmark_name', 'text'),
		"Benchmark MTD": ('benchmark_summary.MTD', 'percentage_raw'),
		"Benchmark QTD": ('benchmark_summary.QTD', 'percentage_raw'),
		"Benchmark YTD": ('benchmark_summary.YTD', 'percentage_raw'),
		"Benchmark 1Y": ('benchmark_summary.1Y', 'percentage_raw'),
		"Benchmark 3Y (ann.)": ('benchmark_summary.3Y', 'percentage_raw'),
		"Benchmark 5Y (ann.)": ('benchmark_summary.5Y', 'percentage_raw'),
		"Benchmark ITD": ('benchmark_summary.ITD', 'percentage_raw'),
		"Excess Return YTD": ('excess_return_ytd', 'percentage_raw'),
		"Active Return YTD (ann.)": ('active_return_ytd', 'percentage_raw'),
		"Excess Return Full Period": ('excess_return_full_period', 'percentage_raw'),
		"Active Return Full Period (ann.)": ('active_return_full_period', 'percentage_raw'),
	},

	# Section 5: Risk Metrics
	"Risk Metrics": {
		"Vol (Recent Window)": ('compute_advanced_annualized_volatility.current_30d', 'percentage_raw'),
		"Vol (Rolling Window)": ('compute_advanced_annualized_volatility.historical_252d', 'percentage_raw'),
		"Vol Recent Obs": ('compute_advanced_annualized_volatility.short_window', 'count'),
		"Vol Rolling Obs": ('compute_advanced_annualized_volatility.long_window', 'count'),
		"Peak Vol (95%)": ('compute_advanced_annualized_volatility.peak_95th', 'percentage_raw'),
		"VaR (95%) CF Normal": ('compute_var_ratio.var_metrics.cornish_fisher.normal', 'percentage'),
		"VaR (95%) Stress Vol": ('compute_var_ratio.var_metrics.cornish_fisher.stress', 'percentage'),
		"Expected Shortfall": ('compute_cvar_ratio.cvar_metrics.cornish_fisher.stress', 'percentage'),
		"Stress Tail Obs": ('compute_var_ratio.regime_info.stress_var_observations', 'count'),
		"Tracking Error": ('tracking_error', 'percentage'),
		"Max Drawdown": ('compute_max_drawdown', 'percentage'),
		"Max Drawdown ($)": ('compute_trade_analytics.max_dd_dollar', 'currency0'),
		"Distribution Skewness": ('compute_var_ratio.regime_info.distribution.normal_skew', 'ratio4'),
		"Distribution Kurtosis": ('compute_var_ratio.regime_info.distribution.normal_kurt', 'ratio4'),
		"Tail Risk Ratio": ('compute_cvar_ratio.tail_metrics.tail_ratio', 'ratio4'),
		"Worst Loss Magnitude": ('compute_cvar_ratio.tail_metrics.worst_loss', 'percentage'),
		"5 Worst Days Avg Loss": ('compute_cvar_ratio.tail_metrics.worst_5_avg', 'percentage'),
		"Outlier Percentage": ('compute_var_ratio.regime_info.outlier_percentage', 'percentage'),
	},

	# Section 6: Market Dynamics
	"Market Dynamics": {
		"CAPM Alpha (ann.)": ('alpha', 'percentage'),
		"Beta": ('beta', 'ratio'),
		"Up Capture": ('up_capture', 'percentage'),
		"Down Capture": ('down_capture', 'percentage'),
		"R-Squared": ('r_squared', 'ratio'),
		"Correlation": ('correlation', 'ratio'),
	},

	# Section 7: Portfolio Characteristics
	"Strat Characteristics": {
		"Inception Date": ('start_date', 'date'),
		"Current Date": ('end_date', 'date'),
		"Track Record": ('duration', 'duration'),
		"Strategy Type": ('strategy_type', 'text'),
		"Base Currency": ('base_currency', 'text'),
		"Long Exposure": ('compute_weight_summary.long_exposure', 'percentage'),
		"Short Exposure": ('compute_weight_summary.short_exposure', 'percentage'),
		"Gross Exposure": ('compute_weight_summary.gross_exposure', 'percentage'),
		"Position Count": ('compute_weight_summary.active_positions', 'count'),
		"Concentration": ('compute_weight_summary.concentration', 'percentage'),
		"Raw Concentration": ('compute_weight_summary.raw_concentration', 'percentage'),
        "Effective Positions": ('compute_weight_summary.effective_positions', 'count'),
	},

	# Section 8: Exposure Path Checks
	"Exposure Path Checks": {
		"Max Gross Exposure": ('compute_exposure_path_checks.max_gross_exposure', 'percentage'),
		"Avg Gross Exposure": ('compute_exposure_path_checks.avg_gross_exposure', 'percentage'),
		"Max Long Exposure": ('compute_exposure_path_checks.max_long_exposure', 'percentage'),
		"Max Short Exposure": ('compute_exposure_path_checks.max_short_exposure', 'percentage'),
		"Min Cash Sleeve": ('compute_exposure_path_checks.min_cash_sleeve', 'percentage'),
		"Max Cash Sleeve": ('compute_exposure_path_checks.max_cash_sleeve', 'percentage'),
		"Cash Sleeve Nonnegative": ('compute_exposure_path_checks.cash_sleeve_nonnegative', 'bool'),
		"Finite Weights": ('compute_exposure_path_checks.finite_weights', 'bool'),
	},

	# Section 9: Trading Analytics
	"Trading Analytics": {
		"Win Rate (Trade)": ('compute_trade_analytics.win_rate_trade', 'percentage'),
		"Win Rate (Day)": ('compute_period_stats.win_days', 'percentage_raw'),
		"Win Rate (Month)": ('compute_period_stats.win_month', 'percentage_raw'),
		"Profit Factor (Trade)": ('compute_trade_analytics.profit_factor_trade', 'ratio'),
		"Recovery Factor": ('compute_recovery_factor', 'ratio'),
		"Expectancy": ('compute_trade_analytics.expectancy', 'currency'),
		"Avg Gain/Loss Ratio (Trade)": ('compute_trade_analytics.avg_gain_loss_ratio_trade', 'ratio'),
		"Largest Win": ('compute_trade_analytics.largest_win', 'currency'),
		"Largest Loss": ('compute_trade_analytics.largest_loss', 'currency'),
		"Average Win": ('compute_trade_analytics.avg_win_dollar', 'currency'),
		"Average Loss": ('compute_trade_analytics.avg_loss_dollar', 'currency'),
		"Max Consecutive Win Days": ('compute_max_consecutive_wins_losses.max_consecutive_wins', 'count'),
		"Max Consecutive Loss Days": ('compute_max_consecutive_wins_losses.max_consecutive_losses', 'count'),
		"Total Trades": ('compute_trade_analytics.total_trades', 'count'),
		"  Buy Trades": ('compute_trade_analytics.total_buy_trades', 'count'),
		"  Sell Trades": ('compute_trade_analytics.total_sell_trades', 'count'),
		"Total FIFO Lot RTs": ('compute_trade_analytics.total_round_trips', 'count'),
		"  Long FIFO Lot RTs": ('compute_trade_analytics.long_round_trips', 'count'),
		"  Short FIFO Lot RTs": ('compute_trade_analytics.short_round_trips', 'count'),
		"Avg Lot Holding Period": ('compute_trade_analytics.avg_holding_days', 'days'),
		"Median Lot Holding Period": ('compute_trade_analytics.median_holding_days', 'days'),
		"Turnover (ann.)": ('compute_trade_analytics.annualized_turnover_pct', 'percentage_raw'),
		"Avg Trade Size": ('compute_trade_analytics.avg_trade_size', 'currency0'),
	},

	# Section 10: Advanced Risk
	"Advanced Risk": {
		"CVaR / Expected Shortfall (95%)": ('compute_cvar_ratio.cvar_metrics.historical.stress', 'percentage'),
		"Ulcer Index": ('compute_advanced_ulcer_index.ulcer_index', 'ratio4'),
		"Pain Index": ('compute_advanced_ulcer_index.pain_index', 'ratio4'),
		"Serenity Index": ('compute_serenity_index', 'ratio'),
		"Max Drawdown Duration (days)": ('compute_drawdown_durations.max_duration_days', 'count'),
		"Avg Drawdown Duration (days)": ('compute_drawdown_durations.avg_duration_days', 'count'),
		"Risk of Ruin": ('compute_statistical_metrics.risk_of_ruin', 'percentage'),
	},

	# Section 11: Advanced Trading
	"Advanced Trading": {
		"Gain-to-Pain Ratio": ('compute_gain_to_pain_ratio', 'ratio'),
		"Drawdown-Vol Adj. Return": ('compute_drawdown_vol_adjusted_return', 'ratio'),
		"Win Streak Concentration": ('compute_max_win_streak_per_win_cluster', 'ratio'),
		"Kelly Criterion (Daily Returns)": ('compute_kelly_criterion', 'percentage'),
		"Tail Ratio": ('compute_cvar_ratio.tail_metrics.tail_ratio', 'ratio'),
		"Downside Deviation": ('compute_statistical_metrics.downside_deviation', 'percentage'),
	},

	# Section 12: Operational Metrics
	"Operational Metrics": {
		"Total Commission": ('compute_trade_analytics.total_commission', 'currency'),
		"Commission (% Vol)": ('compute_trade_analytics.commission_pct', 'percentage_raw'),
		"Total Volume": ('compute_trade_analytics.total_volume', 'currency0'),
		"Instruments": ('compute_trade_analytics.instruments', 'count'),
	},

	# Section 13: Consistency Cross-Checks
	"Consistency Checks": {
		"Trade/2x FIFO Lot RT": ('compute_trade_analytics.trade_to_rt_ratio', 'ratio'),
		"Avg Trades / FIFO Lot RT": ('compute_trade_analytics.trades_per_round_trip', 'ratio'),
		"Strict Lot-Match Check": ('compute_trade_analytics.trade_rt_consistent', 'bool'),
		"Volume Consistency": ('compute_trade_analytics.volume_consistency_ratio', 'ratio'),
		"Volume Consistent": ('compute_trade_analytics.volume_consistent', 'bool'),
		"RT Position-Day Load": ('compute_trade_analytics.rt_position_day_load', 'ratio'),
		"RT Granularity": ('compute_trade_analytics.round_trip_granularity', 'text'),
	},

	# Section 14: Execution Context (reproducibility metadata)
	"Execution Context": {
		"Backtester Version": ('execution_context.backtester_version', 'text'),
		"Data Start": ('execution_context.data_start', 'text'),
		"Data End": ('execution_context.data_end', 'text'),
		"Trading Days": ('execution_context.trading_days', 'count'),
		"Reporting Frequency": ('execution_context.reporting_frequency', 'text'),
		"Reporting Observations": ('execution_context.reporting_observations', 'count'),
		"Reporting Periods/Year": ('execution_context.reporting_periods_per_year', 'count'),
		"Initial Capital": ('execution_context.initial_capital', 'currency0'),
		"Risk-Free Rate": ('execution_context.risk_free_rate', 'percentage'),
		"Slippage Model": ('execution_context.slippage_model', 'text'),
		"Commission Model": ('execution_context.commission_model', 'text'),
		"Commission Rate (bps)": ('execution_context.commission_rate_bps', 'ratio'),
		"Fill At": ('execution_context.fill_at', 'text'),
	},

	# Section 15: Reproducibility Fingerprint
	"Reproducibility": {
		"Fingerprint": ('execution_context.fingerprint', 'text'),
		"Config Hash": ('execution_context.config_hash', 'text'),
		"Data Hash": ('execution_context.data_hash', 'text'),
		"Sanity Checks Passed": ('sanity_passed', 'bool'),
	},

	# Section 16: Definitions (key terms used in this report)
	"Definitions": {
		"Trade": ('_def_trade', 'definition'),
		"Round Trip": ('_def_round_trip', 'definition'),
		"Trade Coverage vs 2xRT": ('_def_trade_coverage_vs_2xrt', 'definition'),
		"RT Position-Day Load": ('_def_rt_position-day_load', 'definition'),
		"Turnover (ann.)": ('_def_turnover_ann', 'definition'),
		"Holding Period": ('_def_holding_period', 'definition'),
		"Win Rate (Trade)": ('_def_win_rate_trade', 'definition'),
		"Expectancy": ('_def_expectancy', 'definition'),
		"Profit Factor (Trade)": ('_def_profit_factor_trade', 'definition'),
		"Worst Loss Magnitude": ('_def_worst_loss_magnitude', 'definition'),
		"Kelly Criterion (Daily Returns)": ('_def_kelly_criterion_daily_returns', 'definition'),
		"Exposure Path Checks": ('_def_exposure_path_checks', 'definition'),
		"Fingerprint": ('_def_fingerprint', 'definition'),
	},
}
