"""
    Module for formatting backtesting metrics.
	---------------------------------------------------------

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from typing import Any, Dict, List, Optional, Text
import pandas as pd

from backtester.metrics.utils import get_nested_value
from backtester.metrics.configs import DEFAULT_METRICS, PORTFOLIO_PERF_METRICS
from backtester.utils.logger import logger

METRIC_FORMATTERS = {
    'percentage': lambda v: f"{v * 100:.2f}%",      # decimal input: 0.35 → "35.00%"
    'percentage_raw': lambda v: f"{v:.2f}%",         # already-% input: 18.5 → "18.50%"
    'ratio': lambda v: f"{v:.2f}",
    'ratio4': lambda v: f"{v:.4f}",
    'ratio6': lambda v: f"{v:.6f}",
    'currency': lambda v: f"-${abs(v):,.2f}" if v < 0 else f"${v:,.2f}",    # -1234 → "-$1,234.00"
    'currency0': lambda v: f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}",  # -1234 → "-$1,234"
    'days': lambda v: f"{v:.0f} days",
    'duration': lambda v: v,
    'date': lambda v: v,
    'count': lambda v: f"{v:.0f}",
    'text': lambda v: str(v),
    'bool': lambda v: "✓" if v else "✗",
    'definition': lambda v: str(v),
    'nan': lambda v: 'nan' if pd.isna(v) else f"{v:.2f}",
}


def format_metric(value: Any, formatter_type: str) -> str:
	"""
	Format a metric value according to its type.
	"""
	# Handle null, nan, and zero values
	if pd.isna(value) or value is None:
		return ""

	# Bool values (must check before numeric, since bool is subclass of int)
	if isinstance(value, bool) or formatter_type == 'bool':
		formatter = METRIC_FORMATTERS.get('bool', str)
		return formatter(value)

	# Handle numeric values
	if isinstance(value, (int, float)):
		if abs(value) < 1e-10:  # Effectively zero
			return ""

	# Special handling for strings that might be dates, durations, or text
	if formatter_type in ['date', 'duration', 'text', 'definition']:
		return str(value)

	formatter = METRIC_FORMATTERS.get(formatter_type)
	if formatter is None:
		return str(value)

	try:
		formatted = formatter(value)
		# Check for various zero formats
		# if formatted in ['0', '0.0', '0.00', '0.000', '0.0000', '0.00000', '0.000000',
		#                 '0%', '0.0%', '0.00%', '0.000%', '0.0000%',
		#                 '0 days', 'nan', 'nan%']:
		#     return ""
		return formatted
	except:
		return ""


def generate_report_sections(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
	"""
	Generate formatted report sections from results.
	"""
	sections = {}
	for section_name, metrics in PORTFOLIO_PERF_METRICS.items():
		sections[section_name] = {}
		for metric_name, (value_path, formatter_type) in metrics.items():
			value = get_nested_value(results, value_path)
			sections[section_name][metric_name] = format_metric(value, formatter_type)
	return sections


def extract_metric_value_safely(
	result_dict: Dict,
	keys: List[str],
	format_str: str
) -> Optional[str]:
	"""
	Safely extracts and formats a metric value from nested dictionary with error handling.

	Args:
		result_dict: Dictionary containing metric results
		keys: Path to the metric in nested dictionary
		format_str: Format string for the metric value

	Returns:
		Formatted metric value or None if extraction fails
	"""
	try:
		current = result_dict
		for key in keys:
			current = current[key]
		return format_str.format(current)
	except (KeyError, TypeError, ValueError) as e:
		logger.error(f"Error accessing metric {'.'.join(keys)}: {str(e)}")
		return None

def format_metric_value_with_color(
	value: str
) -> Text:
	"""
	Format metric value with appropriate color coding based on value type.
	"""
	if isinstance(value, str) and "%" in value:
		try:
			num_value = float(value.replace("%", ""))
			if num_value < 0:
				return Text(value, style="bold red")
			elif num_value > 0:
				return Text(value, style="bold green")
		except ValueError:
			pass
	return Text(value)