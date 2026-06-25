"""
	QuantJourney Framework - Decorators for Timing Functions
	------------------------------------------------------------

	This module provides a set of decorators for the QuantJourney Framework, including
	`timer`, `timefn`, and `asyn_timefn` for timing both synchronous and asynchronous
	functions. These decorators are designed to aid in performance monitoring and
	optimization within the framework.

Institutional-grade QuantJourney Backtester component.
Designed for deterministic strategy simulation, portfolio accounting,
analytics, reporting, and reproducible research workflows.

Copyright (c) 2026 QuantJourney.
Updated: 05.2026.
Licensed under the Apache License 2.0.
"""

import functools
from functools import wraps
import time
import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict, Any, Optional, List as PyList

# Optional numba support: provide graceful fallbacks when unavailable
try:
    from numba import njit  # type: ignore
    from numba.typed import List as NumbaList  # type: ignore
except Exception:  # numba not installed or unsupported python version
    def njit(*args, **kwargs):  # type: ignore
        def _decorator(func):
            return func
        return _decorator

    # Fallback: use built-in list as a stand-in for numba.typed.List
    NumbaList = list  # type: ignore

from backtester.utils.logger import logger


def to_flat_np_array(input_list: PyList[np.ndarray]) -> np.ndarray:
    return np.concatenate(input_list).ravel()


@njit(cache=False, fastmath=False)
def set_time_grid(ttm: float, nb_steps: int = 360) -> Tuple[int, float, np.ndarray]:
    """
    Set daily steps

    Examples:
        nb_steps, dt, grid_t = set_time_grid(1.0, 360)

    Args:
        ttm (float): time to maturity
        nb_steps (int): number of steps

    Returns:
        Tuple[int, float, np.ndarray]: number of steps, time step, grid
    """
    grid_t = np.linspace(0.0, ttm, nb_steps + 1)
    dt = grid_t[1] - grid_t[0]
    return nb_steps, dt, grid_t


@njit(cache=False, fastmath=True)
def set_seed(value):
    """
    Set seed for numba space

    Examples:
        set_seed(1234)

    Args:
        value (int): seed value
    """
    np.random.seed(value)


def compute_histogram_data(
    data: np.ndarray, x_grid: np.ndarray, name: str = "Histogram"
) -> pd.Series:
    """
    Compute histogram on defined discrete grid

    Examples:
        hist_data = compute_histogram_data(data, x_grid, name)

    Args:
        data (np.ndarray): data to compute histogram
        x_grid (np.ndarray): grid for histogram
        name (str): name of the histogram

    Returns:
        pd.Series: histogram data
    """
    hist_data, bin_edges = np.histogram(
        a=data, bins=len(x_grid) - 1, range=(x_grid[0], x_grid[-1])
    )
    hist_data = np.append(np.array(x_grid[0]), hist_data)
    hist_data = hist_data / len(data)
    hist_data = pd.Series(hist_data, index=bin_edges, name=name)
    return hist_data


def timer(func):
    """
    Print the runtime of the decorated function

    Args:
        func: function

    Returns:
        wrapper_timer: wrapper function
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def loggertext(start_message, end_message=None):
    """
    Log messages at the start and optionally at the end of the decorated function.

    Args:
        start_message (str): The message to log at the start of the function.
        end_message (str, optional): The message to log at the end of the function. Defaults to None.

    Returns:
        wrapper_log_start_and_end: wrapper function
    """

    def decorator_log_start_and_end(func):
        @functools.wraps(func)
        def wrapper_log_start_and_end(*args, **kwargs):
            logger.info(start_message)
            value = func(*args, **kwargs)
            if end_message:
                logger.info(end_message)
            return value

        return wrapper_log_start_and_end

    return decorator_log_start_and_end


def update_kwargs(
    kwargs: Dict[Any, Any], new_kwargs: Optional[Dict[Any, Any]]
) -> Dict[Any, Any]:
    """
    Update kwargs with optional kwargs dicts

    Args:
        kwargs (Dict[Any, Any]): kwargs
        new_kwargs (Optional[Dict[Any, Any]]): new kwargs

    Returns:
        Dict[Any, Any]: updated kwargs
    """
    local_kwargs = kwargs.copy()
    if new_kwargs is not None and not len(new_kwargs) == 0:
        local_kwargs.update(new_kwargs)
    return local_kwargs


def timefn(func):
    """
    Print the runtime of the decorated function

    Args:
        func: function

    Returns:
        timediff: wrapper function
    """

    @functools.wraps(func)  # Use functools.wraps to correctly apply the decorator
    def timediff(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"@timefn: {func.__name__} took {end_time - start_time} seconds")
        return result

    return timediff


def asyn_timefn(func):
    """
    Print the runtime of the decorated function

    Args:
        func: function

    Returns:
        timediff: wrapper function
    """

    @wraps(func)
    async def timediff(*args, **kwargs):
        a = time.time()
        result = await func(*args, **kwargs)
        print(f"@timefn: {func.__name__} took {time.time() - a} seconds")
        return result

    return timediff


def error_logger(message=None):
    """
    Error decorator with a message that can also log the file name where an error occurs.
    Supports both sync and async functions.
    Note: __file__ captures the filename of this decorator module.
    """
    import asyncio
    import inspect

    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Get the filename of the decorated function
                    current_file = os.path.basename(func.__code__.co_filename)

                    # Generate a custom message if a message function is provided
                    if callable(message):
                        custom_message = message(*args, **kwargs)
                    else:
                        custom_message = message if message else f"Error in {func.__name__}"

                    # Log the custom or default message along with the error and the filename
                    logger.error(f"File:{current_file} - {custom_message}: {str(e)}")
                    raise

            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Get the filename of the decorated function
                    current_file = os.path.basename(func.__code__.co_filename)

                    # Generate a custom message if a message function is provided
                    if callable(message):
                        custom_message = message(*args, **kwargs)
                    else:
                        custom_message = message if message else f"Error in {func.__name__}"

                    # Log the custom or default message along with the error and the filename
                    logger.error(f"File:{current_file} - {custom_message}: {str(e)}")
                    raise

            return sync_wrapper

    return decorator
