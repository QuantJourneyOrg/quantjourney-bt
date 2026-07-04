# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field


class CalcConfig(BaseModel):
    """Central runtime configuration for portfolio calculations."""

    days_per_year: int = Field(252, description="Trading days per year for annualization")
    use_pandera: bool = Field(False, description="Enable strict Pandera validation")
    use_numba: bool = Field(False, description="Enable numba-accelerated rolling reductions")

    risk_free_rate_annual: Optional[float] = Field(
        None, description="Constant annual RF rate to use where a series is not provided"
    )
    calendar: Optional[str] = Field(
        None, description="Trading calendar identifier (e.g., 'XNYS')"
    )

    @classmethod
    def from_env(cls) -> "CalcConfig":
        def _env_bool(name: str, default: bool = False) -> bool:
            v = os.getenv(name)
            if v is None:
                return default
            return v.lower() in ("1", "true", "yes", "on")

        def _env_int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except Exception:
                return default

        def _env_float_opt(name: str) -> Optional[float]:
            v = os.getenv(name)
            if v is None or v == "":
                return None
            try:
                return float(v)
            except Exception:
                return None

        return cls(
            days_per_year=_env_int("QJ_DAYS_PER_YEAR", 252),
            use_pandera=_env_bool("QJ_USE_PANDERA", False),
            use_numba=_env_bool("QJ_USE_NUMBA", False),
            risk_free_rate_annual=_env_float_opt("QJ_RF_RATE_ANNUAL"),
            calendar=os.getenv("QJ_CALENDAR") or None,
        )


def get_default_config() -> CalcConfig:
    """Return a config instance resolved from environment variables."""
    return CalcConfig.from_env()
