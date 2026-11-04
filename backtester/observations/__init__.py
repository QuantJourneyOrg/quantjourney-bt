# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Optional, single-instrument replay of declared observation availability."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from numbers import Number
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtester.core import Backtester

_RESERVED = {"decision_at", "usable", "data_age_days", "_position"}
_IDENTIFIERS = {"symbol", "ticker", "instrument", "asset"}


def _scalar(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _scalar(value.item())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Observation values must be finite")
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError("Observation values must be finite")
    if isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("Observation values must be scalar")


def _hash_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _hash_scalar(value.item())
    normalized = _scalar(value)
    return None if normalized is None else [type(value).__name__, normalized]


def _received(value: Any) -> pd.Timestamp:
    if value is None or isinstance(value, Number):
        raise ValueError("observed_at requires time-zone-aware timestamps, not epochs")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid observed_at timestamp") from None
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("observed_at requires time-zone-aware timestamps")
    return timestamp.tz_convert("UTC")


def _data_day(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return pd.NaT
    if isinstance(value, Number):
        raise ValueError("data_date requires a UTC calendar date")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid data_date") from None
    if pd.isna(timestamp):
        return pd.NaT
    timestamp = (
        timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    )
    if timestamp != timestamp.normalize():
        raise ValueError("data_date requires a UTC calendar date")
    return timestamp


def _age_limit(value: int | None) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError("max_data_age_days must be a nonnegative integer or None")


class ObservationData:
    """Validate and freeze a scalar observation table for one declared instrument."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        instrument: str,
        source: str,
        synthetic: bool,
        provenance_columns: tuple[str, ...] = (),
        archive_bytes: bytes | None = None,
        file_name: str | None = None,
    ) -> None:
        if not isinstance(instrument, str) or not instrument.strip():
            raise ValueError("Declare one observation instrument")
        if not isinstance(source, str) or not source.strip() or type(synthetic) is not bool:
            raise ValueError("Declare an observation source and a boolean synthetic label")
        if frame.columns.has_duplicates or any(not isinstance(c, str) for c in frame.columns):
            raise ValueError("Observation columns must be unique strings")
        if _RESERVED.intersection(frame.columns) or "observed_at" not in frame:
            raise ValueError("Require observed_at and exclude reserved observation columns")
        self._instrument = instrument.strip().upper()
        self._source = source.strip()
        self._synthetic = synthetic
        data = frame.copy(deep=True).reset_index(drop=True)
        for column in data:
            if isinstance(column, str) and column.lower() in _IDENTIFIERS:
                if (
                    not data[column]
                    .map(
                        lambda value: (
                            isinstance(value, str) and value.strip().upper() == self.instrument
                        )
                    )
                    .all()
                ):
                    raise ValueError("Observation identifier columns must match one instrument")
        data["observed_at"] = pd.DatetimeIndex(
            [_received(value) for value in data["observed_at"]], tz="UTC"
        ).as_unit("ns")
        if data["observed_at"].duplicated().any():
            raise ValueError("Observation receipt timestamps must be unique")
        if "data_date" in data:
            data["data_date"] = pd.DatetimeIndex(
                [_data_day(value) for value in data["data_date"]], tz="UTC"
            ).as_unit("ns")
            if (data["data_date"] > data["observed_at"].dt.normalize()).any():
                raise ValueError("data_date cannot follow its receipt day")
        for column in provenance_columns:
            if column not in data:
                raise ValueError("A requested provenance column is absent")
            for value in data[column]:
                if _scalar(value) is not None and not isinstance(value, (Number, Decimal)):
                    raise ValueError("Provenance columns must contain numeric or boolean values")
        self._provenance_columns = tuple(provenance_columns)
        self._frame = data.sort_values("observed_at").reset_index(drop=True)
        encoded = json.dumps(
            {
                "instrument": self.instrument,
                "source": self.source,
                "synthetic": self.synthetic,
                "columns": list(self._frame.columns),
                "dtypes": [str(dtype) for dtype in self._frame.dtypes],
                "data": [
                    [_hash_scalar(v) for v in row] for row in self._frame.itertuples(index=False)
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        self._metadata: dict[str, Any] = {
            "source": self.source,
            "instrument": self.instrument,
            "synthetic": self.synthetic,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "sha256_basis": "observation-frame-json-v1",
            "file_name": None,
            "archive": (
                {
                    "sha256": hashlib.sha256(archive_bytes).hexdigest(),
                    "sha256_basis": "file_bytes",
                    "file_name": file_name,
                }
                if archive_bytes is not None
                else None
            ),
        }

    @property
    def instrument(self) -> str:
        return self._instrument

    @property
    def source(self) -> str:
        return self._source

    @property
    def synthetic(self) -> bool:
        return self._synthetic

    @property
    def metadata(self) -> dict[str, Any]:
        return deepcopy(self._metadata)

    def align(
        self, index: pd.DatetimeIndex, *, max_data_age_days: int | None = None
    ) -> pd.DataFrame:
        """Select the latest strictly earlier receipt; retain missing revisions."""
        _age_limit(max_data_age_days)
        if not isinstance(index, pd.DatetimeIndex) or index.hasnans:
            raise ValueError("Decision times must be a DatetimeIndex without NaT")
        if index.has_duplicates or not index.is_monotonic_increasing:
            raise ValueError("Decision times must be unique and increasing")
        if index.tz is None and not index.equals(index.normalize()):
            raise ValueError("Naive decision times must be midnight UTC dates")
        if max_data_age_days is not None and "data_date" not in self._frame:
            raise ValueError("A freshness limit requires data_date")
        cutoffs = pd.to_datetime(index, utc=True).as_unit("ns")
        left = pd.DataFrame({"decision_at": cutoffs})
        if self._frame.empty:
            aligned = pd.concat([left, self._frame.reindex(left.index)], axis=1)
        else:
            aligned = pd.merge_asof(
                left,
                self._frame,
                left_on="decision_at",
                right_on="observed_at",
                direction="backward",
                allow_exact_matches=False,
            )
        usable = aligned["observed_at"].notna()
        if "data_date" in aligned:
            aligned["data_age_days"] = (
                aligned["decision_at"].dt.normalize() - aligned["data_date"]
            ).dt.days
            if max_data_age_days is not None:
                usable &= aligned["data_age_days"].le(max_data_age_days)
        else:
            aligned["data_age_days"] = np.nan
        aligned["usable"] = usable.fillna(False)
        aligned.index = index.copy()
        return aligned.copy(deep=True)

    def provenance(self, aligned: pd.DataFrame) -> list[dict[str, Any]]:
        columns = ["decision_at", "observed_at", "data_age_days", "usable"]
        if "data_date" in aligned:
            columns.append("data_date")
        columns.extend(self._provenance_columns)
        return [
            {column: _scalar(value) for column, value in zip(columns, row, strict=True)}
            for row in aligned[columns].itertuples(index=False, name=None)
        ]


class ObservationBacktester(Backtester):
    """Opt-in weight strategy with automatic observation provenance."""

    def __init__(
        self,
        *,
        observation_data: ObservationData,
        max_data_age_days: int | None = None,
        **kwargs: Any,
    ) -> None:
        _age_limit(max_data_age_days)
        if kwargs.get("execution_mode", "weights") != "weights":
            raise ValueError("ObservationBacktester currently supports weight mode only")
        if kwargs.get("strict_reporting") is False:
            raise ValueError("Observation replay requires strict reporting")
        self._observation_data = observation_data
        self._max_data_age_days = max_data_age_days
        self._observation_calls: list[list[dict[str, Any]]] = []
        kwargs["strict_reporting"] = True
        kwargs["strict_data_fetch"] = True
        super().__init__(**kwargs)
        self._strict_reporting = True
        self._strict_data_fetch = True
        self._validate_observation_instrument()
        self._original_observation_strategy_type = self.strategy_type
        self._sample_price_mode = self._source == "sample"
        self._label_observation_research()

    def _validate_observation_instrument(self) -> None:
        if self.instruments != [self._observation_data.instrument]:
            raise ValueError("Observation replay requires exactly one matching instrument")

    async def _process_market_data(self) -> None:
        await super()._process_market_data()
        self._validate_observation_instrument()
        summary = (getattr(self, "_api_response", None) or {}).get("summary") or {}
        self._sample_price_mode = self._source == "sample" or summary.get("source") == "sample"
        self._label_observation_research()
        if list(self.data.get_feature("adj_close").columns) != [self._observation_data.instrument]:
            raise ValueError("Prepared prices must match the observation instrument")

    def _label_observation_research(self) -> None:
        self.strategy_type = (
            f"Synthetic research: {self._original_observation_strategy_type}"
            if self._observation_data.synthetic or self._sample_price_mode
            else self._original_observation_strategy_type
        )

    def observations_at(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        aligned = self._observation_data.align(index, max_data_age_days=self._max_data_age_days)
        self._observation_calls.append(self._observation_data.provenance(aligned))
        return aligned

    async def run_strategy(self) -> None:
        self._observation_calls = []
        await super().run_strategy()
        if not self._observation_calls:
            raise ValueError("Observation strategy must call observations_at")

    async def _generate_strategy_analysis(self) -> None:
        synthetic = self._observation_data.synthetic or self._sample_price_mode
        heading = (
            "SYNTHETIC RESEARCH — not observed market performance.\n"
            f"Synthetic observations: {self._observation_data.synthetic}; "
            f"sample price mode: {self._sample_price_mode}.\n\n"
        )
        if synthetic and self._show_text_reports:
            print(heading)
        await super()._generate_strategy_analysis()
        if synthetic and self._save_text_reports:
            folder = Path(self._reports_directory) / self.strategy_name
            for name in ("performance_report.txt", "summary.txt"):
                path = folder / name
                path.write_text(heading + path.read_text(encoding="utf-8"), encoding="utf-8")

    def _build_run_metadata(self) -> dict[str, Any]:
        if not self._observation_calls:
            raise ValueError("Observation strategy must call observations_at")
        metadata = super()._build_run_metadata()
        if self._sample_price_mode:
            metadata["data_source"] = "sample"
        metadata["external_observations"] = {
            **self._observation_data.metadata,
            "selection": "latest observed_at strictly before decision_at",
            "sample_price_mode": self._sample_price_mode,
            "max_data_age_days": self._max_data_age_days,
            "freshness_basis": "UTC calendar days from data_date",
            "calls": deepcopy(self._observation_calls),
        }
        return metadata

    def _build_strategy_params(self) -> dict[str, str]:
        parameters = super()._build_strategy_params()
        parameters["Observation source"] = self._observation_data.source
        parameters["Synthetic observations"] = str(self._observation_data.synthetic)
        parameters["Sample price mode"] = str(self._sample_price_mode)
        if self._sample_price_mode:
            parameters["Data Source"] = "SAMPLE (synthetic prices)"
        parameters["Observation freshness days"] = str(self._max_data_age_days)
        return parameters
