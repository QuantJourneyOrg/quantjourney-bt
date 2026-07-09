"""
Risk Model ABC and Chain
========================

Every risk model implements ``adjust(weights, returns) → weights``.
``RiskModelChain`` composes multiple models sequentially.

Copyright (c) 2026 QuantJourney.
Licensed under the Apache License 2.0.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


class RiskModel(abc.ABC):
    """
    Abstract risk-adjustment layer.

    Receives raw target weights and historical asset returns,
    returns adjusted weights.  Stateless between calls to ``adjust``
    (state like rolling vol is computed on the fly from the provided
    returns window).
    """

    @abc.abstractmethod
    def adjust(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        *,
        metadata: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Adjust *weights* given historical *returns*.

        Parameters
        ----------
        weights : DataFrame
            Target weights, shape (n_dates, n_instruments).  May sum
            to > 1 (leveraged) or < 1 (partial investment).
        returns : DataFrame
            Daily asset returns, same shape / columns as *weights*.
        metadata : dict, optional
            Extra context (e.g. sector map, factor exposures).

        Returns
        -------
        DataFrame
            Adjusted weights, same shape as input.
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}()"


class RiskModelChain(RiskModel):
    """
    Compose multiple risk models sequentially.

    The output of model *i* is fed as input to model *i+1*.

    Example::

        chain = RiskModelChain([
            VolTargetModel(target_vol=0.15),
            PositionLimitModel(max_weight=0.40),
        ])
        adjusted = chain.adjust(weights, returns)
    """

    def __init__(self, models: List[RiskModel]):
        if not models:
            raise ValueError("RiskModelChain requires at least one model")
        self._models = list(models)

    def adjust(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        *,
        metadata: Optional[Dict] = None,
    ) -> pd.DataFrame:
        w = weights
        for model in self._models:
            w = model.adjust(w, returns, metadata=metadata)
        return w

    @property
    def name(self) -> str:
        names = ", ".join(m.name for m in self._models)
        return f"Chain({names})"

    def __repr__(self) -> str:
        return self.name

    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self):
        return iter(self._models)
