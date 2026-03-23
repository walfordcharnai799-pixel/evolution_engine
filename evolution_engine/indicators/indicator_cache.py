"""
Memoized indicator computation.
Avoids recomputing the same indicator series for multiple genomes in a generation.
"""
from __future__ import annotations

import hashlib
import json
from typing import Union

import pandas as pd

from evolution_engine.indicators.library import INDICATOR_REGISTRY


class IndicatorCache:
    def __init__(self) -> None:
        self._cache: dict[str, Union[pd.Series, pd.DataFrame]] = {}

    def _make_key(
        self,
        df: pd.DataFrame,
        indicator_name: str,
        params: dict,
    ) -> str:
        params_str = json.dumps(params, sort_keys=True)
        data_sig = f"{df.index[0]}_{len(df)}"
        raw = f"{indicator_name}|{params_str}|{data_sig}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get_or_compute(
        self,
        df: pd.DataFrame,
        indicator_name: str,
        params: dict,
    ) -> Union[pd.Series, pd.DataFrame]:
        key = self._make_key(df, indicator_name, params)
        if key not in self._cache:
            spec = INDICATOR_REGISTRY[indicator_name]
            self._cache[key] = spec.fn(df, **params)
        return self._cache[key]

    def clear(self) -> None:
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)
