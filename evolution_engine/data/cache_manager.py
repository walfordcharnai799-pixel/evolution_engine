"""
Cache manager — determinism anchor.
Once data is cached at session start, every backtest in a generation
reads identical data.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from evolution_engine.config.settings import ALLOWED_TIMEFRAMES, DATA_CACHE_DIR, LOOKBACK_BARS
from evolution_engine.data.mt5_connector import MT5Connector


class CacheManager:
    def __init__(self, cache_dir: str = DATA_CACHE_DIR) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_key(self, symbol: str, timeframe: str, n_bars: int) -> str:
        return f"{symbol}_{timeframe}_{n_bars}"

    def _path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.parquet"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def save(self, key: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._path(key))

    def load(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(key))

    def load_or_fetch(
        self,
        connector: MT5Connector,
        symbol: str,
        timeframe: str,
        n_bars: int = LOOKBACK_BARS,
    ) -> pd.DataFrame:
        key = self.cache_key(symbol, timeframe, n_bars)
        if self.exists(key):
            return self.load(key)
        df = connector.fetch_ohlcv(symbol, timeframe, n_bars)
        self.save(key, df)
        return df

    def load_all_symbols(
        self,
        connector: MT5Connector,
        symbols: list[str],
        n_bars: int = LOOKBACK_BARS,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Return {symbol: {timeframe: df}} for all symbols × TFs."""
        result: dict[str, dict[str, pd.DataFrame]] = {}
        for symbol in symbols:
            result[symbol] = {}
            for tf in ALLOWED_TIMEFRAMES:
                result[symbol][tf] = self.load_or_fetch(connector, symbol, tf, n_bars)
        return result

    def invalidate(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()

    def invalidate_all(self) -> None:
        for f in self._cache_dir.glob("*.parquet"):
            f.unlink()
