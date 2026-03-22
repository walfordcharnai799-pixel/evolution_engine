"""
MT5 connection layer.
Fetches OHLCV data only — never computes indicators.
Designed to work via MetaTrader5 Python package (runs under Wine on Linux).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from evolution_engine.config.settings import ALLOWED_TIMEFRAMES, LOOKBACK_BARS


class MT5ConnectionError(Exception):
    pass


class MT5Connector:
    def __init__(
        self,
        login: int = 0,
        password: str = "",
        server: str = "",
        terminal_path: str = "",
    ) -> None:
        self._login = login
        self._password = password
        self._server = server
        self._path = terminal_path
        self._connected = False
        self._mt5 = None  # lazy import

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        except ImportError as e:
            raise MT5ConnectionError(
                "MetaTrader5 Python package not installed. "
                "Install it with: pip install MetaTrader5"
            ) from e

        kwargs: dict = {}
        if self._path:
            kwargs["path"] = self._path
        if self._login:
            kwargs["login"] = self._login
            kwargs["password"] = self._password
            kwargs["server"] = self._server

        if not self._mt5.initialize(**kwargs):
            err = self._mt5.last_error()
            raise MT5ConnectionError(f"MT5 initialize failed: {err}")

        self._connected = True
        return True

    def disconnect(self) -> None:
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _tf_constant(self, timeframe: str):
        if timeframe not in ALLOWED_TIMEFRAMES:
            raise ValueError(f"Timeframe {timeframe!r} not in ALLOWED_TIMEFRAMES")
        tf_map = {
            "H4":  self._mt5.TIMEFRAME_H4,
            "H1":  self._mt5.TIMEFRAME_H1,
            "M30": self._mt5.TIMEFRAME_M30,
            "M15": self._mt5.TIMEFRAME_M15,
        }
        return tf_map[timeframe]

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        n_bars: int = LOOKBACK_BARS,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        if not self._connected:
            raise MT5ConnectionError("Not connected to MT5")

        tf_const = self._tf_constant(timeframe)
        last_error = None

        for attempt in range(3):
            if end_time is not None:
                rates = self._mt5.copy_rates_from(symbol, tf_const, end_time, n_bars)
            else:
                rates = self._mt5.copy_rates_from_pos(symbol, tf_const, 0, n_bars)

            if rates is not None and len(rates) > 0:
                break
            last_error = self._mt5.last_error()
            time.sleep(1)
        else:
            raise MT5ConnectionError(
                f"Failed to fetch {symbol} {timeframe} after 3 attempts: {last_error}"
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)

        # Normalise column names
        rename = {
            "open": "open", "high": "high", "low": "low", "close": "close",
            "tick_volume": "volume", "real_volume": "real_volume",
        }
        df.rename(columns=rename, inplace=True, errors="ignore")

        return df[["open", "high", "low", "close", "volume"]]

    def fetch_all_timeframes(
        self,
        symbol: str,
        n_bars: int = LOOKBACK_BARS,
    ) -> dict[str, pd.DataFrame]:
        return {
            tf: self.fetch_ohlcv(symbol, tf, n_bars)
            for tf in ALLOWED_TIMEFRAMES
        }
