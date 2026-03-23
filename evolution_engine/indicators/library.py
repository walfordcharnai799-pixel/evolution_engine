"""
Pure indicator computation functions.
Each function is stateless: same inputs → same outputs.
All use pandas/numpy only (no TA-Lib dependency).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from evolution_engine.config.settings import INDICATOR_PARAM_RANGES


# ---------------------------------------------------------------------------
# Trend indicators
# ---------------------------------------------------------------------------

def compute_ema(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].ewm(span=int(period), adjust=False).mean()


def compute_sma(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].rolling(window=int(period)).mean()


def compute_wma(df: pd.DataFrame, period: int) -> pd.Series:
    period = int(period)
    weights = np.arange(1, period + 1, dtype=float)

    def _wma(x: np.ndarray) -> float:
        return float(np.dot(x, weights) / weights.sum())

    return df["close"].rolling(window=period).apply(_wma, raw=True)


def compute_macd(
    df: pd.DataFrame, fast: int, slow: int, signal: int
) -> pd.DataFrame:
    fast, slow, signal = int(fast), int(slow), int(signal)
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": histogram},
        index=df.index,
    )


def compute_adx(df: pd.DataFrame, period: int) -> pd.DataFrame:
    period = int(period)
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    dm_plus = high.diff()
    dm_minus = -low.diff()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)

    atr = tr.ewm(span=period, adjust=False).mean()
    di_plus = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()

    return pd.DataFrame(
        {"ADX": adx, "DI_plus": di_plus, "DI_minus": di_minus}, index=df.index
    )


# ---------------------------------------------------------------------------
# Volatility indicators
# ---------------------------------------------------------------------------

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    period = int(period)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_bollinger(
    df: pd.DataFrame, period: int, std_dev: float
) -> pd.DataFrame:
    period = int(period)
    middle = df["close"].rolling(window=period).mean()
    std = df["close"].rolling(window=period).std()
    return pd.DataFrame(
        {
            "upper": middle + std_dev * std,
            "middle": middle,
            "lower": middle - std_dev * std,
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Momentum / oscillator indicators
# ---------------------------------------------------------------------------

def compute_rsi(df: pd.DataFrame, period: int) -> pd.Series:
    period = int(period)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_stochastic(
    df: pd.DataFrame, k_period: int, d_period: int
) -> pd.DataFrame:
    k_period, d_period = int(k_period), int(d_period)
    lowest_low = df["low"].rolling(window=k_period).min()
    highest_high = df["high"].rolling(window=k_period).max()
    k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(window=d_period).mean()
    return pd.DataFrame({"K": k, "D": d}, index=df.index)


def compute_cci(df: pd.DataFrame, period: int) -> pd.Series:
    period = int(period)
    typical = (df["high"] + df["low"] + df["close"]) / 3
    mean_dev = typical.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    return (typical - typical.rolling(window=period).mean()) / (0.015 * mean_dev)


def compute_williams_r(df: pd.DataFrame, period: int) -> pd.Series:
    period = int(period)
    highest_high = df["high"].rolling(window=period).max()
    lowest_low = df["low"].rolling(window=period).min()
    return -100 * (highest_high - df["close"]) / (highest_high - lowest_low).replace(0, np.nan)


def compute_momentum(df: pd.DataFrame, period: int) -> pd.Series:
    period = int(period)
    return df["close"] - df["close"].shift(period)


# ---------------------------------------------------------------------------
# Volume indicators
# ---------------------------------------------------------------------------

def compute_obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    vol = df.get("volume", df.get("tick_volume", pd.Series(0, index=df.index)))
    return (direction * vol).cumsum()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class IndicatorSpec:
    fn: Callable
    params: dict = field(default_factory=dict)


INDICATOR_REGISTRY: dict[str, IndicatorSpec] = {
    "EMA":        IndicatorSpec(fn=compute_ema,        params=INDICATOR_PARAM_RANGES["EMA"]),
    "SMA":        IndicatorSpec(fn=compute_sma,        params=INDICATOR_PARAM_RANGES["SMA"]),
    "WMA":        IndicatorSpec(fn=compute_wma,        params=INDICATOR_PARAM_RANGES["WMA"]),
    "MACD":       IndicatorSpec(fn=compute_macd,       params=INDICATOR_PARAM_RANGES["MACD"]),
    "ADX":        IndicatorSpec(fn=compute_adx,        params=INDICATOR_PARAM_RANGES["ADX"]),
    "RSI":        IndicatorSpec(fn=compute_rsi,        params=INDICATOR_PARAM_RANGES["RSI"]),
    "STOCHASTIC": IndicatorSpec(fn=compute_stochastic, params=INDICATOR_PARAM_RANGES["STOCHASTIC"]),
    "CCI":        IndicatorSpec(fn=compute_cci,        params=INDICATOR_PARAM_RANGES["CCI"]),
    "MOMENTUM":   IndicatorSpec(fn=compute_momentum,   params=INDICATOR_PARAM_RANGES["MOMENTUM"]),
    "BOLLINGER":  IndicatorSpec(fn=compute_bollinger,  params=INDICATOR_PARAM_RANGES["BOLLINGER"]),
    "ATR":        IndicatorSpec(fn=compute_atr,        params=INDICATOR_PARAM_RANGES["ATR"]),
    "WILLIAMS_R": IndicatorSpec(fn=compute_williams_r, params=INDICATOR_PARAM_RANGES["WILLIAMS_R"]),
    "OBV":        IndicatorSpec(fn=compute_obv,        params=INDICATOR_PARAM_RANGES["OBV"]),
}

ALL_INDICATOR_NAMES: list[str] = list(INDICATOR_REGISTRY.keys())
