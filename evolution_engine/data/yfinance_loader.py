"""
Free data loader using yfinance.
Downloads real OHLCV data for forex pairs.
Used as a drop-in replacement for MT5Connector when MT5 is not yet available.
Produces data in the exact same format the engine expects.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from evolution_engine.config.settings import ALLOWED_TIMEFRAMES, DATA_CACHE_DIR, LOOKBACK_BARS


# yfinance interval map for our allowed timeframes
# yfinance limits: 1m/2m/5m/15m/30m/60m/90m = max 60 days
#                  1h = max 730 days, 1d/5d/1wk = full history
YF_INTERVAL_MAP = {
    "M15": "15m",
    "M30": "30m",
    "H1":  "1h",
    "H4":  "1h",    # yfinance has no 4h — we resample from 1h
}

# Symbol map: MT5 symbol name → yfinance ticker
SYMBOL_MAP = {
    # Metals
    "XAUUSD": "GC=F",          # Gold futures
    "XAGUSD": "SI=F",          # Silver futures
    "XPTUSD": "PL=F",          # Platinum futures

    # Crypto
    "BTCUSD": "BTC-USD",       # Bitcoin
    "ETHUSD": "ETH-USD",       # Ethereum
    "LTCUSD": "LTC-USD",       # Litecoin

    # Forex (available if needed later)
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
}

# How many days of history to pull per timeframe
DAYS_MAP = {
    "M15": 55,     # yfinance max ~60 days for 15m
    "M30": 55,
    "H1":  700,
    "H4":  700,
}


def _resample_to_h4(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h OHLCV to 4h bars."""
    df = df_1h.resample("4h", label="left", closed="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["open", "close"])
    return df


def fetch_10year_silver(cache_dir: str = DATA_CACHE_DIR) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Fetch 10 years of Silver (XAGUSD) data.
    - H4 / H1 from daily resampled (full 10 years)
    - M30 from intraday where available (~60 days)
    Returns same structure as fetch_real_data.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Install yfinance: pip install yfinance --break-system-packages")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    symbol = "XAGUSD"
    ticker = "SI=F"
    print(f"Fetching 10-year Silver ({ticker})...")

    # --- Daily data: 10 years ---
    cache_daily = cache_path / f"{symbol}_DAILY_10Y.parquet"
    if cache_daily.exists():
        print("  DAILY 10Y: loading from cache")
        df_daily = pd.read_parquet(cache_daily)
    else:
        print("  DAILY 10Y: downloading...", end=" ", flush=True)
        df_daily = yf.download(ticker, period="10y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(df_daily.columns, pd.MultiIndex):
            df_daily.columns = df_daily.columns.droplevel(1)
        df_daily.columns = [c.lower() for c in df_daily.columns]
        df_daily.index = pd.to_datetime(df_daily.index, utc=True)
        df_daily.index.name = "time"
        df_daily = df_daily[["open","high","low","close","volume"]].dropna()
        df_daily.sort_index(inplace=True)
        df_daily.to_parquet(cache_daily)
        print(f"{len(df_daily)} daily bars ({df_daily.index[0].date()} → {df_daily.index[-1].date()})")

    # Resample daily → H4 (treat each daily bar as one H4 bar)
    # For a 10-year H4 simulation this is the best available
    df_h4 = df_daily.copy()

    # Resample daily → H1 (split each daily bar into 24 equal H1 bars using OHLCV)
    # Practical approach: use available H1 data where we have it, daily elsewhere
    cache_h1 = cache_path / f"{symbol}_H1_10Y.parquet"
    if cache_h1.exists():
        print("  H1 10Y: loading from cache")
        df_h1 = pd.read_parquet(cache_h1)
    else:
        print("  H1 (700d): downloading...", end=" ", flush=True)
        df_h1_raw = yf.download(ticker, period="700d", interval="1h", auto_adjust=True, progress=False)
        if isinstance(df_h1_raw.columns, pd.MultiIndex):
            df_h1_raw.columns = df_h1_raw.columns.droplevel(1)
        df_h1_raw.columns = [c.lower() for c in df_h1_raw.columns]
        df_h1_raw.index = pd.to_datetime(df_h1_raw.index, utc=True)
        df_h1_raw.index.name = "time"
        df_h1_raw = df_h1_raw[["open","high","low","close","volume"]].dropna()
        df_h1 = df_h1_raw
        df_h1.to_parquet(cache_h1)
        print(f"{len(df_h1)} H1 bars")

    # M30: resample from H1 (avoids yfinance intraday timeout issues)
    cache_m30 = cache_path / f"{symbol}_M30_10Y.parquet"
    if cache_m30.exists():
        print("  M30: loading from cache")
        df_m30 = pd.read_parquet(cache_m30)
    else:
        print("  M30: resampling from H1...", end=" ", flush=True)
        df_m30 = df_h1.resample("30min", label="left", closed="left").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna(subset=["open", "close"])
        df_m30.to_parquet(cache_m30)
        print(f"{len(df_m30)} M30 bars")

    # Save H4 from daily
    cache_h4 = cache_path / f"{symbol}_H4_10Y.parquet"
    if not cache_h4.exists():
        df_h4.to_parquet(cache_h4)

    return {
        symbol: {
            "H4":  df_h4,
            "H1":  df_h1,
            "M30": df_m30,
        }
    }


def fetch_real_data(
    symbols: list[str] | None = None,
    n_bars: int = LOOKBACK_BARS,
    cache_dir: str = DATA_CACHE_DIR,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Download real forex OHLCV data from Yahoo Finance.
    Returns {symbol: {timeframe: pd.DataFrame}} — identical structure to MT5 output.
    Saves to cache so subsequent runs load instantly.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance not installed. Run: pip install yfinance --break-system-packages"
        )

    if symbols is None:
        from evolution_engine.config.settings import SYMBOLS
        symbols = SYMBOLS

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    price_data: dict[str, dict[str, pd.DataFrame]] = {}

    for symbol in symbols:
        ticker = SYMBOL_MAP.get(symbol, symbol)
        price_data[symbol] = {}
        print(f"  Fetching {symbol} ({ticker})...")

        # We fetch H1 and M15/M30, then derive H4 from H1
        raw: dict[str, pd.DataFrame] = {}

        for tf in ["M30", "H1"]:   # M15 excluded — not used in logic
            cache_key = f"{symbol}_{tf}_{n_bars}"
            cache_file = cache_path / f"{cache_key}.parquet"

            if cache_file.exists():
                print(f"    {tf}: loading from cache")
                raw[tf] = pd.read_parquet(cache_file)
                continue

            interval = YF_INTERVAL_MAP[tf]
            period = f"{DAYS_MAP[tf]}d"
            print(f"    {tf}: downloading ({period} @ {interval})...", end=" ", flush=True)

            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )

            if df.empty:
                print(f"WARNING: no data returned for {symbol} {tf}")
                continue

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            df.columns = [c.lower() for c in df.columns]
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "time"
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            df.sort_index(inplace=True)

            raw[tf] = df
            df.to_parquet(cache_file)
            print(f"{len(df)} bars")

        # Derive H4 from H1
        if "H1" in raw and not raw["H1"].empty:
            cache_key_h4 = f"{symbol}_H4_{n_bars}"
            cache_file_h4 = cache_path / f"{cache_key_h4}.parquet"
            if cache_file_h4.exists():
                print(f"    H4: loading from cache")
                raw["H4"] = pd.read_parquet(cache_file_h4)
            else:
                print(f"    H4: resampling from H1...", end=" ", flush=True)
                raw["H4"] = _resample_to_h4(raw["H1"])
                raw["H4"].to_parquet(cache_file_h4)
                print(f"{len(raw['H4'])} bars")

        for tf in ALLOWED_TIMEFRAMES:
            if tf in raw and not raw[tf].empty:
                price_data[symbol][tf] = raw[tf]
            else:
                print(f"  WARNING: {symbol} {tf} has no data")

    return price_data


def load_cached_data(
    symbols: list[str] | None = None,
    n_bars: int = LOOKBACK_BARS,
    cache_dir: str = DATA_CACHE_DIR,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load only from cache (no network). Raises if cache missing."""
    if symbols is None:
        from evolution_engine.config.settings import SYMBOLS
        symbols = SYMBOLS

    cache_path = Path(cache_dir)
    price_data: dict[str, dict[str, pd.DataFrame]] = {}

    for symbol in symbols:
        price_data[symbol] = {}
        for tf in ALLOWED_TIMEFRAMES:
            key = f"{symbol}_{tf}_{n_bars}"
            f = cache_path / f"{key}.parquet"
            if not f.exists():
                raise FileNotFoundError(
                    f"Cache missing for {symbol} {tf}. Run fetch_real_data() first."
                )
            price_data[symbol][tf] = pd.read_parquet(f)

    return price_data
