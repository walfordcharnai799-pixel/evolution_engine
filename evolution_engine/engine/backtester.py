"""
Deterministic backtester.
Same genome + same data = same trades, every time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from evolution_engine.evolution.genome import StrategyGenome
from evolution_engine.indicators.indicator_cache import IndicatorCache
from evolution_engine.strategies.strategy import Strategy, Trade


@dataclass
class BacktestResult:
    genome_id: str
    species: str
    trades: list[Trade]
    equity_curve: pd.Series
    symbol: str
    primary_timeframe: str
    n_bars: int
    initial_equity: float = 10_000.0


MAX_BACKTEST_BARS = 2500   # Cap per TF to keep each backtest under ~5s


class Backtester:
    def __init__(
        self,
        price_data: dict[str, dict[str, pd.DataFrame]],
        indicator_cache: IndicatorCache,
        initial_equity: float = 10_000.0,
        max_bars: int = MAX_BACKTEST_BARS,
    ) -> None:
        self._price_data = self._trim_data(price_data, max_bars)
        self._cache = indicator_cache
        self._initial_equity = initial_equity
        self._default_symbol = next(iter(self._price_data))

    @staticmethod
    def _trim_data(
        price_data: dict[str, dict[str, pd.DataFrame]], max_bars: int
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Use the most recent max_bars for each TF to keep backtests fast."""
        trimmed = {}
        for sym, tfs in price_data.items():
            trimmed[sym] = {}
            for tf, df in tfs.items():
                trimmed[sym][tf] = df.iloc[-max_bars:] if len(df) > max_bars else df
        return trimmed

    def run(
        self,
        genome: StrategyGenome,
        symbol: str | None = None,
    ) -> BacktestResult:
        sym = symbol or self._default_symbol
        strategy = Strategy(
            genome=genome,
            indicator_cache=self._cache,
            price_data=self._price_data,
            symbol=sym,
        )

        trades = strategy.run()
        equity_curve = self._build_equity_curve(trades, genome)
        primary_tf = genome.indicators[0].timeframe
        n_bars = len(self._price_data[sym][primary_tf])

        return BacktestResult(
            genome_id=genome.genome_id,
            species=genome.species,
            trades=trades,
            equity_curve=equity_curve,
            symbol=sym,
            primary_timeframe=primary_tf,
            n_bars=n_bars,
            initial_equity=self._initial_equity,
        )

    def _build_equity_curve(
        self, trades: list[Trade], genome: StrategyGenome
    ) -> pd.Series:
        equity = self._initial_equity
        curve: list[float] = [equity]
        risk_frac = genome.risk.risk_per_trade

        for trade in sorted(trades, key=lambda t: t.entry_bar):
            if trade.pnl_r is None:
                continue
            risk_amount = equity * risk_frac
            pnl = trade.pnl_r * risk_amount
            equity += pnl
            curve.append(equity)

        return pd.Series(curve, name="equity")
