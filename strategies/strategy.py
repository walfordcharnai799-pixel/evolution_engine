"""
Strategy — a runnable strategy driven entirely by its genome.
No hardcoded logic: all behavior comes from the genome's genes.
Multi-timeframe aware with strict no-lookahead-bias enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd

from evolution_engine.evolution.genome import (
    Condition,
    EntryLogicGene,
    ExitLogicGene,
    IndicatorGene,
    StrategyGenome,
)
from evolution_engine.indicators.indicator_cache import IndicatorCache


@dataclass
class Trade:
    entry_bar: int
    entry_price: float
    direction: str                  # "long" or "short"
    stop_loss: float
    take_profit: float
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    pnl_r: Optional[float] = None  # profit in R units
    exit_reason: Optional[str] = None  # "tp_hit", "sl_hit", "forced_close"

    def is_open(self) -> bool:
        return self.exit_bar is None


class Strategy:
    def __init__(
        self,
        genome: StrategyGenome,
        indicator_cache: IndicatorCache,
        price_data: dict[str, dict[str, pd.DataFrame]],
        symbol: str,
    ) -> None:
        self._genome = genome
        self._cache = indicator_cache
        self._price_data = price_data
        self._symbol = symbol

        # Pre-compute indicator series for all indicator genes
        self._indicator_series: list[Union[pd.Series, pd.DataFrame]] = []
        for gene in genome.indicators:
            df = price_data[symbol][gene.timeframe]
            series = indicator_cache.get_or_compute(df, gene.name, gene.params)
            self._indicator_series.append(series)

        # Primary timeframe = timeframe of first indicator (entry_trigger)
        self._primary_tf = genome.indicators[0].timeframe
        self._primary_df = price_data[symbol][self._primary_tf]

        # ATR series for stop computation (always computed on primary TF)
        from evolution_engine.indicators.library import compute_atr
        self._atr = compute_atr(self._primary_df, period=14)

        # Pre-build multi-TF alignment lookup (vectorised)
        self._tf_bar_map: dict = {}
        self._tf_aligned: dict[str, np.ndarray] = {}
        self._build_tf_alignment()

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run(self) -> list[Trade]:
        closed_trades: list[Trade] = []
        open_trades: list[Trade] = []
        genome = self._genome
        primary_df = self._primary_df
        n_bars = len(primary_df)
        max_concurrent = genome.risk.max_concurrent_trades

        # Need at least 50 bars of warmup for indicators
        warmup = 50

        for bar_idx in range(warmup, n_bars):
            # 1. Manage open trades (check TP/SL on current bar)
            still_open = []
            for trade in open_trades:
                closed = self._evaluate_exit(trade, bar_idx)
                if closed:
                    closed_trades.append(trade)
                else:
                    still_open.append(trade)
            open_trades = still_open

            # 2. Entry — only if under concurrent trade limit
            if len(open_trades) < max_concurrent:
                direction = self._evaluate_entry(bar_idx)
                if direction is not None:
                    trade = self._open_trade(bar_idx, direction)
                    if trade is not None:
                        open_trades.append(trade)

        # Force-close all open trades at last bar
        last_bar = n_bars - 1
        last_close = float(primary_df["close"].iloc[last_bar])
        for trade in open_trades:
            trade.exit_bar = last_bar
            trade.exit_price = last_close
            trade.exit_reason = "forced_close"
            trade.pnl_r = self._compute_pnl_r(trade, last_close)
            closed_trades.append(trade)

        return closed_trades

    # ------------------------------------------------------------------
    # Entry evaluation
    # ------------------------------------------------------------------

    def _evaluate_entry(self, bar_idx: int) -> Optional[str]:
        genome = self._genome
        entry_gene = genome.entry_logic

        if entry_gene.direction == "both":
            long_ok = self._check_conditions(entry_gene.conditions, bar_idx, "long", entry_gene.logic_type)
            short_ok = self._check_conditions(entry_gene.conditions, bar_idx, "short", entry_gene.logic_type)
            if long_ok:
                return "long"
            if short_ok:
                return "short"
            return None
        else:
            ok = self._check_conditions(
                entry_gene.conditions, bar_idx, entry_gene.direction, entry_gene.logic_type
            )
            return entry_gene.direction if ok else None

    def _check_conditions(
        self,
        conditions: list[Condition],
        bar_idx: int,
        direction: str,
        logic_type: str,
    ) -> bool:
        if not conditions:
            return False
        results = [self._eval_condition(cond, bar_idx, direction) for cond in conditions]
        if logic_type == "AND":
            return all(results)
        else:  # OR
            return any(results)

    def _eval_condition(
        self, cond: Condition, bar_idx: int, direction: str
    ) -> bool:
        val = self._get_scalar(cond.indicator_idx, bar_idx, cond.sub_key)
        if val is None or np.isnan(val):
            return False

        comparison = cond.comparison

        if comparison in ("crossover", "crossunder"):
            if cond.reference_idx is None:
                return False
            val_prev = self._get_scalar(cond.indicator_idx, bar_idx - 1, cond.sub_key)
            ref_curr = self._get_scalar(cond.reference_idx, bar_idx, cond.sub_key)
            ref_prev = self._get_scalar(cond.reference_idx, bar_idx - 1, cond.sub_key)
            if any(x is None or np.isnan(x) for x in [val_prev, ref_curr, ref_prev]):
                return False
            if comparison == "crossover":
                crossed = (val_prev <= ref_prev) and (val > ref_curr)
            else:
                crossed = (val_prev >= ref_prev) and (val < ref_curr)
            # For direction-aware crossover, long uses crossover, short uses crossunder
            if direction == "long":
                return crossed if comparison == "crossover" else not crossed
            else:
                return crossed if comparison == "crossunder" else not crossed

        elif comparison == "above_threshold":
            threshold = cond.threshold if cond.threshold is not None else 50.0
            return float(val) > threshold

        elif comparison == "below_threshold":
            threshold = cond.threshold if cond.threshold is not None else 50.0
            return float(val) < threshold

        elif comparison in ("slope_positive", "slope_negative"):
            n = max(1, min(cond.n_bars, bar_idx))
            val_past = self._get_scalar(cond.indicator_idx, bar_idx - n, cond.sub_key)
            if val_past is None or np.isnan(val_past):
                return False
            slope_positive = float(val) > float(val_past)
            if comparison == "slope_positive":
                return slope_positive if direction == "long" else not slope_positive
            else:
                return not slope_positive if direction == "long" else slope_positive

        elif comparison == "divergence":
            if cond.reference_idx is None:
                return False
            n = max(1, min(cond.n_bars, bar_idx))
            price_now = float(self._primary_df["close"].iloc[bar_idx])
            price_past = float(self._primary_df["close"].iloc[bar_idx - n])
            val_past = self._get_scalar(cond.indicator_idx, bar_idx - n, cond.sub_key)
            if val_past is None or np.isnan(val_past):
                return False
            price_up = price_now > price_past
            indicator_up = float(val) > float(val_past)
            # Divergence: price and indicator move in opposite directions
            return price_up != indicator_up

        return False

    # ------------------------------------------------------------------
    # Exit evaluation
    # ------------------------------------------------------------------

    def _evaluate_exit(self, trade: Trade, bar_idx: int) -> bool:
        df = self._primary_df
        bar = df.iloc[bar_idx]
        high = float(bar["high"])
        low = float(bar["low"])

        if trade.direction == "long":
            if low <= trade.stop_loss:
                trade.exit_bar = bar_idx
                trade.exit_price = trade.stop_loss
                trade.exit_reason = "sl_hit"
                trade.pnl_r = self._compute_pnl_r(trade, trade.stop_loss)
                return True
            if high >= trade.take_profit:
                trade.exit_bar = bar_idx
                trade.exit_price = trade.take_profit
                trade.exit_reason = "tp_hit"
                trade.pnl_r = self._compute_pnl_r(trade, trade.take_profit)
                return True
        else:  # short
            if high >= trade.stop_loss:
                trade.exit_bar = bar_idx
                trade.exit_price = trade.stop_loss
                trade.exit_reason = "sl_hit"
                trade.pnl_r = self._compute_pnl_r(trade, trade.stop_loss)
                return True
            if low <= trade.take_profit:
                trade.exit_bar = bar_idx
                trade.exit_price = trade.take_profit
                trade.exit_reason = "tp_hit"
                trade.pnl_r = self._compute_pnl_r(trade, trade.take_profit)
                return True

        return False

    # ------------------------------------------------------------------
    # Trade construction
    # ------------------------------------------------------------------

    def _open_trade(self, bar_idx: int, direction: str) -> Optional[Trade]:
        df = self._primary_df
        entry_price = float(df["close"].iloc[bar_idx])  # enter on close of signal bar

        atr_val = self._atr.iloc[bar_idx]
        if np.isnan(atr_val) or atr_val <= 0:
            return None

        sl_dist = float(atr_val) * self._genome.exit_logic.stop_loss_atr_mult
        tp_dist = sl_dist * self._genome.exit_logic.take_profit_r

        if direction == "long":
            stop_loss = entry_price - sl_dist
            take_profit = entry_price + tp_dist
        else:
            stop_loss = entry_price + sl_dist
            take_profit = entry_price - tp_dist

        return Trade(
            entry_bar=bar_idx,
            entry_price=entry_price,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def _compute_pnl_r(self, trade: Trade, exit_price: float) -> float:
        sl_dist = abs(trade.entry_price - trade.stop_loss)
        if sl_dist == 0:
            return 0.0
        if trade.direction == "long":
            pnl = exit_price - trade.entry_price
        else:
            pnl = trade.entry_price - exit_price
        return pnl / sl_dist

    # ------------------------------------------------------------------
    # Indicator value access
    # ------------------------------------------------------------------

    def _get_scalar(
        self,
        indicator_idx: int,
        bar_idx: int,
        sub_key: Optional[str] = None,
    ) -> Optional[float]:
        if bar_idx < 0 or bar_idx >= len(self._primary_df):
            return None

        gene = self._genome.indicators[indicator_idx]
        series_or_df = self._indicator_series[indicator_idx]

        # Align bar_idx from primary TF to this indicator's TF
        target_tf = gene.timeframe
        if target_tf != self._primary_tf:
            target_bar_idx = self._get_aligned_tf_idx(bar_idx, target_tf)
            if target_bar_idx is None:
                return None
        else:
            target_bar_idx = bar_idx

        # Use bar_idx - 1 (completed bar) to prevent lookahead bias
        target_bar_idx = max(0, target_bar_idx - 1)

        try:
            if isinstance(series_or_df, pd.DataFrame):
                # Multi-column indicator: pick sub_key or first column
                col = sub_key if sub_key and sub_key in series_or_df.columns else series_or_df.columns[0]
                val = series_or_df.iloc[target_bar_idx][col]
            else:
                val = series_or_df.iloc[target_bar_idx]
            return float(val)
        except (IndexError, KeyError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Multi-TF alignment (vectorised — built once, shared via cache)
    # ------------------------------------------------------------------

    def _build_tf_alignment(self) -> None:
        primary_times = self._primary_df.index.asi8   # nanosecond int array — fast
        tf_dfs = {
            tf: self._price_data[self._symbol][tf]
            for tf in self._price_data[self._symbol]
            if tf != self._primary_tf
        }

        # Build a {tf: np.ndarray} of aligned bar indices
        aligned: dict[str, np.ndarray] = {}
        for tf, df in tf_dfs.items():
            target_times = df.index.asi8
            # searchsorted over the full primary array in one numpy call
            raw = np.searchsorted(target_times, primary_times, side="right") - 1
            aligned[tf] = np.clip(raw, 0, len(df) - 1)

        # Store as dict-of-arrays for O(1) column access per bar
        self._tf_aligned = aligned          # {tf: np.ndarray of shape (n_primary_bars,)}
        self._tf_bar_map = {}               # kept for compatibility but no longer used in hot loop

    def _get_aligned_tf_idx(self, bar_idx: int, target_tf: str) -> Optional[int]:
        arr = self._tf_aligned.get(target_tf)
        if arr is None:
            return None
        return int(arr[bar_idx])
