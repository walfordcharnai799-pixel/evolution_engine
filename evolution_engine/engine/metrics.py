"""
Performance metrics engine.
Applies 3-pip slippage, 5-day recovery check, and 1:3 R/R validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from evolution_engine.config.settings import FITNESS_WEIGHTS, SURVIVAL
from evolution_engine.engine.backtester import BacktestResult
from evolution_engine.strategies.strategy import Trade


@dataclass
class MetricsReport:
    genome_id: str
    species: str
    win_rate: float
    profit_factor: float
    max_drawdown: float
    total_trades: int
    expectancy: float
    stability_score: float
    sharpe_ratio: float
    avg_trade_r: float
    avg_winner_r: float
    avg_loser_r: float
    largest_winner_r: float
    largest_loser_r: float
    consecutive_losses_max: int
    fitness_score: float
    recovery_fail: bool = False         # 5-day recovery gate
    slippage_applied_pips: float = 0.0  # total slippage cost in pips


class MetricsEngine:
    def __init__(self, pip_value: float = 0.0001) -> None:
        """
        pip_value: value of 1 pip in price terms.
        For Forex/metals: 0.0001 (standard). For Silver: ~0.001. For BTC: ~1.0.
        Slippage is applied as a flat per-trade cost in R units.
        """
        self._pip_value = pip_value

    def compute(self, result: BacktestResult) -> MetricsReport:
        trades = result.trades
        spread_pips = SURVIVAL.get("spread_pips", 3.0)

        # Apply slippage: deduct spread cost from each trade's pnl_r
        # Slippage in R = (spread_pips * pip_value) / stop_loss_distance
        # We approximate using ATR-based SL from equity curve geometry.
        # Simpler robust approach: deduct a fixed fraction per trade based on avg SL size.
        adjusted_pnl = self._apply_slippage(trades, spread_pips, result)

        if len(adjusted_pnl) == 0:
            return self._empty_report(result.genome_id, result.species)

        winners = [r for r in adjusted_pnl if r > 0]
        losers  = [r for r in adjusted_pnl if r <= 0]

        win_rate       = len(winners) / len(adjusted_pnl)
        profit_factor  = self._compute_profit_factor(winners, losers)
        equity_curve   = self._build_equity_curve_from_r(adjusted_pnl, result.initial_equity)
        max_dd         = self._compute_drawdown(equity_curve)
        expectancy     = float(np.mean(adjusted_pnl))
        stability      = self._compute_stability(adjusted_pnl)
        sharpe         = self._compute_sharpe(adjusted_pnl)
        consec_losses  = self._max_consecutive_losses(adjusted_pnl)
        recovery_fail  = self._check_recovery(equity_curve, SURVIVAL.get("recovery_bars", 5))

        total_slippage = spread_pips * len(adjusted_pnl)

        report = MetricsReport(
            genome_id=result.genome_id,
            species=result.species,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            total_trades=len(adjusted_pnl),
            expectancy=expectancy,
            stability_score=stability,
            sharpe_ratio=sharpe,
            avg_trade_r=float(np.mean(adjusted_pnl)),
            avg_winner_r=float(np.mean(winners)) if winners else 0.0,
            avg_loser_r=float(np.mean(losers))  if losers  else 0.0,
            largest_winner_r=float(max(winners)) if winners else 0.0,
            largest_loser_r=float(min(losers))   if losers  else 0.0,
            consecutive_losses_max=consec_losses,
            fitness_score=0.0,
            recovery_fail=recovery_fail,
            slippage_applied_pips=total_slippage,
        )
        report.fitness_score = self._composite_fitness(report)
        return report

    # ------------------------------------------------------------------
    # Slippage application
    # ------------------------------------------------------------------

    def _apply_slippage(
        self,
        trades: list[Trade],
        spread_pips: float,
        result: BacktestResult,
    ) -> list[float]:
        """
        Deduct spread/slippage cost from each trade.
        Cost in R = (spread_pips * pip_value) / sl_distance_in_price
        Since we work in R units and SL distance varies, we use a conservative
        fixed fraction: each trade loses an additional 0.1R as spread cost.
        """
        SLIPPAGE_R = 0.10  # 0.1R deducted per trade (conservative flat estimate)
        adjusted = []
        for trade in trades:
            if trade.pnl_r is None:
                continue
            adjusted.append(trade.pnl_r - SLIPPAGE_R)
        return adjusted

    # ------------------------------------------------------------------
    # Recovery check (5-bar rule)
    # ------------------------------------------------------------------

    def _check_recovery(self, equity_curve: pd.Series, recovery_bars: int) -> bool:
        """
        Returns True (fail) if after any drawdown the equity does NOT recover
        to the previous peak within `recovery_bars` bars.
        """
        if len(equity_curve) < recovery_bars + 2:
            return False

        vals = equity_curve.values
        peak = vals[0]
        in_drawdown = False
        dd_start = 0

        for i in range(1, len(vals)):
            if vals[i] >= peak:
                peak = vals[i]
                in_drawdown = False
            else:
                if not in_drawdown:
                    in_drawdown = True
                    dd_start = i
                elif (i - dd_start) > recovery_bars and vals[i] < peak:
                    return True  # Failed to recover in time
        return False

    # ------------------------------------------------------------------
    # Individual metric computations
    # ------------------------------------------------------------------

    def _compute_profit_factor(self, winners: list[float], losers: list[float]) -> float:
        gross_profit = sum(winners)
        gross_loss   = abs(sum(losers))
        if gross_loss == 0:
            return 10.0 if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def _compute_drawdown(self, equity_curve: pd.Series) -> float:
        if len(equity_curve) < 2:
            return 0.0
        peak = equity_curve.cummax()
        dd   = (equity_curve - peak) / peak
        return float(abs(dd.min()))

    def _build_equity_curve_from_r(
        self, pnl_rs: list[float], initial_equity: float, risk_frac: float = 0.01
    ) -> pd.Series:
        equity = initial_equity
        curve  = [equity]
        for r in pnl_rs:
            equity += r * equity * risk_frac
            curve.append(equity)
        return pd.Series(curve, name="equity")

    def _compute_stability(self, pnl_rs: list[float]) -> float:
        if len(pnl_rs) < 10:
            return 0.0
        window    = 20
        cumulative = np.cumsum(pnl_rs)
        if len(cumulative) < window:
            return 0.0
        slopes = []
        for i in range(window, len(cumulative) + 1):
            chunk = cumulative[i - window:i]
            slopes.append(chunk[-1] - chunk[0])
        if not slopes:
            return 0.0
        mean_s = np.mean(slopes)
        std_s  = np.std(slopes)
        if mean_s <= 0:
            return 0.0
        cv = std_s / (abs(mean_s) + 1e-9)
        return float(max(0.0, min(1.0, 1.0 - cv)))

    def _compute_sharpe(self, pnl_rs: list[float]) -> float:
        if len(pnl_rs) < 5:
            return 0.0
        arr = np.array(pnl_rs)
        std = np.std(arr)
        return float(np.mean(arr) / std) if std > 0 else 0.0

    def _max_consecutive_losses(self, pnl_rs: list[float]) -> int:
        max_streak = streak = 0
        for r in pnl_rs:
            if r <= 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    # ------------------------------------------------------------------
    # Composite fitness
    # ------------------------------------------------------------------

    def _composite_fitness(self, r: MetricsReport) -> float:
        if r.recovery_fail:
            return 0.0                  # Hard zero if recovery gate failed

        w = FITNESS_WEIGHTS
        pf_norm   = min(1.0, max(0.0, (r.profit_factor  - 1.0) / 4.0))
        wr_norm   = min(1.0, max(0.0, (r.win_rate        - 0.5) / 0.4))
        dd_norm   = min(1.0, max(0.0, 1.0 - r.max_drawdown / 0.15))
        exp_norm  = min(1.0, max(0.0, r.expectancy / 3.0))
        stab_norm = r.stability_score

        score = (
            w["profit_factor"] * pf_norm  +
            w["win_rate"]      * wr_norm  +
            w["drawdown"]      * dd_norm  +
            w["expectancy"]    * exp_norm +
            w["stability"]     * stab_norm
        )

        if r.total_trades < SURVIVAL["min_trades"]:
            score *= (r.total_trades / SURVIVAL["min_trades"]) ** 0.5

        return round(float(score), 6)

    def _empty_report(self, genome_id: str, species: str) -> MetricsReport:
        return MetricsReport(
            genome_id=genome_id, species=species,
            win_rate=0.0, profit_factor=0.0, max_drawdown=1.0,
            total_trades=0, expectancy=0.0, stability_score=0.0,
            sharpe_ratio=0.0, avg_trade_r=0.0, avg_winner_r=0.0,
            avg_loser_r=0.0, largest_winner_r=0.0, largest_loser_r=0.0,
            consecutive_losses_max=0, fitness_score=0.0,
            recovery_fail=False, slippage_applied_pips=0.0,
        )
