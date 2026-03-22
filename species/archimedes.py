"""
Archimedes — The Leverage King.
Targets exactly 1:3+ R/R on every trade. Never compromises the ratio.
Precise stop/TP placement using ATR. Medium mutation.
Uses MACD, EMA, ADX for trend confirmation before entry.
"""
from __future__ import annotations

import numpy as np

from evolution_engine.evolution.genome import ExitLogicGene
from evolution_engine.species.base_species import BaseSpecies, SpeciesTraits


class ArchimedesSpecies(BaseSpecies):
    traits = SpeciesTraits(
        name="archimedes",
        mutation_rate=0.15,
        mutation_strength=0.18,
        preferred_timeframes=["H4", "H1"],
        preferred_indicators=["MACD", "EMA", "ADX", "ATR", "RSI"],
        risk_tolerance=0.8,
        tf_weight_vector={
            "H4":  0.45,
            "H1":  0.40,
            "M30": 0.15,
        },
        indicator_weight_vector={
            "MACD":       0.28,
            "EMA":        0.22,
            "ADX":        0.18,
            "ATR":        0.14,
            "RSI":        0.10,
            "BOLLINGER":  0.05,
            "SMA":        0.02,
            "CCI":        0.01,
            "MOMENTUM":   0.00,
            "STOCHASTIC": 0.00,
            "WILLIAMS_R": 0.00,
            "WMA":        0.00,
            "OBV":        0.00,
        },
        logic_type_preference="AND",
        take_profit_r_range=(3.0, 5.0),    # Always >= 3R
        stop_loss_atr_range=(1.0, 2.0),
        max_concurrent_trades_range=(1, 2),
        daily_loss_limit_range=(0.01, 0.02),
        entry_conditions_count=(2, 3),
    )

    def _build_exit_logic(self, rng: np.random.Generator) -> ExitLogicGene:
        # Archimedes enforces minimum 3:1 R/R — hardwired
        tp_r = float(rng.uniform(3.0, 5.0))
        sl_mult = float(rng.uniform(*self.traits.stop_loss_atr_range))
        return ExitLogicGene(
            exit_type="fixed_rr",
            take_profit_r=round(tp_r, 2),
            stop_loss_atr_mult=round(sl_mult, 2),
        )
