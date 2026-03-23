"""
Curie — The Stoic.
Hardworking, precise, disciplined. Prefers H4. Very low mutation rate.
Always exits via ATR-based stops. Strict risk control.
"""
from __future__ import annotations

import numpy as np

from evolution_engine.evolution.genome import ExitLogicGene, RiskGene
from evolution_engine.species.base_species import BaseSpecies, SpeciesTraits


class CurieSpecies(BaseSpecies):
    traits = SpeciesTraits(
        name="curie",
        mutation_rate=0.08,
        mutation_strength=0.10,
        preferred_timeframes=["H4", "H1"],
        preferred_indicators=["EMA", "ADX", "OBV", "MACD", "ATR"],
        risk_tolerance=0.6,
        tf_weight_vector={
            "H4":  0.65,
            "H1":  0.25,
            "M30": 0.10,
        },
        indicator_weight_vector={
            "EMA":        0.25,
            "ADX":        0.22,
            "OBV":        0.18,
            "MACD":       0.15,
            "ATR":        0.12,
            "SMA":        0.04,
            "RSI":        0.02,
            "BOLLINGER":  0.01,
            "WMA":        0.01,
            "CCI":        0.00,
            "STOCHASTIC": 0.00,
            "MOMENTUM":   0.00,
            "WILLIAMS_R": 0.00,
        },
        logic_type_preference="AND",
        take_profit_r_range=(1.5, 3.5),
        stop_loss_atr_range=(1.5, 3.0),     # stoic: wider stops, patient
        max_concurrent_trades_range=(1, 1), # never stack: 1 trade only
        daily_loss_limit_range=(0.01, 0.03),
        entry_conditions_count=(2, 3),       # thorough confirmation
    )

    def _build_exit_logic(self, rng: np.random.Generator) -> ExitLogicGene:
        # Curie always uses fixed_rr — no ambiguity in exit
        tp_r = float(rng.uniform(*self.traits.take_profit_r_range))
        sl_mult = float(rng.uniform(*self.traits.stop_loss_atr_range))
        return ExitLogicGene(
            exit_type="fixed_rr",
            take_profit_r=round(tp_r, 2),
            stop_loss_atr_mult=round(sl_mult, 2),
        )

    def _build_risk(self, rng: np.random.Generator) -> RiskGene:
        # Very conservative risk
        risk = float(rng.uniform(0.005, 0.015))
        return RiskGene(
            risk_per_trade=round(risk, 4),
            max_concurrent_trades=1,
            daily_loss_limit=round(float(rng.uniform(0.01, 0.03)), 4),
        )
