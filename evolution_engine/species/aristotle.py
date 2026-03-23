"""
Aristotle — The Pure Logician.
Only trades when ALL timeframes agree. Strict IF-THEN logic chains.
Low mutation. Highest confirmation requirement of all species.
Trades very rarely but with extreme precision.
"""
from __future__ import annotations

import numpy as np

from evolution_engine.evolution.genome import EntryLogicGene
from evolution_engine.species.base_species import BaseSpecies, SpeciesTraits


class AristotleSpecies(BaseSpecies):
    traits = SpeciesTraits(
        name="aristotle",
        mutation_rate=0.07,
        mutation_strength=0.08,
        preferred_timeframes=["H4", "H1"],
        preferred_indicators=["EMA", "MACD", "ADX", "RSI", "ATR"],
        risk_tolerance=0.6,
        tf_weight_vector={
            "H4":  0.50,
            "H1":  0.35,
            "M30": 0.15,
        },
        indicator_weight_vector={
            "EMA":        0.25,
            "MACD":       0.22,
            "ADX":        0.20,
            "RSI":        0.15,
            "ATR":        0.10,
            "SMA":        0.05,
            "BOLLINGER":  0.02,
            "CCI":        0.01,
            "MOMENTUM":   0.00,
            "STOCHASTIC": 0.00,
            "WILLIAMS_R": 0.00,
            "WMA":        0.00,
            "OBV":        0.00,
        },
        logic_type_preference="AND",     # ALWAYS AND — pure logician
        take_profit_r_range=(3.0, 5.0),
        stop_loss_atr_range=(1.5, 2.5),
        max_concurrent_trades_range=(1, 1),
        daily_loss_limit_range=(0.01, 0.02),
        entry_conditions_count=(3, 3),    # Always uses maximum 3 conditions
    )

    def _build_entry_logic(self, rng: np.random.Generator) -> EntryLogicGene:
        # Force AND logic with 3 conditions — strict multi-TF confirmation
        entry = super()._build_entry_logic(rng)
        return EntryLogicGene(
            direction=entry.direction,
            logic_type="AND",             # Aristotle never uses OR
            conditions=entry.conditions[:3] if len(entry.conditions) >= 3 else entry.conditions,
        )
