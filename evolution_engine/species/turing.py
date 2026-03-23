"""
Turing — The Codebreaker.
Finds recurring sequences and cyclic patterns.
Uses Momentum, Stochastic, Williams %R, OBV for cycle detection.
High mutation rate — constantly probing for new patterns.
Favours slope and divergence conditions.
"""
from __future__ import annotations

import numpy as np

from evolution_engine.evolution.genome import Condition, EntryLogicGene
from evolution_engine.species.base_species import BaseSpecies, SpeciesTraits


class TuringSpecies(BaseSpecies):
    traits = SpeciesTraits(
        name="turing",
        mutation_rate=0.25,
        mutation_strength=0.30,
        preferred_timeframes=["H1", "M30"],
        preferred_indicators=["MOMENTUM", "STOCHASTIC", "WILLIAMS_R", "OBV", "CCI"],
        risk_tolerance=1.0,
        tf_weight_vector={
            "H4":  0.15,
            "H1":  0.45,
            "M30": 0.40,
        },
        indicator_weight_vector={
            "MOMENTUM":   0.25,
            "STOCHASTIC": 0.20,
            "WILLIAMS_R": 0.18,
            "OBV":        0.15,
            "CCI":        0.12,
            "RSI":        0.06,
            "MACD":       0.03,
            "ATR":        0.01,
            "EMA":        0.00,
            "SMA":        0.00,
            "ADX":        0.00,
            "WMA":        0.00,
            "BOLLINGER":  0.00,
        },
        logic_type_preference="OR",    # Turing looks for ANY matching pattern
        take_profit_r_range=(3.0, 4.5),
        stop_loss_atr_range=(0.8, 1.5),
        max_concurrent_trades_range=(1, 2),
        daily_loss_limit_range=(0.01, 0.03),
        entry_conditions_count=(1, 3),
    )

    def _build_entry_logic(self, rng: np.random.Generator) -> EntryLogicGene:
        # Turing favours slope and divergence conditions — cycle detection
        from evolution_engine.config.settings import NUM_INDICATORS
        entry = super()._build_entry_logic(rng)
        new_conditions = []
        for cond in entry.conditions:
            if rng.random() < 0.50:
                comparison = rng.choice(["slope_positive", "slope_negative", "divergence"])
                reference_idx = None
                if comparison == "divergence":
                    others = [i for i in range(NUM_INDICATORS) if i != cond.indicator_idx]
                    reference_idx = int(rng.choice(others)) if others else cond.indicator_idx
                new_conditions.append(Condition(
                    indicator_idx=cond.indicator_idx,
                    comparison=str(comparison),
                    reference_idx=reference_idx,
                    threshold=None,
                    n_bars=int(rng.integers(2, 8)),
                    sub_key=None,
                ))
            else:
                new_conditions.append(cond)
        return EntryLogicGene(
            direction=entry.direction,
            logic_type="OR",
            conditions=new_conditions,
        )
