"""
Hypatia — Precision Math.
Obsessed with exact price levels and mathematical relationships.
Uses WMA, Momentum, OBV for precision timing.
Medium-low mutation. Prefers exact threshold conditions over crossovers.
"""
from __future__ import annotations

import numpy as np

from evolution_engine.evolution.genome import Condition, EntryLogicGene
from evolution_engine.species.base_species import BaseSpecies, SpeciesTraits


class HypatiaSpecies(BaseSpecies):
    traits = SpeciesTraits(
        name="hypatia",
        mutation_rate=0.12,
        mutation_strength=0.15,
        preferred_timeframes=["H1", "M30"],
        preferred_indicators=["WMA", "MOMENTUM", "OBV", "RSI", "ATR"],
        risk_tolerance=0.8,
        tf_weight_vector={
            "H4":  0.20,
            "H1":  0.45,
            "M30": 0.35,
        },
        indicator_weight_vector={
            "WMA":        0.25,
            "MOMENTUM":   0.22,
            "OBV":        0.18,
            "RSI":        0.15,
            "ATR":        0.10,
            "EMA":        0.05,
            "CCI":        0.03,
            "MACD":       0.02,
            "BOLLINGER":  0.00,
            "ADX":        0.00,
            "STOCHASTIC": 0.00,
            "SMA":        0.00,
            "WILLIAMS_R": 0.00,
        },
        logic_type_preference="AND",
        take_profit_r_range=(3.0, 4.0),
        stop_loss_atr_range=(1.0, 1.8),
        max_concurrent_trades_range=(1, 2),
        daily_loss_limit_range=(0.01, 0.03),
        entry_conditions_count=(2, 3),
    )

    def _build_entry_logic(self, rng: np.random.Generator) -> EntryLogicGene:
        # Hypatia favours above/below threshold conditions (precision math)
        from evolution_engine.config.settings import NUM_INDICATORS
        entry = super()._build_entry_logic(rng)
        new_conditions = []
        for cond in entry.conditions:
            if rng.random() < 0.60:
                # Force a threshold-based condition
                comparison = "above_threshold" if rng.random() > 0.5 else "below_threshold"
                threshold = float(rng.uniform(30, 70))
                new_conditions.append(Condition(
                    indicator_idx=cond.indicator_idx,
                    comparison=comparison,
                    reference_idx=None,
                    threshold=threshold,
                    n_bars=cond.n_bars,
                    sub_key=None,
                ))
            else:
                new_conditions.append(cond)
        return EntryLogicGene(
            direction=entry.direction,
            logic_type="AND",
            conditions=new_conditions,
        )
