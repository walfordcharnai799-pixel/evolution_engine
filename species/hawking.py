"""
Hawking — The Black Hole.
Trades explosive bounces from extreme compression zones.
Waits for the singularity — then bets on the explosion out.
Medium mutation. Uses ATR contraction → expansion trigger logic.
"""
from __future__ import annotations

import numpy as np

from evolution_engine.evolution.genome import IndicatorGene
from evolution_engine.species.base_species import BaseSpecies, SpeciesTraits


class HawkingSpecies(BaseSpecies):
    traits = SpeciesTraits(
        name="hawking",
        mutation_rate=0.20,
        mutation_strength=0.25,
        preferred_timeframes=["H1", "H4"],
        preferred_indicators=["ATR", "BOLLINGER", "ADX", "MACD", "RSI"],
        risk_tolerance=1.0,
        tf_weight_vector={
            "H4":  0.35,
            "H1":  0.45,
            "M30": 0.20,
        },
        indicator_weight_vector={
            "ATR":        0.30,
            "BOLLINGER":  0.25,
            "ADX":        0.18,
            "MACD":       0.12,
            "RSI":        0.08,
            "CCI":        0.04,
            "MOMENTUM":   0.02,
            "EMA":        0.01,
            "STOCHASTIC": 0.00,
            "SMA":        0.00,
            "WMA":        0.00,
            "WILLIAMS_R": 0.00,
            "OBV":        0.00,
        },
        logic_type_preference="AND",
        take_profit_r_range=(3.0, 6.0),    # Explosive bounce = big R targets
        stop_loss_atr_range=(0.8, 1.5),    # Tight stops — inside the compression zone
        max_concurrent_trades_range=(1, 1),
        daily_loss_limit_range=(0.01, 0.02),
        entry_conditions_count=(2, 3),
    )

    def _build_indicators(self, rng: np.random.Generator) -> list[IndicatorGene]:
        # 5% chance of a "singularity jump" — completely random indicator
        genes = super()._build_indicators(rng)
        for i, gene in enumerate(genes):
            if rng.random() < 0.05:
                from evolution_engine.indicators.library import ALL_INDICATOR_NAMES
                name = str(rng.choice(ALL_INDICATOR_NAMES))
                params = self._sample_indicator_params(name, rng)
                genes[i] = IndicatorGene(
                    name=name, params=params,
                    timeframe=gene.timeframe, role=gene.role
                )
        return genes
