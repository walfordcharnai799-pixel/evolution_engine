"""
Einstein — The Logician.
Slow, stable, rational. Prefers higher timeframes and trend indicators.
Multi-condition AND logic. Conservative risk.
"""
from __future__ import annotations

import numpy as np

from evolution_engine.species.base_species import BaseSpecies, SpeciesTraits


class EinsteinSpecies(BaseSpecies):
    traits = SpeciesTraits(
        name="einstein",
        mutation_rate=0.10,
        mutation_strength=0.15,
        preferred_timeframes=["H4", "H1"],
        preferred_indicators=["EMA", "SMA", "MACD", "ADX"],
        risk_tolerance=0.5,
        tf_weight_vector={
            "H4":  0.50,
            "H1":  0.35,
            "M30": 0.15,
        },
        indicator_weight_vector={
            "EMA":        0.25,
            "SMA":        0.20,
            "MACD":       0.20,
            "ADX":        0.15,
            "ATR":        0.08,
            "RSI":        0.04,
            "BOLLINGER":  0.03,
            "WMA":        0.02,
            "CCI":        0.01,
            "STOCHASTIC": 0.01,
            "MOMENTUM":   0.01,
            "WILLIAMS_R": 0.00,
            "OBV":        0.00,
        },
        logic_type_preference="AND",
        take_profit_r_range=(1.5, 3.0),
        stop_loss_atr_range=(1.5, 3.0),
        max_concurrent_trades_range=(1, 1),     # conservative: 1 trade at a time
        daily_loss_limit_range=(0.02, 0.04),
        entry_conditions_count=(2, 3),           # multi-condition AND = thorough
    )
