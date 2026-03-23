"""
Newton — The Gravity Player.
What goes up must come down. Trades mean reversion and trend gravity.
Favours RSI extremes, Bollinger Band touches, ADX trend exhaustion.
Low mutation. Patient, systematic, waits for overextension.
"""
from __future__ import annotations

from evolution_engine.species.base_species import BaseSpecies, SpeciesTraits


class NewtonSpecies(BaseSpecies):
    traits = SpeciesTraits(
        name="newton",
        mutation_rate=0.10,
        mutation_strength=0.12,
        preferred_timeframes=["H4", "H1"],
        preferred_indicators=["RSI", "BOLLINGER", "ADX", "ATR", "CCI"],
        risk_tolerance=0.7,
        tf_weight_vector={
            "H4":  0.55,
            "H1":  0.30,
            "M30": 0.15,
        },
        indicator_weight_vector={
            "RSI":        0.28,
            "BOLLINGER":  0.22,
            "ADX":        0.18,
            "ATR":        0.14,
            "CCI":        0.10,
            "MACD":       0.05,
            "EMA":        0.02,
            "MOMENTUM":   0.01,
            "STOCHASTIC": 0.00,
            "WILLIAMS_R": 0.00,
            "SMA":        0.00,
            "WMA":        0.00,
            "OBV":        0.00,
        },
        logic_type_preference="AND",
        take_profit_r_range=(3.0, 4.5),
        stop_loss_atr_range=(1.0, 2.0),
        max_concurrent_trades_range=(1, 1),
        daily_loss_limit_range=(0.01, 0.02),
        entry_conditions_count=(2, 3),
    )
