"""
Da Vinci — The Pattern Master.
Sees structure everywhere. Seeks recurring geometric patterns in price.
Uses Bollinger Bands, CCI, and visual structure indicators.
Medium-high mutation. Favours multi-timeframe pattern confirmation.
"""
from __future__ import annotations

from evolution_engine.species.base_species import BaseSpecies, SpeciesTraits


class DaVinciSpecies(BaseSpecies):
    traits = SpeciesTraits(
        name="davinci",
        mutation_rate=0.18,
        mutation_strength=0.22,
        preferred_timeframes=["H4", "H1"],
        preferred_indicators=["BOLLINGER", "CCI", "EMA", "SMA", "ATR"],
        risk_tolerance=0.9,
        tf_weight_vector={
            "H4":  0.40,
            "H1":  0.40,
            "M30": 0.20,
        },
        indicator_weight_vector={
            "BOLLINGER":  0.25,
            "CCI":        0.20,
            "EMA":        0.18,
            "SMA":        0.12,
            "ATR":        0.10,
            "MACD":       0.06,
            "RSI":        0.05,
            "ADX":        0.03,
            "STOCHASTIC": 0.01,
            "MOMENTUM":   0.00,
            "WILLIAMS_R": 0.00,
            "WMA":        0.00,
            "OBV":        0.00,
        },
        logic_type_preference="AND",
        take_profit_r_range=(3.0, 5.0),    # Pattern targets = extended R/R
        stop_loss_atr_range=(1.0, 2.0),
        max_concurrent_trades_range=(1, 2),
        daily_loss_limit_range=(0.01, 0.03),
        entry_conditions_count=(2, 3),
    )
