"""
Tesla — The Frequency Hunter.
Only trades high-volatility time cycles. Obsessed with ATR explosions
and momentum surges. High mutation. Aggressive volatility seeker.
"""
from __future__ import annotations

from evolution_engine.species.base_species import BaseSpecies, SpeciesTraits


class TeslaSpecies(BaseSpecies):
    traits = SpeciesTraits(
        name="tesla",
        mutation_rate=0.28,
        mutation_strength=0.35,
        preferred_timeframes=["M30", "H1"],
        preferred_indicators=["ATR", "BOLLINGER", "MACD", "MOMENTUM", "RSI"],
        risk_tolerance=1.2,
        tf_weight_vector={
            "H4":  0.10,
            "H1":  0.35,
            "M30": 0.55,
        },
        indicator_weight_vector={
            "ATR":        0.28,
            "BOLLINGER":  0.22,
            "MACD":       0.18,
            "MOMENTUM":   0.15,
            "RSI":        0.10,
            "CCI":        0.04,
            "ADX":        0.02,
            "STOCHASTIC": 0.01,
            "EMA":        0.00,
            "SMA":        0.00,
            "WMA":        0.00,
            "WILLIAMS_R": 0.00,
            "OBV":        0.00,
        },
        logic_type_preference="OR",
        take_profit_r_range=(3.0, 5.0),
        stop_loss_atr_range=(0.8, 1.5),
        max_concurrent_trades_range=(1, 2),
        daily_loss_limit_range=(0.01, 0.03),
        entry_conditions_count=(1, 2),
    )
