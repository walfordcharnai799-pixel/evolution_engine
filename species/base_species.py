"""
Abstract base class for all species.
Encodes personality as numeric traits that shape genome generation and mutation.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from evolution_engine.config.settings import (
    ALLOWED_TIMEFRAMES,
    INDICATOR_PARAM_RANGES,
    MAX_RISK_PER_TRADE,
    MIN_RISK_PER_TRADE,
    NUM_INDICATORS,
)
from evolution_engine.evolution.genome import (
    Condition,
    EntryLogicGene,
    ExitLogicGene,
    IndicatorGene,
    RiskGene,
    StrategyGenome,
    COMPARISON_TYPES,
    DIRECTION_TYPES,
    EXIT_TYPES,
)
from evolution_engine.indicators.library import ALL_INDICATOR_NAMES


INDICATOR_ROLES = ["entry_trigger", "entry_filter", "exit_trigger"]


@dataclass
class SpeciesTraits:
    name: str
    mutation_rate: float
    mutation_strength: float
    preferred_timeframes: list[str]
    preferred_indicators: list[str]
    risk_tolerance: float                       # scales risk_per_trade range
    tf_weight_vector: dict[str, float]
    indicator_weight_vector: dict[str, float]
    logic_type_preference: str = "AND"          # "AND" or "OR"
    take_profit_r_range: tuple = (1.5, 3.0)
    stop_loss_atr_range: tuple = (1.0, 2.5)
    max_concurrent_trades_range: tuple = (1, 2)
    daily_loss_limit_range: tuple = (0.02, 0.05)
    entry_conditions_count: tuple = (1, 3)      # min/max conditions per entry gene


class BaseSpecies(ABC):
    traits: SpeciesTraits

    # ------------------------------------------------------------------
    # Genome factory (called by PopulationFactory)
    # ------------------------------------------------------------------

    def spawn_genome(
        self, generation: int, rng: np.random.Generator
    ) -> StrategyGenome:
        indicators = self._build_indicators(rng)
        entry = self._build_entry_logic(rng)
        exit_gene = self._build_exit_logic(rng)
        risk = self._build_risk(rng)
        genome = StrategyGenome(
            genome_id=str(uuid.uuid4()),
            species=self.traits.name,
            generation=generation,
            indicators=indicators,
            entry_logic=entry,
            exit_logic=exit_gene,
            risk=risk,
            parent_ids=[],
        )
        try:
            genome.validate()
        except AssertionError:
            # If validation fails, try once more with fresh params
            return self.spawn_genome(generation, rng)
        return genome

    # ------------------------------------------------------------------
    # Gene builders (subclasses may override)
    # ------------------------------------------------------------------

    def _build_indicators(self, rng: np.random.Generator) -> list[IndicatorGene]:
        genes = []
        for role in INDICATOR_ROLES:
            name = self._sample_indicator(rng)
            params = self._sample_indicator_params(name, rng)
            tf = self._sample_timeframe(rng)
            genes.append(IndicatorGene(name=name, params=params, timeframe=tf, role=role))
        return genes

    def _build_entry_logic(self, rng: np.random.Generator) -> EntryLogicGene:
        n_conditions = int(rng.integers(
            self.traits.entry_conditions_count[0],
            self.traits.entry_conditions_count[1] + 1,
        ))
        conditions = []
        for _ in range(n_conditions):
            comparison = rng.choice(COMPARISON_TYPES)
            indicator_idx = int(rng.integers(0, NUM_INDICATORS))
            reference_idx = None
            threshold = None
            sub_key = None

            if comparison in ("crossover", "crossunder", "divergence"):
                # reference must differ from indicator_idx
                others = [i for i in range(NUM_INDICATORS) if i != indicator_idx]
                reference_idx = int(rng.choice(others)) if others else indicator_idx
            elif comparison in ("above_threshold", "below_threshold"):
                threshold = float(rng.uniform(20, 80))

            n_bars = int(rng.integers(1, 6))
            conditions.append(
                Condition(
                    indicator_idx=indicator_idx,
                    comparison=comparison,
                    reference_idx=reference_idx,
                    threshold=threshold,
                    n_bars=n_bars,
                    sub_key=sub_key,
                )
            )

        direction = str(rng.choice(DIRECTION_TYPES))
        logic_type = self.traits.logic_type_preference
        return EntryLogicGene(
            direction=direction,
            logic_type=logic_type,
            conditions=conditions,
        )

    def _build_exit_logic(self, rng: np.random.Generator) -> ExitLogicGene:
        exit_type = str(rng.choice(EXIT_TYPES))
        tp_r = float(rng.uniform(*self.traits.take_profit_r_range))
        sl_mult = float(rng.uniform(*self.traits.stop_loss_atr_range))
        return ExitLogicGene(
            exit_type=exit_type,
            take_profit_r=round(tp_r, 2),
            stop_loss_atr_mult=round(sl_mult, 2),
        )

    def _build_risk(self, rng: np.random.Generator) -> RiskGene:
        base_risk_min = MIN_RISK_PER_TRADE
        base_risk_max = MAX_RISK_PER_TRADE * self.traits.risk_tolerance
        risk = float(rng.uniform(base_risk_min, min(base_risk_max, MAX_RISK_PER_TRADE)))
        max_concurrent = int(rng.integers(
            self.traits.max_concurrent_trades_range[0],
            self.traits.max_concurrent_trades_range[1] + 1,
        ))
        daily_loss = float(rng.uniform(*self.traits.daily_loss_limit_range))
        return RiskGene(
            risk_per_trade=round(risk, 4),
            max_concurrent_trades=max_concurrent,
            daily_loss_limit=round(daily_loss, 4),
        )

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_timeframe(self, rng: np.random.Generator) -> str:
        weights_arr = [self.traits.tf_weight_vector.get(tf, 0.0) for tf in ALLOWED_TIMEFRAMES]
        total = sum(weights_arr)
        probs = [w / total for w in weights_arr]
        return str(rng.choice(ALLOWED_TIMEFRAMES, p=probs))

    def _sample_indicator(self, rng: np.random.Generator) -> str:
        names = ALL_INDICATOR_NAMES
        weights_arr = [self.traits.indicator_weight_vector.get(n, 0.0) for n in names]
        total = sum(weights_arr)
        if total == 0:
            return str(rng.choice(names))
        probs = [w / total for w in weights_arr]
        return str(rng.choice(names, p=probs))

    def _sample_indicator_params(
        self, indicator_name: str, rng: np.random.Generator
    ) -> dict:
        ranges = INDICATOR_PARAM_RANGES.get(indicator_name, {})
        params = {}
        for param_name, (lo, hi) in ranges.items():
            if isinstance(lo, int) and isinstance(hi, int):
                params[param_name] = int(rng.integers(lo, hi + 1))
            else:
                params[param_name] = round(float(rng.uniform(lo, hi)), 2)
        return params
