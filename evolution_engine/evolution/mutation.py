"""
Species-aware mutation engine.
Never mutates in place — always returns a new StrategyGenome.
"""
from __future__ import annotations

import copy
import uuid

import numpy as np

from evolution_engine.config.settings import (
    ALLOWED_TIMEFRAMES,
    INDICATOR_PARAM_RANGES,
    MAX_RISK_PER_TRADE,
    MIN_RISK_PER_TRADE,
    NUM_INDICATORS,
)
from evolution_engine.evolution.genome import (
    COMPARISON_TYPES,
    EXIT_TYPES,
    Condition,
    EntryLogicGene,
    ExitLogicGene,
    IndicatorGene,
    RiskGene,
    StrategyGenome,
)
from evolution_engine.indicators.library import ALL_INDICATOR_NAMES


class MutationEngine:
    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def mutate(
        self,
        genome: StrategyGenome,
        mutation_rate: float,
        mutation_strength: float,
        species_indicator_weights: dict[str, float] | None = None,
    ) -> StrategyGenome:
        """Return a mutated copy of genome. Parent genome is never modified."""
        child = copy.deepcopy(genome)
        child.genome_id = str(uuid.uuid4())
        child.parent_ids = [genome.genome_id]
        child.fitness_score = None
        child.generation = genome.generation + 1

        rng = self._rng

        # Each gene slot has an independent mutation_rate chance of mutating
        new_indicators = list(child.indicators)
        for i, gene in enumerate(new_indicators):
            if rng.random() < mutation_rate:
                sub = rng.random()
                if sub < 0.40:
                    new_indicators[i] = self._mutate_indicator_params(gene, mutation_strength)
                elif sub < 0.65:
                    new_indicators[i] = self._mutate_indicator_swap(
                        gene, mutation_strength, species_indicator_weights
                    )
                elif sub < 0.80:
                    new_indicators[i] = self._mutate_timeframe(gene)
                else:
                    new_indicators[i] = self._mutate_indicator_swap(
                        gene, mutation_strength, species_indicator_weights
                    )
        child.indicators = new_indicators

        if rng.random() < mutation_rate:
            child.entry_logic = self._mutate_entry_conditions(child.entry_logic)

        if rng.random() < mutation_rate:
            child.exit_logic = self._mutate_exit_logic(child.exit_logic, mutation_strength)

        if rng.random() < mutation_rate:
            child.risk = self._mutate_risk(child.risk, mutation_strength)

        try:
            child.validate()
        except AssertionError:
            # Return original genome if mutation produces an invalid genome
            return genome
        return child

    # ------------------------------------------------------------------
    # Sub-mutators
    # ------------------------------------------------------------------

    def _mutate_indicator_params(
        self, gene: IndicatorGene, strength: float
    ) -> IndicatorGene:
        new_params = {}
        ranges = INDICATOR_PARAM_RANGES.get(gene.name, {})
        for param_name, value in gene.params.items():
            if param_name not in ranges:
                new_params[param_name] = value
                continue
            lo, hi = ranges[param_name]
            span = hi - lo
            delta = self._rng.normal(0, strength * span)
            new_val = value + delta
            new_val = float(max(lo, min(hi, new_val)))
            if isinstance(lo, int) and isinstance(hi, int):
                new_val = round(new_val)
            else:
                new_val = round(new_val, 2)
            new_params[param_name] = new_val
        return IndicatorGene(
            name=gene.name,
            params=new_params,
            timeframe=gene.timeframe,
            role=gene.role,
        )

    def _mutate_indicator_swap(
        self,
        gene: IndicatorGene,
        strength: float,
        indicator_weights: dict[str, float] | None,
    ) -> IndicatorGene:
        if indicator_weights:
            names = ALL_INDICATOR_NAMES
            w = [indicator_weights.get(n, 0.01) for n in names]
            total = sum(w)
            probs = [x / total for x in w]
            new_name = str(self._rng.choice(names, p=probs))
        else:
            new_name = str(self._rng.choice(ALL_INDICATOR_NAMES))

        new_params = self._sample_params(new_name)
        return IndicatorGene(
            name=new_name,
            params=new_params,
            timeframe=gene.timeframe,
            role=gene.role,
        )

    def _mutate_timeframe(self, gene: IndicatorGene) -> IndicatorGene:
        others = [tf for tf in ALLOWED_TIMEFRAMES if tf != gene.timeframe]
        new_tf = str(self._rng.choice(others)) if others else gene.timeframe
        return IndicatorGene(
            name=gene.name,
            params=gene.params,
            timeframe=new_tf,
            role=gene.role,
        )

    def _mutate_entry_conditions(self, entry: EntryLogicGene) -> EntryLogicGene:
        conditions = list(entry.conditions)
        rng = self._rng

        action = rng.random()
        if action < 0.33 and conditions:
            # Modify one condition
            idx = int(rng.integers(0, len(conditions)))
            old = conditions[idx]
            new_comparison = str(rng.choice(COMPARISON_TYPES))
            reference_idx = old.reference_idx
            threshold = old.threshold

            if new_comparison in ("crossover", "crossunder", "divergence"):
                others = [i for i in range(NUM_INDICATORS) if i != old.indicator_idx]
                reference_idx = int(rng.choice(others)) if others else old.indicator_idx
                threshold = None
            elif new_comparison in ("above_threshold", "below_threshold"):
                threshold = float(rng.uniform(20, 80))
                reference_idx = None

            conditions[idx] = Condition(
                indicator_idx=old.indicator_idx,
                comparison=new_comparison,
                reference_idx=reference_idx,
                threshold=threshold,
                n_bars=int(rng.integers(1, 6)),
                sub_key=old.sub_key,
            )
        elif action < 0.66:
            # Flip direction
            new_dir = str(rng.choice(["long", "short", "both"]))
            return EntryLogicGene(
                direction=new_dir,
                logic_type=entry.logic_type,
                conditions=conditions,
            )
        else:
            # Flip logic type
            new_logic = "OR" if entry.logic_type == "AND" else "AND"
            return EntryLogicGene(
                direction=entry.direction,
                logic_type=new_logic,
                conditions=conditions,
            )

        return EntryLogicGene(
            direction=entry.direction,
            logic_type=entry.logic_type,
            conditions=conditions,
        )

    def _mutate_exit_logic(
        self, exit_gene: ExitLogicGene, strength: float
    ) -> ExitLogicGene:
        rng = self._rng
        tp = exit_gene.take_profit_r + rng.normal(0, strength * 1.0)
        tp = float(max(0.5, min(10.0, tp)))
        sl = exit_gene.stop_loss_atr_mult + rng.normal(0, strength * 0.5)
        sl = float(max(0.5, min(5.0, sl)))
        exit_type = exit_gene.exit_type
        if rng.random() < 0.15:
            exit_type = str(rng.choice(EXIT_TYPES))
        return ExitLogicGene(
            exit_type=exit_type,
            take_profit_r=round(tp, 2),
            stop_loss_atr_mult=round(sl, 2),
        )

    def _mutate_risk(self, risk: RiskGene, strength: float) -> RiskGene:
        rng = self._rng
        new_risk = risk.risk_per_trade + rng.normal(0, strength * 0.01)
        new_risk = float(max(MIN_RISK_PER_TRADE, min(MAX_RISK_PER_TRADE, new_risk)))
        daily_loss = risk.daily_loss_limit + rng.normal(0, strength * 0.01)
        daily_loss = float(max(0.005, min(0.10, daily_loss)))
        return RiskGene(
            risk_per_trade=round(new_risk, 4),
            max_concurrent_trades=risk.max_concurrent_trades,
            daily_loss_limit=round(daily_loss, 4),
        )

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _sample_params(self, indicator_name: str) -> dict:
        ranges = INDICATOR_PARAM_RANGES.get(indicator_name, {})
        params = {}
        for param_name, (lo, hi) in ranges.items():
            if isinstance(lo, int) and isinstance(hi, int):
                params[param_name] = int(self._rng.integers(lo, hi + 1))
            else:
                params[param_name] = round(float(self._rng.uniform(lo, hi)), 2)
        return params
