"""
StrategyGenome — the DNA of a trading strategy.
Fully serializable to/from dict for JSON export.
No hardcoded logic: entry/exit conditions are parameterized relationships
between the 3 indicator slots.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from evolution_engine.config.settings import (
    ALLOWED_TIMEFRAMES,
    MAX_RISK_PER_TRADE,
    MIN_RISK_PER_TRADE,
    NUM_INDICATORS,
)


# ---------------------------------------------------------------------------
# Condition types
# ---------------------------------------------------------------------------

COMPARISON_TYPES = [
    "crossover",        # indicator_idx line crosses above reference_idx
    "crossunder",       # crosses below
    "above_threshold",  # indicator value > float threshold
    "below_threshold",  # indicator value < float threshold
    "slope_positive",   # n-bar slope of indicator is positive
    "slope_negative",
    "divergence",       # price and indicator diverge
]

EXIT_TYPES = ["indicator_signal", "fixed_rr", "trailing_stop"]
DIRECTION_TYPES = ["long", "short", "both"]


# ---------------------------------------------------------------------------
# Gene dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IndicatorGene:
    name: str           # key in INDICATOR_REGISTRY
    params: dict        # e.g. {"period": 14}
    timeframe: str      # "H4", "H1", "M30", "M15"
    role: str           # "entry_trigger", "entry_filter", "exit_trigger"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "params": self.params,
            "timeframe": self.timeframe,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IndicatorGene":
        return cls(
            name=d["name"],
            params=d["params"],
            timeframe=d["timeframe"],
            role=d["role"],
        )


@dataclass
class Condition:
    """A single logical condition referencing indicator slots by index."""
    indicator_idx: int              # 0..NUM_INDICATORS-1
    comparison: str                 # one of COMPARISON_TYPES
    reference_idx: Optional[int]    # for crossover/crossunder/divergence
    threshold: Optional[float]      # for above/below_threshold
    n_bars: int = 1                 # lookback for slope/divergence checks
    sub_key: Optional[str] = None   # e.g. "macd" for MACD line vs signal

    def to_dict(self) -> dict:
        return {
            "indicator_idx": self.indicator_idx,
            "comparison": self.comparison,
            "reference_idx": self.reference_idx,
            "threshold": self.threshold,
            "n_bars": self.n_bars,
            "sub_key": self.sub_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Condition":
        return cls(
            indicator_idx=d["indicator_idx"],
            comparison=d["comparison"],
            reference_idx=d.get("reference_idx"),
            threshold=d.get("threshold"),
            n_bars=d.get("n_bars", 1),
            sub_key=d.get("sub_key"),
        )


@dataclass
class EntryLogicGene:
    direction: str              # "long", "short", "both"
    logic_type: str             # "AND" (all conditions must be true) / "OR"
    conditions: list[Condition]

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "logic_type": self.logic_type,
            "conditions": [c.to_dict() for c in self.conditions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EntryLogicGene":
        return cls(
            direction=d["direction"],
            logic_type=d.get("logic_type", "AND"),
            conditions=[Condition.from_dict(c) for c in d["conditions"]],
        )


@dataclass
class ExitLogicGene:
    exit_type: str              # "indicator_signal", "fixed_rr", "trailing_stop"
    take_profit_r: float        # R multiples
    stop_loss_atr_mult: float   # stop = entry +/- ATR * mult

    def to_dict(self) -> dict:
        return {
            "exit_type": self.exit_type,
            "take_profit_r": self.take_profit_r,
            "stop_loss_atr_mult": self.stop_loss_atr_mult,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExitLogicGene":
        return cls(
            exit_type=d["exit_type"],
            take_profit_r=float(d["take_profit_r"]),
            stop_loss_atr_mult=float(d["stop_loss_atr_mult"]),
        )


@dataclass
class RiskGene:
    risk_per_trade: float           # fraction of equity
    max_concurrent_trades: int      # 1-3
    daily_loss_limit: float         # fraction of equity

    def to_dict(self) -> dict:
        return {
            "risk_per_trade": self.risk_per_trade,
            "max_concurrent_trades": self.max_concurrent_trades,
            "daily_loss_limit": self.daily_loss_limit,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RiskGene":
        return cls(
            risk_per_trade=float(d["risk_per_trade"]),
            max_concurrent_trades=int(d["max_concurrent_trades"]),
            daily_loss_limit=float(d["daily_loss_limit"]),
        )


# ---------------------------------------------------------------------------
# Strategy genome
# ---------------------------------------------------------------------------

@dataclass
class StrategyGenome:
    species: str
    generation: int
    indicators: list[IndicatorGene]     # always exactly NUM_INDICATORS
    entry_logic: EntryLogicGene
    exit_logic: ExitLogicGene
    risk: RiskGene
    genome_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_ids: list[str] = field(default_factory=list)
    fitness_score: Optional[float] = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        assert len(self.indicators) == NUM_INDICATORS, (
            f"Genome must have exactly {NUM_INDICATORS} indicators, "
            f"got {len(self.indicators)}"
        )
        for gene in self.indicators:
            assert gene.timeframe in ALLOWED_TIMEFRAMES, (
                f"Timeframe {gene.timeframe!r} not in {ALLOWED_TIMEFRAMES}"
            )
        assert self.entry_logic.direction in DIRECTION_TYPES
        assert self.entry_logic.logic_type in ("AND", "OR")
        assert self.exit_logic.exit_type in EXIT_TYPES
        assert 0 < self.exit_logic.take_profit_r <= 10
        assert 0.5 <= self.exit_logic.stop_loss_atr_mult <= 5.0
        assert MIN_RISK_PER_TRADE <= self.risk.risk_per_trade <= MAX_RISK_PER_TRADE
        assert 1 <= self.risk.max_concurrent_trades <= 3
        assert 0 < self.risk.daily_loss_limit <= 0.10

        for cond in self.entry_logic.conditions:
            assert 0 <= cond.indicator_idx < NUM_INDICATORS
            assert cond.comparison in COMPARISON_TYPES
            if cond.comparison in ("crossover", "crossunder", "divergence"):
                assert cond.reference_idx is not None
                assert 0 <= cond.reference_idx < NUM_INDICATORS
            if cond.comparison in ("above_threshold", "below_threshold"):
                assert cond.threshold is not None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "species": self.species,
            "generation": self.generation,
            "indicators": [g.to_dict() for g in self.indicators],
            "entry_logic": self.entry_logic.to_dict(),
            "exit_logic": self.exit_logic.to_dict(),
            "risk": self.risk.to_dict(),
            "parent_ids": self.parent_ids,
            "fitness_score": self.fitness_score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyGenome":
        return cls(
            genome_id=d["genome_id"],
            species=d["species"],
            generation=d["generation"],
            indicators=[IndicatorGene.from_dict(g) for g in d["indicators"]],
            entry_logic=EntryLogicGene.from_dict(d["entry_logic"]),
            exit_logic=ExitLogicGene.from_dict(d["exit_logic"]),
            risk=RiskGene.from_dict(d["risk"]),
            parent_ids=d.get("parent_ids", []),
            fitness_score=d.get("fitness_score"),
        )
