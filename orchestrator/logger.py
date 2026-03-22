"""
Evolution logger — structured JSON Lines output + human-readable console.
Every generation, species, mutation, metric, and failure is logged.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evolution_engine.engine.backtester import BacktestResult
from evolution_engine.engine.metrics import MetricsReport
from evolution_engine.engine.survival_filter import SurvivalResult
from evolution_engine.evolution.genome import StrategyGenome


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvolutionLogger:
    def __init__(self, log_dir: str = "evolution_engine/logs", generation: int = 0) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._generation = generation

        # Rolling log that accumulates all generations
        self._rolling_path = self._log_dir / "evolution.jsonl"

        # Console logger
        self._console = logging.getLogger("evolution_engine")
        if not self._console.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
            )
            self._console.addHandler(handler)
        self._console.setLevel(logging.INFO)

    def _write(self, event: dict[str, Any]) -> None:
        event["timestamp"] = _now()
        line = json.dumps(event, default=str)
        with self._rolling_path.open("a") as f:
            f.write(line + "\n")
        # Also write per-generation file
        gen_path = self._log_dir / f"generation_{self._generation:04d}.jsonl"
        with gen_path.open("a") as f:
            f.write(line + "\n")

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def set_generation(self, generation: int) -> None:
        self._generation = generation

    def log_generation_start(self, generation: int, population_size: int) -> None:
        self._generation = generation
        self._write({
            "event": "generation_start",
            "generation": generation,
            "population_size": population_size,
        })
        self._console.info(
            f"=== Generation {generation:04d} | Population: {population_size} ==="
        )

    def log_genome_backtest(
        self,
        genome: StrategyGenome,
        result: BacktestResult,
        report: MetricsReport,
    ) -> None:
        self._write({
            "event": "genome_backtested",
            "generation": self._generation,
            "genome_id": genome.genome_id,
            "species": genome.species,
            "total_trades": report.total_trades,
            "win_rate": round(report.win_rate, 4),
            "profit_factor": round(report.profit_factor, 4),
            "max_drawdown": round(report.max_drawdown, 4),
            "fitness_score": round(report.fitness_score, 6),
            "indicators": [
                {"name": g.name, "tf": g.timeframe} for g in genome.indicators
            ],
        })

    def log_survival_result(self, result: SurvivalResult) -> None:
        if not result.passed:
            self._write({
                "event": "genome_eliminated",
                "generation": self._generation,
                "genome_id": result.genome_id,
                "species": result.species,
                "failures": result.failures,
            })

    def log_mutation(
        self,
        parent: StrategyGenome,
        child: StrategyGenome,
        mutation_type: str,
    ) -> None:
        self._write({
            "event": "mutation",
            "generation": self._generation,
            "parent_id": parent.genome_id,
            "child_id": child.genome_id,
            "species": parent.species,
            "mutation_type": mutation_type,
        })

    def log_generation_summary(
        self,
        generation: int,
        survivors: list[StrategyGenome],
        best_genome: StrategyGenome,
        best_report: MetricsReport,
        species_counts: dict[str, int],
    ) -> None:
        survival_rate = len(survivors) / max(1, sum(species_counts.values()))
        self._write({
            "event": "generation_summary",
            "generation": generation,
            "survivors": len(survivors),
            "survival_rate": round(survival_rate, 3),
            "best_genome_id": best_genome.genome_id,
            "best_species": best_genome.species,
            "best_fitness": round(best_report.fitness_score, 6),
            "best_win_rate": round(best_report.win_rate, 4),
            "best_profit_factor": round(best_report.profit_factor, 4),
            "best_max_drawdown": round(best_report.max_drawdown, 4),
            "best_total_trades": best_report.total_trades,
            "species_counts": species_counts,
        })
        self._console.info(
            f"Gen {generation:04d} | Survivors: {len(survivors)} ({survival_rate:.0%}) | "
            f"Best [{best_genome.species}] fitness={best_report.fitness_score:.4f} | "
            f"WR={best_report.win_rate:.1%} PF={best_report.profit_factor:.2f} "
            f"DD={best_report.max_drawdown:.1%} Trades={best_report.total_trades}"
        )

        if survival_rate < 0.20:
            self._console.warning(
                f"Gen {generation}: LOW SURVIVAL RATE {survival_rate:.0%} — "
                "consider relaxing filters or increasing population diversity"
            )

    def log_error(self, context: str, exception: Exception) -> None:
        self._write({
            "event": "error",
            "generation": self._generation,
            "context": context,
            "error_type": type(exception).__name__,
            "error_message": str(exception),
        })
        self._console.error(f"ERROR [{context}]: {type(exception).__name__}: {exception}")
