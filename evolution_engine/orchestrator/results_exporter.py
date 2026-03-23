"""
Results exporter — saves structured JSON output after every generation.
Maintains a hall of fame across all generations.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evolution_engine.engine.metrics import MetricsReport
from evolution_engine.engine.survival_filter import SurvivalResult
from evolution_engine.evolution.genome import StrategyGenome


class ResultsExporter:
    def __init__(self, results_dir: str = "evolution_engine/results") -> None:
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-generation export
    # ------------------------------------------------------------------

    def export_generation(
        self,
        generation: int,
        survivors: list[StrategyGenome],
        reports: dict[str, MetricsReport],
        all_genomes: list[StrategyGenome],
        all_survival_results: list[SurvivalResult],
    ) -> None:
        gen_dir = self._results_dir / f"gen_{generation:04d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # --- Summary ---
        species_stats = self._compute_species_stats(all_genomes, reports, all_survival_results)
        summary = {
            "generation": generation,
            "total_genomes": len(all_genomes),
            "survivors": len(survivors),
            "survival_rate": round(len(survivors) / max(1, len(all_genomes)), 4),
            "species_stats": species_stats,
        }
        if survivors and reports:
            best = max(survivors, key=lambda g: reports.get(g.genome_id, self._null_report()).fitness_score)
            best_report = reports[best.genome_id]
            summary["best_genome_id"] = best.genome_id
            summary["best_species"] = best.species
            summary["best_fitness"] = best_report.fitness_score
            summary["best_win_rate"] = best_report.win_rate
            summary["best_profit_factor"] = best_report.profit_factor
            summary["best_max_drawdown"] = best_report.max_drawdown
            summary["best_total_trades"] = best_report.total_trades
        self._write_json(gen_dir / "summary.json", summary)

        # --- Survivors ---
        survivors_data = [
            self._genome_to_export(g, reports.get(g.genome_id))
            for g in sorted(
                survivors,
                key=lambda g: (reports.get(g.genome_id) or self._null_report()).fitness_score,
                reverse=True,
            )
        ]
        self._write_json(gen_dir / "survivors.json", survivors_data)

        # --- Best genome ---
        if survivors:
            best = max(survivors, key=lambda g: (reports.get(g.genome_id) or self._null_report()).fitness_score)
            self._write_json(
                gen_dir / "best_genome.json",
                self._genome_to_export(best, reports.get(best.genome_id)),
            )

        # --- Full population dump ---
        pop_data = [
            self._genome_to_export(g, reports.get(g.genome_id))
            for g in all_genomes
        ]
        self._write_json(gen_dir / "population.json", pop_data)

        # --- Species stats ---
        self._write_json(gen_dir / "species_stats.json", species_stats)

    # ------------------------------------------------------------------
    # Hall of fame
    # ------------------------------------------------------------------

    def export_hall_of_fame(
        self,
        hall_of_fame: list[tuple[StrategyGenome, MetricsReport]],
    ) -> None:
        hof_data = [
            self._genome_to_export(genome, report)
            for genome, report in hall_of_fame
        ]
        self._write_json(self._results_dir / "hall_of_fame.json", hof_data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _genome_to_export(
        self,
        genome: StrategyGenome,
        report: MetricsReport | None,
    ) -> dict[str, Any]:
        result = genome.to_dict()
        if report is not None:
            result["metrics"] = {
                "win_rate": report.win_rate,
                "profit_factor": report.profit_factor,
                "max_drawdown": report.max_drawdown,
                "total_trades": report.total_trades,
                "expectancy": report.expectancy,
                "stability_score": report.stability_score,
                "sharpe_ratio": report.sharpe_ratio,
                "avg_trade_r": report.avg_trade_r,
                "avg_winner_r": report.avg_winner_r,
                "avg_loser_r": report.avg_loser_r,
                "largest_winner_r": report.largest_winner_r,
                "largest_loser_r": report.largest_loser_r,
                "consecutive_losses_max": report.consecutive_losses_max,
                "fitness_score": report.fitness_score,
            }
        return result

    def _compute_species_stats(
        self,
        genomes: list[StrategyGenome],
        reports: dict[str, MetricsReport],
        survival_results: list[SurvivalResult],
    ) -> dict[str, Any]:
        from collections import defaultdict
        stats: dict[str, dict] = defaultdict(lambda: {
            "total": 0, "survived": 0, "avg_fitness": 0.0, "scores": []
        })

        survived_ids = {sr.genome_id for sr in survival_results if sr.passed}

        for genome in genomes:
            sp = genome.species
            stats[sp]["total"] += 1
            if genome.genome_id in survived_ids:
                stats[sp]["survived"] += 1
            report = reports.get(genome.genome_id)
            if report:
                stats[sp]["scores"].append(report.fitness_score)

        # Finalize
        final: dict[str, Any] = {}
        for sp, s in stats.items():
            scores = s["scores"]
            final[sp] = {
                "total": s["total"],
                "survived": s["survived"],
                "survival_rate": round(s["survived"] / max(1, s["total"]), 3),
                "avg_fitness": round(float(sum(scores) / len(scores)), 6) if scores else 0.0,
                "best_fitness": round(max(scores), 6) if scores else 0.0,
            }
        return final

    def _null_report(self) -> MetricsReport:
        from evolution_engine.engine.metrics import MetricsReport
        return MetricsReport(
            genome_id="", species="", win_rate=0.0, profit_factor=0.0,
            max_drawdown=1.0, total_trades=0, expectancy=0.0,
            stability_score=0.0, sharpe_ratio=0.0, avg_trade_r=0.0,
            avg_winner_r=0.0, avg_loser_r=0.0, largest_winner_r=0.0,
            largest_loser_r=0.0, consecutive_losses_max=0, fitness_score=0.0,
        )

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
