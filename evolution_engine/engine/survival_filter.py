"""
Survival filter — The Referee.
Hard gates every strategy must pass. No exceptions.
Weak strategies die. Eliminated species mutate toward the winner.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from evolution_engine.config.settings import PROGRESSIVE_GATES, SURVIVAL
from evolution_engine.engine.metrics import MetricsReport
from evolution_engine.evolution.genome import StrategyGenome


@dataclass
class SurvivalResult:
    genome_id: str
    species: str
    passed: bool
    failures: list[str] = field(default_factory=list)


def get_gates_for_generation(generation: int) -> dict:
    """Return the correct threshold dict for the given generation."""
    base = dict(SURVIVAL)
    for (lo, hi, overrides) in PROGRESSIVE_GATES:
        if lo <= generation <= hi:
            base.update(overrides)
            return base
    return base


class SurvivalFilter:
    def __init__(self, thresholds: dict | None = None, generation: int = 0) -> None:
        self._base_thresholds = thresholds  # None = use progressive schedule
        self._generation = generation
        self._t = self._resolve_thresholds(generation)

    def _resolve_thresholds(self, generation: int) -> dict:
        if self._base_thresholds is not None:
            return self._base_thresholds
        return get_gates_for_generation(generation)

    def set_generation(self, generation: int) -> None:
        """Call at the start of each generation to apply the right gate level."""
        self._generation = generation
        self._t = self._resolve_thresholds(generation)

    def evaluate(self, report: MetricsReport) -> SurvivalResult:
        failures: list[str] = []

        if report.win_rate < self._t["min_win_rate"]:
            failures.append(
                f"win_rate={report.win_rate:.1%} < {self._t['min_win_rate']:.0%} required"
            )

        if report.profit_factor < self._t["min_profit_factor"]:
            failures.append(
                f"profit_factor={report.profit_factor:.2f} < {self._t['min_profit_factor']:.2f} required"
            )

        if report.max_drawdown > self._t["max_drawdown"]:
            failures.append(
                f"ELIMINATED: drawdown={report.max_drawdown:.1%} > {self._t['max_drawdown']:.0%} limit"
            )

        if report.total_trades < self._t["min_trades"]:
            failures.append(
                f"total_trades={report.total_trades} < {self._t['min_trades']} required"
            )

        if report.expectancy < self._t["min_expectancy"]:
            failures.append(
                f"expectancy={report.expectancy:.3f}R < {self._t['min_expectancy']:.2f}R required"
            )

        if report.avg_winner_r > 0 and report.avg_loser_r < 0:
            actual_rr = abs(report.avg_winner_r / report.avg_loser_r) if report.avg_loser_r != 0 else 0
            if actual_rr < self._t["min_rr_ratio"]:
                failures.append(
                    f"rr_ratio={actual_rr:.2f} < {self._t['min_rr_ratio']:.1f} required"
                )

        stability_floor = 1.0 - self._t["max_stability_var"]
        if report.stability_score < stability_floor:
            failures.append(
                f"stability={report.stability_score:.3f} < {stability_floor:.3f} required"
            )

        if report.recovery_fail:
            failures.append("recovery_fail: did not recover to peak within 5 bars")

        return SurvivalResult(
            genome_id=report.genome_id,
            species=report.species,
            passed=len(failures) == 0,
            failures=failures,
        )

    def filter_population(
        self,
        population: list[StrategyGenome],
        reports: dict[str, MetricsReport],
        generation: int | None = None,
    ) -> tuple[list[StrategyGenome], list[SurvivalResult]]:
        survivors: list[StrategyGenome] = []
        all_results: list[SurvivalResult] = []

        if generation is not None:
            self.set_generation(generation)

        for genome in population:
            report = reports.get(genome.genome_id)
            if report is None:
                all_results.append(SurvivalResult(
                    genome_id=genome.genome_id,
                    species=genome.species,
                    passed=False,
                    failures=["no_backtest_result"],
                ))
                continue
            result = self.evaluate(report)
            all_results.append(result)
            if result.passed:
                survivors.append(genome)

        return survivors, all_results
