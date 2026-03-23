"""
Selection engine — elites and tournament selection.
No crossover: mutation-only evolution for genome structural stability.
"""
from __future__ import annotations

import numpy as np

from evolution_engine.evolution.genome import StrategyGenome


class SelectionEngine:
    def select_elites(
        self,
        population: list[StrategyGenome],
        elite_fraction: float,
    ) -> list[StrategyGenome]:
        """Return top elite_fraction of population by fitness_score."""
        ranked = self.rank_population(population)
        n = max(1, int(len(ranked) * elite_fraction))
        return ranked[:n]

    def tournament_select(
        self,
        population: list[StrategyGenome],
        n_select: int,
        tournament_size: int = 4,
        rng: np.random.Generator | None = None,
    ) -> list[StrategyGenome]:
        """Select n_select parents via repeated tournaments."""
        if rng is None:
            rng = np.random.default_rng()
        eligible = [g for g in population if g.fitness_score is not None and g.fitness_score >= 0]
        if not eligible:
            return list(population[:n_select]) if population else []

        selected = []
        for _ in range(n_select):
            size = min(tournament_size, len(eligible))
            contestants_idx = rng.choice(len(eligible), size=size, replace=False)
            contestants = [eligible[i] for i in contestants_idx]
            winner = max(contestants, key=lambda g: g.fitness_score or 0.0)
            selected.append(winner)
        return selected

    def rank_population(
        self,
        population: list[StrategyGenome],
    ) -> list[StrategyGenome]:
        """Return a new list sorted by fitness_score descending. Does not modify originals."""
        return sorted(
            population,
            key=lambda g: g.fitness_score if g.fitness_score is not None else -999.0,
            reverse=True,
        )
