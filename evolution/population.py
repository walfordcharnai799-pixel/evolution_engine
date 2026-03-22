"""
Population factory — generate initial population and replenish after selection.
"""
from __future__ import annotations

import numpy as np

from evolution_engine.config.settings import (
    MIN_POPULATION_SIZE,
    SPECIES_DISTRIBUTION,
)
from evolution_engine.evolution.genome import StrategyGenome
from evolution_engine.evolution.mutation import MutationEngine
from evolution_engine.evolution.selection import SelectionEngine
from evolution_engine.species.base_species import BaseSpecies


class PopulationFactory:
    def __init__(
        self,
        species_registry: dict[str, BaseSpecies],
        rng: np.random.Generator,
    ) -> None:
        self._species = species_registry
        self._rng = rng

    def generate_initial(
        self,
        size: int = MIN_POPULATION_SIZE,
        generation: int = 0,
    ) -> list[StrategyGenome]:
        population: list[StrategyGenome] = []
        species_names = list(self._species.keys())
        distribution = SPECIES_DISTRIBUTION

        for species_name, fraction in distribution.items():
            count = max(1, int(size * fraction))
            sp = self._species[species_name]
            for _ in range(count):
                genome = sp.spawn_genome(generation, self._rng)
                population.append(genome)

        # Fill up to exact size if rounding left gaps
        while len(population) < size:
            sp_name = str(self._rng.choice(species_names))
            genome = self._species[sp_name].spawn_genome(generation, self._rng)
            population.append(genome)

        return population[:size]

    def replenish(
        self,
        elites: list[StrategyGenome],
        target_size: int,
        generation: int,
        mutation_engine: MutationEngine,
        selection_engine: SelectionEngine,
    ) -> list[StrategyGenome]:
        """
        Build next generation:
        - Elites pass through unchanged
        - 75 % of remainder = mutated offspring from tournament-selected parents
        - 15 % = guaranteed fresh genomes per species (diversity protection)
        - 10 % = fully random injection (prevents premature convergence)
        """
        if not elites:
            return self.generate_initial(target_size, generation)

        next_gen: list[StrategyGenome] = list(elites)
        n_remaining = target_size - len(next_gen)
        species_names = list(self._species.keys())

        # Determine which species survived vs were eliminated
        elite_species = {g.species for g in elites}
        eliminated_species = [s for s in species_names if s not in elite_species]
        winner_species = elites[0].species if elites else None

        # Guarantee minimum slots per species (diversity protection)
        n_guaranteed_per_species = max(1, int(n_remaining * 0.15) // len(species_names))
        n_guaranteed = n_guaranteed_per_species * len(species_names)
        n_random = max(1, int(n_remaining * 0.10))
        n_offspring = n_remaining - n_guaranteed - n_random

        # Offspring via mutation from elites
        parents = selection_engine.tournament_select(
            elites, n_offspring, tournament_size=4, rng=self._rng
        )
        for parent in parents:
            sp = self._species.get(parent.species)
            if sp is None:
                continue
            child = mutation_engine.mutate(
                parent,
                mutation_rate=sp.traits.mutation_rate,
                mutation_strength=sp.traits.mutation_strength,
                species_indicator_weights=sp.traits.indicator_weight_vector,
            )
            child.generation = generation
            next_gen.append(child)

        # Guaranteed fresh genomes per species
        # Eliminated species mutate toward the winner's DNA
        for sp_name in species_names:
            for _ in range(n_guaranteed_per_species):
                if sp_name in eliminated_species and winner_species and elites:
                    # Eliminated species: spawn from winner genome + high mutation
                    winner_parent = next(
                        (g for g in elites if g.species == winner_species), elites[0]
                    )
                    winner_sp = self._species.get(winner_species)
                    child = mutation_engine.mutate(
                        winner_parent,
                        mutation_rate=0.80,           # Very high — aggressive learning
                        mutation_strength=0.50,
                        species_indicator_weights=self._species[sp_name].traits.indicator_weight_vector,
                    )
                    # Re-brand child to the eliminated species
                    import copy
                    child = copy.copy(child)
                    child.species = sp_name
                    child.generation = generation
                    next_gen.append(child)
                else:
                    genome = self._species[sp_name].spawn_genome(generation, self._rng)
                    next_gen.append(genome)

        # Fully random injection
        for _ in range(n_random):
            sp_name = str(self._rng.choice(species_names))
            genome = self._species[sp_name].spawn_genome(generation, self._rng)
            next_gen.append(genome)

        return next_gen[:target_size]
