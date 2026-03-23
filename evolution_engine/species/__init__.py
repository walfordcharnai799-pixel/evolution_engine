from .base_species import BaseSpecies, SpeciesTraits  # noqa: F401
from .newton import NewtonSpecies  # noqa: F401
from .turing import TuringSpecies  # noqa: F401

SPECIES_REGISTRY: dict[str, BaseSpecies] = {
    "newton":     NewtonSpecies(),
    "turing":     TuringSpecies(),
}
