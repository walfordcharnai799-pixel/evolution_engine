from .base_species import BaseSpecies, SpeciesTraits  # noqa: F401
from .davinci import DaVinciSpecies  # noqa: F401
from .newton import NewtonSpecies  # noqa: F401
from .tesla import TeslaSpecies  # noqa: F401
from .aristotle import AristotleSpecies  # noqa: F401
from .hypatia import HypatiaSpecies  # noqa: F401
from .archimedes import ArchimedesSpecies  # noqa: F401
from .hawking import HawkingSpecies  # noqa: F401
from .turing import TuringSpecies  # noqa: F401

SPECIES_REGISTRY: dict[str, BaseSpecies] = {
    "davinci":    DaVinciSpecies(),
    "newton":     NewtonSpecies(),
    "tesla":      TeslaSpecies(),
    "aristotle":  AristotleSpecies(),
    "hypatia":    HypatiaSpecies(),
    "archimedes": ArchimedesSpecies(),
    "hawking":    HawkingSpecies(),
    "turing":     TuringSpecies(),
}
