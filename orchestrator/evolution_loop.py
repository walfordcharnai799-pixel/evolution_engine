"""
Evolution loop — the main orchestrator.
Wires all components together and drives the generation-by-generation loop.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from evolution_engine.config import settings as _s
from evolution_engine.data.cache_manager import CacheManager
from evolution_engine.data.mt5_connector import MT5Connector, MT5ConnectionError
from evolution_engine.engine.backtester import Backtester
from evolution_engine.engine.metrics import MetricsEngine, MetricsReport
from evolution_engine.engine.survival_filter import SurvivalFilter
from evolution_engine.evolution.genome import StrategyGenome
from evolution_engine.evolution.mutation import MutationEngine
from evolution_engine.evolution.population import PopulationFactory
from evolution_engine.evolution.selection import SelectionEngine
from evolution_engine.indicators.indicator_cache import IndicatorCache
from evolution_engine.orchestrator.logger import EvolutionLogger
from evolution_engine.orchestrator.results_exporter import ResultsExporter
from evolution_engine.species import SPECIES_REGISTRY


class EvolutionLoop:
    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self._cfg = config or {}
        self._rng: Optional[np.random.Generator] = None
        self._mt5: Optional[MT5Connector] = None
        self._cache_mgr: Optional[CacheManager] = None
        self._indicator_cache: Optional[IndicatorCache] = None
        self._backtester: Optional[Backtester] = None
        self._metrics_engine: Optional[MetricsEngine] = None
        self._survival_filter: Optional[SurvivalFilter] = None
        self._population_factory: Optional[PopulationFactory] = None
        self._mutation_engine: Optional[MutationEngine] = None
        self._selection_engine: Optional[SelectionEngine] = None
        self._logger: Optional[EvolutionLogger] = None
        self._exporter: Optional[ResultsExporter] = None
        self._price_data: dict = {}

    # ------------------------------------------------------------------
    # Config accessors (local override > settings.py)
    # ------------------------------------------------------------------

    def _get(self, key: str):
        return self._cfg.get(key, getattr(_s, key))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        seed = self._get("RANDOM_SEED")
        self._rng = np.random.default_rng(seed)

        self._logger = EvolutionLogger(
            log_dir=self._get("DATA_CACHE_DIR").replace("data/cache", "logs")
        )
        self._logger._console.info("Setting up Evolution Engine...")

        # Data
        self._cache_mgr = CacheManager(self._get("DATA_CACHE_DIR"))
        self._connect_and_load_data()

        # Indicators
        self._indicator_cache = IndicatorCache()

        # Backtester
        self._backtester = Backtester(
            price_data=self._price_data,
            indicator_cache=self._indicator_cache,
        )

        # Metrics & survival
        self._metrics_engine = MetricsEngine()
        self._survival_filter = SurvivalFilter()

        # Evolution
        self._mutation_engine = MutationEngine(self._rng)
        self._selection_engine = SelectionEngine()
        self._population_factory = PopulationFactory(SPECIES_REGISTRY, self._rng)

        # Results
        self._exporter = ResultsExporter(
            results_dir=self._get("DATA_CACHE_DIR").replace("data/cache", "results")
        )

        self._logger._console.info("Setup complete. Ready to evolve.")

    def _connect_and_load_data(self) -> None:
        symbols = self._get("SYMBOLS")
        n_bars = self._get("LOOKBACK_BARS")
        data_source = self._cfg.get("DATA_SOURCE", "auto")  # "mt5", "yfinance", "auto"

        # Check if all data is cached
        all_cached = all(
            self._cache_mgr.exists(self._cache_mgr.cache_key(sym, tf, n_bars))
            for sym in symbols
            for tf in _s.ALLOWED_TIMEFRAMES
        )

        if all_cached:
            self._logger._console.info("All data found in cache.")
            for sym in symbols:
                self._price_data[sym] = {}
                for tf in _s.ALLOWED_TIMEFRAMES:
                    key = self._cache_mgr.cache_key(sym, tf, n_bars)
                    self._price_data[sym][tf] = self._cache_mgr.load(key)
            return

        # Decide data source
        if data_source == "yfinance" or (data_source == "auto" and not self._get("MT5_LOGIN")):
            self._logger._console.info("Fetching real market data via yfinance...")
            from evolution_engine.data.yfinance_loader import fetch_real_data
            self._price_data = fetch_real_data(
                symbols=symbols,
                n_bars=n_bars,
                cache_dir=self._get("DATA_CACHE_DIR"),
            )
        else:
            self._logger._console.info("Connecting to MT5...")
            self._mt5 = MT5Connector(
                login=self._get("MT5_LOGIN"),
                password=self._get("MT5_PASSWORD"),
                server=self._get("MT5_SERVER"),
                terminal_path=self._get("MT5_PATH"),
            )
            try:
                self._mt5.connect()
            except MT5ConnectionError as e:
                raise RuntimeError(
                    f"MT5 connection failed: {e}\n"
                    "Tip: set DATA_SOURCE=yfinance to use free data instead."
                ) from e
            self._price_data = self._cache_mgr.load_all_symbols(
                self._mt5, symbols, n_bars
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        max_gen = self._get("MAX_GENERATIONS")
        pop_size = self._get("MIN_POPULATION_SIZE")
        elite_frac = self._get("ELITE_FRACTION")

        population = self._population_factory.generate_initial(pop_size, generation=0)
        hall_of_fame: list[tuple[StrategyGenome, MetricsReport]] = []

        for generation in range(max_gen):
            self._logger.log_generation_start(generation, len(population))

            # --- Backtest all genomes ---
            results = {}
            reports = {}
            for genome in population:
                try:
                    result = self._backtester.run(genome)
                    report = self._metrics_engine.compute(result)
                    genome.fitness_score = report.fitness_score
                    results[genome.genome_id] = result
                    reports[genome.genome_id] = report
                    self._logger.log_genome_backtest(genome, result, report)
                except Exception as e:
                    self._logger.log_error(
                        f"backtest gen={generation} genome={genome.genome_id}", e
                    )
                    genome.fitness_score = -1.0

            # --- Survival filter (progressive gates) ---
            survivors, survival_results = self._survival_filter.filter_population(
                population, reports, generation=generation
            )
            for sr in survival_results:
                self._logger.log_survival_result(sr)

            # --- Rank survivors ---
            ranked = self._selection_engine.rank_population(survivors)

            # Fallback: if 0 survivors, use top 20% by raw fitness as provisional elites
            # so evolution can climb toward the strict gates rather than random-restart each gen
            if not ranked:
                all_ranked = self._selection_engine.rank_population(population)
                n_fallback = max(2, int(len(all_ranked) * 0.20))
                ranked = all_ranked[:n_fallback]
                self._logger._console.warning(
                    f"Gen {generation}: 0 survivors — using top {n_fallback} by fitness as provisional elites"
                )

            # --- Update hall of fame ---
            if ranked and ranked[0].genome_id in reports:
                best = ranked[0]
                best_report = reports[best.genome_id]
                self._update_hall_of_fame(hall_of_fame, best, best_report)

            # --- Export generation results ---
            self._exporter.export_generation(
                generation, survivors, reports, population, survival_results
            )

            # --- Log generation summary ---
            if ranked:
                self._logger.log_generation_summary(
                    generation,
                    survivors,
                    ranked[0],
                    reports[ranked[0].genome_id],
                    self._count_species(population),
                )
            else:
                self._logger._console.warning(
                    f"Gen {generation}: NO SURVIVORS. Full population regeneration."
                )

            # --- Evolve next generation ---
            elites = self._selection_engine.select_elites(ranked, elite_frac)
            population = self._population_factory.replenish(
                elites,
                pop_size,
                generation + 1,
                self._mutation_engine,
                self._selection_engine,
            )

            # Clear indicator cache between generations (data unchanged, genomes change)
            self._indicator_cache.clear()

        # Export hall of fame
        self._exporter.export_hall_of_fame(hall_of_fame)
        self._logger._console.info(
            f"Evolution complete. Hall of fame has {len(hall_of_fame)} entries. "
            f"Results saved to {self._exporter._results_dir}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_hall_of_fame(
        self,
        hof: list[tuple[StrategyGenome, MetricsReport]],
        genome: StrategyGenome,
        report: MetricsReport,
        max_size: int = 10,
    ) -> None:
        existing_ids = {g.genome_id for g, _ in hof}
        if genome.genome_id not in existing_ids:
            hof.append((genome, report))
        hof.sort(key=lambda x: x[1].fitness_score, reverse=True)
        del hof[max_size:]

    def _count_species(self, population: list[StrategyGenome]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for g in population:
            counts[g.species] = counts.get(g.species, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def teardown(self) -> None:
        if self._mt5 and self._mt5.is_connected():
            self._mt5.disconnect()
            self._logger._console.info("MT5 disconnected.")
