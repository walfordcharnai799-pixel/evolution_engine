"""
Multi-Species Trading Evolution Engine — entry point.
"""
from __future__ import annotations

import argparse
import json
import sys

from evolution_engine.orchestrator.evolution_loop import EvolutionLoop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Species Trading Evolution Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults (uses cached data if available):
  python main.py

  # Run 50 generations with a population of 60:
  python main.py --generations 50 --population 60

  # Run on specific symbols:
  python main.py --symbols EURUSD GBPUSD

  # Override MT5 connection details:
  python main.py --mt5-login 12345 --mt5-password secret --mt5-server MyBroker-Demo

  # Load config from JSON file:
  python main.py --config my_config.json

  # Clear cached data and re-fetch:
  python main.py --clear-cache

  # Use only cached data (no MT5 connection):
  python main.py --use-cached
        """,
    )

    parser.add_argument("--generations", type=int, default=None, help="Max generations to run")
    parser.add_argument("--population", type=int, default=None, help="Population size per generation")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to trade (e.g. EURUSD GBPUSD)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for determinism")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")
    parser.add_argument("--mt5-login", type=int, default=None)
    parser.add_argument("--mt5-password", type=str, default=None)
    parser.add_argument("--mt5-server", type=str, default=None)
    parser.add_argument("--mt5-path", type=str, default=None, help="Path to MT5 terminal64.exe")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cached data before starting")
    parser.add_argument("--use-cached", action="store_true", help="Use only cached data, no MT5 connection")
    parser.add_argument(
        "--create-video",
        action="store_true",
        help="After evolution completes, render an animated video of results",
    )
    parser.add_argument("--video-fps", type=int, default=8, help="Video frames per second (default: 8)")
    parser.add_argument("--video-dpi", type=int, default=120, help="Video render DPI (default: 120)")
    parser.add_argument("--video-format", choices=["mp4", "gif"], default="mp4", help="Video format (default: mp4)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Start with file-based config overrides
    config: dict = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # CLI overrides (highest priority)
    if args.generations is not None:
        config["MAX_GENERATIONS"] = args.generations
    if args.population is not None:
        config["MIN_POPULATION_SIZE"] = args.population
    if args.symbols is not None:
        config["SYMBOLS"] = args.symbols
    if args.seed is not None:
        config["RANDOM_SEED"] = args.seed
    if args.mt5_login is not None:
        config["MT5_LOGIN"] = args.mt5_login
    if args.mt5_password is not None:
        config["MT5_PASSWORD"] = args.mt5_password
    if args.mt5_server is not None:
        config["MT5_SERVER"] = args.mt5_server
    if args.mt5_path is not None:
        config["MT5_PATH"] = args.mt5_path

    # Handle cache options
    if args.clear_cache:
        from evolution_engine.data.cache_manager import CacheManager
        CacheManager().invalidate_all()
        print("Cache cleared.")

    loop = EvolutionLoop(config=config)
    loop.setup()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving state and exiting...")
    finally:
        loop.teardown()

    if args.create_video:
        from evolution_engine.orchestrator.video_creator import VideoCreator
        results_dir = config.get("DATA_CACHE_DIR", "evolution_engine/data/cache").replace(
            "data/cache", "results"
        )
        output_path = f"{results_dir}/evolution.{args.video_format}"
        creator = VideoCreator(results_dir=results_dir, output_path=output_path)
        creator.create(fps=args.video_fps, dpi=args.video_dpi)


if __name__ == "__main__":
    main()
