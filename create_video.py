"""
CLI: generate an animated video from evolution results.

Usage
-----
  python create_video.py
  python create_video.py --results-dir evolution_engine/results --output evolution.mp4
  python create_video.py --fps 12 --dpi 150 --format gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render evolution results as an animated video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: reads evolution_engine/results, writes evolution.mp4 there
  python create_video.py

  # Custom directory and output
  python create_video.py --results-dir my_run/results --output my_run/replay.mp4

  # GIF output at 12 fps
  python create_video.py --format gif --fps 12

  # Higher resolution
  python create_video.py --dpi 180
        """,
    )
    p.add_argument(
        "--results-dir",
        default="evolution_engine/results",
        help="Directory containing gen_XXXX/ sub-directories (default: evolution_engine/results)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output file path. Defaults to <results-dir>/evolution.<format>",
    )
    p.add_argument(
        "--format",
        choices=["mp4", "gif"],
        default="mp4",
        help="Output format (default: mp4; falls back to gif if ffmpeg is absent)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second (default: 8)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Render resolution in dots per inch (default: 120)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir
    output_path = args.output or str(
        Path(results_dir) / f"evolution.{args.format}"
    )

    try:
        from evolution_engine.orchestrator.video_creator import VideoCreator
    except ImportError as e:
        print(f"Import error: {e}", file=sys.stderr)
        print(
            "Make sure you have installed the requirements:\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    creator = VideoCreator(results_dir=results_dir, output_path=output_path)
    try:
        out = creator.create(fps=args.fps, dpi=args.dpi)
        print(f"\nDone. Video written to: {out}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
