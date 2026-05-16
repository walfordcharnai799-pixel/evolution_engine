"""
Video creator — renders an animated evolution replay from per-generation
summary.json files produced by ResultsExporter.

Outputs MP4 (requires ffmpeg) or falls back to GIF (requires Pillow).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


_BG      = "#0D1117"
_PANEL   = "#161B22"
_ACCENT  = "#58A6FF"
_TEXT    = "#E6EDF3"
_GRID    = "#30363D"
_TARGET  = "#FFD700"
_SURV    = "#56D364"
_DOT     = "#FF6B6B"

_SPECIES_COLORS: dict[str, str] = {
    "newton":     "#4C72B0",
    "turing":     "#DD8452",
    "einstein":   "#55A868",
    "curie":      "#C44E52",
    "tesla":      "#8172B3",
    "archimedes": "#937860",
    "aristotle":  "#DA8BC3",
    "hawking":    "#8C8C8C",
    "hypatia":    "#CCB974",
    "davinci":    "#64B5CD",
}
_DEFAULT_COLOR = "#888888"


class VideoCreator:
    """
    Reads gen_XXXX/summary.json files from *results_dir* and produces an
    animated video where each frame represents one generation.

    Parameters
    ----------
    results_dir:
        Directory containing gen_0000/, gen_0001/, … sub-directories.
    output_path:
        Destination file.  Extension determines format: .mp4 or .gif.
        Defaults to ``<results_dir>/evolution.mp4``.
    """

    def __init__(
        self,
        results_dir: str = "evolution_engine/results",
        output_path: str | None = None,
    ) -> None:
        self._results_dir = Path(results_dir)
        self._output_path = (
            Path(output_path) if output_path
            else self._results_dir / "evolution.mp4"
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def create(self, fps: int = 8, dpi: int = 120) -> str:
        """
        Build the animation and write it to disk.

        Returns the output path as a string.
        Raises ValueError if no generation summaries are found.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

        frames = self._load_frames()
        if not frames:
            raise ValueError(
                f"No generation summaries found under {self._results_dir!r}. "
                "Run the evolution engine first."
            )

        fig = plt.figure(figsize=(16, 9), facecolor=_BG)
        gs  = gridspec.GridSpec(
            2, 3, figure=fig,
            hspace=0.44, wspace=0.36,
            left=0.07, right=0.97, top=0.90, bottom=0.10,
        )

        ax_fit  = fig.add_subplot(gs[0, :2])  # top-left wide  — fitness history
        ax_sp   = fig.add_subplot(gs[1, :2])  # bottom-left    — species bars
        ax_met  = fig.add_subplot(gs[0, 2])   # top-right      — best-genome metrics
        ax_pop  = fig.add_subplot(gs[1, 2])   # bottom-right   — population stats

        _style([ax_fit, ax_sp, ax_met, ax_pop])

        fig.text(
            0.5, 0.96,
            "Multi-Species Trading Evolution Engine",
            ha="center", va="top",
            color=_TEXT, fontsize=14, fontweight="bold",
        )

        # Precomputed series
        gens        = [f["generation"]              for f in frames]
        best_fit    = [f.get("best_fitness", 0.0)  for f in frames]
        surv_rates  = [f.get("survival_rate", 0.0) for f in frames]
        max_gen     = max(gens) if gens else 1

        # ---- Fitness axis -------------------------------------------
        ax_fit.set_title("Best Fitness Score", color=_TEXT, fontsize=10, pad=6)
        ax_fit.set_xlabel("Generation", color=_TEXT, fontsize=8)
        ax_fit.set_ylabel("Fitness", color=_TEXT, fontsize=8)
        ax_fit.set_xlim(-0.5, max_gen + 0.5)
        ax_fit.set_ylim(0, 1.05)
        ax_fit.axhline(y=0.74, color=_TARGET, lw=0.8, ls="--", alpha=0.6, label="Target 0.74")
        fit_line,  = ax_fit.plot([], [], color=_ACCENT,  lw=1.8, zorder=3, label="Best fitness")
        surv_line, = ax_fit.plot([], [], color=_SURV, lw=1.2, ls=":", alpha=0.7, label="Survival rate")
        gen_dot,   = ax_fit.plot([], [], "o", color=_DOT, ms=6, zorder=4)
        ax_fit.legend(
            fontsize=7, facecolor=_PANEL, edgecolor=_GRID, labelcolor=_TEXT, loc="upper left"
        )

        # ---- Species bars (rebuilt each frame) ----------------------

        # ---- Metrics text panel -------------------------------------
        ax_met.set_title("Best Genome Metrics", color=_TEXT, fontsize=10, pad=6)
        ax_met.axis("off")
        met_txt = ax_met.text(
            0.05, 0.95, "", transform=ax_met.transAxes,
            color=_TEXT, fontsize=9, va="top", fontfamily="monospace",
        )

        # ---- Population stats text ----------------------------------
        ax_pop.set_title("Population Stats", color=_TEXT, fontsize=10, pad=6)
        ax_pop.axis("off")
        pop_txt = ax_pop.text(
            0.05, 0.95, "", transform=ax_pop.transAxes,
            color=_TEXT, fontsize=9, va="top", fontfamily="monospace",
        )

        # ---- Animation update ---------------------------------------
        def update(i: int):
            f = frames[i]
            gen = f["generation"]

            # Fitness + survival lines
            xs = gens[:i + 1]
            fit_line.set_data(xs, best_fit[:i + 1])
            surv_line.set_data(xs, surv_rates[:i + 1])
            gen_dot.set_data([gen], [best_fit[i]])

            # Species bars
            ax_sp.cla()
            _style([ax_sp])
            ax_sp.set_title(
                "Species Survival Rate (current gen)", color=_TEXT, fontsize=10, pad=6
            )
            ax_sp.set_xlim(0, 1.05)
            ax_sp.set_xlabel("Survival Rate", color=_TEXT, fontsize=8)
            sp_stats = f.get("species_stats", {})
            if sp_stats:
                names  = list(sp_stats.keys())
                rates  = [sp_stats[s].get("survival_rate", 0.0) for s in names]
                colors = [_SPECIES_COLORS.get(s, _DEFAULT_COLOR) for s in names]
                ypos   = np.arange(len(names))
                bars   = ax_sp.barh(ypos, rates, color=colors, alpha=0.85, height=0.6)
                ax_sp.set_yticks(ypos)
                ax_sp.set_yticklabels(names, color=_TEXT, fontsize=8)
                for bar, rate in zip(bars, rates):
                    ax_sp.text(
                        min(rate + 0.02, 1.02),
                        bar.get_y() + bar.get_height() / 2,
                        f"{rate:.0%}", va="center", color=_TEXT, fontsize=7,
                    )

            # Best-genome metrics
            met_txt.set_text(
                "\n".join([
                    f"Gen:      {gen:04d}",
                    f"Species:  {f.get('best_species', '—')}",
                    "",
                    f"Fitness:  {f.get('best_fitness', 0.0):.4f}",
                    f"Win Rate: {f.get('best_win_rate', 0.0):.1%}",
                    f"Prof.Fac: {f.get('best_profit_factor', 0.0):.2f}",
                    f"Drawdown: {f.get('best_max_drawdown', 0.0):.1%}",
                    f"Trades:   {f.get('best_total_trades', 0)}",
                ])
            )

            # Population stats
            pop_txt.set_text(
                "\n".join([
                    f"Population: {f.get('total_genomes', 0)}",
                    f"Survivors:  {f.get('survivors', 0)}",
                    f"Surv. Rate: {f.get('survival_rate', 0.0):.1%}",
                    "",
                    f"# Species:  {len(sp_stats)}",
                ])
            )

            return fit_line, surv_line, gen_dot, met_txt, pop_txt

        anim = FuncAnimation(
            fig, update, frames=len(frames),
            interval=1000 // fps, blit=False,
        )

        out = self._resolved_output()
        if out.suffix.lower() == ".gif":
            writer = PillowWriter(fps=fps)
        else:
            try:
                writer = FFMpegWriter(fps=fps, bitrate=1800)
            except Exception:
                out = out.with_suffix(".gif")
                writer = PillowWriter(fps=fps)

        print(f"Rendering {len(frames)} frames → {out}")
        anim.save(str(out), writer=writer, dpi=dpi)
        plt.close(fig)
        print(f"Saved: {out}")
        return str(out)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_frames(self) -> list[dict[str, Any]]:
        summaries = sorted(self._results_dir.glob("gen_*/summary.json"))
        frames = []
        for path in summaries:
            with open(path) as fh:
                frames.append(json.load(fh))
        return frames

    def _resolved_output(self) -> Path:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        return self._output_path


def _style(axes: list) -> None:
    for ax in axes:
        ax.set_facecolor(_PANEL)
        ax.tick_params(colors=_TEXT, labelsize=7)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color(_GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
