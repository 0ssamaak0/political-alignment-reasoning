"""2x2 figure: DPO-LoRA trait-expression vs LoRA scale, marker-coloured by coherence.

The post-training analog of 4_steering's trait_vs_alpha_coherence figure, but the
x-axis is the LoRA multiplier (1.0 = trained strength) instead of the steering
coefficient. Reads results/scale_sweep/sweep_{family}_{direction}.json.

Per panel: a line coloured by political lean (left=blue, right=red), markers
coloured by coherence on RdYlGn (red=degenerate, green=fluent), and a grey base
(scale=0) anchor taken from 4_steering/docs/RESULTS_trait_eval.md's alpha=0 rows.

Run: python3 -m src.plot_scale_sweep
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Locate the brand-viz helper (repo root holds .claude/skills/...).
for _d in Path(__file__).resolve().parents:
    _s = _d / ".claude/skills/visualizations-designer/scripts"
    if _s.is_dir():
        sys.path.insert(0, str(_s))
        break
from polireason_viz import apply_theme, LEAN, BASE, save_fig  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results" / "scale_sweep"

# Base (scale=0) anchors = the alpha=0 rows in 4_steering/docs/RESULTS_trait_eval.md
# (trait, coherence), i.e. the unsteered base model scored by each lean's judge.
BASE_ANCHOR = {
    ("mistral", "left"): (32.2, 71.2),
    ("mistral", "right"): (1.0, 71.5),
    ("llama", "left"): (31.2, 71.2),
    ("llama", "right"): (7.9, 71.5),
}

CMAP = plt.get_cmap("RdYlGn")
NORM = Normalize(vmin=0, vmax=100)


def load_cell(family: str, direction: str):
    j = json.loads((RES / f"sweep_{family}_{direction}.json").read_text())
    ps = j["per_scale"]
    scales = sorted(float(s) for s in ps)
    # JSON dict keys are strings; match by float value to be key-format agnostic.
    trait, coh = [], []
    for s in scales:
        key = next(k for k in ps if float(k) == s)
        trait.append(ps[key]["trait_mean"])
        coh.append(ps[key]["coh_mean"])
    return scales, trait, coh


def main() -> None:
    apply_theme()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), sharex=True, sharey=True)
    families = ["mistral", "llama"]
    directions = ["left", "right"]

    for r, family in enumerate(families):
        for c, direction in enumerate(directions):
            ax = axes[r][c]
            scales, trait, coh = load_cell(family, direction)
            b_trait, b_coh = BASE_ANCHOR[(family, direction)]
            lean_color = LEAN[direction]

            xs = [0.0] + scales
            ys = [b_trait] + trait
            # line coloured by lean, behind the markers
            ax.plot(xs, ys, color=lean_color, lw=1.6, zorder=2, alpha=0.9)
            # swept markers coloured by coherence
            ax.scatter(scales, trait, c=coh, cmap=CMAP, norm=NORM,
                       s=90, edgecolor="#1A1A1A", linewidth=0.6, zorder=3)
            # grey base anchor
            ax.scatter([0.0], [b_trait], color=BASE, s=70, zorder=3,
                       edgecolor="#1A1A1A", linewidth=0.6)
            ax.annotate("base", (0.0, b_trait), textcoords="offset points",
                        xytext=(4, -11), fontsize=8, color="#6B6B6B")
            # trained-strength reference at scale = 1.0
            ax.axvline(1.0, color="#8A8A8A", ls=":", lw=1.0, alpha=0.7, zorder=1)

            ax.set_title(f"{family.capitalize()} · {direction}", loc="left")
            ax.set_ylim(-3, 107)
            ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
            if r == 1:
                ax.set_xlabel("LoRA scale  (1.0 = trained strength)")
            if c == 0:
                ax.set_ylabel("trait expression (0–100)")

    # shared coherence colorbar
    sm = ScalarMappable(norm=NORM, cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.03)
    cbar.set_label("coherence (0–100)  —  red = degenerate, green = fluent")

    fig.suptitle("PoliTune-HF DPO-LoRA: trait expression vs LoRA scale",
                 x=0.012, ha="left", fontweight="semibold", fontsize=12.5)

    save_fig(fig, str(RES / "figures" / "scale_vs_trait_coherence"))
    print(f"saved {RES / 'figures' / 'scale_vs_trait_coherence'}.{{pdf,svg}}")


if __name__ == "__main__":
    main()
