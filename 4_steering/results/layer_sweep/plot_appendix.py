"""Appendix figure: single-layer trait-expression sweep, 2x2 cross-family.

Regenerates layer_curve_2x2 in the thesis house style (vector PDF+SVG,
left=blue, right=red, title-free, minimal panel labels) for the
single-layer steering appendix. Reads the 4 live sweep JSONs.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

for _d in Path(__file__).resolve().parents:
    _s = _d / ".claude/skills/visualizations-designer/scripts"
    if _s.is_dir():
        sys.path.insert(0, str(_s)); break
from polireason_viz import apply_theme, LEAN, save_fig
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = Path("/Users/0ssamaak0/Documents/Writing/tempelate/images/app_single_layer_sweep")

FAMILIES = ("mistral", "llama")
DIRECTIONS = ("left", "right")


def means_by_layer(data):
    items = sorted(((int(L), c["mean"]) for L, c in data["per_layer"].items()))
    return [L for L, _ in items], [m for _, m in items]


apply_theme()
fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.8), sharey=True, sharex=True)
for i, fam in enumerate(FAMILIES):
    for j, d in enumerate(DIRECTIONS):
        ax = axes[i, j]
        data = json.loads((BASE / fam / f"sweep_{d}.json").read_text())
        Ls, means = means_by_layer(data)
        ax.plot(Ls, means, marker="o", markersize=3.5, lw=1.4,
                color=LEAN[d], alpha=0.95)
        be = data["best_layer_excl_last_2"]
        ax.axvline(be, color=LEAN[d], ls=":", lw=1.0, alpha=0.8)
        ax.set_title(f"{fam.title()} · {d}", fontsize=11)
        ax.set_ylim(0, 105)
        if i == 1:
            ax.set_xlabel("Layer")
        if j == 0:
            ax.set_ylabel("Trait-expression score")
fig.tight_layout()
save_fig(fig, str(OUT))
print("wrote", OUT)
