"""Plot the persona-vector trait-expression dose-response.

x = α (steering coefficient), y = mean trait-expression score (0-100),
marker colour = mean coherence (0-100) on an RdYlGn map so low coherence
reads red ("less accurate / degenerate") and high coherence reads green.

2x2 panel: rows = family (Mistral / Llama), cols = lean (left / right).
Shared colorbar. Reads the four sweep jsons written by coef_sweep.py.

    python3 -m src.plot_trait_eval
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

_ROOT = Path(__file__).resolve().parent.parent
_SWEEP = _ROOT / "results" / "coef_sweep"
_OUT = _SWEEP / "figures" / "trait_vs_alpha_coherence.png"

CMAP = "RdYlGn"          # low coherence -> red (bad), high -> green (good)
NORM = Normalize(vmin=0, vmax=100)

CELLS = [  # (family, lean) -> grid position (row, col)
    ("mistral", "left"), ("mistral", "right"),
    ("llama", "left"), ("llama", "right"),
]


def _load(family: str, lean: str):
    d = json.loads((_SWEEP / f"sweep_{family}_{lean}.json").read_text())
    coefs = sorted(float(c) for c in d["per_coef"])
    trait = [d["per_coef"][_key(d, c)]["trait_mean"] for c in coefs]
    coh = [d["per_coef"][_key(d, c)]["coh_mean"] for c in coefs]
    return coefs, trait, coh


def _key(d, c):
    # per_coef keys may be "2.0" or "2"; match by float value
    for k in d["per_coef"]:
        if float(k) == c:
            return k
    raise KeyError(c)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    pos = {("mistral", "left"): axes[0, 0], ("mistral", "right"): axes[0, 1],
           ("llama", "left"): axes[1, 0], ("llama", "right"): axes[1, 1]}

    for (family, lean) in CELLS:
        ax = pos[(family, lean)]
        coefs, trait, coh = _load(family, lean)
        # connecting line (neutral grey) under the coherence-coloured markers
        ax.plot(coefs, trait, color="0.6", lw=1.2, zorder=1)
        sc = ax.scatter(coefs, trait, c=coh, cmap=CMAP, norm=NORM,
                        s=130, edgecolor="black", linewidth=0.6, zorder=2)
        # annotate the α=0 base point
        ax.axvline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_title(f"{family.capitalize()} — {lean}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_xticks(coefs)

    for ax in axes[-1, :]:
        ax.set_xlabel("Steering coefficient α  (α=0 = unsteered base)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean trait-expression score (0–100)")

    cbar = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), ax=axes,
                        fraction=0.046, pad=0.04)
    cbar.set_label("Mean coherence (0–100) — red = degenerate, green = fluent")

    fig.suptitle("Persona-vector trait expression vs steering coefficient\n"
                 "(colour = coherence; trait collapses past the α≈4 coherence cliff)",
                 fontsize=13)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=130, bbox_inches="tight")
    print(f"saved {_OUT}")


if __name__ == "__main__":
    main()
