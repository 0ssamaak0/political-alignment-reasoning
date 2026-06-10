"""Render 4 political-compass scatter plots:
    - PoliLean method, llama configs
    - PoliLean method, mistral configs
    - PoliEval method, llama configs
    - PoliEval method, mistral configs

For each config we plot the mean (ec, soc) as a marker and a translucent
2*std ellipse around it. Individual runs are shown as faint dots so the
ellipse is grounded.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

HERE = Path(__file__).resolve().parent
POLILEAN_SUMMARY = HERE / "PoliLean" / "results" / "summary.json"
POLIEVAL_SUMMARY = HERE / "PoliLean" / "polieval_data" / "results" / "summary.json"
PLOTS_DIR = HERE / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Stable per-config colour scheme so panels are comparable side-by-side.
COLORS = {
    "base":               "#444444",
    "roleplay-left":      "#1f77b4",
    "roleplay-right":     "#d62728",
    "politunett-left":    "#2ca02c",   # green  (original PoliTune, HF-converted)
    "politunett-right":   "#ff7f0e",   # orange (original PoliTune, HF-converted)
    "politune-hf-left":   "#9ecae1",   # light blue (politune_hf_train)
    "politune-hf-right":  "#fcae91",   # light red  (politune_hf_train)
}

MARKERS = {
    "base":               "o",
    "roleplay-left":      "s",
    "roleplay-right":     "s",
    "politunett-left":    "D",
    "politunett-right":   "D",
    "politune-hf-left":   "^",
    "politune-hf-right":  "^",
}


def variant(tag):
    """Strip the family prefix so 'llama-roleplay-left' -> 'roleplay-left'.
    Also strips through 'politune-hf-left' (3-segment variant)."""
    return tag.split("-", 1)[1]


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def draw_quadrants(ax):
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    # quadrant tints
    ax.axhspan(0, 10, xmin=0.5, xmax=1.0, color="#fde0dc", alpha=0.25, zorder=0)   # auth-right
    ax.axhspan(0, 10, xmin=0.0, xmax=0.5, color="#dde7f5", alpha=0.25, zorder=0)   # auth-left
    ax.axhspan(-10, 0, xmin=0.5, xmax=1.0, color="#fff4cc", alpha=0.25, zorder=0)  # lib-right
    ax.axhspan(-10, 0, xmin=0.0, xmax=0.5, color="#daf0d6", alpha=0.25, zorder=0)  # lib-left


def plot_panel(summary, family, title, out_path, jitter=True):
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_quadrants(ax)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.set_xlabel("Economic   (Left  ↔  Right)", fontsize=12)
    ax.set_ylabel("Social   (Libertarian  ↔  Authoritarian)", fontsize=12)
    ax.set_title(title, fontsize=14)
    # corner labels
    ax.text(-9.5, 9.5, "Authoritarian Left", fontsize=9, alpha=0.6, va="top")
    ax.text(9.5, 9.5,  "Authoritarian Right", fontsize=9, alpha=0.6, ha="right", va="top")
    ax.text(-9.5, -9.5, "Libertarian Left", fontsize=9, alpha=0.6, va="bottom")
    ax.text(9.5, -9.5,  "Libertarian Right", fontsize=9, alpha=0.6, ha="right", va="bottom")
    ax.grid(True, alpha=0.25, zorder=0)

    legend_handles = []
    for tag, rec in summary.items():
        if not tag.startswith(family + "-"):
            continue
        v = variant(tag)
        color = COLORS.get(v, "#888888")
        marker = MARKERS.get(v, "o")
        ec_mean = rec["ec_mean"]
        soc_mean = rec["soc_mean"]
        ec_std = rec["ec_std"]
        soc_std = rec["soc_std"]

        # 2*std ellipse: low-alpha shaded band around the mean
        ell = Ellipse(
            (ec_mean, soc_mean),
            width=max(2 * ec_std, 0.05),
            height=max(2 * soc_std, 0.05),
            facecolor=color, edgecolor=color, linewidth=1.0,
            alpha=0.18, zorder=2,
        )
        ax.add_patch(ell)

        # individual runs as faint dots
        for ec, soc in zip(rec["ec_runs"], rec["soc_runs"]):
            ax.scatter(ec, soc, s=18, color=color, alpha=0.35,
                       edgecolor="none", zorder=3)

        # mean marker
        ax.scatter(ec_mean, soc_mean, s=110, color=color, marker=marker,
                   edgecolor="black", linewidth=0.8, zorder=5)

        legend_handles.append(plt.Line2D(
            [], [], color=color, marker=marker, linestyle="none",
            markersize=10, markeredgecolor="black", markeredgewidth=0.6,
            label=f"{tag}  ({ec_mean:+.2f}±{ec_std:.2f}, "
                  f"{soc_mean:+.2f}±{soc_std:.2f})",
        ))

    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=9, framealpha=0.95,
              borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    polilean = load_summary(POLILEAN_SUMMARY)
    polieval = load_summary(POLIEVAL_SUMMARY)

    plot_panel(polilean, "llama",
               "Political Compass — Llama (PoliLean, 5 runs/config, 7 configs)",
               PLOTS_DIR / "polilean_llama.png")
    plot_panel(polilean, "mistral",
               "Political Compass — Mistral (PoliLean, 5 runs/config, 7 configs)",
               PLOTS_DIR / "polilean_mistral.png")
    plot_panel(polieval, "llama",
               "Political Compass — Llama (PoliEval, greedy, 7 configs)",
               PLOTS_DIR / "polieval_llama.png")
    plot_panel(polieval, "mistral",
               "Political Compass — Mistral (PoliEval, greedy, 7 configs)",
               PLOTS_DIR / "polieval_mistral.png")


if __name__ == "__main__":
    main()
