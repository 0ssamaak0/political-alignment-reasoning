"""Render a single 2D Political Compass figure overlaying PoliLean and
PoliEval results for the 14 Mistral pvsteer-ml cells documented in
``4_steering/docs/RESULTS_polieval.md``.

Each model variant (tag) gets one stable colour; PoliLean is drawn as a
circle and PoliEval as a square. A thin grey segment connects the two
markers for the same variant so the methodology shift is visually
explicit.

Output: ``4_steering/docs/compass_polilean_vs_polieval.png``.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "4_steering" / "docs" / "compass_polilean_vs_polieval.png"

# Per-cell numbers transcribed verbatim from
# 4_steering/docs/RESULTS_polieval.md (Per-cell numbers table).
ALPHAS = [2.0, 2.2, 2.4, 2.5, 2.6, 2.8, 3.0]
ALPHA_SUFFIX = {2.0: "a2", 2.2: "a2_2", 2.4: "a2_4", 2.5: "a2_5",
                2.6: "a2_6", 2.8: "a2_8", 3.0: "a3"}

# (direction, alpha) -> {"polilean": (ec, soc), "polieval": (ec, soc)}
DATA = {
    ("left", 2.0): {"polilean": (-5.73, -5.64), "polieval": (-0.68, +1.08)},
    ("left", 2.2): {"polilean": (-4.58, -5.26), "polieval": (+0.18, +0.94)},
    ("left", 2.4): {"polilean": (-4.58, -4.32), "polieval": (+1.50, +1.36)},
    ("left", 2.5): {"polilean": (-4.03, -4.49), "polieval": (+1.18, +1.53)},
    ("left", 2.6): {"polilean": (-3.65, -4.46), "polieval": (+0.03, -0.86)},
    ("left", 2.8): {"polilean": (-3.05, -3.85), "polieval": (+0.03, -0.84)},
    ("left", 3.0): {"polilean": (-3.13, -2.74), "polieval": (-0.28, -2.32)},
    ("right", 2.0): {"polilean": (-1.23, -2.21), "polieval": (+3.40, +2.31)},
    ("right", 2.2): {"polilean": (+0.93, -2.90), "polieval": (+3.18, +2.47)},
    ("right", 2.4): {"polilean": (+1.30, -2.58), "polieval": (+2.05, +2.39)},
    ("right", 2.5): {"polilean": (+1.20, -2.75), "polieval": (+2.15, +2.21)},
    ("right", 2.6): {"polilean": (+0.70, -2.21), "polieval": (+1.95, +2.04)},
    ("right", 2.8): {"polilean": (+1.90, -1.43), "polieval": (+2.43, +2.56)},
    ("right", 3.0): {"polilean": (+1.85, -2.50), "polieval": (+1.90, +2.31)},
}


def variant_color(direction, alpha):
    # Blue gradient for left, red gradient for right; saturation tracks α
    # (α=2.0 lightest, α=3.0 darkest).
    norm = Normalize(vmin=1.6, vmax=3.2)
    if direction == "left":
        return cm.Blues(norm(alpha))
    return cm.Reds(norm(alpha))


def draw_quadrants(ax):
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.axhspan(0, 10, xmin=0.5, xmax=1.0, color="#fde0dc", alpha=0.25, zorder=0)
    ax.axhspan(0, 10, xmin=0.0, xmax=0.5, color="#dde7f5", alpha=0.25, zorder=0)
    ax.axhspan(-10, 0, xmin=0.5, xmax=1.0, color="#fff4cc", alpha=0.25, zorder=0)
    ax.axhspan(-10, 0, xmin=0.0, xmax=0.5, color="#daf0d6", alpha=0.25, zorder=0)


def main():
    fig, ax = plt.subplots(figsize=(10, 9))
    draw_quadrants(ax)
    ax.set_xlim(-7, 5)
    ax.set_ylim(-7, 5)
    ax.set_aspect("equal")
    ax.set_xlabel("Economic   (Left  ↔  Right)", fontsize=12)
    ax.set_ylabel("Social   (Libertarian  ↔  Authoritarian)", fontsize=12)
    ax.set_title(
        "Political Compass — Mistral pvsteer-ml (PoliLean ○ vs PoliEval □)",
        fontsize=13,
    )
    ax.text(-6.7, 4.7, "Authoritarian Left", fontsize=9, alpha=0.6, va="top")
    ax.text(4.7, 4.7, "Authoritarian Right", fontsize=9, alpha=0.6, ha="right", va="top")
    ax.text(-6.7, -6.7, "Libertarian Left", fontsize=9, alpha=0.6, va="bottom")
    ax.text(4.7, -6.7, "Libertarian Right", fontsize=9, alpha=0.6, ha="right", va="bottom")
    ax.grid(True, alpha=0.25, zorder=0)

    legend_handles = []
    for (direction, alpha), pts in DATA.items():
        color = variant_color(direction, alpha)
        ec_pl, soc_pl = pts["polilean"]
        ec_pe, soc_pe = pts["polieval"]

        # Connector so the per-variant pair is obvious.
        ax.plot([ec_pl, ec_pe], [soc_pl, soc_pe],
                color=color, linewidth=0.9, alpha=0.55, zorder=2)

        # PoliLean = circle, PoliEval = square (same color per variant).
        ax.scatter(ec_pl, soc_pl, s=130, color=color, marker="o",
                   edgecolor="black", linewidth=0.8, zorder=5)
        ax.scatter(ec_pe, soc_pe, s=130, color=color, marker="s",
                   edgecolor="black", linewidth=0.8, zorder=5)

        label = f"pvsteer-ml-{direction}-α={alpha}"
        legend_handles.append(plt.Line2D(
            [], [], color=color, marker="o", linestyle="none",
            markersize=10, markeredgecolor="black", markeredgewidth=0.6,
            label=label,
        ))

    # Method-shape legend (top-left, inside the axes).
    method_legend = ax.legend(
        handles=[
            plt.Line2D([], [], color="#444", marker="o", linestyle="none",
                       markersize=11, markeredgecolor="black",
                       markeredgewidth=0.6, label="PoliLean (bart-mnli)"),
            plt.Line2D([], [], color="#444", marker="s", linestyle="none",
                       markersize=11, markeredgecolor="black",
                       markeredgewidth=0.6, label="PoliEval (integer 0–3)"),
        ],
        loc="upper left", fontsize=10, framealpha=0.95, title="Method",
    )
    ax.add_artist(method_legend)

    # Variant-color legend (outside, to the right).
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=9,
              framealpha=0.95, borderaxespad=0, title="Variant (color)")

    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
