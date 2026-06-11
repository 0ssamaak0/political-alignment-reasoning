"""Political Compass scatter — steering cells only.

Renders all `mistral-pvsteer-*` cells (single-layer + multi-layer) on the
standard Political Compass canvas, with `mistral-base` and the
`mistral-politune-hf-*` cells as reference points. ml cells use star markers
and saturated colors to make them visually pop against the single-layer and
DPO references.

Outputs `4_steering/results/figures/political_compass.png`.

Run from anywhere:
    python3 -m 4_steering.src.plot_compass
or, from inside `4_steering/`:
    python3 -m src.plot_compass
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

POLILEAN_SUMMARY = _REPO / "political_compass" / "PoliLean" / "results" / "summary.json"
OUT_PNG = _REPO / "4_steering" / "results" / "figures" / "political_compass.png"

# Cells to plot, ordered for legend grouping (references first, then steering).
REFS = ["mistral-base", "mistral-politune-hf-left", "mistral-politune-hf-right"]
SL = [
    "mistral-pvsteer-left-a3", "mistral-pvsteer-left-a5",
    "mistral-pvsteer-right-a3", "mistral-pvsteer-right-a5",
]
ML = [
    "mistral-pvsteer-ml-left-a2", "mistral-pvsteer-ml-left-a3",
    "mistral-pvsteer-ml-right-a2", "mistral-pvsteer-ml-right-a3",
]

STYLE = {
    "mistral-base":              dict(color="#444444", marker="o", size=140, alpha=1.0, edge=1.2, label="base"),
    "mistral-politune-hf-left":  dict(color="#1f77b4", marker="^", size=140, alpha=1.0, edge=1.2, label="politune-hf-left (DPO)"),
    "mistral-politune-hf-right": dict(color="#d62728", marker="^", size=140, alpha=1.0, edge=1.2, label="politune-hf-right (DPO)"),

    "mistral-pvsteer-left-a3":   dict(color="#7eb6e6", marker="o", size=80, alpha=0.85, edge=0.7, label="pvsteer-left α=3 (single L17)"),
    "mistral-pvsteer-left-a5":   dict(color="#4a90d9", marker="o", size=80, alpha=0.85, edge=0.7, label="pvsteer-left α=5 (single L17)"),
    "mistral-pvsteer-right-a3":  dict(color="#f0a3a3", marker="o", size=80, alpha=0.85, edge=0.7, label="pvsteer-right α=3 (single L9)"),
    "mistral-pvsteer-right-a5":  dict(color="#dd5757", marker="o", size=80, alpha=0.85, edge=0.7, label="pvsteer-right α=5 (single L9)"),

    "mistral-pvsteer-ml-left-a2":  dict(color="#0b3d91", marker="*", size=380, alpha=1.0, edge=1.4, label="pvsteer-ml-left α=2 (multi 1–31)"),
    "mistral-pvsteer-ml-left-a3":  dict(color="#1f5fbf", marker="*", size=300, alpha=1.0, edge=1.4, label="pvsteer-ml-left α=3 (multi 1–31)"),
    "mistral-pvsteer-ml-right-a2": dict(color="#8b1a1a", marker="*", size=380, alpha=1.0, edge=1.4, label="pvsteer-ml-right α=2 (multi 1–31)"),
    "mistral-pvsteer-ml-right-a3": dict(color="#c81818", marker="*", size=300, alpha=1.0, edge=1.4, label="pvsteer-ml-right α=3 (multi 1–31)"),
}


def draw_quadrants(ax):
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.axhspan(0, 10, xmin=0.5, xmax=1.0, color="#fde0dc", alpha=0.30, zorder=0)
    ax.axhspan(0, 10, xmin=0.0, xmax=0.5, color="#dde7f5", alpha=0.30, zorder=0)
    ax.axhspan(-10, 0, xmin=0.5, xmax=1.0, color="#fff4cc", alpha=0.30, zorder=0)
    ax.axhspan(-10, 0, xmin=0.0, xmax=0.5, color="#daf0d6", alpha=0.30, zorder=0)


def main():
    if not POLILEAN_SUMMARY.exists():
        raise FileNotFoundError(f"PoliLean summary not found: {POLILEAN_SUMMARY}")
    summary = json.loads(POLILEAN_SUMMARY.read_text())

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    draw_quadrants(ax)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.set_xlabel("Economic   (Left  ←→  Right)", fontsize=12)
    ax.set_ylabel("Social   (Libertarian  ←→  Authoritarian)", fontsize=12)
    ax.set_title(
        "Political Compass — Mistral steering cells (PoliLean, 5 seeds/cell)\n"
        "★ = multi-layer ml hook  ·  ● = single-layer pvsteer  ·  ▲ = PoliTune-DPO  ·  ○ = base",
        fontsize=13,
    )
    ax.text(-9.5, 9.5, "Authoritarian Left", fontsize=9, alpha=0.65, va="top")
    ax.text(9.5, 9.5,  "Authoritarian Right", fontsize=9, alpha=0.65, ha="right", va="top")
    ax.text(-9.5, -9.5, "Libertarian Left", fontsize=9, alpha=0.65, va="bottom")
    ax.text(9.5, -9.5,  "Libertarian Right", fontsize=9, alpha=0.65, ha="right", va="bottom")
    ax.grid(True, alpha=0.25, zorder=0)

    legend_handles = []
    for tag in REFS + SL + ML:
        if tag not in summary:
            print(f"[plot_compass] WARN: {tag} not in PoliLean summary; skipping")
            continue
        rec = summary[tag]
        style = STYLE[tag]
        ec_mean, soc_mean = rec["ec_mean"], rec["soc_mean"]
        ec_std, soc_std = rec["ec_std"], rec["soc_std"]

        # 2σ ellipse (lighter for refs/sl, slightly more visible for ml)
        ell_alpha = 0.10 if tag.startswith("mistral-base") or "politune-hf" in tag else 0.13
        if "pvsteer-ml" in tag:
            ell_alpha = 0.20
        ax.add_patch(Ellipse(
            (ec_mean, soc_mean),
            width=max(2 * ec_std, 0.05),
            height=max(2 * soc_std, 0.05),
            facecolor=style["color"], edgecolor=style["color"], linewidth=0.8,
            alpha=ell_alpha, zorder=2,
        ))

        # individual seeds as faint dots
        for ec, soc in zip(rec["ec_runs"], rec["soc_runs"]):
            ax.scatter(ec, soc, s=14, color=style["color"], alpha=0.35,
                       edgecolor="none", zorder=3)

        # mean marker
        ax.scatter(
            ec_mean, soc_mean,
            s=style["size"], color=style["color"], marker=style["marker"],
            alpha=style["alpha"],
            edgecolor="black", linewidth=style["edge"], zorder=6,
        )

        legend_handles.append(plt.Line2D(
            [], [], color=style["color"], marker=style["marker"],
            linestyle="none",
            markersize=12 if "ml" in tag else 9,
            markeredgecolor="black", markeredgewidth=0.6,
            label=f"{style['label']}  ({ec_mean:+.2f}±{ec_std:.2f}, {soc_mean:+.2f}±{soc_std:.2f})",
        ))

    # Arrow base → ml-left-a2 and base → ml-right-a3, to highlight the
    # cells that match/exceed DPO. Subtle.
    if "mistral-base" in summary:
        b = summary["mistral-base"]
        for target in ("mistral-pvsteer-ml-left-a2", "mistral-pvsteer-ml-right-a3"):
            if target not in summary:
                continue
            t = summary[target]
            color = STYLE[target]["color"]
            ax.annotate(
                "",
                xy=(t["ec_mean"], t["soc_mean"]),
                xytext=(b["ec_mean"], b["soc_mean"]),
                arrowprops=dict(
                    arrowstyle="->", color=color, lw=1.3,
                    alpha=0.55, shrinkA=10, shrinkB=14,
                ),
                zorder=4,
            )

    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=9, framealpha=0.95,
              borderaxespad=0, title="cell (ec, soc)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
