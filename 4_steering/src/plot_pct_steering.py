"""PCT economic-axis plot across Campaigns 1-3 (pvsteer single-layer + multi-layer).

Drops α=7 cells (collapsed per coherence judge). Reference lines for mistral-base,
politune-hf, and roleplay anchor the visual scale.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SUMMARY = ROOT / "political_compass" / "PoliLean" / "results" / "summary.json"

S = json.loads(SUMMARY.read_text())


def ec(tag: str):
    v = S[tag]
    return v["ec_mean"], v["ec_std"]


CELLS_LEFT = [
    ("sl", 3, "mistral-pvsteer-left-a3"),
    ("sl", 5, "mistral-pvsteer-left-a5"),
    ("ml", 2, "mistral-pvsteer-ml-left-a2"),
    ("ml", 3, "mistral-pvsteer-ml-left-a3"),
]
CELLS_RIGHT = [
    ("sl", 3, "mistral-pvsteer-right-a3"),
    ("sl", 5, "mistral-pvsteer-right-a5"),
    ("ml", 2, "mistral-pvsteer-ml-right-a2"),
    ("ml", 3, "mistral-pvsteer-ml-right-a3"),
]

REF_LINES = {
    "base":           ("mistral-base",              "grey",   "-",  "base"),
    "politune-left":  ("mistral-politune-hf-left",  "navy",   "--", "PoliTune-DPO left"),
    "politune-right": ("mistral-politune-hf-right", "darkred","--", "PoliTune-DPO right"),
    "roleplay-left":  ("mistral-roleplay-left",     "navy",   ":",  "Roleplay left"),
    "roleplay-right": ("mistral-roleplay-right",    "darkred",":",  "Roleplay right"),
}


def panel(ax, cells, dir_label, color):
    for mode, alpha, tag in cells:
        mean, std = ec(tag)
        marker = "o" if mode == "sl" else "*"
        size = 11 if mode == "sl" else 17
        ax.errorbar(alpha, mean, yerr=std, fmt=marker, color=color,
                    markersize=size, capsize=4, capthick=1.2,
                    markeredgecolor="black", markeredgewidth=0.6,
                    label=f"{'single-layer' if mode == 'sl' else 'multi-layer'} α={alpha}")

    for key, (tag, c, ls, label) in REF_LINES.items():
        if dir_label == "left" and "right" in key:
            continue
        if dir_label == "right" and "left" in key:
            continue
        if key == "base":
            ax.axhline(ec(tag)[0], color=c, ls=ls, lw=1.0, alpha=0.7, label=label)
        else:
            ax.axhline(ec(tag)[0], color=c, ls=ls, lw=1.0, alpha=0.7, label=label)

    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.set_xlabel("α (steering coefficient)")
    ax.set_ylabel("PCT economic axis (ec)  [← left   right →]")
    ax.set_title(f"{dir_label.capitalize()}-direction steering")
    ax.set_xticks([2, 3, 5])
    ax.set_xlim(1.5, 5.7)
    ax.grid(alpha=0.25)
    by_label = dict(zip(*reversed(ax.get_legend_handles_labels())))
    ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=8, framealpha=0.9)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
panel(axes[0], CELLS_LEFT,  "left",  "tab:blue")
panel(axes[1], CELLS_RIGHT, "right", "tab:red")

# Common y limits so both panels share scale
ymin = min(ax.get_ylim()[0] for ax in axes)
ymax = max(ax.get_ylim()[1] for ax in axes)
for ax in axes:
    ax.set_ylim(ymin - 0.5, ymax + 0.5)

fig.suptitle("Political Compass (ec) across pvsteer campaigns — Mistral-7B-Instruct-v0.2\n"
             "(α=7 collapsed cells excluded; n=5 PoliLean runs per cell, errorbar = ±1σ)",
             fontsize=11)
fig.tight_layout()

OUT = Path(__file__).parent / "pct_steering.png"
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"saved: {OUT}")
