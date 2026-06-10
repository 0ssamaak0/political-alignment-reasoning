"""Combined cross-family layer-sweep plot.

Reads all 4 sweep JSONs (mistral|llama × left|right) and produces a single
2x2 figure plus a flat 1x2 (one panel per direction) for direct family
comparison. Highlights both `best_layer` (raw argmax, often L32 due to the
extraction artifact) and `best_layer_excl_last_2` (the "real" peak).

Usage:
  python -m 4_steering.src.plot_layer_sweep_combined
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "4_steering" / "results" / "layer_sweep"

FAMILIES = ("mistral", "llama")
DIRECTIONS = ("left", "right")
COLOR = {"left": "tab:blue", "right": "tab:red"}
LINESTYLE = {"mistral": "-", "llama": "--"}
MARKER = {"mistral": "o", "llama": "s"}


def _load(family: str, direction: str) -> dict | None:
    p = BASE / family / f"sweep_{direction}.json"
    if not p.exists():
        print(f"[plot_combined] missing {p}")
        return None
    return json.loads(p.read_text())


def _means_by_layer(data: dict) -> tuple[list[int], list[float]]:
    items = sorted(((int(L), cell["mean"]) for L, cell in data["per_layer"].items()),
                   key=lambda t: t[0])
    return [L for L, _ in items], [m for _, m in items]


def plot_2x2():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True, sharex=True)
    for i, family in enumerate(FAMILIES):
        for j, direction in enumerate(DIRECTIONS):
            ax = axes[i, j]
            data = _load(family, direction)
            if data is None:
                ax.set_title(f"{family} {direction} — missing")
                continue
            Ls, means = _means_by_layer(data)
            ax.plot(Ls, means, marker=MARKER[family], color=COLOR[direction], lw=1.5,
                    markersize=6, alpha=0.9)
            ax.axvline(data["best_layer"], color="gray", linestyle="--", alpha=0.5,
                       label=f"best L (raw)={data['best_layer']}")
            ax.axvline(data["best_layer_excl_last_2"], color=COLOR[direction], linestyle=":",
                       alpha=0.9,
                       label=f"best L (excl last 2)={data['best_layer_excl_last_2']}")
            ax.set_title(f"{family.title()} — {direction}-leaning (α={data['alpha']})",
                         fontsize=12)
            ax.set_xlabel("Layer (1-indexed post-block)")
            ax.set_ylabel("Mean trait-expression score (0..100)")
            ax.set_ylim(0, 105)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9, loc="best")
    fig.suptitle(
        "Stage A layer-effectiveness sweep — cross-family\n"
        "Same recipe (response_avg_diff, α=5, 20 latter-half trait questions, Gemini judge)",
        fontsize=13,
    )
    fig.tight_layout()
    out = BASE / "figures" / "layer_curve_2x2_xfam.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_overlay():
    """One panel per direction; both families overlaid for direct comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for j, direction in enumerate(DIRECTIONS):
        ax = axes[j]
        for family in FAMILIES:
            data = _load(family, direction)
            if data is None:
                continue
            Ls, means = _means_by_layer(data)
            ax.plot(Ls, means,
                    marker=MARKER[family], color=COLOR[direction],
                    linestyle=LINESTYLE[family], lw=1.8, markersize=6, alpha=0.85,
                    label=f"{family.title()} (raw L={data['best_layer']}, "
                          f"excl-2 L={data['best_layer_excl_last_2']})")
        ax.set_title(f"{direction.capitalize()}-leaning steering, α=5", fontsize=12)
        ax.set_xlabel("Layer (1-indexed post-block)")
        ax.set_ylabel("Mean trait-expression score (0..100)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(
        "Layer-effectiveness — Mistral (solid/circle) vs Llama (dashed/square)",
        fontsize=13,
    )
    fig.tight_layout()
    out = BASE / "figures" / "layer_curve_overlay_xfam.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_2x2()
    plot_overlay()
