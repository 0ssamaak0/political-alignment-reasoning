"""Plot the LoRA-scale sweep MMLU formal_logic accuracy, steering-style.

Signed x-axis: left adapter -> negative scale, right adapter -> positive scale,
base (no adapter) at scale=0 — mirroring the pvsteer-ml signed-alpha sweep panel
(classic_evals/results/mistral_sweep/alpha_sweep_4panel.png).

Marks:
  * black diamond + dashed line at the BASE value (scale=0).
  * gold stars + vertical guides at scale=+/-2.0 (the as-trained default,
    lora_alpha=16 / r=8 = 2.0 — the cell actually shipped as politune-hf).

Usage:
    conda run -n main python -m classic_evals.plot_lora_sweep
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SWEEP_DIR = HERE / "results" / "lora_sweep"
BASE_SUMMARY = HERE / "results" / "mistral_sweep" / "mistral-base" / "summary.json"
OUT_PNG = SWEEP_DIR / "lora_sweep_mmlu.png"

SCALES = [1.0, 1.5, 2.0, 2.5, 3.0]
METRIC = "mmlu_formal_logic"


def load_acc(tag: str) -> float:
    d = json.loads((SWEEP_DIR / tag / "summary.json").read_text())
    return float(d[METRIC]) * 100.0


def main() -> None:
    base_acc = json.loads(BASE_SUMMARY.read_text())[METRIC] * 100.0

    left = {s: load_acc(f"mistral-politune-hf-left-lora{str(s).replace('.', '_')}") for s in SCALES}
    right = {s: load_acc(f"mistral-politune-hf-right-lora{str(s).replace('.', '_')}") for s in SCALES}

    # Signed-scale series: left arm (negative) -> base (0) -> right arm (positive).
    xs = [-s for s in reversed(SCALES)] + [0.0] + [s for s in SCALES]
    ys = [left[s] for s in reversed(SCALES)] + [base_acc] + [right[s] for s in SCALES]

    fig, ax = plt.subplots(figsize=(8, 5))

    # main curve
    ax.plot(xs, ys, "-o", color="#1f77b4", lw=1.8, ms=6, zorder=3,
            label=f"{METRIC} accuracy")

    # base diamond + reference line
    ax.plot(0, base_acc, "D", color="black", ms=11, zorder=5,
            label=f"base (scale=0) = {base_acc:.1f}%")
    ax.axhline(base_acc, color="black", ls=":", lw=1.0, alpha=0.6, zorder=1)

    # as-trained scale=2.0 markers (the shipped politune-hf cell)
    for sx in (-2.0, 2.0):
        ax.axvline(sx, color="#d62728", ls="--", lw=1.0, alpha=0.45, zorder=1)
    ax.plot([-2.0, 2.0], [left[2.0], right[2.0]], "*", color="#ff9900",
            ms=18, mec="black", mew=0.6, zorder=6,
            label="as-trained (scale=2.0, the shipped cell)")

    # 4-way random chance
    ax.axhline(25.0, color="grey", ls="--", lw=1.0, alpha=0.7, zorder=1,
               label="4-way random (25%)")

    # lean banners
    ax.text(0.02, 0.96, "LEFT", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color="#1f77b4", va="top", ha="left")
    ax.text(0.98, 0.96, "RIGHT", transform=ax.transAxes, fontsize=13,
            fontweight="bold", color="#d62728", va="top", ha="right")

    ax.set_xlabel("Signed LoRA scale  (negative = LEFT adapter, positive = RIGHT adapter)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Mistral-7B-Instruct-v0.2 — LoRA-scale sweep on classic_evals\n"
                 "MMLU formal_logic (multiple_choice loglik) · 10 cells · "
                 "lm-eval 0.4.9 · limit=150")
    ax.set_xticks([-3, -2.5, -2, -1.5, -1, 0, 1, 1.5, 2, 2.5, 3])
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}")
    print(f"base={base_acc:.1f}%  left2.0={left[2.0]:.1f}%  right2.0={right[2.0]:.1f}%")


if __name__ == "__main__":
    main()
