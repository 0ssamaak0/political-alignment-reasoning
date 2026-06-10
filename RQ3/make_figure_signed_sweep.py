"""Signed-axis variant of the RQ3 reasoning dose-response figure.

Produces the thesis figure `ch4_accuracy_collapse_strength_sweep` (PDF + SVG)
in the polireason house style. Unlike rq3_1_dose_response in make_figures_rq3.py
(which draws the left and right arms as two series over the same positive grid),
this version puts BOTH arms on a single signed strength axis:

    negative alpha / s  ->  LEFT arm     positive alpha / s  ->  RIGHT arm

The sign is a visualization convention only. There is no negative steering
coefficient or negative LoRA scale; the magnitude |x| is the real strength and
the sign just says which side (left or right) was induced. The two arms share
the base model at x = 0 (strength 0), so each panel is a single trajectory that
runs from the left extreme through the base to the right extreme.

Run from anywhere:

    python3 RQ3/make_figure_signed_sweep.py

Data: copied verbatim from RQ3/make_figures_rq3.py (the rq3_1 dose-response
block), which is itself verified against RQ3/RESULTS.md and RESULTS_llama.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Walk up to the brand helper so the import works from any cwd.
for _d in Path(__file__).resolve().parents:
    _s = _d / ".claude/skills/visualizations-designer/scripts"
    if _s.is_dir():
        sys.path.insert(0, str(_s))
        break
from polireason_viz import apply_theme, LEFT, RIGHT, BASE, MUTED, save_fig  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

apply_theme()

# Both-model figure lands in the thesis images dir, thesis filename.
OUT = "/Users/0ssamaak0/Documents/Writing/tempelate/images"

# ---------------------------------------------------------------- data (verified)
# Strength grids. A = steering coefficient alpha, S = DPO LoRA scale s.
A = [0, 0.5, 1, 2, 3, 4]
S = [0, 0.25, 0.5, 1.0, 1.5, 2.0]

# Per-row base accuracy anchor (dotted grey line). Different per model row.
BASE_MISTRAL = 74
BASE_LLAMA = 88

# ===== LLAMA arrays (verbatim from make_figures_rq3.py) =====
steerL_bbh = [88, 89, 86, 78, 64, 41]
steerL_col = [3, 2, 0, 0, 0, 22]
steerR_bbh = [88, 87, 80, 67, 62, 55]
steerR_col = [3, 5, 11, 23, 22, 10]
dpoL_bbh = [88, 88, 87, 89, 85, 79]
dpoL_col = [3, 4, 3, 3, 4, 5]
dpoR_bbh = [88, 88, 88, 80, 31, 0]
dpoR_col = [3, 1, 1, 0, 48, 99]

# ===== MISTRAL arrays (verbatim from make_figures_rq3.py) =====
m_steerL_bbh = [74, 77, 78, 71, 52, 3];   m_steerL_col = [5, 3, 2, 0, 7, 92]
m_steerR_bbh = [74, 75, 72, 69, 45, 0];   m_steerR_col = [5, 6, 9, 11, 34, 99]
m_dpoL_bbh   = [74, 74, 72, 65, 22, 0];   m_dpoL_col   = [5, 6, 8, 14, 70, 100]
m_dpoR_bbh   = [74, 75, 75, 72, 24, 18];  m_dpoR_col   = [5, 4, 4, 4, 68, 71]

ML, MR = "o", "s"  # markers: left = circle, right = square


def _signed(grid):
    """Negate the grid for the left arm; the base point (0) stays at 0."""
    return [-v for v in grid]


# x-axis label is two lines: the quantity, then the sign convention, restated
# under every panel so the figure reads on its own.
XLAB_STEER = "steering coefficient  α\nnegative = left, positive = right"
XLAB_DPO = "LoRA adapter scale  s\nnegative = left, positive = right"


def _panel(ax, grid, Lb, Lc, Rb, Rc, xlabel, title, base):
    xneg = _signed(grid)   # left arm at negative x
    xpos = grid            # right arm at positive x

    # reference lines: base accuracy (horizontal) and the x = 0 base centre.
    ax.axhline(base, color=BASE, ls=":", lw=1.0, zorder=1)
    ax.axvline(0, color=MUTED, ls=":", lw=0.8, zorder=1)

    # LEFT arm (blue) on the negative axis, RIGHT arm (red) on the positive axis.
    # color encodes the side, line style encodes the quantity (solid = accuracy,
    # dashed = collapse). The shared legend at the bottom of the figure spells
    # both keys out, so the per-line labels here are intentionally omitted.
    ax.plot(xneg, Lb, color=LEFT, marker=ML, lw=1.8, zorder=3)
    ax.plot(xpos, Rb, color=RIGHT, marker=MR, lw=1.8, zorder=3)
    ax.plot(xneg, Lc, color=LEFT, marker=ML, ls="--", lw=1.3, ms=4, zorder=2)
    ax.plot(xpos, Rc, color=RIGHT, marker=MR, ls="--", lw=1.3, ms=4, zorder=2)

    # the shared base point at x = 0 (strength 0 = the unaligned base model).
    ax.scatter([0], [base], color=BASE, s=28, zorder=4)

    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylim(-4, 104)
    lim = max(grid) * 1.08
    ax.set_xlim(-lim, lim)


# 2x2: rows = model (top Mistral, bottom Llama), cols = method (steering, DPO).
fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.6), sharey=True)

panels = [
    (axes[0, 0], A, m_steerL_bbh, m_steerL_col, m_steerR_bbh, m_steerR_col,
     XLAB_STEER, "Mistral · Steering", BASE_MISTRAL),
    (axes[0, 1], S, m_dpoL_bbh, m_dpoL_col, m_dpoR_bbh, m_dpoR_col,
     XLAB_DPO, "Mistral · DPO", BASE_MISTRAL),
    (axes[1, 0], A, steerL_bbh, steerL_col, steerR_bbh, steerR_col,
     XLAB_STEER, "Llama · Steering", BASE_LLAMA),
    (axes[1, 1], S, dpoL_bbh, dpoL_col, dpoR_bbh, dpoR_col,
     XLAB_DPO, "Llama · DPO", BASE_LLAMA),
]
for ax, grid, Lb, Lc, Rb, Rc, xlab, title, base in panels:
    _panel(ax, grid, Lb, Lc, Rb, Rc, xlab, title, base)

axes[0, 0].set_ylabel("Percent of reasoning items")
axes[1, 0].set_ylabel("Percent of reasoning items")

# single shared legend along the bottom. Two decoupled keys:
#   color  -> which side was induced (blue left, red right)
#   style  -> what the line measures (solid accuracy, dashed collapse, dotted base)
legend_handles = [
    Line2D([0], [0], color=LEFT, lw=1.8, marker=ML, label="left-aligned"),
    Line2D([0], [0], color=RIGHT, lw=1.8, marker=MR, label="right-aligned"),
    Line2D([0], [0], color=MUTED, lw=1.8, ls="-", label="accuracy"),
    Line2D([0], [0], color=MUTED, lw=1.3, ls="--", label="collapse rate"),
    Line2D([0], [0], color=BASE, lw=1.0, ls=":", label="base accuracy"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=5,
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.012),
           columnspacing=1.6, handlelength=1.8)

fig.suptitle("Reasoning response to alignment strength",
             x=0.012, ha="left", fontsize=12.5, fontweight="semibold")
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
print("wrote:", save_fig(fig, f"{OUT}/ch4_accuracy_collapse_strength_sweep"))
plt.close(fig)
