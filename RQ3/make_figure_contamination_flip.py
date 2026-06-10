"""Thesis figure `ch4_contamination_flip` (PDF + SVG), annotated-bar design.

Eight horizontal bars, one per arm (2 models x 2 methods x 2 sides), bar
length = contamination share at MAXIMUM strength (steering alpha=4, DPO
s=2.0), bar color = the arm's political direction (house blue/red). The
no-answer collapse context appears only inside three small grey annotations
that carry the qualifications the prose leans on:

  - Mistral steering: both arms collapse (92 / 99 no-answer), so both
    contaminate.
  - Mistral DPO right arm: collapses with no political text (71 no-answer,
    0 contamination).
  - Llama steering left arm: floods while mostly still answering
    (63 contamination, only 22 no-answer).

This replaces two earlier diverging-bar designs that were rejected for
carrying too many encodings. The onset values and the exact (no-answer,
contamination) pairs live in the prose and the table above the figure
(tab:rq3-contamination-flip-table), not here.

Run from anywhere:

    python3 RQ3/make_figure_contamination_flip.py

Data: contamination at maximum strength from RQ3/RESULTS.md s5 and
RESULTS_llama.md s6 (judge tables), identical to the thesis table.
"""
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

apply_theme()

OUT = "/Users/0ssamaak0/Documents/Writing/tempelate/images"

# ---- data: contamination (%) at maximum strength (alpha=4 / s=2.0) -------
# (row label, contamination, color, y position)  -- y grows downward
ROWS = [
    ("Steering (left)",  92, LEFT,  0.0),   # Mistral
    ("Steering (right)", 58, RIGHT, 1.0),
    ("DPO (left)",      100, LEFT,  2.4),
    ("DPO (right)",       0, RIGHT, 3.4),
    ("Steering (left)",  63, LEFT,  5.9),   # Llama
    ("Steering (right)",  1, RIGHT, 6.9),
    ("DPO (left)",       11, LEFT,  8.3),
    ("DPO (right)",      98, RIGHT, 9.3),
]
HEADERS = [("Mistral-7B", -1.05), ("Llama-3-8B", 4.85)]
BAR_H = 0.65
LABEL_X = -0.27          # axes-fraction x where row labels and headers start

fig, ax = plt.subplots(figsize=(6.4, 3.4))

for label, val, color, y in ROWS:
    ax.barh(y, val, height=BAR_H, color=color)
    # value just past the bar end; zeros get an explicit number at the origin
    ax.text(val + 1.8, y, f"{val}", color=BASE, fontsize=8, va="center", ha="left")

# ---- row labels and section headers (left gutter, all left-aligned) ------
ytrans = ax.get_yaxis_transform()   # x in axes fraction, y in data
for label, _, _, y in ROWS:
    ax.text(LABEL_X, y, label, transform=ytrans, fontsize=9,
            color="#1A1A1A", va="center", ha="left")
for name, y in HEADERS:
    ax.text(LABEL_X, y, name, transform=ytrans, fontsize=10,
            color=BASE, va="center", ha="left", fontweight="bold")

# ---- three annotations with thin grey leader lines ------------------------
ANN_FS = 8
ANN_LW = 0.8

# 1) bracket over the two Mistral steering rows: both arms collapse
bx = 104.5
ax.plot([bx - 2.2, bx, bx, bx - 2.2], [-0.28, -0.28, 1.28, 1.28],
        color=MUTED, lw=ANN_LW, solid_capstyle="butt", clip_on=False)
ax.text(bx + 2.8, 0.5, "both sides collapse,\nno-answer 92 and 99 percent",
        color=MUTED, fontsize=ANN_FS, va="center", ha="left")

# 2) Mistral DPO right arm: zero contamination, silent collapse
ax.plot([6.0, 11.5], [3.4, 3.4], color=MUTED, lw=ANN_LW, solid_capstyle="butt")
ax.text(13.5, 3.4, "collapses with no political text, no-answer 71 percent",
        color=MUTED, fontsize=ANN_FS, va="center", ha="left")

# 3) Llama steering left arm: floods while mostly still answering
ax.plot([71.5, 77.0], [5.9, 5.9], color=MUTED, lw=ANN_LW, solid_capstyle="butt")
ax.text(79.0, 5.9, "mostly still answers, no-answer 22 percent",
        color=MUTED, fontsize=ANN_FS, va="center", ha="left")

# ---- axes -----------------------------------------------------------------
ax.set_xlim(0, 158)
ax.set_ylim(10.05, -1.8)            # inverted: first row on top
ax.set_xticks([0, 50, 100])
ax.set_yticks([])
ax.grid(False)
ax.spines["bottom"].set_bounds(0, 100)
ax.spines["left"].set_bounds(-0.4, 9.7)
ax.set_xlabel("Contamination at maximum strength, percent of items")
ax.xaxis.set_label_coords(50 / 150, -0.13)

print("wrote:", save_fig(fig, f"{OUT}/ch4_contamination_flip"))
plt.close(fig)
