"""Generate the three RQ3 Llama figures in the polireason house style.

  rq3_1_dose_response_llama   reasoning dose-response (BBHmean + collapse% vs knob)
  rq3_2_coherence_gap_llama   coherence gate vs reasoning collapse across DPO scale
  rq3_3_contamination_llama   contamination of the neutral task vs knob (the arm flip)

All numbers are the verified values in RQ3/RESULTS_llama.md (computed by
consolidate.py llama / deeper tables / judge_dose_response_llama.txt). Run from the
polireason repo root:  python3 RQ3/make_figures_llama.py
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

apply_theme()

NAN = float("nan")
OUT = "RQ3/figures"

# ---------------------------------------------------------------- data (verified)
# Steering coefficient grid (alpha) and DPO LoRA-scale grid (s).
A = [0, 0.5, 1, 2, 3, 4]
S = [0, 0.25, 0.5, 1.0, 1.5, 2.0]

# Reasoning dose-response: BBHmean (reparsed accuracy) and collapse% per cell.
steerL_bbh = [88, 89, 86, 78, 64, 41]
steerL_col = [3, 2, 0, 0, 0, 22]
steerR_bbh = [88, 87, 80, 67, 62, 55]
steerR_col = [3, 5, 11, 23, 22, 10]
dpoL_bbh = [88, 88, 87, 89, 85, 79]
dpoL_col = [3, 4, 3, 3, 4, 5]
dpoR_bbh = [88, 88, 88, 80, 31, 0]
dpoR_col = [3, 1, 1, 0, 48, 99]

# Trait-sweep coherence (gate). DPO s0.25 has no Llama coherence file -> NaN.
dpoL_coh = [71, NAN, 85, 94, 69, 62]
dpoR_coh = [71, NAN, 73, 79, 26, 13]

# Contamination of the neutral task (% items flagged), judge over 21,000 rows.
steerL_con = [0, 0, 0, 0, 2, 63]
steerR_con = [0, 0, 0, 0, 0, 1]
dpoL_con = [0, 0, 0, 0, 0, 11]
dpoR_con = [0, 0, 0, 0, 47, 98]

ML, MR = "o", "s"  # markers: left=circle, right=square


def _style_panel(ax, xlabel, ylabel, title):
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-4, 104)


# =============================================================== rq3_1 dose-response
fig, (axS, axD) = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)

for ax, x, Lb, Lc, Rb, Rc, xlab, title in [
    (axS, A, steerL_bbh, steerL_col, steerR_bbh, steerR_col,
     "steering coefficient  α", "Steering"),
    (axD, S, dpoL_bbh, dpoL_col, dpoR_bbh, dpoR_col,
     "DPO LoRA scale  s", "DPO"),
]:
    ax.plot(x, Lb, color=LEFT, marker=ML, lw=1.8, label="left  accuracy")
    ax.plot(x, Rb, color=RIGHT, marker=MR, lw=1.8, label="right  accuracy")
    ax.plot(x, Lc, color=LEFT, marker=ML, ls="--", lw=1.3, ms=4, label="left  collapse%")
    ax.plot(x, Rc, color=RIGHT, marker=MR, ls="--", lw=1.3, ms=4, label="right  collapse%")
    ax.axhline(88, color=BASE, ls=":", lw=1.0)
    _style_panel(ax, xlab, "% of items", title)

axS.set_ylabel("% of items  (BBHmean accuracy / collapse)")
axS.text(0, 90.5, "base accuracy", color=MUTED, fontsize=8)
axS.legend(loc="center left", fontsize=8)
fig.suptitle("RQ3 Llama — reasoning dose-response (solid = accuracy, dashed = collapse)",
             x=0.012, ha="left", fontsize=12.5, fontweight="semibold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
print("wrote:", save_fig(fig, f"{OUT}/rq3_1_dose_response_llama"))
plt.close(fig)


# =============================================================== rq3_2 coherence gap
def _drop_nan(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if y == y]  # y == y is False for NaN
    return [p[0] for p in pairs], [p[1] for p in pairs]


fig, ax = plt.subplots(figsize=(5.6, 3.9))
_xL, _yL = _drop_nan(S, dpoL_coh)   # DPO s0.25 has no Llama coherence point
_xR, _yR = _drop_nan(S, dpoR_coh)
ax.plot(_xL, _yL, color=LEFT, marker=ML, lw=1.8, label="left  coherence")
ax.plot(_xR, _yR, color=RIGHT, marker=MR, lw=1.8, label="right  coherence")
ax.plot(S, dpoL_col, color=LEFT, marker=ML, ls="--", lw=1.3, ms=4, label="left  collapse%")
ax.plot(S, dpoR_col, color=RIGHT, marker=MR, ls="--", lw=1.3, ms=4, label="right  collapse%")
_style_panel(ax, "DPO LoRA scale  s", "% (coherence gate / collapse)",
             "Coherence gate tracks the collapse on Llama (DPO)")
ax.annotate("right: coherence falls AND collapse rises together\n(the gate does not miss it)",
            xy=(1.5, 37), xytext=(0.35, 62), color=MUTED, fontsize=8,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
ax.legend(loc="center left", fontsize=8)
fig.tight_layout()
print("wrote:", save_fig(fig, f"{OUT}/rq3_2_coherence_gap_llama"))
plt.close(fig)


# =============================================================== rq3_3 contamination
fig, (axS, axD) = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)
axS.plot(A, steerL_con, color=LEFT, marker=ML, lw=1.8, label="left")
axS.plot(A, steerR_con, color=RIGHT, marker=MR, lw=1.8, label="right")
_style_panel(axS, "steering coefficient  α", "contaminated %", "Steering")
axS.legend(loc="upper left", fontsize=8)

axD.plot(S, dpoL_con, color=LEFT, marker=ML, lw=1.8, label="left")
axD.plot(S, dpoR_con, color=RIGHT, marker=MR, lw=1.8, label="right")
_style_panel(axD, "DPO LoRA scale  s", "contaminated %", "DPO")
axD.legend(loc="upper left", fontsize=8)

axS.set_ylabel("contaminated % of the neutral task")
fig.suptitle("RQ3 Llama — contamination follows the collapsing arm (left on steering, right on DPO)",
             x=0.012, ha="left", fontsize=12.0, fontweight="semibold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
print("wrote:", save_fig(fig, f"{OUT}/rq3_3_contamination_llama"))
plt.close(fig)
