"""Generate the three BOTH-MODEL RQ3 thesis figures in the polireason house style.

  rq3_1_dose_response   reasoning dose-response (BBHmean accuracy + collapse% vs knob)
  rq3_2_coherence_gap   coherence gate vs reasoning collapse across DPO scale
  rq3_3_contamination   contamination of the neutral task vs knob (the arm flip)

Each figure stitches the Mistral row (top) and the Llama row (bottom) into one
figure. Run from the polireason repo root:

    python3 RQ3/make_figures_rq3.py

Data provenance:
  - LLAMA arrays: copied verbatim from RQ3/make_figures_llama.py (verified against
    RQ3/RESULTS_llama.md). Same names, prefixed where a Mistral twin exists.
  - MISTRAL arrays: copied verbatim from the "Mistral arrays for the builder" block
    of /Users/0ssamaak0/Documents/Writing/brainstorming/RQ3.md section 8 (verified
    against RQ3/RESULTS.md).

House style (from RQ3.md section 8 + make_figures_llama.py):
  left = blue (LEFT), marker circle 'o'; right = red (RIGHT), marker square 's';
  base = dotted grey (BASE). Solid = accuracy / coherence; dashed = collapse%.
  y fixed to ylim(-4, 104). Output PDF + SVG (never PNG).
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
# Both-model figures land in the thesis images dir, filenames WITHOUT _llama.
OUT = "/Users/0ssamaak0/Documents/Writing/tempelate/images"

# ---------------------------------------------------------------- data (verified)
# Steering coefficient grid (alpha) and DPO LoRA-scale grid (s). Same for both models.
A = [0, 0.5, 1, 2, 3, 4]
S = [0, 0.25, 0.5, 1.0, 1.5, 2.0]

# Per-row base accuracy anchor (dotted grey line). DIFFERENT per model row.
BASE_MISTRAL = 74
BASE_LLAMA = 88

# ===== LLAMA arrays (copied verbatim from RQ3/make_figures_llama.py) =====
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

# ===== MISTRAL arrays (copied verbatim from RQ3.md section 8 builder block) =====
# --- Mistral dose-response (RESULTS.md §3): BBHmean and collapse% ---
m_steerL_bbh = [74, 77, 78, 71, 52, 3];   m_steerL_col = [5, 3, 2, 0, 7, 92]
m_steerR_bbh = [74, 75, 72, 69, 45, 0];   m_steerR_col = [5, 6, 9, 11, 34, 99]
m_dpoL_bbh   = [74, 74, 72, 65, 22, 0];   m_dpoL_col   = [5, 6, 8, 14, 70, 100]
m_dpoR_bbh   = [74, 75, 75, 72, 24, 18];  m_dpoR_col   = [5, 4, 4, 4, 68, 71]

# --- Mistral coherence (RESULTS.md §2 dial + §3 coh column) ---
# Base coherence = 71 (RESULTS.md §3 canonical dose-response), NOT the §2 dial's 72.
m_dpoL_coh = [71, 74, 78, 89, 89, 78]
m_dpoR_coh = [71, 71, 79, 100, 99, 83]

# --- Mistral contamination (real per-point, from RQ3 results/judge_summary_mistral.json) ---
# contam% on the four neutral BBH tasks, measured at every knob value. Onset is at
# alpha3 / s1.5, one step below the alpha4 / s2 collapse cliff, while the model still
# mostly answers; it floods at the cliff. (s2 contam for DPO: left 100, right 0, from
# the §5 cliff bucket; judge_summary does not store an s2_0 contam field.)
m_steerL_con = [0, 0, 0, 0.9, 23.5, 91.5]
m_steerR_con = [0, 0, 0, 0, 19.3, 57.8]
m_dpoL_con   = [0, 0, 0, 0.4, 41.5, 100]
m_dpoR_con   = [0, 0, 0, 0.1, 0.1, 0]

ML, MR = "o", "s"  # markers: left=circle, right=square


def _style_panel(ax, xlabel, ylabel, title):
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-4, 104)


def _drop_nan(xs, ys):
    """Drop (x, y) pairs where y is NaN (Llama DPO s0.25 coherence is missing)."""
    pairs = [(x, y) for x, y in zip(xs, ys) if y == y]  # y == y is False for NaN
    return [p[0] for p in pairs], [p[1] for p in pairs]


# =============================================================== rq3_1 dose-response
# 2x2: rows = model (top Mistral, bottom Llama), cols = method (left steering, right DPO).
fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.0), sharey=True)

# (ax, x-grid, Lbbh, Lcol, Rbbh, Rcol, xlabel, title, base)
dose_panels = [
    (axes[0, 0], A, m_steerL_bbh, m_steerL_col, m_steerR_bbh, m_steerR_col,
     "steering coefficient  α", "Mistral - Steering", BASE_MISTRAL),
    (axes[0, 1], S, m_dpoL_bbh, m_dpoL_col, m_dpoR_bbh, m_dpoR_col,
     "DPO LoRA scale  s", "Mistral - DPO", BASE_MISTRAL),
    (axes[1, 0], A, steerL_bbh, steerL_col, steerR_bbh, steerR_col,
     "steering coefficient  α", "Llama - Steering", BASE_LLAMA),
    (axes[1, 1], S, dpoL_bbh, dpoL_col, dpoR_bbh, dpoR_col,
     "DPO LoRA scale  s", "Llama - DPO", BASE_LLAMA),
]
for ax, x, Lb, Lc, Rb, Rc, xlab, title, base in dose_panels:
    ax.plot(x, Lb, color=LEFT, marker=ML, lw=1.8, label="left  accuracy")
    ax.plot(x, Rb, color=RIGHT, marker=MR, lw=1.8, label="right  accuracy")
    ax.plot(x, Lc, color=LEFT, marker=ML, ls="--", lw=1.3, ms=4, label="left  collapse%")
    ax.plot(x, Rc, color=RIGHT, marker=MR, ls="--", lw=1.3, ms=4, label="right  collapse%")
    ax.axhline(base, color=BASE, ls=":", lw=1.0)  # per-ROW base line
    _style_panel(ax, xlab, "% of items", title)

axes[0, 0].set_ylabel("% of items  (BBHmean accuracy / collapse)")
axes[1, 0].set_ylabel("% of items  (BBHmean accuracy / collapse)")
axes[0, 0].legend(loc="center left", fontsize=8)
fig.suptitle("RQ3 reasoning dose-response (solid = accuracy, dashed = collapse, dotted = base)",
             x=0.012, ha="left", fontsize=12.5, fontweight="semibold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
print("wrote:", save_fig(fig, f"{OUT}/rq3_1_dose_response"))
plt.close(fig)


# =============================================================== rq3_2 coherence gap
# 1x2 (NOT 2x2): left = Mistral DPO, right = Llama DPO. x = s.
fig, (axM, axL) = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)

# --- Mistral DPO panel ---
_mxL, _myL = _drop_nan(S, m_dpoL_coh)
_mxR, _myR = _drop_nan(S, m_dpoR_coh)
axM.plot(_mxL, _myL, color=LEFT, marker=ML, lw=1.8, label="left  coherence")
axM.plot(_mxR, _myR, color=RIGHT, marker=MR, lw=1.8, label="right  coherence")
axM.plot(S, m_dpoL_col, color=LEFT, marker=ML, ls="--", lw=1.3, ms=4, label="left  collapse%")
axM.plot(S, m_dpoR_col, color=RIGHT, marker=MR, ls="--", lw=1.3, ms=4, label="right  collapse%")
_style_panel(axM, "DPO LoRA scale  s", "% (coherence gate / collapse)",
             "Mistral - the gate UNDER-detects")
axM.legend(loc="center left", fontsize=8)

# --- Llama DPO panel (DPO s0.25 coherence is missing -> _drop_nan) ---
_lxL, _lyL = _drop_nan(S, dpoL_coh)
_lxR, _lyR = _drop_nan(S, dpoR_coh)
axL.plot(_lxL, _lyL, color=LEFT, marker=ML, lw=1.8, label="left  coherence")
axL.plot(_lxR, _lyR, color=RIGHT, marker=MR, lw=1.8, label="right  coherence")
axL.plot(S, dpoL_col, color=LEFT, marker=ML, ls="--", lw=1.3, ms=4, label="left  collapse%")
axL.plot(S, dpoR_col, color=RIGHT, marker=MR, ls="--", lw=1.3, ms=4, label="right  collapse%")
_style_panel(axL, "DPO LoRA scale  s", "% (coherence gate / collapse)",
             "Llama - the gate TRACKS the collapse")

fig.suptitle("RQ3 coherence gate vs reasoning collapse under DPO (solid = coherence, dashed = collapse)",
             x=0.012, ha="left", fontsize=12.0, fontweight="semibold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
print("wrote:", save_fig(fig, f"{OUT}/rq3_2_coherence_gap"))
plt.close(fig)


# =============================================================== rq3_3 contamination
# 2x2: same row/column layout as rq3_1. Per panel: left contam% (blue), right (red).
# NOTE: the Mistral contamination arrays are BUCKETED. The four below-cliff points
# are plotted at 0 (the RESULTS.md §5 "0% below the cliff" bucket), NOT six measured
# points. Only the alpha3/s1.5 and alpha4/s2 values are measured per-knob.
fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.0), sharey=True)

contam_panels = [
    (axes[0, 0], A, m_steerL_con, m_steerR_con, "steering coefficient  α", "Mistral - Steering"),
    (axes[0, 1], S, m_dpoL_con, m_dpoR_con, "DPO LoRA scale  s", "Mistral - DPO"),
    (axes[1, 0], A, steerL_con, steerR_con, "steering coefficient  α", "Llama - Steering"),
    (axes[1, 1], S, dpoL_con, dpoR_con, "DPO LoRA scale  s", "Llama - DPO"),
]
for ax, x, Lc, Rc, xlab, title in contam_panels:
    ax.plot(x, Lc, color=LEFT, marker=ML, lw=1.8, label="left")
    ax.plot(x, Rc, color=RIGHT, marker=MR, lw=1.8, label="right")
    _style_panel(ax, xlab, "contaminated %", title)
    ax.legend(loc="upper left", fontsize=8)

axes[0, 0].set_ylabel("contaminated % of the neutral task")
axes[1, 0].set_ylabel("contaminated % of the neutral task")
fig.suptitle("RQ3 contamination of the neutral task (contamination follows the collapsing arm)",
             x=0.012, ha="left", fontsize=12.0, fontweight="semibold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
print("wrote:", save_fig(fig, f"{OUT}/rq3_3_contamination"))
plt.close(fig)
