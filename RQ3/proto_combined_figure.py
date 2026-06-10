"""
PROTOTYPE (throwaway) for the RQ3 thesis discussion.

One combined RQ3 figure that puts the dose-response, the failure mechanism, and
the contamination coloring together, on the data we actually have (the neutral
BBH strength sweep), Mistral only.

Design decisions (the faithful version of the user's mockup):
  - x-axis is a MIRRORED MAGNITUDE axis: left intervention on the negative side,
    right on the positive side, base shared at 0. Legitimate for BOTH methods
    because steering (separate left/right persona vectors, alpha>=0) and DPO
    (separate left/right adapters, s>=0) are both magnitudes on two separate
    mechanisms, not one coefficient swept through zero.
  - the stacked area is the REPARSE three-regime split (correct / wrong /
    collapse), NOT the judge categories. Accuracy = top of the green band.
  - contamination is an ORTHOGONAL OVERLAY LINE (judge layer), never a stack
    slice. It must be a line because the data forbids a sub-slice: at steering
    -left alpha3 contamination (24%) EXCEEDS collapse (7%), so contaminated is
    not a subset of collapsed. The line sits at 0 across the usable band and
    rises only on the flanks = the gating rule, drawn.
  - bands avoid RED so collapse is not confused with right-lean red.

Numbers: blueprint section 8 (verified against RQ3/RESULTS.md section 3/5).
"""
import sys
from pathlib import Path

for _d in Path(__file__).resolve().parents:
    _s = _d / ".claude/skills/visualizations-designer/scripts"
    if _s.is_dir():
        sys.path.insert(0, str(_s))
        break
from polireason_viz import apply_theme, LEAN, BASE, save_fig  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

apply_theme()

# --- categorical band colors (Okabe-Ito green/amber + grey), NO red ---
C_CORRECT = "#009E73"   # bluish green
C_WRONG = "#E69F00"     # orange/amber
C_COLLAPSE = "#666666"  # neutral grey, "the model went dark"
C_CONTAM = "#7B3294"    # purple, clearly a different (judge) layer
C_ACC = "#1A1A1A"       # ink, accuracy edge

# --- Mistral RQ3 data (reparse) ---
A = [0, 0.5, 1, 2, 3, 4]              # steering coefficient
S = [0, 0.25, 0.5, 1.0, 1.5, 2.0]    # DPO adapter scale

methods = {
    "Steering": dict(
        grid=A, knob=r"steering coefficient $\alpha$",
        L_bbh=[74, 77, 78, 71, 52, 3], L_col=[5, 3, 2, 0, 7, 92], L_con=[0, 0, 0, 0, 24, 92],
        R_bbh=[74, 75, 72, 69, 45, 0], R_col=[5, 6, 9, 11, 34, 99], R_con=[0, 0, 0, 0, 19, 58],
    ),
    "DPO fine-tuning": dict(
        grid=S, knob=r"adapter scale $s$",
        L_bbh=[74, 74, 72, 65, 22, 0], L_col=[5, 6, 8, 14, 70, 100], L_con=[0, 0, 0, 0, 42, 100],
        R_bbh=[74, 75, 75, 72, 24, 18], R_col=[5, 4, 4, 4, 68, 71], R_con=[0, 0, 0, 0, 0, 0],
    ),
}


def signed(grid, Lser, Rser):
    """Left series on negative x (base excluded), base at 0, right series on positive x."""
    n = len(grid)
    neg_x = [-grid[i] for i in range(n - 1, 0, -1)]
    neg_v = [Lser[i] for i in range(n - 1, 0, -1)]
    pos_x = [grid[i] for i in range(n)]      # includes base at index 0
    pos_v = [Rser[i] for i in range(n)]
    return np.array(neg_x + pos_x, float), np.array(neg_v + pos_v, float)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)

for ax, (name, d) in zip(axes, methods.items()):
    x, correct = signed(d["grid"], d["L_bbh"], d["R_bbh"])
    _, collapse = signed(d["grid"], d["L_col"], d["R_col"])
    _, contam = signed(d["grid"], d["L_con"], d["R_con"])
    wrong = np.clip(100 - correct - collapse, 0, None)

    ax.stackplot(
        x, correct, wrong, collapse,
        colors=[C_CORRECT, C_WRONG, C_COLLAPSE],
        labels=["correct (reparse)", "wrong", "collapse / no-answer"],
        edgecolor="white", linewidth=0.4,
    )
    # accuracy = top of the green band
    ax.plot(x, correct, color=C_ACC, lw=1.3, label="accuracy (top of green)")
    # contamination, orthogonal overlay (judge layer), never a slice
    ax.plot(
        x, contam, ls=":", lw=1.6, color=C_CONTAM, marker="D", ms=5,
        mec="white", mew=0.6, label="% contaminated (judge layer)",
    )
    # base anchor
    ax.axvline(0, color=BASE, ls=(0, (1, 2)), lw=1.0)
    ax.text(0, 103, "base", color=BASE, ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{abs(v):g}" for v in x], fontsize=8)
    for lbl, v in zip(ax.get_xticklabels(), x):
        if v < 0:
            lbl.set_color(LEAN["left"])
        elif v > 0:
            lbl.set_color(LEAN["right"])
    ax.set_xlabel(d["knob"] + "   (magnitude; left = negative, right = positive)")
    ax.set_ylim(-2, 108)
    ax.set_title(name)
    # lean color language without fighting the fill
    ax.text(0.02, 0.96, "LEFT", color=LEAN["left"], weight="bold", fontsize=9,
            transform=ax.transAxes, ha="left", va="top")
    ax.text(0.98, 0.96, "RIGHT", color=LEAN["right"], weight="bold", fontsize=9,
            transform=ax.transAxes, ha="right", va="top")

axes[0].set_ylabel("% of responses")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, -0.04), fontsize=9)
fig.suptitle("RQ3 combined prototype, Mistral. Reparse three-regime stack with contamination overlay",
             x=0.01, ha="left", fontsize=12.5, weight="semibold")
fig.tight_layout(rect=(0, 0.02, 1, 0.97))

save_fig(fig, str(Path(__file__).resolve().parent / "figures" / "rq3_combined_proto_mistral"))
print("wrote figures/rq3_combined_proto_mistral.pdf and .svg")
