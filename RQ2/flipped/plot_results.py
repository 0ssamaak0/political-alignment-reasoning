"""Figures for RQ2/flipped/RESULTS.md — brand-styled, vector output (PDF+SVG).
MRfal = % responses with fallacy_lens in {motivational_reasoning,
premise_truth_conflation}. Data mirrors RESULTS.md (14/14 cells, verdict-first)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.cm as cm  # noqa: E402

# --- brand identity inlined (skill ships SKILL.md only; no helper module) ---
LEAN = {"left": "#2166AC", "right": "#B2182B"}
BASE = "#4D4D4D"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 12.5,
    "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.edgecolor": "#8A8A8A", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": "#1A1A1A", "axes.labelcolor": "#1A1A1A",
    "xtick.color": "#6B6B6B", "ytick.color": "#6B6B6B",
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": "#E6E6E6", "grid.linewidth": 0.8,
    "legend.frameon": False, "figure.facecolor": "white", "savefig.bbox": "tight",
})


def save_fig(fig, path, pad_inches=None):  # vector only: PDF (canonical) + SVG
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    kw = {} if pad_inches is None else {"bbox_inches": "tight", "pad_inches": pad_inches}
    for ext in ("pdf", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), **kw)


OUT = Path(__file__).resolve().parent / "figures"

cells = [
    "llama Base", "llama RP-L", "llama RP-R", "llama Steer-L", "llama Steer-R",
    "llama DPO-L", "llama DPO-R", "mistral Base", "mistral RP-L", "mistral RP-R",
    "mistral Steer-L", "mistral Steer-R", "mistral DPO-L", "mistral DPO-R",
]
# acc / contam unaffected by the MRfal typo fix; MRfal corrected to include
# motivational_reasoning (was dropped). Order: Neu / PolC / PolF.
acc = np.array([
    [68, 75, 72], [70, 65, 64], [65, 69, 68], [50, 50, 50], [48, 46, 44],
    [72, 60, 61], [50, 33, 33], [77, 62, 51], [72, 62, 52], [72, 59, 50],
    [67, 71, 66], [53, 58, 60], [50, 55, 47], [73, 63, 55],
])
contam = np.array([
    [0, 0, 0], [5, 4, 5], [8, 9, 10], [17, 47, 42], [2, 15, 15],
    [0, 29, 49], [28, 100, 100], [0, 0, 0], [0, 4, 3], [0, 1, 1],
    [65, 76, 73], [40, 51, 56], [50, 98, 99], [0, 7, 6],
])
mrfal = np.array([
    [0, 0, 3], [0, 5, 8], [0, 8, 9], [0, 5, 3], [22, 49, 57],
    [0, 16, 33], [5, 21, 22], [2, 10, 18], [7, 27, 40], [8, 28, 34],
    [25, 34, 37], [10, 29, 34], [45, 79, 87], [8, 30, 32],
])
conds = ["Neu", "PolC", "PolF"]
# Conditions are a politicization-intensity sequence, NOT a lean axis →
# grey control + two viridis samples (sequential magnitude), per brand.
cond_colors = [BASE, cm.viridis(0.45), cm.viridis(0.80)]


def grouped_percell(data, title, fname):
    x = np.arange(len(cells)); w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, c in enumerate(conds):
        ax.bar(x + (i - 1) * w, data[:, i], w, label=c, color=cond_colors[i])
    ax.set_xticks(x); ax.set_xticklabels(cells, rotation=45, ha="right")
    ax.set_ylabel("%"); ax.set_title(title)
    ax.legend(title="condition", ncol=3, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.02))
    save_fig(fig, OUT / fname); plt.close(fig)


grouped_percell(acc, "Accuracy by condition", "acc_by_condition")
grouped_percell(contam, "Contamination by condition", "contam_by_condition")
grouped_percell(mrfal, "Content-over-form fallacy by condition", "mrfal_by_condition")

# --- Pooled metrics across conditions (corrected MRfal) ---
# Thesis pooled figure (ch4_judge_measures_by_condition): readable condition
# names + spelled-out measures, no "%" suffix on the ticks. Labels are local
# here so the per-cell figures above keep their compact Neu/PolC/PolF legend.
pooled = {"accuracy": [63, 59, 55], "contamination": [15, 31, 33],
          "content-over-form\nfallacy": [9, 24, 30], "invalid\nreasoning": [46, 54, 60]}
pooled_conds = ["neutral", "political", "political flipped"]
fig, ax = plt.subplots(figsize=(7.0, 4.3))
x = np.arange(len(pooled)); w = 0.27
for i, c in enumerate(pooled_conds):
    ax.bar(x + (i - 1) * w, [pooled[m][i] for m in pooled], w, label=c, color=cond_colors[i])
ax.set_xticks(x); ax.set_xticklabels(list(pooled.keys()))
ax.set_ylabel("percent of responses")
# Legend on its own row just above the axes.
ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0))
save_fig(fig, OUT / "pooled_by_condition", pad_inches=0.22); plt.close(fig)

# --- signed_bias per cell: lean axis → red=right(>0), blue=left(<0) ---
# llama DPO-R's +0.19 is a refusal artifact: asymmetric disengagement (~59% left-arm
# refusal vs ~10% right-arm) inflates the engaged signed bias. On matched-engaged
# pairs it collapses to +0.02 (n.s.). Render it muted + hatched and annotate the
# corrected value so it does not read as a genuine directional outlier.
signed_bias = [0.029, 0.039, -0.003, 0.000, -0.009, 0.055, 0.19,
               0.008, 0.021, 0.042, -0.021, 0.008, -0.024, 0.000]
ARTIFACT = 6             # llama DPO-R
ARTIFACT_MATCHED = 0.02  # matched-engaged signed bias (n.s.)
fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = np.arange(len(cells))
colors = [LEAN["right"] if v > 0 else LEAN["left"] for v in signed_bias]
bars = ax.bar(x, signed_bias, color=colors)
bars[ARTIFACT].set(color=BASE, alpha=0.4, hatch="///", edgecolor=BASE)
ax.plot(ARTIFACT, ARTIFACT_MATCHED, marker="D", ms=5, color="#1A1A1A", zorder=5)
ax.annotate("refusal artifact",
            xy=(ARTIFACT, 0.18), xytext=(ARTIFACT - 3.4, 0.15),
            fontsize=8, color="#6B6B6B", ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="#6B6B6B", lw=0.8))
ax.axhline(0, color="#1A1A1A", lw=0.8)
for y in (0.1, -0.1):
    ax.axhline(y, color="#6B6B6B", ls="--", lw=0.8)
ax.set_ylim(-0.06, 0.23)
ax.set_ylabel("signed bias")
ax.set_xticks(x)
ax.set_xticklabels(cells, rotation=45, ha="right")
save_fig(fig, OUT / "signed_bias"); plt.close(fig)

print("wrote brand-styled PDF+SVG to", OUT)
