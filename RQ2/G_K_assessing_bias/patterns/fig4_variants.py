"""Scratch: two redesign variants of fig4 (reasoning-failure composition).
Both split mistral | llama into 2 panels in one row (shared y, single grouped
legend, family as panel title, n preserved in xticks). Severity ramp kept
(green -> khaki -> orange/red -> grey); confusable pairs separated.

  V1  full 7-category stack  (no information loss)
  V2  simplified 4-band stack (faithful / other-uncontaminated /
                               political-contamination / format-failure)

Writes PDF+SVG to patterns/figures_scratch/. Does NOT touch the live figure.
Run:  python -m RQ2.G_K_assessing_bias.patterns.fig4_variants
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from collections import Counter

# --- brand helper (walk up to the skill's scripts dir) ----------------------
for _d in Path(__file__).resolve().parents:
    _s = _d / ".claude/skills/visualizations-designer/scripts"
    if _s.is_dir():
        sys.path.insert(0, str(_s)); break
from polireason_viz import apply_theme, save_fig  # noqa: E402
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.patches import Patch               # noqa: E402

apply_theme()

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures_scratch"; OUT.mkdir(exist_ok=True)
ROWS = [json.loads(l) for l in (HERE / "judge_long.jsonl").open()]
REGIMES = ["base", "roleplay", "steering", "DPO"]
FAMILIES = ["mistral", "llama"]

# raw judge categories (severity order, bottom -> top)
RAW = ["faithful_task_performance", "post_hoc_reasoning", "capability_error",
       "motivational_framing_bias", "viewpoint_bias", "generation_collapse",
       "instruction_following_failure"]


def counts(fam, rg):
    sub = [r for r in ROWS if r["family"] == fam and r["regime"] == rg]
    return Counter(r["primary_category"] for r in sub), len(sub)


def render(name, cat_order, members, color, dark_text, legend_groups,
           figsize=(7.6, 3.9)):
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, fam in zip(axes, FAMILIES):
        xlabels = []
        for gi, rg in enumerate(REGIMES):
            cc, n = counts(fam, rg)
            bottom = 0.0
            for cat in cat_order:
                h = sum(cc.get(m, 0) for m in members[cat]) / n * 100
                ax.bar(gi, h, bottom=bottom, color=color[cat],
                       edgecolor="white", linewidth=0.6, width=0.82)
                if h >= 3.0:
                    ax.text(gi, bottom + h / 2, f"{h:.0f}", ha="center",
                            va="center", fontsize=7,
                            color="#1A1A1A" if cat in dark_text else "white")
                bottom += h
            xlabels.append(f"{rg}\n(n={n})")
        ax.set_xticks(range(len(REGIMES)))
        ax.set_xticklabels(xlabels, fontsize=8.5)
        ax.set_ylim(0, 100)
        ax.set_title(fam.capitalize(), fontsize=11, loc="center")
        ax.tick_params(axis="x", length=0)
        ax.grid(False)
    axes[0].set_ylabel("% of responses")

    # grouped legend on the right: bold band header rows + colour swatches
    handles, labels, header_idx = [], [], []
    for header, cats in legend_groups:
        handles.append(Patch(facecolor="none", edgecolor="none"))
        labels.append(header); header_idx.append(len(labels) - 1)
        for cat in cats:
            handles.append(Patch(facecolor=color[cat], edgecolor="white", linewidth=0.5))
            labels.append(LBL[cat])
    leg = fig.legend(handles, labels, loc="center left",
                     bbox_to_anchor=(0.99, 0.5), fontsize=8, frameon=False,
                     handlelength=1.1, handletextpad=0.6, labelspacing=0.5)
    for i in header_idx:
        leg.get_texts()[i].set_fontweight("bold")
        leg.get_texts()[i].set_fontsize(8.3)
    fig.text(0.0, 1.02, "Reasoning-failure composition by method and family",
             fontsize=12.5, fontweight="semibold", ha="left", va="bottom")
    save_fig(fig, str(OUT / name))
    plt.close(fig)
    print(f"[fig] wrote {name}.pdf/.svg")


# legend labels (shared)
LBL = {
    "faithful_task_performance": "faithful (correct, clean)",
    "post_hoc_reasoning": "post-hoc (correct, broken)",
    "capability_error": "capability error",
    "motivational_framing_bias": "motiv. framing bias",
    "viewpoint_bias": "viewpoint bias",
    "generation_collapse": "generation collapse",
    "instruction_following_failure": "instruction-follow fail",
    # banded
    "faithful_band": "faithful (correct, clean)",
    "uncontam_band": "other uncontaminated failure",
    "contam_band": "political contamination",
    "format_band": "format failure",
}

# ============================ V1 — full 7-category =========================
COLOR7 = {
    "faithful_task_performance": "#1B7837",   # deep green
    "post_hoc_reasoning": "#7FBF7B",          # medium green (separated)
    "capability_error": "#C2A33B",            # khaki  (wrong, non-political)
    "motivational_framing_bias": "#E8883A",   # orange (contam, correct)
    "viewpoint_bias": "#B2182B",              # red    (contam, wrong)
    "generation_collapse": "#595959",         # dark grey
    "instruction_following_failure": "#ADADAD",  # light grey (separated)
}
DARK7 = {"post_hoc_reasoning", "capability_error", "instruction_following_failure"}
MEMBERS7 = {c: [c] for c in RAW}
LEGEND7 = [
    ("No political contamination",
     ["faithful_task_performance", "post_hoc_reasoning", "capability_error"]),
    ("Political contamination",
     ["motivational_framing_bias", "viewpoint_bias"]),
    ("Format failure",
     ["generation_collapse", "instruction_following_failure"]),
]
render("fig4_v1_7cat_2panel", RAW, MEMBERS7, COLOR7, DARK7, LEGEND7)

# ============================ V2 — simplified 4-band =======================
BAND_ORDER = ["faithful_band", "uncontam_band", "contam_band", "format_band"]
MEMBERS4 = {
    "faithful_band": ["faithful_task_performance"],
    "uncontam_band": ["post_hoc_reasoning", "capability_error"],
    "contam_band": ["motivational_framing_bias", "viewpoint_bias"],
    "format_band": ["generation_collapse", "instruction_following_failure"],
}
COLOR4 = {
    "faithful_band": "#1B7837",
    "uncontam_band": "#C2A33B",
    "contam_band": "#C0392B",
    "format_band": "#7F7F7F",
}
DARK4 = {"uncontam_band"}
LEGEND4 = [("", BAND_ORDER)]  # single flat group; no headers needed
# flat legend: drop the empty header row
LEGEND4 = [(None, BAND_ORDER)]


def render4():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=True)
    for ax, fam in zip(axes, FAMILIES):
        xlabels = []
        for gi, rg in enumerate(REGIMES):
            cc, n = counts(fam, rg)
            bottom = 0.0
            for cat in BAND_ORDER:
                h = sum(cc.get(m, 0) for m in MEMBERS4[cat]) / n * 100
                ax.bar(gi, h, bottom=bottom, color=COLOR4[cat],
                       edgecolor="white", linewidth=0.6, width=0.82)
                if h >= 3.0:
                    ax.text(gi, bottom + h / 2, f"{h:.0f}", ha="center",
                            va="center", fontsize=7.5,
                            color="#1A1A1A" if cat in DARK4 else "white")
                bottom += h
            xlabels.append(f"{rg}\n(n={n})")
        ax.set_xticks(range(len(REGIMES)))
        ax.set_xticklabels(xlabels, fontsize=8.5)
        ax.set_ylim(0, 100)
        ax.set_title(fam.capitalize(), fontsize=11)
        ax.tick_params(axis="x", length=0)
        ax.grid(False)
    axes[0].set_ylabel("% of responses")
    handles = [Patch(facecolor=COLOR4[c], edgecolor="white", linewidth=0.5) for c in BAND_ORDER]
    labels = [LBL[c] for c in BAND_ORDER]
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.99, 0.5),
               fontsize=8.5, frameon=False, handlelength=1.2, labelspacing=0.6)
    fig.text(0.0, 1.02, "Reasoning-failure composition (simplified bands)",
             fontsize=12.5, fontweight="semibold", ha="left", va="bottom")
    save_fig(fig, str(OUT / "fig4_v2_4band_2panel"))
    plt.close(fig)
    print("[fig] wrote fig4_v2_4band_2panel.pdf/.svg")


render4()
print(f"[fig] all variants in {OUT}")
