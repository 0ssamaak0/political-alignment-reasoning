"""Render the 4 reportable RQ2 judge-pattern figures (G&K cohort).
All quantities are recomputed here from the canonical table so the figures match
JUDGE_PATTERNS.md exactly. House style: left=tab:blue, right=tab:red, base=gray.

Run from repo root:
    python -m RQ2.G_K_assessing_bias.patterns.make_figures
Outputs PDF+SVG to patterns/figures/.
"""
from __future__ import annotations
import json, math
from collections import defaultdict, Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
GK = HERE.parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
rows = [json.loads(l) for l in (HERE / "judge_long.jsonl").open()]
pid = {k["gidx"]: k["pair_id"] for k in (json.loads(l) for l in (GK / "verify" / "gemini_key.jsonl").open())}

CLR = {"left": "#2166AC", "right": "#B2182B", "none": "#4D4D4D"}
BASE_ACC = {"mistral": 0.625, "llama": 0.599}
BASE_BIASCAT = {"mistral": 0.0, "llama": 0.005}
ALPHA_BONF = 0.05 / 15
CELLS = ["mistral-base", "llama-base", "mistral-roleplay-left", "mistral-roleplay-right",
         "llama-roleplay-left", "llama-roleplay-right", "mistral-steering-left", "mistral-steering-right",
         "llama-steering-left", "llama-steering-right", "mistral-DPO-left", "mistral-DPO-right",
         "llama-DPO-left", "llama-DPO-right"]  # llama-DPO-right-2nd excluded from final RQ2 analysis
def fam(c): return "mistral" if c.startswith("mistral") else "llama"
def lean(c): return "none" if c.endswith("base") else ("left" if "left" in c else "right")
def short(c): return c.replace("mistral", "M").replace("llama", "L").replace("-roleplay-", "-RP-").replace("-steering-", "-ST-").replace("-right-2nd", "-R2")
def binom(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)

# ---- recompute per-cell metrics -------------------------------------------------
M = {}
for c in CELLS:
    cr = [r for r in rows if r["cell"] == c]
    eng = [r for r in cr if r["engaged"]]
    # matched-pair net (engaged-only discordant)
    pairs = defaultdict(dict)
    for r in eng:
        if r["gidx"] in pid and r["parsed_verdict"] in ("VALID", "INVALID"):
            pairs[pid[r["gidx"]]][r["item_lean"]] = r["parsed_verdict"]
    fR = fL = 0
    for m in pairs.values():
        if "left" in m and "right" in m and m["left"] != m["right"]:
            if m["right"] == "VALID" and m["left"] == "INVALID": fR += 1
            elif m["left"] == "VALID" and m["right"] == "INVALID": fL += 1
    net = fR - fL; pv = binom(fR, fL)
    M[c] = {
        "fam": fam(c), "lean": lean(c), "n_eng": len(eng),
        "eng_rate": len(eng) / len(cr),
        "acc_all": sum(r["correct"] for r in cr) / len(cr),
        "acc_eng": sum(r["correct"] for r in eng) / max(len(eng), 1),
        "biascat_d": sum(r["is_bias_cat"] for r in eng) / max(len(eng), 1) - BASE_BIASCAT[fam(c)],
        "mp_net": net, "mp_disc": fR + fL, "mp_p": pv, "bonf": pv < ALPHA_BONF,
    }

def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", dpi=140, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig); print(f"[fig] wrote {name}.pdf/.svg")

# =================================================================================
# FIG 1 — Matched-pair partisan double-standard (headline) + silent-cell acceptance panel
# =================================================================================
# authored narrower than before so the fonts survive the in-thesis downscale to 0.8\textwidth
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 3.5), gridspec_kw={"width_ratios": [2.1, 1]})
order = sorted(CELLS, key=lambda c: M[c]["mp_net"])
ys = range(len(order))
axA.barh(list(ys), [M[c]["mp_net"] for c in order],
         color=[CLR[M[c]["lean"]] for c in order], edgecolor="black", linewidth=0.5)
for i, c in enumerate(order):
    m = M[c]; x = m["mp_net"]
    axA.text(x + (1.2 if x >= 0 else -1.2), i, short(c), va="center",
             ha="left" if x >= 0 else "right", fontsize=9.5)
axA.axvline(0, color="black", lw=0.8)
axA.set_yticks([]); axA.set_xlim(-50, 42)
axA.tick_params(axis="x", labelsize=11)
axA.set_xlabel("net partisan double standard\n<- favors left            favors right ->", fontsize=12)
axA.set_title("A. Net double standard by configuration", fontsize=13)
leg = [Line2D([0], [0], marker="s", color="w", markerfacecolor=CLR[k], markersize=12,
              label={"left": "Left", "right": "Right", "none": "Base"}[k]) for k in ("left", "right", "none")]
axA.legend(handles=leg, loc="lower right", fontsize=10.5, frameon=False)

# Panel B — mistral-DPO-right acceptance by party on matched-validity items
e = [r for r in rows if r["cell"] == "mistral-DPO-right" and r["engaged"]]
def vrate(gv, ln):
    s = [r for r in e if r["gold_valid"] == gv and r["item_lean"] == ln]
    return sum(1 for r in s if r["parsed_verdict"] == "VALID") / max(len(s), 1)
groups = ["gold-VALID\n(should accept)", "gold-INVALID\n(should reject)"]
demv = [vrate(1, "left"), vrate(0, "left")]; repv = [vrate(1, "right"), vrate(0, "right")]
xpos = range(len(groups)); w = 0.36
axB.bar([x - w/2 for x in xpos], [v*100 for v in demv], w, color="#2166AC", label="Left", edgecolor="black", lw=0.5)
axB.bar([x + w/2 for x in xpos], [v*100 for v in repv], w, color="#B2182B", label="Right", edgecolor="black", lw=0.5)
for x, v in zip(xpos, demv): axB.text(x - w/2, v*100 + 1.5, f"{v*100:.0f}%", ha="center", fontsize=10.5)
for x, v in zip(xpos, repv): axB.text(x + w/2, v*100 + 1.5, f"{v*100:.0f}%", ha="center", fontsize=10.5)
axB.set_xticks(list(xpos)); axB.set_xticklabels(groups, fontsize=10.5)
axB.tick_params(axis="y", labelsize=11)
axB.set_ylabel("P(model says VALID)  %", fontsize=12); axB.set_ylim(0, 65)
axB.set_title("B. Mistral DPO-right\nacceptance by party", fontsize=12.5)
axB.legend(fontsize=10.5, loc="upper right", frameon=False)
# suptitle removed: the thesis caption carries the title role
# fig.suptitle("Partisan double standard in validity judgements", fontsize=14, y=1.02)
save(fig, "fig1_matched_pair_double_standard")

# =================================================================================
# FIG 2 — Silent/Loud decoupling scatter: rhetoric (x) vs verdict double-standard (y)
# =================================================================================
fig, ax = plt.subplots(figsize=(9.5, 7))
RM = {"base": "o", "roleplay": "s", "steering": "^", "DPO": "D"}
def regime(c): return "roleplay" if "roleplay" in c else ("steering" if "steering" in c else ("DPO" if "DPO" in c else "base"))
for c in CELLS:
    m = M[c]
    ax.scatter(m["biascat_d"] * 100, m["mp_net"], s=150, marker=RM[regime(c)],
               color=CLR[m["lean"]], edgecolor="black", linewidth=0.7, zorder=3,
               alpha=0.9)
for c, dx, dy, ha in [("mistral-DPO-right", 2, 1.5, "left"), ("llama-DPO-right", -2, -2.5, "right"),
                      ("llama-DPO-left", 2, -1, "left"), ("mistral-steering-left", 2, -1.5, "left"),
                      ("llama-steering-left", 2, 0.5, "left"), ("mistral-steering-right", 2, 0.5, "left")]:
    m = M[c]; ax.annotate(short(c), (m["biascat_d"]*100, m["mp_net"]), xytext=(m["biascat_d"]*100+dx, m["mp_net"]+dy), fontsize=8.5, ha=ha)
ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="0.7", lw=0.8, ls="--")
ax.set_ylim(-42, 34); ax.set_xlim(-6, 104)
ax.set_xlabel("LOUDNESS:  added motivated-reasoning / bias rhetoric  (Δ bias-category rate over base, pp, engaged)")
ax.set_ylabel("VERDICT skew:  matched-pair partisan double-standard (net)")
ax.text(16, 18, "SILENT skew\n(verdicts move, ~no rhetoric)", fontsize=9, color="0.35", style="italic")
ax.text(56, 27, "LOUD\n(rhetoric ≫ net verdict skew)", fontsize=9, color="0.35", style="italic", ha="center")
leg1 = [Line2D([0],[0], marker=RM[r], color="w", markerfacecolor="0.5", markeredgecolor="black", markersize=10, label=r) for r in ["roleplay","steering","DPO","base"]]
leg2 = [Line2D([0],[0], marker="o", color="w", markerfacecolor=CLR[k], markersize=10, label=k) for k in ["left","right","none"]]
l1 = ax.legend(handles=leg1, loc="upper right", fontsize=8.5, title="regime", frameon=True); ax.add_artist(l1)
ax.legend(handles=leg2, loc="lower right", fontsize=8.5, title="induced alignment", frameon=True)
ax.set_title("Same verdict double-standard, opposite mechanism: DPO can skew verdicts silently\n(mistral-DPO-right: high skew, ~0 rhetoric) or loudly (llama-DPO-right: saturated rhetoric)", fontsize=11)
save(fig, "fig2_silent_loud_decoupling")

# =================================================================================
# FIG 3 — Hollow accuracy dumbbell: acc_all -> acc_engaged
# =================================================================================
fig, ax = plt.subplots(figsize=(10, 6.8))
order = sorted(CELLS, key=lambda c: M[c]["eng_rate"])
for i, c in enumerate(order):
    m = M[c]
    ax.plot([m["acc_all"], m["acc_eng"]], [i, i], color="0.6", lw=2, zorder=1)
    ax.scatter(m["acc_all"], i, s=70, color="white", edgecolor=CLR[m["lean"]], linewidth=1.8, zorder=2)
    ax.scatter(m["acc_eng"], i, s=70, color=CLR[m["lean"]], edgecolor="black", linewidth=0.5, zorder=3)
    ax.text(0.012, i, short(c), transform=ax.get_yaxis_transform(), va="center", fontsize=8.5)
    ax.text(max(m["acc_all"], m["acc_eng"]) + 0.028, i, f"eng {m['eng_rate']*100:.0f}%", va="center", fontsize=7.5, color="0.4")
for famn, x in BASE_ACC.items():
    ax.axvline(x, color="0.5", ls=":", lw=1)
    ax.text(x, len(order)-0.45, f"{famn} base", rotation=90, fontsize=7.5, color="0.4", va="bottom", ha="center")
ax.set_yticks([]); ax.set_xlim(0.22, 0.78); ax.set_ylim(-0.7, len(order)+1.1)
ax.set_xlabel("accuracy")
ax.set_title("Hollow accuracy: the penalty is mostly disengagement", fontsize=11, loc="left", pad=70)
# --- legend lifted out of the plot body into a strip above the axes --------------
# primary key: open vs filled circle (this distinction is what "hollow accuracy" means)
fill_leg = [
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="0.2",
           markeredgewidth=1.8, markersize=10, label="open circle = accuracy over all 192 prompts (deployed)"),
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="0.3", markeredgecolor="black",
           markeredgewidth=0.5, markersize=10, label="filled circle = accuracy over engaged items only"),
]
# secondary key: colour = induced alignment (already echoed in the row labels)
lean_leg = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CLR[k], markeredgecolor="black",
            markeredgewidth=0.5, markersize=8,
            label={"left": "left", "right": "right", "none": "base"}[k]) for k in ("left", "right", "none")]
l1 = ax.legend(handles=fill_leg, loc="lower left", bbox_to_anchor=(0.0, 1.055), fontsize=9.5,
               frameon=False, title="marker fill", title_fontsize=9.5, handletextpad=0.6)
l1._legend_box.align = "left"
ax.add_artist(l1)
l2 = ax.legend(handles=lean_leg, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=8.5,
               frameon=False, title="colour = induced alignment", title_fontsize=8.5,
               handletextpad=0.4, columnspacing=1.1)
l2._legend_box.align = "left"
save(fig, "fig3_hollow_accuracy")

# =================================================================================
# FIG 4 — Reasoning-failure composition by regime x family
# =================================================================================
CATS = ["faithful_task_performance", "post_hoc_reasoning", "capability_error",
        "motivational_framing_bias", "viewpoint_bias", "generation_collapse", "instruction_following_failure"]
CATCLR = {"faithful_task_performance": "#2ca02c", "post_hoc_reasoning": "#98df8a",
          "capability_error": "#bcbd22", "motivational_framing_bias": "#ff7f0e",
          "viewpoint_bias": "#d62728", "generation_collapse": "#7f7f7f", "instruction_following_failure": "#c7c7c7"}
CATLBL = {"faithful_task_performance":"faithful (correct, valid reasoning)","post_hoc_reasoning":"post-hoc (correct, invalid reasoning)",
          "capability_error":"capability error","motivational_framing_bias":"editorial framing bias (correct, contaminated)",
          "viewpoint_bias":"viewpoint bias (wrong, contaminated)","generation_collapse":"generation collapse","instruction_following_failure":"instruction-follow fail"}
from itertools import groupby
def cfg_lean(c):
    return "base" if c.endswith("base") else ("left" if "left" in c else "right")

# one bar per (family, regime, induced alignment); base has no alignment so it stays single
specs = []
for f in ("mistral", "llama"):
    specs.append((f, "base", "base"))
    for rg in ("roleplay", "steering", "DPO"):
        specs += [(f, rg, "left"), (f, rg, "right")]

# x positions: hug the left/right pair, gap between methods, wider gap between families
xs = []; x = 0.0; prev = None
for spec in specs:
    if prev is not None:
        if prev[0] != spec[0]:     x += 1.6     # new family
        elif prev[1] != spec[1]:   x += 1.15    # new method
        else:                      x += 0.66    # left -> right within a method
    xs.append(x); prev = spec

BAR_W = 0.42
# width kept near the printed \textwidth so fonts survive the in-thesis downscale;
# height kept low so the figure is wide, not tall, once scaled to \textwidth
fig, ax = plt.subplots(figsize=(11.0, 5.0))
for (f, rg, ln), xc in zip(specs, xs):
    sub = [r for r in rows if r["family"] == f and r["regime"] == rg and cfg_lean(r["cell"]) == ln]
    n = len(sub); cc = Counter(r["primary_category"] for r in sub)
    first = (f == "mistral" and rg == "base")    # attach the 7 legend labels once
    bottom = 0
    for cat in CATS:
        h = cc.get(cat, 0) / n * 100
        ax.bar(xc, h, width=BAR_W, bottom=bottom, color=CATCLR[cat], edgecolor="white",
               linewidth=0.4, label=CATLBL[cat] if first else None)
        bottom += h

# tier 1: induced alignment under each bar (the base bar just reads "base")
ax.set_xticks(xs); ax.set_xticklabels([ln for (_, _, ln) in specs], fontsize=13)
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", labelsize=12)
# tier 2: method, centered under each left/right pair
for (f, rg), grp in groupby(zip(specs, xs), key=lambda t: (t[0][0], t[0][1])):
    gx = [px for _, px in grp]
    if rg != "base":
        ax.text(sum(gx) / len(gx), -0.135, rg, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=13.5)
# tier 3: family, centered under each family block
for f, grp in groupby(zip(specs, xs), key=lambda t: t[0][0]):
    gx = [px for _, px in grp]
    ax.text(sum(gx) / len(gx), -0.275, f, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=16, fontweight="semibold")

mx = [px for s, px in zip(specs, xs) if s[0] == "mistral"]
lx = [px for s, px in zip(specs, xs) if s[0] == "llama"]
ax.axvline((max(mx) + min(lx)) / 2, color="black", lw=0.8, ls="--")
ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7); ax.set_ylim(0, 100)
ax.set_ylabel("% of responses", fontsize=14)
ax.set_title("Reasoning-failure composition by method, family, and alignment",
             fontsize=13, loc="left")
# two columns so the bigger font still fits; sits just below the three-tier x-axis
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), fontsize=15,
          frameon=False, ncol=2, handletextpad=0.6, columnspacing=2.2)
save(fig, "fig4_failure_composition")

print(f"\n[fig] all figures in {FIG}")
# echo the headline numbers used, for cross-check against the doc
print("[fig] matched-pair net (Bonferroni-sig ★):",
      {short(c): (M[c]["mp_net"], "★" if M[c]["bonf"] else "") for c in CELLS if abs(M[c]["mp_net"]) >= 5})
