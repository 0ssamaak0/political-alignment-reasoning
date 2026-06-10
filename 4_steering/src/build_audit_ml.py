#!/usr/bin/env python3
"""Build AUDIT_ml.md — Judge audit limited to multi-layer pvsteer-ml cells
plus the Mistral comparison rows that matter. Mirrors the structure of
Judge/results/audit/AUDIT.md."""
import json, csv
from pathlib import Path
from collections import Counter

ROOT = Path("/Users/0ssamaak0/Documents/polievalpp")
AUTO = ROOT / "Judge" / "raw" / "auto"

_MISTRAL_AS = ("a2", "a2_2", "a2_4", "a2_5", "a2_6", "a2_8", "a3")
_LLAMA_AS = ("a2", "a2_2", "a2_4", "a2_5", "a2_6", "a2_8", "a3", "a4")
ML_CELLS = (
    [f"mistral-pvsteer-ml-left-{a}" for a in _MISTRAL_AS]
    + [f"mistral-pvsteer-ml-right-{a}" for a in _MISTRAL_AS]
    + [f"llama-pvsteer-ml-left-{a}" for a in _LLAMA_AS]
    + [f"llama-pvsteer-ml-right-{a}" for a in _LLAMA_AS]
)
COMPARE_CELLS = [
    # Single-layer pvsteer family — the activation-space comparison set.
    # Other Mistral cells (base/roleplay/politune-hf) live in
    # Judge/results/audit/AUDIT.md; this audit is scoped to STEERING.
    "mistral-pvsteer-left-a3", "mistral-pvsteer-left-a5",
    "mistral-pvsteer-right-a3", "mistral-pvsteer-right-a5",
]


def load(tag):
    p = AUTO / f"{tag}.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


cells = {}
for tag in ML_CELLS + COMPARE_CELLS:
    rows = load(tag)
    if rows is not None:
        cells[tag] = rows

pct = json.loads((ROOT / "political_compass/PoliLean/results/summary.json").read_text())

coh = {}
for path in sorted((ROOT / "4_steering/results/coherence").glob("pvsteer_ml*.csv")):
    if not path.exists():
        continue
    with open(path) as fh:
        for row in csv.DictReader(fh):
            tag = row.get("tag") or row.get("config")
            try:
                coh[tag] = {
                    "mean": float(row.get("mean") or row.get("coherence_mean")),
                    "std": float(row.get("std", "nan") or "nan"),
                    "p10": float(row.get("p10", "nan") or "nan"),
                }
            except (ValueError, TypeError):
                pass

sl_coh = {
    "mistral-pvsteer-left-a3": (73.5, None, None),
    "mistral-pvsteer-left-a5": (72.4, None, None),
    "mistral-pvsteer-left-a7": (37.5, None, None),
    "mistral-pvsteer-right-a3": (72.9, None, None),
    "mistral-pvsteer-right-a5": (72.8, None, None),
    "mistral-pvsteer-right-a7": (60.3, None, None),
}

sweep = {}
for d in ("left", "right"):
    p = ROOT / "4_steering/results/coef_sweep" / f"sweep_{d}.json"
    if p.exists():
        sweep[d] = json.loads(p.read_text())

# f4 bias values pulled dynamically from the latest aggregator output.
def _load_f4_bias_ml():
    out = {}
    summary_path = ROOT / "1_benchmarking/runs/f4/political/summary.json"
    if not summary_path.exists():
        return out
    summary = json.loads(summary_path.read_text())
    cfgs = summary.get("configs", {})
    for tag, v in cfgs.items():
        if "pvsteer-ml" not in tag:
            continue
        by_lean = v.get("by_lean", {})
        n = by_lean.get("neutral", {})
        l = by_lean.get("left", {})
        r = by_lean.get("right", {})
        out[tag] = {
            "bias_signed_FPFN": v.get("bias_signed_FPFN"),
            "acc_N": n.get("accuracy"),
            "acc_L": l.get("accuracy"),
            "acc_R": r.get("accuracy"),
            "wall_sec": None,
        }
    return out

f4_bias_ml = _load_f4_bias_ml()


def rates(rows):
    n = len(rows)
    if n == 0:
        return None
    contam = sum(1 for r in rows if r.get("contaminated") is True)
    collapsed = sum(1 for r in rows if r.get("collapsed") is True)
    both = sum(1 for r in rows if r.get("contaminated") is True and r.get("collapsed") is True)
    clean = sum(1 for r in rows if r.get("contaminated") is False and r.get("collapsed") is False)
    contam_only = contam - both
    collapse_only = collapsed - both
    acc = sum(1 for r in rows if r.get("outcome") == "correct") / n
    correct_rows = [r for r in rows if r.get("outcome") == "correct"]
    contam_correct = sum(1 for r in correct_rows if r.get("contaminated") is True)
    conf = sum(r.get("confidence", 0) for r in rows) / n
    pc = Counter(r.get("primary_category") for r in rows)
    rv = Counter(r.get("reasoning_validity") for r in rows)
    fal = Counter(r.get("fallacy_lens") for r in rows if r.get("fallacy_lens"))
    p_coll_contam = (both / contam) if contam else None
    p_coll_clean = (collapse_only / (clean + collapse_only)) if (clean + collapse_only) > 0 else None
    return {
        "n": n, "acc": acc, "contam": contam, "collapsed": collapsed, "both": both,
        "clean": clean, "contam_only": contam_only, "collapse_only": collapse_only,
        "contam_correct": contam_correct, "n_correct": len(correct_rows), "conf": conf,
        "primary_category": pc, "reasoning_validity": rv, "fallacy_lens": fal,
        "p_coll_contam": p_coll_contam, "p_coll_clean": p_coll_clean,
    }


def stratify(rows, key, levels):
    out = {}
    for lvl in levels:
        sub = [r for r in rows if r.get(key) == lvl]
        out[lvl] = (len(sub), Counter(r.get("primary_category") for r in sub))
    return out


def t7_stratify(rows):
    out = {}
    for label, sub in [
        ("T7", [r for r in rows if r.get("template_id", "").startswith("T7")]),
        ("T1-T6", [r for r in rows if not r.get("template_id", "").startswith("T7")]),
    ]:
        out[label] = (len(sub), Counter(r.get("primary_category") for r in sub))
    return out


def fmt_ec(tag):
    v = pct.get(tag, {})
    em = v.get("ec_mean")
    es = v.get("ec_std")
    return f"{em:+.2f} ± {es:.2f}" if em is not None else "—"


def fmt_coh(tag):
    if tag in coh:
        return f"{coh[tag]['mean']:.1f}"
    if tag in sl_coh:
        return f"{sl_coh[tag][0]:.1f}"
    return "—"


def fmt_signed(x, d=3):
    return f"{x:+.{d}f}" if x is not None else "—"


lines = []
P = lines.append

P("# Judge audit — activation-space steering on Mistral")
P("")
P("> Auto-classifier: `gemini-3-flash-preview` via Vertex with v3 schema")
P("> (orthogonal `contaminated`/`collapsed` flags). Per-cell jsonls in")
P("> [`4_steering/runs/judge/`](.) (mirrors of `Judge/raw/auto/`).")
P("> Same format as [`Judge/results/audit/AUDIT.md`](../../../Judge/results/audit/AUDIT.md),")
P("> **scoped to STEERING methods only** — single-layer `pvsteer-α{3,5}` ×")
P("> {left, right} + multi-layer `pvsteer-ml-α{2,3}` × {left, right}. The")
P("> α=2 cells are the \"pv3C-analogy\" relaxed coef pass — added to test whether")
P("> the 25.6% collapse rate on ml-left-a3 is reducible by subthreshold coef.")
P("> Comparisons to base / roleplay / politune-hf live in the project-wide audit.")
P("")

P("## 1. Cross-method headline — pvsteer family")
P("")
P("PCT (5 seeds), coherence (Gemini, 0–100), Judge contam / partisan over 336")
P("f4/political rows, plus f4 bias_signed_FPFN. Same persona vectors across")
P("all cells (`3_persona_vectors/shared/vectors/mistral/{lean}_response_avg_diff.pt`),")
P("only the hook shape varies.")
P("")
P("| cell | hook | α | PCT ec (μ±σ) | Coherence | acc | contam% | collapsed% | both% | partisan% (vp+mfb) | C-among-correct | f4 bias_FPFN | mean conf |")
P("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

def _alpha_str(suffix):
    """Convert a coef suffix like a2_5 to a display string like '2.5'."""
    s = suffix.replace("a", "").replace("_", ".")
    return s

CELL_META = {
    "mistral-pvsteer-left-a3":   ("single L17", "3"),
    "mistral-pvsteer-left-a5":   ("single L17", "5"),
    "mistral-pvsteer-right-a3":  ("single L9",  "3"),
    "mistral-pvsteer-right-a5":  ("single L9",  "5"),
}
for fam, alphas in (("mistral", _MISTRAL_AS), ("llama", _LLAMA_AS)):
    for d in ("left", "right"):
        for a in alphas:
            tag = f"{fam}-pvsteer-ml-{d}-{a}"
            CELL_META[tag] = ("multi 1-31 incr.", _alpha_str(a))

ordered = (
    ["mistral-pvsteer-left-a3", "mistral-pvsteer-left-a5"]
    + [f"mistral-pvsteer-ml-left-{a}" for a in _MISTRAL_AS]
    + ["mistral-pvsteer-right-a3", "mistral-pvsteer-right-a5"]
    + [f"mistral-pvsteer-ml-right-{a}" for a in _MISTRAL_AS]
    + [f"llama-pvsteer-ml-left-{a}" for a in _LLAMA_AS]
    + [f"llama-pvsteer-ml-right-{a}" for a in _LLAMA_AS]
)
for tag in ordered:
    r = rates(cells[tag]) if tag in cells else None
    ec = fmt_ec(tag)
    ch = fmt_coh(tag)
    bias = f4_bias_ml.get(tag, {}).get("bias_signed_FPFN")
    hook, alpha = CELL_META.get(tag, ("?", "?"))
    if r is None:
        P(f"| `{tag}` | {hook} | {alpha} | {ec} | {ch} | — | — | — | — | — | — | — | — |")
        continue
    cc = f"{100*r['contam_correct']/r['n_correct']:.1f}% ({r['contam_correct']}/{r['n_correct']})" if r['n_correct'] else "—"
    partisan_n = r["primary_category"].get("viewpoint_bias", 0) + r["primary_category"].get("motivational_framing_bias", 0)
    partisan_pct = 100 * partisan_n / r["n"]
    bold = "**" if "ml" in tag else ""
    P(f"| {bold}`{tag}`{bold} | {hook} | {alpha} | {ec} | {ch} | {100*r['acc']:.1f}% | {100*r['contam']/r['n']:.1f}% | {100*r['collapsed']/r['n']:.1f}% | {100*r['both']/r['n']:.1f}% | {partisan_pct:.1f}% | {cc} | {fmt_signed(bias) if bias is not None else '—'} | {r['conf']:.2f} |")
P("")

for ml_tag in ML_CELLS:
    if ml_tag not in cells:
        continue
    rows = cells[ml_tag]
    r = rates(rows)
    direction = "left" if "left" in ml_tag else "right"
    soc_m = pct.get(ml_tag, {}).get("soc_mean")
    soc_s = pct.get(ml_tag, {}).get("soc_std")
    coh_p10 = coh[ml_tag]["p10"] if ml_tag in coh else float("nan")

    P(f"## 2. `{ml_tag}` — full breakdown")
    P("")
    P(f"**n={r['n']}** · accuracy {100*r['acc']:.1f}% · contam {100*r['contam']/r['n']:.1f}% ({r['contam']}/{r['n']}) · collapsed {100*r['collapsed']/r['n']:.1f}% ({r['collapsed']}/{r['n']}) · both {100*r['both']/r['n']:.1f}% ({r['both']}/{r['n']}) · mean confidence {r['conf']:.2f}")
    P("")
    soc_str = f"{soc_m:+.2f} ± {soc_s:.2f}" if soc_m is not None else "—"
    P(f"PCT ec = {fmt_ec(ml_tag)}, soc = {soc_str} · coherence {fmt_coh(ml_tag)} (p10={coh_p10:.0f}) · f4 bias_signed_FPFN = {fmt_signed(f4_bias_ml[ml_tag]['bias_signed_FPFN'])}")
    P("")

    P("### Contamination × Collapse 2×2 (orthogonal flags, fresh v3 schema)")
    P("")
    P("Counts; rows = `contaminated`, columns = `collapsed`. P(collapse | contam) probes")
    P("whether contamination gates collapse on this cell.")
    P("")
    P("|  | collapsed=False | collapsed=True | total |")
    P("| --- | ---: | ---: | ---: |")
    P(f"| contaminated=False | {r['clean']} | {r['collapse_only']} | {r['clean']+r['collapse_only']} |")
    P(f"| contaminated=True  | {r['contam_only']} | {r['both']} | {r['contam_only']+r['both']} |")
    P(f"| total | {r['clean']+r['contam_only']} | {r['collapse_only']+r['both']} | {r['n']} |")
    P("")
    p_cc = r['p_coll_contam']
    p_cl = r['p_coll_clean']
    P(f"- P(collapse | contam) = {p_cc:.3f}" if p_cc is not None else "- P(collapse | contam) = —")
    P(f"- P(collapse | clean)  = {p_cl:.3f}" if p_cl is not None else "- P(collapse | clean)  = —")
    P("")

    P("### primary_category counts")
    P("")
    pc_order = ["faithful_task_performance", "post_hoc_reasoning", "capability_error",
                "quality_of_service_failure", "viewpoint_bias", "motivational_framing_bias",
                "generation_collapse"]
    P("| category | count | % |")
    P("| --- | ---: | ---: |")
    for cat in pc_order:
        v = r["primary_category"].get(cat, 0)
        P(f"| {cat} | {v} | {100*v/r['n']:.1f}% |")
    P("")

    P("### T7 (value-loaded) vs T1-T6 (strict-identity) — primary_category")
    P("")
    t7 = t7_stratify(rows)
    P(f"T7 n={t7['T7'][0]}, T1-T6 n={t7['T1-T6'][0]}")
    P("")
    P("| category | T7 | T1-T6 |")
    P("| --- | ---: | ---: |")
    for cat in pc_order:
        tv, t6 = t7["T7"][1].get(cat, 0), t7["T1-T6"][1].get(cat, 0)
        if tv == 0 and t6 == 0:
            continue
        P(f"| {cat} | {tv} ({100*tv/max(t7['T7'][0],1):.1f}%) | {t6} ({100*t6/max(t7['T1-T6'][0],1):.1f}%) |")
    P("")

    P("### Per-lean primary_category (counts)")
    P("")
    P(f"Stimuli lean = left/neutral/right (conclusion direction). Response steered by")
    P(f"the {direction} persona vector.")
    P("")
    lean = stratify(rows, "lean", ["left", "neutral", "right"])
    P(f"| category | left (n={lean['left'][0]}) | neutral (n={lean['neutral'][0]}) | right (n={lean['right'][0]}) |")
    P("| --- | ---: | ---: | ---: |")
    for cat in pc_order:
        L = lean['left'][1].get(cat, 0)
        N = lean['neutral'][1].get(cat, 0)
        R = lean['right'][1].get(cat, 0)
        if L == 0 and N == 0 and R == 0:
            continue
        P(f"| {cat} | {L} ({100*L/max(lean['left'][0],1):.1f}%) | {N} ({100*N/max(lean['neutral'][0],1):.1f}%) | {R} ({100*R/max(lean['right'][0],1):.1f}%) |")
    P("")

    P("### reasoning_validity + top fallacy lenses")
    P("")
    rv = r["reasoning_validity"]
    fal = r["fallacy_lens"]
    rv_str = " / ".join(f"{rv.get(k,0)}" for k in ["valid", "invalid", "opaque", "n/a"])
    fal_top = ", ".join(f"{k}({v})" for k, v in fal.most_common(4))
    P(f"- RV (valid / invalid / opaque / n/a): {rv_str}")
    P(f"- Top fallacy lenses: {fal_top if fal_top else '(none)'}")
    P("")

    P("### f4 accuracy by stimulus lean")
    P("")
    f4 = f4_bias_ml.get(ml_tag, {})
    def _f3(v): return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"
    P(f"- acc_N (neutral) = {_f3(f4.get('acc_N'))}")
    P(f"- acc_L (left-coded stimuli) = {_f3(f4.get('acc_L'))}")
    P(f"- acc_R (right-coded stimuli) = {_f3(f4.get('acc_R'))}")
    P(f"- bias_signed_FPFN = {fmt_signed(f4.get('bias_signed_FPFN')) if f4.get('bias_signed_FPFN') is not None else 'n/a'}")
    ws = f4.get('wall_sec')
    P(f"- wall time on L4: {ws:.1f}s" if isinstance(ws, (int, float)) else "- wall time on L4: n/a")
    P("")

P("## 3. Stage A — coefficient sweep (paper §A.3 layers 1..31 incremental)")
P("")
P("20 trait-questions per coef per direction. Gemini judges trait-expression (0..100)")
P("and coherence (0..100). Pick rule: largest coef where coh ≥ 50 AND trait still climbing.")
P("")
P("| direction | α=2 | α=3 | α=4 | α=5 | α=8 | α=12 |")
P("| --- | --- | --- | --- | --- | --- | --- |")
for d in ("left", "right"):
    sw = sweep.get(d, {})
    per = sw.get("per_coef", {})
    cells_md = []
    for c in (2.0, 3.0, 4.0, 5.0, 8.0, 12.0):
        m = None
        for k in (str(c), f"{c:.1f}", str(int(c))):
            if k in per:
                m = per[k]
                break
        if m is None:
            cells_md.append("—")
            continue
        t = m.get("trait_mean", 0)
        h = m.get("coh_mean", 0)
        flag = "✅" if h >= 50 and t >= 50 else "❌"
        cells_md.append(f"trait={t:.1f} coh={h:.1f} {flag}")
    P("| " + d + " | " + " | ".join(cells_md) + " |")
P("")
P("**Finding:** Sharp coherence cliff between α=3 and α=4 for both directions. Paper §A.3 \"better")
P("preserves capability\" claim does NOT hold for Mistral + `response_avg_diff.pt` at these layers.")
P("Only α=3 survives the coh ≥ 50 gate as a strong-stance cell.")
P("")

P("## 4. Pareto comparison — ml vs single-layer pvsteer (Mistral)")
P("")
P("Plot: [`4_steering/results/figures/pareto_ml.png`](../../results/figures/pareto_ml.png) (★ = ml, ○ = sl).")
P("")
P("| Pareto axis | sl-left-a3 | sl-left-a5 | **ml-left-a2** | **ml-left-a3** | sl-right-a3 | sl-right-a5 | sl-right-a7 | **ml-right-a2** | **ml-right-a3** |")
P("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")


def cell_val(tag, key):
    if tag not in cells:
        return "—"
    r = rates(cells[tag])
    if key == "contam_pct":
        return f"{100*r['contam']/r['n']:.1f}%"
    if key == "collapsed_pct":
        return f"{100*r['collapsed']/r['n']:.1f}%"
    if key == "partisan_pct":
        p = r["primary_category"].get("viewpoint_bias", 0) + r["primary_category"].get("motivational_framing_bias", 0)
        return f"{100*p/r['n']:.1f}%"
    return "—"


axis_rows = [
    ("PCT ec (μ)", lambda t: (f"{pct.get(t,{}).get('ec_mean','—'):+.2f}" if pct.get(t, {}).get("ec_mean") is not None else "—")),
    ("Coherence",   fmt_coh),
    ("Contam %",   lambda t: cell_val(t, "contam_pct")),
    ("Collapsed %", lambda t: cell_val(t, "collapsed_pct")),
    ("Partisan %", lambda t: cell_val(t, "partisan_pct")),
]
ordered_pareto = ["mistral-pvsteer-left-a3", "mistral-pvsteer-left-a5",
                  "mistral-pvsteer-ml-left-a2", "mistral-pvsteer-ml-left-a3",
                  "mistral-pvsteer-right-a3", "mistral-pvsteer-right-a5", "mistral-pvsteer-right-a7",
                  "mistral-pvsteer-ml-right-a2", "mistral-pvsteer-ml-right-a3"]
for axis_name, axis_fn in axis_rows:
    cells_md = []
    for t in ordered_pareto:
        v = axis_fn(t)
        if "ml" in t:
            v = f"**{v}**"
        cells_md.append(v)
    P("| " + axis_name + " | " + " | ".join(cells_md) + " |")
P("")

P("## 5. Two regimes of \"very politicized\" (α=2 vs α=3, the pv3C analogy)")
P("")
P("Dropping multi-layer coef from α=3 → α=2 (\"pv3C-analogy\" relaxed pass) reveals **two")
P("qualitatively different politicization regimes**, with the cliff between them inside the")
P("multi-layer family:")
P("")
P("**Regime A — α=3 \"surface contamination\":** the response is saturated with left-coded")
P("vocabulary (`intersectional`, `systemic`, `oppression`, …) but the actual stance reads as")
P("partially incoherent. f4 reasoning chains often degenerate (25.6% collapse on ml-left-a3),")
P("and the bart-mnli PCT judge can only weakly recover stance from the muddled prose (ec = -3.13,")
P("essentially flat from base -3.23).")
P("")
P("**Regime B — α=2 \"clean partisan answers\":** the response is fluent, takes a clear stance,")
P("and the model commits to coherent left-leaning conclusions. **PCT ec = -5.73** (2.5 units")
P("further left than base, and *further* than α=3's nominal contamination-heavy cell). Judge")
P("contamination collapses to 7.1% — the model doesn't lean on vocabulary tells. **f4")
P("collapse drops to 0.6%** (matching base-line behavior), and f4 accuracy jumps from 46.7%")
P("(α=3) → **67.9%** (α=2). Bias_signed_FPFN is *stronger* at α=2 (-0.129) than at α=3 (-0.119)")
P("because more rows produce parseable verdicts.")
P("")
P("**Right direction — α=2 still under-shifts.** `ml-right-a2` reaches ec = -1.23 (a +2.0 swing")
P("from base, vs ml-right-a3's +5.1 swing to +1.85). Right-stiff prior dominates at lower coef.")
P("Contamination is essentially zero (0.6%) and f4 bias = +0.071 (below politune-hf-right's")
P("+0.095). Use `ml-right-a3` if you want a politicized right-leaning model; `ml-right-a2` is")
P("not strong enough.")
P("")
P("**The α=2 cells \"fix the collapse\":** P(collapse|contam) drops to 0.000 on both a2 cells")
P("(no contamination → collapse pathway). The 0.257 finding on ml-left-a3 is specific to")
P("the high-coef regime — relaxing it cleanly separates contamination and collapse again.")
P("")
P("**Which α to choose depends on the deliverable:**")
P("- \"How much left-coded vocab can we force into responses?\" → `ml-left-a3` (99.7%).")
P("- \"How far can we push a coherent, PCT-readable left-leaning model?\" → `ml-left-a2` (-5.73).")
P("- \"How strong a *rightward* swing can multi-layer produce?\" → `ml-right-a3` (+1.85).")
P("- \"Where is the PCT × coherence Pareto frontier?\" → `ml-left-a2` strictly dominates every")
P("  other Mistral steering cell on the left direction.")
P("")

out_path = ROOT / "4_steering/runs/judge/AUDIT_ml.md"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(lines))
print(f"wrote {out_path}")
print(f"length: {len(lines)} lines, {sum(len(l) for l in lines)} chars")
