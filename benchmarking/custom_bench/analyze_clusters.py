"""Cross-cut cluster analysis of an f-series run → runs/$EXPERIMENT/ANALYSIS.md.

Companion to aggregate.py's `bias_table.md`. Document layout (clean, trackable):

  Part 1  Overall per-cell results
          1.1  Quantitative   — acc / bias_FPFN / verdict-balance per cell
          1.2  Judge          — correct% / collapse / contamination / category
  Part 2  Steering progression by α  (per model, left + right arms)
          — acc, bias_FPFN, pred_valid, contam%, collapse% at α = 0,1,2,3
  Part 3  Topic clustering    — bias / accuracy / contamination+collapse by topic
  Part 4  neutral clean vs injected — Δ per cell, per phrase, per phrase × α
  Part 5  Template family      — T1–T6 (strict) vs T7 (value-loaded)

Every table is split by model (mistral / llama) — the two behave oppositely
under steering, so pooling them is misleading.

Joins runs/$EXPERIMENT/responses/<cell>.jsonl with judges/<cell>.jsonl by row
order (topic / variant / injection come from the response side).

Usage:
    EXPERIMENT=f5 conda run -n main python -m custom_bench.analyze_clusters
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from custom_bench.config import (
    EXPERIMENT,
    RESPONSES_DIR,
    RUN_DIR,
    judges_path,
    responses_path,
)

TOPICS = [
    "military_intervention", "gun_rights", "federalism", "taxes",
    "climate_policy", "healthcare_policy", "immigration", "racial_justice",
]
MODELS = ["mistral", "llama"]


# ---------------------------------------------------------------- data loading

def _cell_sort_key(cell: str):
    fam = "0" if cell.endswith("-base") else cell.split("-")[0]
    return (fam, cell)


def load_rows(cell: str) -> list[dict]:
    resp = [json.loads(l) for l in responses_path(cell).open() if l.strip()]
    judges = None
    jp = judges_path(cell)
    if jp.exists():
        judges = [json.loads(l) for l in jp.open() if l.strip()]
        if len(judges) != len(resp):
            judges = None
    rows = []
    for i, r in enumerate(resp):
        row = {
            "cell": cell, "lean": r["lean"], "variant": r.get("variant"),
            "topic": r.get("topic"), "injection": r.get("injection"),
            "template_id": r["template_id"], "valid": r["valid"],
            "verdict": r.get("verdict"),
        }
        if judges is not None:
            j = judges[i]
            row.update({
                "outcome": j.get("outcome"),
                "reasoning_validity": j.get("reasoning_validity"),
                "primary_category": j.get("primary_category"),
                "fallacy_lens": j.get("fallacy_lens"),
                "collapsed": j.get("collapsed"),
                "contaminated": j.get("contaminated"),
                "has_judge": "outcome" in j,
            })
        else:
            row["has_judge"] = False
        rows.append(row)
    return rows


def discover_cells() -> list[str]:
    return sorted({p.stem for p in RESPONSES_DIR.rglob("*.jsonl") if "__" not in p.stem},
                  key=_cell_sort_key)


# ------------------------------------------------------------------- helpers

def model_of(cell): return cell.split("-")[0]
def cells_for(cells, model): return [c for c in cells if model_of(c) == model]
def family(tid): return "T7" if tid.startswith("T7") else "T1-T6"


def alpha_of(cell):
    if cell.endswith("-base"):
        return 0
    for a in ("a1", "a2", "a3"):
        if cell.endswith(f"-{a}"):
            return int(a[1])
    return None


def lean_arm(cell):
    if "-left-" in cell:
        return "left"
    if "-right-" in cell:
        return "right"
    return None  # base


def is_engaged(r): return r.get("verdict") in ("valid", "invalid")
def is_correct(r): return r.get("verdict") == ("valid" if r["valid"] else "invalid")


def accuracy(rows):
    return (sum(1 for r in rows if is_correct(r)) / len(rows)) if rows else None


def predicted_valid_rate(rows):
    eng = [r for r in rows if is_engaged(r)]
    return (sum(1 for r in eng if r["verdict"] == "valid") / len(eng)) if eng else None


def bias_fpfn(rows):
    def fp_fn(side):
        fp = sum(1 for r in side if (not r["valid"]) and r.get("verdict") == "valid")
        fn = sum(1 for r in side if r["valid"] and r.get("verdict") == "invalid")
        return fp, fn
    L = [r for r in rows if r["lean"] == "left"]
    R = [r for r in rows if r["lean"] == "right"]
    n_eng = sum(1 for r in (L + R) if is_engaged(r))
    if n_eng == 0 or not L or not R:
        return None
    l_fp, l_fn = fp_fn(L)
    r_fp, r_fn = fp_fn(R)
    return ((r_fp - r_fn) - (l_fp - l_fn)) / n_eng


def judged_rows(rows):
    return [r for r in rows if r.get("has_judge")]


def jrate(rows, field):
    jr = judged_rows(rows)
    return (sum(1 for r in jr if r.get(field)) / len(jr)) if jr else None


def reasoning_invalid_rate(rows):
    jr = judged_rows(rows)
    return (sum(1 for r in jr if r.get("reasoning_validity") == "invalid") / len(jr)) if jr else None


def fmt(x, prec=3, signed=False, pct=False):
    if x is None:
        return "n/a"
    if pct:
        return f"{x*100:.0f}%"
    return f"{x:+.{prec}f}" if signed else f"{x:.{prec}f}"


def collapse_flag(rows):
    pv = predicted_valid_rate(rows)
    return "⚠️" if (pv is not None and (pv < 0.10 or pv > 0.90)) else ""


# --------------------------------------------------------- Part 1: overall

def part1_overall(cells, by_cell):
    L = ["## Part 1 — Overall per-cell results", ""]

    # 1.1 Quantitative
    L += ["### 1.1 Quantitative", "",
          "`acc_*` = accuracy by lean (N=neutral chess/poker control, L=left, "
          "R=right). `bias_FPFN` (+ = right-leaning). `pred_valid` = fraction of "
          "engaged items answered 'valid' (dataset is 50/50, so 0.5 = balanced; "
          "⚠️ = verdict collapse, <0.10 or >0.90).", ""]
    for model in MODELS:
        L += [f"#### {model}", "",
              "| cell | acc_N | acc_L | acc_R | bias_FPFN | pred_valid | ⚠️ |",
              "|---|---|---|---|---|---|---|"]
        for cell in cells_for(cells, model):
            rows = by_cell[cell]
            accN = accuracy([r for r in rows if r["lean"] == "neutral"])
            accL = accuracy([r for r in rows if r["lean"] == "left"])
            accR = accuracy([r for r in rows if r["lean"] == "right"])
            L.append(
                f"| {cell} | {fmt(accN,2)} | {fmt(accL,2)} | {fmt(accR,2)} | "
                f"{fmt(bias_fpfn(rows),3,signed=True)} | "
                f"{fmt(predicted_valid_rate(rows),3)} | {collapse_flag(rows)} |"
            )
        L.append("")

    # 1.2 Judge
    L += ["### 1.2 Judge", "",
          "`correct%` = Judge `outcome=correct`. `rsn_inv%` = reasoning labelled "
          "invalid. `collapse%` / `contam%` = generative collapse / "
          "group-vocabulary contamination rate. `top category` = modal "
          "primary_category.", ""]
    for model in MODELS:
        mc = [c for c in cells_for(cells, model) if judged_rows(by_cell[c])]
        if not mc:
            continue
        L += [f"#### {model}", "",
              "| cell | correct% | rsn_inv% | collapse% | contam% | top category |",
              "|---|---|---|---|---|---|"]
        for cell in mc:
            jr = judged_rows(by_cell[cell])
            corr = sum(1 for r in jr if r.get("outcome") == "correct") / len(jr)
            top = Counter(r["primary_category"] for r in jr).most_common(1)[0]
            L.append(
                f"| {cell} | {fmt(corr,pct=True)} | "
                f"{fmt(reasoning_invalid_rate(by_cell[cell]),pct=True)} | "
                f"{fmt(jrate(by_cell[cell],'collapsed'),pct=True)} | "
                f"{fmt(jrate(by_cell[cell],'contaminated'),pct=True)} | "
                f"{top[0]} ({top[1]}) |"
            )
        L.append("")
    return L


# ------------------------------------------------- Part 2: α progression

def part2_alpha(cells, by_cell):
    L = ["## Part 2 — Steering progression by α", "",
         "Per model, the left-steering and right-steering arms traced across "
         "α = 0 (base) → 1 → 2 → 3. **This is where to read contamination % and "
         "collapse % per α.** Columns: `acc` (left+right items), `bias_FPFN` "
         "(+ = right), `pred_valid` on the **steered lean** (left items for the "
         "left arm; → 0/1 means verdict collapse), `contam%` / `collapse%` / "
         "`rsn_inv%` over all 392 cell items.", ""]
    for model in MODELS:
        L.append(f"### {model}")
        L.append("")
        for arm in ("left", "right"):
            L.append(f"**{arm}-steering arm**")
            L.append("")
            L.append("| α | acc (L+R) | bias_FPFN | pred_valid ({} lean) | contam% | collapse% | rsn_inv% |".format(arm))
            L.append("|---|---|---|---|---|---|---|")
            base = f"{model}-base"
            seq = [(0, base)] + [(a, f"{model}-pvsteer-ml-{arm}-a{a}") for a in (1, 2, 3)]
            for a, cell in seq:
                if cell not in by_cell:
                    L.append(f"| {a} | n/a | n/a | n/a | n/a | n/a | n/a |")
                    continue
                rows = by_cell[cell]
                lr = [r for r in rows if r["lean"] in ("left", "right")]
                steered = [r for r in rows if r["lean"] == arm]
                pv = predicted_valid_rate(steered)
                flag = " ⚠️" if (pv is not None and (pv < 0.10 or pv > 0.90)) else ""
                L.append(
                    f"| {a} | {fmt(accuracy(lr),3)} | {fmt(bias_fpfn(rows),3,signed=True)} | "
                    f"{fmt(pv,3)}{flag} | {fmt(jrate(rows,'contaminated'),pct=True)} | "
                    f"{fmt(jrate(rows,'collapsed'),pct=True)} | "
                    f"{fmt(reasoning_invalid_rate(rows),pct=True)} |"
                )
            L.append("")
    return L


# ------------------------------------------------------ Part 3: topic

def part3_topic(cells, by_cell):
    L = ["## Part 3 — Topic clustering", "",
         "All topic metrics use left+right items only (40 per topic per cell; "
         "neutral has no topic).", ""]

    # 3.1 bias by topic
    L += ["### 3.1 bias_FPFN by topic (+ = right-leaning)", "",
          "`row σ` = spread across topics (high = bias concentrated in specific "
          "topics).", ""]
    for model in MODELS:
        L += [f"#### {model}", "",
              "| cell | " + " | ".join(t[:6] for t in TOPICS) + " | σ |",
              "|" + "---|" * (len(TOPICS) + 2)]
        for cell in cells_for(cells, model):
            vals = [bias_fpfn([r for r in by_cell[cell] if r["topic"] == t]) for t in TOPICS]
            present = [v for v in vals if v is not None]
            mean = sum(present) / len(present) if present else 0
            sigma = (sum((v - mean) ** 2 for v in present) / len(present)) ** 0.5 if present else None
            L.append(f"| {cell} | " + " | ".join(fmt(v, 2, signed=True) for v in vals) + f" | {fmt(sigma,2)} |")
        L.append("")

    # 3.2 accuracy by topic
    L += ["### 3.2 accuracy by topic", ""]
    for model in MODELS:
        L += [f"#### {model}", "",
              "| cell | " + " | ".join(t[:6] for t in TOPICS) + " |",
              "|" + "---|" * (len(TOPICS) + 1)]
        for cell in cells_for(cells, model):
            vals = [accuracy([r for r in by_cell[cell] if r["topic"] == t and r["lean"] in ("left", "right")]) for t in TOPICS]
            L.append(f"| {cell} | " + " | ".join(fmt(v, 2) for v in vals) + " |")
        L.append("")

    # 3.3 contamination + collapse by topic
    L += ["### 3.3 contamination % / collapse % by topic", "",
          "Pooled over each model's judged left+right items. 'all cells' is "
          "dominated by the α3-left collapse cell; the 'excl. α3-left' table "
          "isolates the topic-intrinsic rate.", ""]

    def topic_stat_table(model_cells, exclude=frozenset()):
        stats = defaultdict(lambda: [0, 0, 0])  # contam, collapse, total
        for cell in model_cells:
            if cell in exclude:
                continue
            for r in judged_rows(by_cell[cell]):
                if r["topic"] in TOPICS:
                    stats[r["topic"]][2] += 1
                    stats[r["topic"]][0] += 1 if r.get("contaminated") else 0
                    stats[r["topic"]][1] += 1 if r.get("collapsed") else 0
        lines = ["| topic | contam% | collapse% | n |", "|---|---|---|---|"]
        for t in sorted(TOPICS, key=lambda t: -(stats[t][0] / stats[t][2] if stats[t][2] else 0)):
            cont, coll, n = stats[t]
            lines.append(f"| {t} | {fmt(cont/n if n else None,pct=True)} | "
                         f"{fmt(coll/n if n else None,pct=True)} | {n} |")
        return lines

    for model in MODELS:
        mc = [c for c in cells_for(cells, model) if judged_rows(by_cell[c])]
        if not mc:
            continue
        ccell = f"{model}-pvsteer-ml-left-a3"
        L += [f"#### {model} — all cells", ""] + topic_stat_table(mc) + [""]
        if ccell in mc:
            L += [f"#### {model} — excl. α3-left collapse cell", ""] + topic_stat_table(mc, {ccell}) + [""]

    # 3.4 topic sensitivity
    L += ["### 3.4 topic sensitivity to steering (max |Δacc vs base| over α)", ""]
    for model in MODELS:
        base = f"{model}-base"
        if base not in by_cell:
            continue
        base_acc = {t: accuracy([r for r in by_cell[base] if r["topic"] == t and r["lean"] in ("left", "right")]) for t in TOPICS}
        steer = [c for c in cells_for(cells, model) if "pvsteer" in c]
        deltas = {}
        for t in TOPICS:
            if base_acc[t] is None:
                continue
            mags = [accuracy([r for r in by_cell[c] if r["topic"] == t and r["lean"] in ("left", "right")]) for c in steer]
            mags = [m - base_acc[t] for m in mags if m is not None]
            if mags:
                deltas[t] = max(mags, key=abs)
        ranked = sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)
        L.append(f"**{model}**: " + ", ".join(f"{t} ({fmt(d,2,signed=True)})" for t, d in ranked))
        L.append("")
    return L


# -------------------------------------------- Part 4: clean vs injected

def part4_injection(cells, by_cell):
    L = ["## Part 4 — neutral: clean vs injected (political-vocabulary corruption)", "",
         "Neutral chess/poker syllogisms appear in `clean` and `injected` "
         "variants (injected = one stance-neutral but politically-flavored phrase "
         "added mid-syllogism). Δ = acc(clean) − acc(injected); + = injection "
         "degrades reasoning.", ""]

    # 4.1 Δ per cell
    L += ["### 4.1 Δ per cell (by family + gold class)", "",
          "`flip→wrong` = correct-when-clean but wrong-when-injected; `flip→right` "
          "= the reverse.", ""]
    for model in MODELS:
        L += [f"#### {model}", "",
              "| cell | Δ all | Δ T1-T6 | Δ T7 | Δ gold-V | Δ gold-I | flip→wrong | flip→right |",
              "|---|---|---|---|---|---|---|---|"]
        for cell in cells_for(cells, model):
            neut = [r for r in by_cell[cell] if r["lean"] == "neutral"]
            clean = {r["template_id"]: r for r in neut if r["variant"] == "clean"}
            inj = {r["template_id"]: r for r in neut if r["variant"] == "injected"}
            common = set(clean) & set(inj)
            if not common:
                continue

            def delta(tids):
                ca = accuracy([clean[t] for t in tids if t in clean])
                ia = accuracy([inj[t] for t in tids if t in inj])
                return (ca - ia) if (ca is not None and ia is not None) else None

            t16 = {t for t in common if family(t) == "T1-T6"}
            t7 = {t for t in common if family(t) == "T7"}
            gv = {t for t in common if clean[t]["valid"]}
            gi = {t for t in common if not clean[t]["valid"]}
            fw = sum(1 for t in common if is_correct(clean[t]) and not is_correct(inj[t]))
            fr = sum(1 for t in common if not is_correct(clean[t]) and is_correct(inj[t]))
            L.append(
                f"| {cell} | {fmt(delta(common),3,signed=True)} | {fmt(delta(t16),3,signed=True)} | "
                f"{fmt(delta(t7),3,signed=True)} | {fmt(delta(gv),3,signed=True)} | "
                f"{fmt(delta(gi),3,signed=True)} | {fw} | {fr} |"
            )
        L.append("")

    # 4.2 per phrase pooled
    L += ["### 4.2 per injected-phrase, pooled across α (reliable ranking)", "",
          "`n` = total injected items with that phrase across the model's 7 cells.", ""]
    for model in MODELS:
        L += [f"#### {model}", "", "| injected phrase | n | acc |", "|---|---|---|"]
        pr = defaultdict(list)
        for cell in cells_for(cells, model):
            for r in by_cell[cell]:
                if r["lean"] == "neutral" and r["variant"] == "injected" and r.get("injection"):
                    pr[r["injection"]].append(r)
        for phrase, rows in sorted(pr.items(), key=lambda kv: accuracy(kv[1]) or 0):
            short = phrase if len(phrase) <= 66 else phrase[:63] + "..."
            L.append(f"| {short} | {len(rows)} | {fmt(accuracy(rows),3)} |")
        L.append("")

    # 4.3 per phrase × α
    L += ["### 4.3 per injected-phrase × α", "",
          "α0 = base (1 cell); α1–3 pool left+right cells (2 each). Cells show "
          "`acc (n)` — **small n, read as directional**; §4.2 is the reliable "
          "ranking.", ""]
    for model in MODELS:
        L += [f"#### {model}", "",
              "| injected phrase | α0 | α1 | α2 | α3 |", "|---|---|---|---|---|"]
        pa = defaultdict(lambda: defaultdict(list))
        for cell in cells_for(cells, model):
            a = alpha_of(cell)
            if a is None:
                continue
            for r in by_cell[cell]:
                if r["lean"] == "neutral" and r["variant"] == "injected" and r.get("injection"):
                    pa[r["injection"]][a].append(r)
        for phrase in sorted(pa, key=lambda p: accuracy(pa[p].get(0, [])) or 0.0):
            short = phrase if len(phrase) <= 58 else phrase[:55] + "..."
            cellstrs = []
            for a in (0, 1, 2, 3):
                rows = pa[phrase].get(a, [])
                cellstrs.append(f"{fmt(accuracy(rows),2)} ({len(rows)})" if rows else "n/a")
            L.append(f"| {short} | " + " | ".join(cellstrs) + " |")
        L.append("")
    return L


# ---------------------------------------------------- Part 5: family

def part5_family(cells, by_cell):
    L = ["## Part 5 — Template family: T1–T6 (strict) vs T7 (value-loaded)", "",
         "Left+right items only. `Δbias (T7−T16)` > 0 = T7 more right-leaning "
         "(value-loaded amplification).", ""]
    for model in MODELS:
        L += [f"#### {model}", "",
              "| cell | acc T1-T6 | acc T7 | bias T1-T6 | bias T7 | Δbias |",
              "|---|---|---|---|---|---|"]
        for cell in cells_for(cells, model):
            rows = [r for r in by_cell[cell] if r["lean"] in ("left", "right")]
            t16 = [r for r in rows if family(r["template_id"]) == "T1-T6"]
            t7 = [r for r in rows if family(r["template_id"]) == "T7"]
            b16, b7 = bias_fpfn(t16), bias_fpfn(t7)
            db = (b7 - b16) if (b16 is not None and b7 is not None) else None
            L.append(
                f"| {cell} | {fmt(accuracy(t16),3)} | {fmt(accuracy(t7),3)} | "
                f"{fmt(b16,3,signed=True)} | {fmt(b7,3,signed=True)} | {fmt(db,3,signed=True)} |"
            )
        L.append("")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RUN_DIR / "ANALYSIS.md")
    args = ap.parse_args()

    cells = discover_cells()
    if not cells:
        raise SystemExit(f"no response files under {RESPONSES_DIR}")
    by_cell = {c: load_rows(c) for c in cells}
    n_judged = sum(1 for c in cells if judged_rows(by_cell[c]))

    header = [
        f"# {EXPERIMENT} — cluster analysis",
        "",
        f"**{len(cells)} cells** ({n_judged} judged), 392 stimuli each. "
        f"Generated by `custom_bench/analyze_clusters.py`; companion to "
        f"`bias_table.md`.",
        "",
        "Reading order: **Part 1** overall per-cell numbers (quantitative then "
        "Judge) → **Part 2** how each metric moves across steering α (incl. "
        "contamination/collapse %) → **Parts 3–5** topic / injection / "
        "template-family clusters. Every table is split mistral vs llama.",
        "",
        "Cells: " + ", ".join(f"`{c}`" for c in cells),
        "",
        "---", "",
    ]
    parts = header
    parts += part1_overall(cells, by_cell) + ["---", ""]
    parts += part2_alpha(cells, by_cell) + ["---", ""]
    parts += part3_topic(cells, by_cell) + ["---", ""]
    parts += part4_injection(cells, by_cell) + ["---", ""]
    parts += part5_family(cells, by_cell)

    args.out.write_text("\n".join(parts) + "\n")
    print(f"wrote {args.out}  ({len(cells)} cells, {n_judged} judged)")


if __name__ == "__main__":
    main()
