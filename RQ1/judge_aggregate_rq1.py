#!/usr/bin/env python3
"""Aggregate the RQ1 judge layer (copied + freshly run) into a contamination /
failure-mode summary, at two levels, for each family:

  * TASK level  -- one row per (cell, task): what happened on that task
  * POOLED      -- one row per cell pooling the 4 BBH-CoT tasks (the "full" view,
                   same shape as RQ3/judge_aggregate.py)

Cells are RQ1's 7 deployed cells (Base, RP-L/R, Steer-L/R, DPO-L/R), reading the
judge.jsonl that judge_copy.py imported or judge_rq1.py produced, in place under
RQ1/<fam>/<cell>/<task>/judge.jsonl.

Metrics (identical definitions to RQ3/judge_aggregate.py so the two are comparable):
  jAcc    -- % outcome==correct          (judge's own accuracy read)
  contam  -- % contaminated==True        (politics injected into a NEUTRAL task)
  jColl   -- % collapsed==True           (judge repetition/gibberish axis)
  noVerd  -- % outcome in {no_answer, off_format}
  + the 7-way primary_category composition.

Run (repo root):
  python RQ1/judge_aggregate_rq1.py            # mistral + llama
  python RQ1/judge_aggregate_rq1.py mistral
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "judge")

TASKS = ["boolean_expressions", "logical_deduction_three_objects",
         "web_of_lies", "navigate"]

# RQ1's 7 deployed cells -> (display label, on-disk cell dir under RQ1/<fam>/).
# right_2nd is Llama's deployed DPO-R checkpoint; right is Mistral's.
CELLS = {
    "mistral": [
        ("Base",    "base"),
        ("RP-L",    "roleplay/left"),
        ("RP-R",    "roleplay/right"),
        ("Steer-L", "steering/left_a2_5"),
        ("Steer-R", "steering/right_a3"),
        ("DPO-L",   "DPO/left"),
        ("DPO-R",   "DPO/right"),
    ],
    "llama": [
        ("Base",    "base"),
        ("RP-L",    "roleplay/left"),
        ("RP-R",    "roleplay/right"),
        ("Steer-L", "steering/left_a2_5"),
        ("Steer-R", "steering/right_a3"),
        ("DPO-L",   "DPO/left"),
        ("DPO-R",   "DPO/right_2nd"),
    ],
}

CATS = ["faithful_task_performance", "post_hoc_reasoning", "capability_error",
        "instruction_following_failure", "motivational_framing_bias",
        "viewpoint_bias", "generation_collapse"]
CAT_SHORT = {"faithful_task_performance": "faith", "post_hoc_reasoning": "posthoc",
             "capability_error": "caperr", "instruction_following_failure": "instr_f",
             "motivational_framing_bias": "mframe", "viewpoint_bias": "vbias",
             "generation_collapse": "gcollap"}


def load_rows(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path)]


def metrics(rows):
    n = len(rows)
    if n == 0:
        return None
    cat = Counter(r.get("primary_category") for r in rows)
    return {
        "n": n,
        "judge_acc": 100 * sum(1 for r in rows if r.get("outcome") == "correct") / n,
        "contam": 100 * sum(1 for r in rows if r.get("contaminated")) / n,
        "jcollapse": 100 * sum(1 for r in rows if r.get("collapsed")) / n,
        "no_verdict": 100 * sum(1 for r in rows if r.get("outcome") in ("no_answer", "off_format")) / n,
        "cat": {c: 100 * cat.get(c, 0) / n for c in CATS},
    }


def fmt(x):
    return ("%.0f" % x).rjust(7) if isinstance(x, (int, float)) else "   -- "


def header(first_col):
    return "  ".join([first_col.rjust(9), "n".rjust(5), "jAcc", "contam", "jColl", "noVerd",
                      *(CAT_SHORT[c].rjust(7) for c in CATS)])


def row_line(name, m):
    return "  ".join([name.rjust(9), str(m["n"]).rjust(5),
                      fmt(m["judge_acc"]), fmt(m["contam"]), fmt(m["jcollapse"]),
                      fmt(m["no_verdict"]), *(fmt(m["cat"][c]) for c in CATS)])


def aggregate_family(fam):
    blocks = [f"\n############ {fam} ############"]
    summary = {"task_level": {}, "pooled": {}}

    # ---- TASK level: per cell, a small block of its 4 tasks ----
    blocks.append("\n=== TASK level (per cell x task) ===")
    for label, reldir in CELLS[fam]:
        sub = [f"\n[{label}]  ({reldir})", header("task")]
        any_row = False
        for t in TASKS:
            rows = load_rows(os.path.join(HERE, fam, reldir, f"bbh_cot_fewshot_{t}", "judge.jsonl"))
            m = metrics(rows)
            if m:
                any_row = True
                sub.append(row_line(t.replace("logical_deduction_three_objects", "logic_3")
                                     .replace("boolean_expressions", "boolean")
                                     .replace("web_of_lies", "web_lies"), m))
                summary["task_level"][f"{fam}/{label}/{t}"] = m
            else:
                sub.append(f"{t[:9].rjust(9)}  (no judge.jsonl)")
        if any_row:
            blocks.append("\n".join(sub))

    # ---- POOLED: one row per cell over the 4 tasks ----
    blocks.append("\n=== POOLED (per cell, 4 BBH tasks) ===")
    blocks.append(header("cell"))
    for label, reldir in CELLS[fam]:
        rows = []
        for t in TASKS:
            rows.extend(load_rows(os.path.join(HERE, fam, reldir, f"bbh_cot_fewshot_{t}", "judge.jsonl")))
        m = metrics(rows)
        if m:
            blocks.append(row_line(label, m))
            summary["pooled"][f"{fam}/{label}"] = m
        else:
            blocks.append(f"{label.rjust(9)}  (no judge.jsonl)")

    return "\n".join(blocks), summary


def main():
    fams = sys.argv[1:] or ["mistral", "llama"]
    os.makedirs(OUT, exist_ok=True)
    legend = ("columns: jAcc=outcome correct%  contam=contaminated%  "
              "jColl=collapsed%(judge repetition axis)  noVerd=no_answer/off_format%\n"
              "primary_category %: faith faithful  posthoc post-hoc  caperr capability_error  "
              "instr_f instruction_following_failure  mframe motivational_framing_bias  "
              "vbias viewpoint_bias  gcollap generation_collapse")
    for fam in fams:
        if not os.path.isdir(os.path.join(HERE, fam)):
            continue
        text, summary = aggregate_family(fam)
        out_txt = legend + "\n" + text + "\n"
        print(out_txt)
        with open(os.path.join(OUT, f"judge_contam_{fam}.txt"), "w") as f:
            f.write(out_txt)
        with open(os.path.join(OUT, f"judge_summary_{fam}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[judge_aggregate_rq1] {fam}: wrote RQ1/judge/judge_contam_{fam}.txt "
              f"+ judge_summary_{fam}.json")


if __name__ == "__main__":
    main()
