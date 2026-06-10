#!/usr/bin/env python3
"""Aggregate the RQ3 judge outputs into a dose-response of failure modes.

Walks every `judge.jsonl` written by judge_rq3.py and, per cell (pooling the 4
BBH-CoT tasks), reports:

  * judge_acc  -- % outcome==correct (the judge's own accuracy read)
  * contam     -- % contaminated=True   (politics injected into a NEUTRAL task)
  * jcollapse  -- % collapsed=True       (repetition/gibberish, the judge's axis)
  * no_verdict -- % outcome in {no_answer, off_format} (failed to commit)
  * the 7-way primary_category composition

The headline RQ3 question the judge answers that consolidate.py's collapse%
cannot: as the knob turns, does the degradation stay a clean capability/format
failure, or does the intervention start *contaminating* neutral reasoning with
political content? The two axes (contam vs jcollapse) separate "floods politics"
from "loops/gibberish".

  python3 RQ3/judge_aggregate.py            # mistral
  python3 RQ3/judge_aggregate.py mistral llama
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

TASKS = ["boolean_expressions", "logical_deduction_three_objects",
         "web_of_lies", "navigate"]
STEER_GRID = [("a0_5", 0.5), ("a1", 1.0), ("a2", 2.0), ("a3", 3.0), ("a4", 4.0)]
DPO_GRID = [("s0_25", 0.25), ("s0_5", 0.5), ("s1_0", 1.0), ("s1_5", 1.5), ("s2", 2.0)]

CATS = ["faithful_task_performance", "post_hoc_reasoning", "capability_error",
        "instruction_following_failure", "motivational_framing_bias",
        "viewpoint_bias", "generation_collapse"]
CAT_SHORT = {"faithful_task_performance": "faith", "post_hoc_reasoning": "posthoc",
             "capability_error": "caperr", "instruction_following_failure": "instr_f",
             "motivational_framing_bias": "mframe", "viewpoint_bias": "vbias",
             "generation_collapse": "gcollap"}


def pool_cell(cell_dir):
    """Pool judge rows across the 4 tasks for one cell. Returns None if no judge
    output present."""
    rows = []
    for t in TASKS:
        p = os.path.join(cell_dir, f"bbh_cot_fewshot_{t}", "judge.jsonl")
        if os.path.exists(p):
            rows.extend(json.loads(l) for l in open(p))
    if not rows:
        return None
    n = len(rows)
    cat = Counter(r.get("primary_category") for r in rows)
    return {
        "n": n,
        "ntasks": sum(1 for t in TASKS
                      if os.path.exists(os.path.join(cell_dir, f"bbh_cot_fewshot_{t}", "judge.jsonl"))),
        "judge_acc": 100 * sum(1 for r in rows if r.get("outcome") == "correct") / n,
        "contam": 100 * sum(1 for r in rows if r.get("contaminated")) / n,
        "jcollapse": 100 * sum(1 for r in rows if r.get("collapsed")) / n,
        "no_verdict": 100 * sum(1 for r in rows if r.get("outcome") in ("no_answer", "off_format")) / n,
        "cat": {c: 100 * cat.get(c, 0) / n for c in CATS},
    }


def fmt(x):
    return ("%.0f" % x).rjust(7) if isinstance(x, (int, float)) else "   -- "


def dose(fam, method, lean, grid):
    base = pool_cell(os.path.join(RESULTS, fam, "base"))
    rows = [("0(base)", base)]
    for suf, _ in grid:
        rows.append((suf, pool_cell(os.path.join(RESULTS, fam, method, lean, suf))))
    rows = [(name, m) for name, m in rows if m]
    if len(rows) <= 1:
        return None

    out = [f"\n=== {fam}  {method}  {lean} ===",
           "  ".join(["strength".rjust(9), "n".rjust(5), "jAcc", "contam", "jCollap", "noVerd",
                      *(CAT_SHORT[c].rjust(7) for c in CATS)])]
    for name, m in rows:
        note = "" if m["ntasks"] == 4 else f"  ({m['ntasks']}tasks)"
        line = [name.rjust(9), str(m["n"]).rjust(5),
                fmt(m["judge_acc"]), fmt(m["contam"]), fmt(m["jcollapse"]), fmt(m["no_verdict"]),
                *(fmt(m["cat"][c]) for c in CATS)]
        out.append("  ".join(line) + note)
    return "\n".join(out)


def main():
    fams = sys.argv[1:] or ["mistral"]
    blocks, summary = [], {}
    legend = ("columns: jAcc=outcome correct%  contam=contaminated%  "
              "jCollap=collapsed%(judge repetition axis)  noVerd=no_answer/off_format%\n"
              "primary_category %: faith faithful  posthoc post-hoc  caperr capability_error  "
              "instr_f instruction_following_failure  mframe motivational_framing_bias  "
              "vbias viewpoint_bias  gcollap generation_collapse")
    for fam in fams:
        if not os.path.isdir(os.path.join(RESULTS, fam)):
            continue
        for lean in ("left", "right"):
            b = dose(fam, "steering", lean, STEER_GRID)
            if b:
                blocks.append(b)
        for lean in ("left", "right"):
            b = dose(fam, "dpo", lean, DPO_GRID)
            if b:
                blocks.append(b)
        # per-cell summary json
        for cell in (["base"]
                     + [f"steering/{l}/{s}" for l in ("left", "right") for s, _ in STEER_GRID]
                     + [f"dpo/{l}/{s}" for l in ("left", "right") for s, _ in DPO_GRID]):
            m = pool_cell(os.path.join(RESULTS, fam, cell))
            if m:
                summary[f"{fam}/{cell}"] = m

    text = legend + "\n" + "\n".join(blocks)
    print(text)
    with open(os.path.join(RESULTS, "judge_dose_response.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(RESULTS, "judge_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[judge_aggregate] wrote results/judge_dose_response.txt + judge_summary.json "
          f"({len(summary)} cells)")


if __name__ == "__main__":
    main()
