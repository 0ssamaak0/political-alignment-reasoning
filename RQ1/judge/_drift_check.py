#!/usr/bin/env python3
"""Instrument-drift check: re-judge an already-COPIED, contaminated cell-task TODAY
and compare per-row labels against the RQ3-original labels stored in its copied
judge.jsonl. If today's run uses the same yardstick (rubric + gemini-3-flash-preview
unchanged), contaminated should agree ~100% and outcome high-90s (RQ3 noted outcome
noise on collapsed rows). Validates that the run-cell numbers are comparable to the
copied-cell numbers. Scratch: writes nothing permanent."""
import asyncio, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent           # RQ1/judge
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "RQ3"))
sys.path.insert(0, str(REPO))
import judge_rq3 as J
from Judge.src.qualitative_classifier import (
    classify_batch, enable_prefix_cache, disable_prefix_cache, DEFAULT_MODEL)

CELL = REPO / "RQ1" / "mistral" / "steering" / "right_a3"
TASK = "navigate"                                # 37% contaminated in RQ3's labels

clf, meta = J.build_rows(CELL, "drift-check", "right", TASK)
old = {json.loads(l)["doc_id"]: json.loads(l)
       for l in (CELL / f"bbh_cot_fewshot_{TASK}" / "judge.jsonl").open()}

enable_prefix_cache(DEFAULT_MODEL)
try:
    res = asyncio.run(classify_batch(DEFAULT_MODEL, clf, 16, desc="drift"))
finally:
    disable_prefix_cache()

n = cont_agree = out_scored = out_agree = err = 0
old_cont = new_cont = 0
for c, m in zip(res, meta):
    if "classifier_error" in c:
        err += 1; continue
    o = old[m["doc_id"]]
    n += 1
    old_cont += int(bool(o.get("contaminated")))
    new_cont += int(bool(c.get("contaminated")))
    if bool(o.get("contaminated")) == bool(c.get("contaminated")):
        cont_agree += 1
    if o.get("outcome") and c.get("outcome"):
        out_scored += 1
        out_agree += int(o["outcome"] == c["outcome"])

print(f"\n=== drift check: mistral/steering/right_a3/{TASK} (n={n}, {err} errored) ===")
print(f"contaminated rate  RQ3-stored {100*old_cont/n:.0f}%  vs  today {100*new_cont/n:.0f}%")
print(f"contaminated per-row agreement: {cont_agree}/{n} ({100*cont_agree/n:.0f}%)")
print(f"outcome per-row agreement:      {out_agree}/{out_scored} ({100*out_agree/out_scored:.0f}%)")
