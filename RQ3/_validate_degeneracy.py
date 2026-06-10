#!/usr/bin/env python3
"""Validate the from-text degeneracy reimplementation against Mistral judge.jsonl.

The Llama analysis computes n_tokens_generated / max_4gram_repeat /
distinct_ratio_last_50 DIRECTLY from samples.jsonl (resps[0][0]) because Llama has
no judge.jsonl yet. This script proves the reimplementation reproduces the judge's
exact values on Mistral cells that DO have judge.jsonl, by:
  (a) confirming judge.raw_response == samples.resps[0][0] (same text), and
  (b) recomputing the three fields from that text and matching judge's stored values.

Definitions copied verbatim from Judge/src/qualitative_classifier.py:loop_signals
and 1_benchmarking/classic_evals/judge_bbh.py (n_tokens = len(resp.split())).
"""
import os, sys, json
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
MIS = os.path.join(HERE, "results", "mistral")
TASKS = ["bbh_cot_fewshot_boolean_expressions",
         "bbh_cot_fewshot_logical_deduction_three_objects",
         "bbh_cot_fewshot_web_of_lies", "bbh_cot_fewshot_navigate"]


def loop_signals(text):
    """Verbatim copy of Judge/src/qualitative_classifier.py:loop_signals."""
    tokens = text.split()
    if len(tokens) < 4:
        return {"max_4gram_repeat": 0, "distinct_ratio_last_50": 1.0}
    fourgrams = [" ".join(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    max_rep = max(Counter(fourgrams).values()) if fourgrams else 0
    last = tokens[-50:]
    distinct_ratio = len(set(last)) / max(len(last), 1)
    return {"max_4gram_repeat": max_rep,
            "distinct_ratio_last_50": round(distinct_ratio, 3)}


def n_tokens(text):
    return len(text.split())


CELLS = ["steering/left/a4", "steering/right/a4", "dpo/right/s2",
         "dpo/left/s1_5", "base"]


def main():
    total = mism_text = mism_tok = mism_rep = mism_dist = checked = 0
    for cell in CELLS:
        cdir = os.path.join(MIS, *cell.split("/"))
        for task in TASKS:
            sp = os.path.join(cdir, task, "samples.jsonl")
            jp = os.path.join(cdir, task, "judge.jsonl")
            if not (os.path.exists(sp) and os.path.exists(jp)):
                continue
            samples = {d["doc_id"]: (d["resps"][0][0] if d.get("resps") else "")
                       for d in (json.loads(l) for l in open(sp))}
            for r in (json.loads(l) for l in open(jp)):
                total += 1
                jtext = r.get("raw_response") or ""
                stext = samples.get(r["doc_id"])
                if stext is None:
                    continue
                checked += 1
                if jtext != stext:
                    mism_text += 1
                    continue  # text differs -> can't compare derived fields fairly
                sig = loop_signals(stext)
                if n_tokens(stext) != r.get("n_tokens_generated"):
                    mism_tok += 1
                if sig["max_4gram_repeat"] != r.get("max_4gram_repeat"):
                    mism_rep += 1
                if sig["distinct_ratio_last_50"] != r.get("distinct_ratio_last_50"):
                    mism_dist += 1
    print(f"rows total={total} checked(doc_id matched)={checked}")
    print(f"raw_response==resps[0][0] mismatches: {mism_text}")
    print(f"n_tokens_generated mismatches:        {mism_tok}")
    print(f"max_4gram_repeat mismatches:          {mism_rep}")
    print(f"distinct_ratio_last_50 mismatches:    {mism_dist}")
    ok = (mism_text == 0 and mism_tok == 0 and mism_rep == 0 and mism_dist == 0)
    print("VALIDATION:", "PASS — reimplementation reproduces judge fields exactly"
          if ok else "FAIL — see mismatches above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
