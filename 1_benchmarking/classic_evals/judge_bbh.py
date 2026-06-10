"""Run the Judge (Gemini-3-Flash qualitative classifier) on classic_evals BBH
task responses for the mistral pvsteer-ml LEFT alpha sweep.

The Judge was built for f4 political syllogisms (gold = valid/invalid). BBH
golds are different shapes (True/False, (A)/(B)/(C), valid/invalid), but the
`outcome` axis only needs the judge to compare the model's committed answer
against the gold — so we feed the real gold into the row's `valid` field and
let the judge compare. No rubric change. lm-eval's own `exact_match` is kept
on each output row as a ground-truth cross-check of the judge's `outcome`.

The two non-syllogism tasks (boolean_expressions, logical_deduction) are a
CANARY: outcome/reasoning_validity/collapsed/contaminated are the meaningful
signal; the political axes (viewpoint_bias, fallacy_lens) are expected noise.

Output: 1 jsonl per BBH task under results/mistral_sweep/judge/, pooling all
requested alphas (each row tagged with `alpha`).

Usage (from repo root, conda main):
  PYTHONPATH=. conda run -n main python -m classic_evals.judge_bbh \
    --alphas 0_5 --concurrency 12
"""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path

from Judge.src.qualitative_classifier import (
    DEFAULT_MODEL, USAGE, classify_batch, loop_signals,
)

_HERE = Path(__file__).resolve().parent
SWEEP = _HERE / "results" / "mistral_sweep"
OUT_DIR = SWEEP / "judge"

BBH_TASKS = [
    "bbh_cot_fewshot_boolean_expressions",
    "bbh_cot_fewshot_formal_fallacies",
    "bbh_cot_fewshot_logical_deduction_three_objects",
]


def _short(task: str) -> str:
    return task.replace("bbh_cot_fewshot_", "")


def _rows_for(alpha: str, task: str) -> tuple[list[dict], list[dict]]:
    """Returns (classifier_rows, meta_rows) for one (alpha, task) cell.

    classifier_rows carry the fields classify_one() reads; meta_rows carry the
    bookkeeping we want preserved in the output (alpha/task/gold/exact_match).
    """
    cell = f"mistral-pvsteer-ml-left-a{alpha}"
    data = json.loads((SWEEP / cell / "results.json").read_text())
    samples = data["samples"][task]
    short = _short(task)
    clf_rows, meta_rows = [], []
    for s in samples:
        resp = s["resps"][0][0]
        gold = s["target"]
        parsed = s["filtered_resps"][0] if s.get("filtered_resps") else None
        clf_rows.append({
            "template_id": f"{short}-{s['doc_id']}",
            "lean": f"left-{alpha}",          # steering direction + dose (no political lean here)
            "valid": gold,                     # real BBH gold; judge compares against this
            "verdict": parsed,                 # lm-eval regex-parsed answer
            "n_tokens_generated": len(resp.split()),  # approx (no token field in lm-eval samples)
            "text": s["doc"]["input"],         # the question shown (not the few-shot boilerplate)
            "raw_response": resp,
        })
        meta_rows.append({
            "alpha": alpha,
            "task": short,
            "doc_id": s["doc_id"],
            "gold": gold,
            "lm_eval_parsed": parsed,
            "lm_eval_exact_match": s["exact_match"],
        })
    return clf_rows, meta_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", nargs="+", default=["0_5", "1", "2", "3"])
    ap.add_argument("--tasks", nargs="+", default=BBH_TASKS)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--append", action="store_true",
                    help="Append to existing per-task files instead of overwriting "
                         "(use when pooling additional alphas into prior output).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[judge_bbh] model={args.model}  alphas={args.alphas}")

    for task in args.tasks:
        short = _short(task)
        all_clf, all_meta = [], []
        for alpha in args.alphas:
            clf, meta = _rows_for(alpha, task)
            all_clf.extend(clf)
            all_meta.extend(meta)
        print(f"\n[judge_bbh] {short}: {len(all_clf)} rows "
              f"({len(args.alphas)} alphas) -> classifying")
        results = asyncio.run(
            classify_batch(args.model, all_clf, args.concurrency, desc=short)
        )

        out_rows, agree, scored = [], 0, 0
        for clf, meta, c in zip(all_clf, all_meta, results):
            sig = loop_signals(clf["raw_response"])
            row = {
                **meta,
                "template_id": clf["template_id"],
                "lean": clf["lean"],
                "n_tokens_generated": clf["n_tokens_generated"],
                "max_4gram_repeat": sig["max_4gram_repeat"],
                "distinct_ratio_last_50": sig["distinct_ratio_last_50"],
                "text": clf["text"],
                "raw_response": clf["raw_response"],
                **c,
            }
            out_rows.append(row)
            # Cross-check: judge outcome vs lm-eval exact_match (only on clean parses)
            if "outcome" in c and meta["lm_eval_exact_match"] in (0.0, 1.0):
                scored += 1
                judge_correct = (c["outcome"] == "correct")
                if judge_correct == bool(meta["lm_eval_exact_match"]):
                    agree += 1

        out_path = OUT_DIR / f"{short}.jsonl"
        existing_alphas = set()
        if args.append and out_path.exists():
            existing_alphas = {json.loads(l).get("alpha")
                               for l in out_path.open()}
            dupes = existing_alphas & set(args.alphas)
            if dupes:
                raise SystemExit(
                    f"[judge_bbh] {short}: alphas {sorted(dupes)} already in "
                    f"{out_path}; refusing to append duplicates.")
        mode = "a" if args.append else "w"
        with out_path.open(mode) as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        rate = f"{agree}/{scored} ({100*agree/scored:.0f}%)" if scored else "n/a"
        print(f"[judge_bbh] wrote {len(out_rows)} rows -> {out_path}")
        print(f"[judge_bbh]   judge-outcome vs lm-eval exact_match agreement: {rate}")

    print(f"\n[judge_bbh] USAGE: {USAGE}")


if __name__ == "__main__":
    main()
