"""Aggregate Judge label jsonls into a long-format qualitative summary.

This is the QUALITATIVE counterpart to compute_bias.py. Where compute_bias
scores the validity verdict (FP/FN bias, accuracy, unmappable), this scans
the LLM-as-judge labels written by Judge.src.qualitative_classifier into
judges/<cell>.jsonl and aggregates the 4-axis schema:

    Axis 1  outcome             correct | wrong | off_format | no_answer
    Axis 2  contaminated (bool) group-coded vocab imported into the rationale
            collapsed    (bool) generative pathology (repetition / drift)
    Axis 3  reasoning_validity  valid | invalid | opaque | n/a
    (cat)   primary_category    7-way tuple of the axes above
    Axis 4  fallacy_lens        failure-mechanism tag (nullable)

Rows that the classifier failed on carry a `classifier_error` key and null
judge fields; they are excluded from every rate (the per-cell denominator is
`n_judged = N - n_error`, reported explicitly).

Output is LONG format: one row per (cell x metric), so the same file can be
pivoted on either axis. Reads every judges/*.jsonl unless --names limits it.

Usage (from 1_benchmarking/):
    conda run -n main python -m G_K_assessing_bias.aggregate_judges
    conda run -n main python -m G_K_assessing_bias.aggregate_judges --names mistral-base-nosys
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from G_K_assessing_bias.gk_paths import JUDGES_DIR as JUDGE_DIR, judge_jsonl

OUTCOMES = ["correct", "wrong", "off_format", "no_answer"]
VALIDITIES = ["valid", "invalid", "opaque", "n/a"]
CATEGORIES = [
    "faithful_task_performance",
    "post_hoc_reasoning",
    "capability_error",
    "instruction_following_failure",
    "motivational_framing_bias",
    "viewpoint_bias",
    "generation_collapse",
]
FALLACIES = [
    "premise_truth_conflation",
    "motivational_reasoning",
    "equivocation",
    "false_dilemma",
    "illicit_premise_insertion",
    "token_bias_shortcut",
]


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else float("nan")


def summarize(cell: str, rows: list[dict]) -> dict:
    n = len(rows)
    judged = [r for r in rows if "classifier_error" not in r]
    nj = len(judged)
    oc = Counter(r["outcome"] for r in judged)
    rv = Counter(r["reasoning_validity"] for r in judged)
    pc = Counter(r["primary_category"] for r in judged)
    fl = Counter(r["fallacy_lens"] for r in judged)

    out = {
        "cell": cell,
        "N": n,
        "n_judged": nj,
        "n_error": n - nj,
        "contaminated_rate": _rate(sum(1 for r in judged if r["contaminated"]), nj),
        "collapsed_rate": _rate(sum(1 for r in judged if r["collapsed"]), nj),
    }
    for k in OUTCOMES:
        out[f"outcome.{k}"] = _rate(oc.get(k, 0), nj)
    for k in VALIDITIES:
        out[f"rv.{k}"] = _rate(rv.get(k, 0), nj)
    for k in CATEGORIES:
        out[f"cat.{k}"] = _rate(pc.get(k, 0), nj)
    for k in FALLACIES:
        out[f"fallacy.{k}"] = _rate(fl.get(k, 0), nj)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--names", nargs="*", default=None,
                    help="cell names to include (default: all judges/*.jsonl)")
    args = ap.parse_args()

    if args.names:
        files = [judge_jsonl(n) for n in args.names]
    else:
        files = sorted(p for p in JUDGE_DIR.rglob("*.jsonl")
                       if p.stem not in {"judges_summary", "judges_summary_long"})

    rows = []
    for fp in files:
        if not fp.exists():
            print(f"skipping missing {fp.name}")
            continue
        records = [json.loads(line) for line in fp.open() if line.strip()]
        rows.append(summarize(fp.stem, records))

    if not rows:
        print(f"No judge jsonls found in {JUDGE_DIR}")
        return

    wide = pd.DataFrame(rows)
    out_wide = JUDGE_DIR / "judges_summary.csv"
    wide.to_csv(out_wide, index=False)

    # Long format: one row per (cell, metric, value) for easy pivoting.
    long = wide.melt(id_vars=["cell", "N", "n_judged", "n_error"],
                     var_name="metric", value_name="value")
    out_long = JUDGE_DIR / "judges_summary_long.csv"
    long.to_csv(out_long, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 100)
    print(wide.to_string(index=False))
    print(f"\nwrote {out_wide}\nwrote {out_long}")


if __name__ == "__main__":
    main()
