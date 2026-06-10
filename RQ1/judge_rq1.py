#!/usr/bin/env python3
"""Run the Gemini judge on the RQ1 deployed cells that judge_copy.py could NOT
import, because RQ3 has no byte-identical, already-judged twin for them:

  * roleplay (RP-L, RP-R)   -- RQ3 has no roleplay arm (no strength dial)
  * Steer-L (alpha 2.5)     -- off RQ3's steering grid on Mistral; unjudged on Llama
  * Llama Steer-R (alpha 3) -- RQ3's copy differs on 13/250 rows (bf16 greedy
                               nondeterminism), so judge RQ1's OWN generations to
                               stay exact rather than borrow 237 and re-judge 13

This is a thin driver: it reuses judge_rq3.run_cell_task / backfill_cell_task
verbatim (same rubric, model, prefix cache, resume cache, escalation ladder) and
only feeds them RQ1 paths + the correct steering/roleplay lean. judge.jsonl is
written next to each samples.jsonl, exactly as in RQ3.

Run (repo root, conda main):
  conda run -n main python RQ1/judge_rq1.py --family mistral --dry-run
  conda run -n main python RQ1/judge_rq1.py --family mistral
  conda run -n main python RQ1/judge_rq1.py --family mistral --backfill   # tail mop-up
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../polireason/RQ1
REPO = HERE.parent                               # .../polireason
sys.path.insert(0, str(REPO / "RQ3"))            # import judge_rq3
sys.path.insert(0, str(REPO))                    # Judge package

import judge_rq3 as J                            # noqa: E402
from Judge.src.qualitative_classifier import (   # noqa: E402
    DEFAULT_MODEL, USAGE, enable_prefix_cache, disable_prefix_cache,
)

# (cell_label, rq1_cell_reldir, lean) for the cells judge_copy.py refuses.
# lean = steering DIRECTION / roleplay persona (political-axis canary on neutral tasks).
RUN_MAP = {
    "mistral": [
        ("steer-left-a2_5", "steering/left_a2_5", "left"),
        ("roleplay-left",   "roleplay/left",      "left"),
        ("roleplay-right",  "roleplay/right",     "right"),
    ],
    "llama": [
        ("steer-left-a2_5", "steering/left_a2_5", "left"),
        ("steer-right-a3",  "steering/right_a3",  "right"),
        ("roleplay-left",   "roleplay/left",      "left"),
        ("roleplay-right",  "roleplay/right",     "right"),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="mistral", choices=["mistral", "llama"])
    ap.add_argument("--cells", nargs="+", default=None,
                    help="cell labels to restrict to (default: all RUN_MAP cells).")
    ap.add_argument("--tasks", nargs="+", default=J.TASKS,
                    help="short BBH task names (no bbh_cot_fewshot_ prefix).")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="cap rows per cell x task (pilot).")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--backfill", action="store_true",
                    help="sync mop-up of missing rows (the async-empty collapse tail).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = RUN_MAP[args.family]
    if args.cells:
        spec = [s for s in spec if s[0] in set(args.cells)]
    use_cache = not args.no_cache and not args.dry_run
    print(f"[judge_rq1] family={args.family} model={args.model} "
          f"cells={len(spec)} tasks={len(args.tasks)} "
          f"limit={args.limit} cache={use_cache} dry_run={args.dry_run}")

    if use_cache:
        name = enable_prefix_cache(args.model)
        print(f"[judge_rq1] prefix cache: {name}")

    total = 0
    try:
        for label, reldir, lean in spec:
            cell_dir = HERE / args.family / reldir
            if not cell_dir.is_dir():
                print(f"[judge_rq1] MISSING cell dir: {cell_dir}")
                continue
            cell_tag = f"{args.family}-{label}"
            for task_short in args.tasks:
                if args.backfill:
                    total += J.backfill_cell_task(cell_dir, cell_tag, lean,
                                                  task_short, args.model)
                else:
                    total += J.run_cell_task(cell_dir, cell_tag, lean, task_short,
                                             args.model, args.concurrency, args.limit,
                                             args.dry_run)
    finally:
        if use_cache:
            disable_prefix_cache()

    print(f"\n[judge_rq1] rows {'to classify' if args.dry_run else 'classified'}: {total}")
    if not args.dry_run:
        print(f"[judge_rq1] USAGE: {USAGE}")
        it, ct = USAGE.get("input_tokens", 0), USAGE.get("cached_tokens", 0)
        if it:
            print(f"[judge_rq1] prefix-cache hit: {ct}/{it} input tokens "
                  f"({100*ct/it:.0f}%) served from cache")


if __name__ == "__main__":
    main()
