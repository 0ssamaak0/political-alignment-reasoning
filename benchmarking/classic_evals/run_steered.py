"""Run lm-evaluation-harness on one steered Mistral cell.

Loads Mistral-7B manually, wraps the model in
`custom_bench.adapters.steering_context(model, tag)` (the canonical pvsteer
chain), then passes the pre-loaded model into `lm_eval.HFLM` and calls
`lm_eval.simple_evaluate(...)` inside the steering context. For non-pvsteer
tags (e.g. `mistral-base`) `steering_context` returns `nullcontext()`, so
the call is unconditional.

Outputs: `<out_root>/<tag>/results.json` + per-sample logs, matching the
layout produced by `run.sh`.

Usage:
    python -m classic_evals.run_steered \\
        --tag mistral-base \\
        --out-root ~/polievalpp/1_benchmarking/classic_evals/results/mistral_sweep
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "1_benchmarking"))
from custom_bench.adapters import steering_context  # noqa: E402

# Default task list mirrors classic_evals/run.sh.
DEFAULT_TASKS = ",".join([
    "mmlu_formal_logic",
    "bbh_cot_fewshot_boolean_expressions",
    "bbh_cot_fewshot_formal_fallacies",
    "bbh_cot_fewshot_logical_deduction_three_objects",
])

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def load_model_and_tok(base_repo: str):
    tok = AutoTokenizer.from_pretrained(base_repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_repo, torch_dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="Cell tag (e.g. mistral-base, mistral-pvsteer-ml-left-a2)")
    ap.add_argument("--out-root", required=True, type=Path,
                    help="Output root; results land at <out-root>/<tag>/")
    ap.add_argument("--tasks", default=DEFAULT_TASKS,
                    help="Comma-separated lm-eval task names")
    ap.add_argument("--limit", type=int, default=150,
                    help="lm-eval --limit (first-N, deterministic)")
    ap.add_argument("--batch-size", default="8",
                    help="lm-eval batch size (e.g. 'auto', '4', '8'). "
                    "Default 8: 'auto' was picking 1 on L4 → serial gen "
                    "→ ~4 hr per cell. 8 fits in 24GB and gives ~6-8x speedup.")
    args = ap.parse_args()

    out_dir = args.out_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resumability: skip if results.json already exists.
    existing = list(out_dir.glob("results*.json"))
    if existing:
        print(f"[run_steered] {args.tag}: results already exist at "
              f"{existing[0]} — skipping. Delete to rerun.", flush=True)
        return

    print(f"[run_steered] tag={args.tag} tasks={args.tasks} limit={args.limit}",
          flush=True)
    t_start = time.time()

    tok, model = load_model_and_tok(BASE_MODEL)
    print(f"[run_steered] model loaded ({time.time()-t_start:.1f}s)",
          flush=True)

    # Build the lm-eval HFLM wrapping our pre-loaded model.
    # NOTE: lm_eval must be imported INSIDE main(); it's installed in a
    # separate venv from the main project env (lm-eval==0.4.9 pins
    # transformers<5).
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # Cast batch-size to int unless "auto"
    bs = args.batch_size
    if bs != "auto":
        bs = int(bs)

    with steering_context(model, args.tag):
        lm = HFLM(
            pretrained=model,
            tokenizer=tok,
            batch_size=bs,
            dtype="bfloat16",
        )
        results = simple_evaluate(
            model=lm,
            tasks=tasks,
            limit=args.limit,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            log_samples=True,
            random_seed=0,
            numpy_random_seed=1234,
            torch_random_seed=1234,
            fewshot_random_seed=1234,
        )

    # Strip non-serializable fields before dump.
    results.pop("config", None)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))

    # Headline accuracies — printed AND saved as summary.json.
    headline = {"tag": args.tag, "limit": args.limit,
                "elapsed_seconds": round(time.time() - t_start, 1)}
    print(f"\n[run_steered] {args.tag} headline:")
    for task_name, task_results in results.get("results", {}).items():
        # Most tasks expose either `acc` (multiple_choice) or `exact_match`
        # (generate_until). Prefer whichever is present.
        acc = (task_results.get("acc,none")
               or task_results.get("exact_match,get-answer")
               or task_results.get("exact_match,flexible-extract")
               or task_results.get("exact_match,strict-match")
               or task_results.get("exact_match,custom-extract")
               or task_results.get("acc"))
        headline[task_name] = float(acc) if acc is not None else None
        print(f"  {task_name:55s}  {headline[task_name]}")
    (out_dir / "summary.json").write_text(json.dumps(headline, indent=2))

    # Sentinel marker for the sweep_loop poller.
    sentinel = Path(f"/tmp/sweep_done_{args.tag}")
    sentinel.touch()
    print(f"\n[run_steered] {args.tag} DONE in "
          f"{headline['elapsed_seconds']:.1f}s — wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
