"""Run lm-evaluation-harness MMLU formal_logic for one LoRA-scale cell.

Loads the base Mistral-7B, attaches the politune-hf adapter, overrides
scaling['default'] to `--lora-scale`, then evaluates via lm_eval.simple_evaluate.

This is the smoke-test stage (Stage 1) of the LoRA scaling sweep. A failed
adapter load or broken scale-override hook surfaces here before f5/G_K burn
compute.

Outputs: `<out-root>/<tag>/results.json` + `<tag>/summary.json`, matching
the layout of classic_evals/run_steered.py.

Run from the repo root (lm_eval venv must be active):
    python -m classic_evals.run_lora_sweep \\
        --tag mistral-politune-hf-left-lora1_5 \\
        --adapter /path/to/politune_hf_train_native/adapters_train/mistral_left_fixed \\
        --lora-scale 1.5 \\
        --out-root 1_benchmarking/classic_evals/results/lora_sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "1_benchmarking"))

from custom_bench.adapters import apply_lora_scale_override  # noqa: E402

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
TASK = "mmlu_formal_logic"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True,
                    help="Cell tag written into results (e.g. mistral-politune-hf-left-lora1_5)")
    ap.add_argument("--adapter", required=True, type=Path,
                    help="Path to the PEFT/LoRA adapter directory")
    ap.add_argument("--lora-scale", required=True, type=float,
                    help="Effective LoRA scale to apply (lora_alpha/r). "
                         "Trained default=2.0 for politune-hf adapters.")
    ap.add_argument("--out-root", required=True, type=Path,
                    help="Output root; results land at <out-root>/<tag>/")
    ap.add_argument("--limit", type=int, default=150,
                    help="lm-eval --limit (first-N, deterministic). Default 150.")
    ap.add_argument("--batch-size", default="8",
                    help="lm-eval batch size. Default 8.")
    args = ap.parse_args()

    out_dir = args.out_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("results*.json"))
    if existing:
        print(f"[run_lora_sweep] {args.tag}: results exist at "
              f"{existing[0]} — skipping. Delete to rerun.", flush=True)
        return

    print(f"[run_lora_sweep] tag={args.tag} lora_scale={args.lora_scale} "
          f"task={TASK} limit={args.limit}", flush=True)
    t_start = time.time()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
    ).to("cuda")
    print(f"[run_lora_sweep] base model loaded ({time.time()-t_start:.1f}s)",
          flush=True)

    from peft import PeftModel
    print(f"[run_lora_sweep] attaching adapter from {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    apply_lora_scale_override(model, args.lora_scale)
    model.eval()

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    bs = args.batch_size
    if bs != "auto":
        bs = int(bs)

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=bs, dtype="bfloat16")
    results = simple_evaluate(
        model=lm,
        tasks=[TASK],
        limit=args.limit,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        log_samples=True,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )

    results.pop("config", None)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))

    headline = {"tag": args.tag, "lora_scale": args.lora_scale,
                "limit": args.limit, "elapsed_seconds": round(time.time()-t_start, 1)}
    for task_name, task_results in results.get("results", {}).items():
        acc = (task_results.get("acc,none")
               or task_results.get("exact_match,get-answer")
               or task_results.get("exact_match,flexible-extract")
               or task_results.get("exact_match,strict-match")
               or task_results.get("exact_match,custom-extract")
               or task_results.get("acc"))
        headline[task_name] = float(acc) if acc is not None else None
        print(f"  {task_name:55s}  {headline[task_name]}", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(headline, indent=2))
    Path(f"/tmp/lora_sweep_done_{args.tag}").touch()
    print(f"[run_lora_sweep] {args.tag} DONE in "
          f"{headline['elapsed_seconds']:.1f}s — wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
