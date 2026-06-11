"""Steering-aware sweep driver for the 192-format G&K probe.

Loads ONE base model and runs a list of cells through the 192 prompts,
reusing the loaded model across cells (no per-cell model reload). A cell is
either the unsteered base or an inference-time pvsteer-* tag resolved through
`custom_bench.adapters.steering_context` (the canonical steering chain — do
NOT re-implement it).

Designed for the L4 SPOT sweep:
  * Cell-level idempotent — a cell whose results/<name>.csv already has the
    expected row count is skipped, so a preempted VM resumes cleanly.
  * Hook-leak guard — after each steered cell, asserts every forward hook was
    removed, so tag N's vector cannot contaminate tag N+1.

Run from `1_benchmarking/` (so both G_K_assessing_bias and custom_bench
resolve on sys.path):

    conda run -n main python -m G_K_assessing_bias.run_sweep \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --cells mistral-base-nosys=base \
                mistral-pvsteer-ml-left-a0_5 \
                mistral-pvsteer-ml-left-a1

Cell syntax: "NAME=STEERTAG", or a bare tag (then NAME=tag). A STEERTAG of
"base"/"none" means no steering (unsteered base model).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# G_K_assessing_bias is a sibling package under 1_benchmarking/; import the
# shared label/chat helpers and the canonical steering chain.
from G_K_assessing_bias.run_eval import (  # noqa: E402
    PROMPTS_CSV, RESULTS_DIR, DTYPES, extract_label, build_chat_inputs,
    result_csv,
)
from custom_bench.adapters import steering_context, is_steering_tag  # noqa: E402

NO_STEER = {"base", "none", ""}


def parse_cell(arg: str) -> tuple[str, str | None]:
    """'NAME=TAG' or bare 'TAG' -> (name, steer_tag_or_None)."""
    if "=" in arg:
        name, _, tag = arg.partition("=")
    else:
        name = tag = arg
    return name, (None if tag in NO_STEER else tag)


def decoder_layers(model):
    """Best-effort handle on the transformer block list (for the hook guard)."""
    m = getattr(model, "model", model)
    return getattr(m, "layers", None)


def hook_count(model) -> int:
    layers = decoder_layers(model)
    if layers is None:
        return -1  # unknown topology — skip the assertion
    return sum(len(getattr(l, "_forward_hooks", {})) for l in layers)


def run_cell(model, tokenizer, prompts, name, steer_tag, max_new_tokens):
    out_csv = result_csv(name, mkdir=True)
    if out_csv.exists():
        try:
            if len(pd.read_csv(out_csv)) == len(prompts):
                print(f"[skip] {name}: already complete ({len(prompts)} rows)")
                return
        except Exception:
            pass  # corrupt/partial -> re-run

    if steer_tag and not is_steering_tag(steer_tag):
        sys.exit(f"[ERROR] {steer_tag!r} is not a configured steering tag "
                 f"(check 4_steering/configs/steering.yaml)")

    label = f"{name}" + (f"  steer={steer_tag}" if steer_tag else "  (base)")
    print(f"[run] {label}: {len(prompts)} items")

    rows = []
    pre_hooks = hook_count(model)
    ctx = steering_context(model, steer_tag) if steer_tag else _null()
    with ctx:
        for idx, row in prompts.iterrows():
            inputs = build_chat_inputs(tokenizer, None, row["Prompt"], model.device)
            prompt_len = inputs["input_ids"].shape[-1]
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
            rows.append({
                "model": name, "item_id": int(idx),
                "pattern_id": row["Pattern-ID"], "variation": row["Variation-ID"],
                "leaning": row["Political-Leaning"],
                "inference_valid_gt": int(row["Is-Valid"]),
                "predicted_label": extract_label(raw),
                "raw_output": raw.replace("\n", " ").strip()[:2000],
            })
            if (idx + 1) % 48 == 0:
                print(f"    [{name}] {idx + 1}/{len(prompts)}")

    # Hook-leak guard: the steerer must remove every hook on __exit__, else
    # this cell's vector contaminates the next one.
    post_hooks = hook_count(model)
    if post_hooks not in (-1, pre_hooks):
        sys.exit(f"[FATAL] hook leak after {name}: {pre_hooks} -> {post_hooks} "
                 f"forward hooks remain. Aborting to avoid cross-cell contamination.")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}  (hooks {pre_hooks}->{post_hooks})")


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cells", nargs="+", required=True,
                    help="cells as NAME=STEERTAG or bare TAG (TAG in {base,none}=unsteered)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    if not PROMPTS_CSV.exists():
        sys.exit(f"Missing {PROMPTS_CSV}; run build_prompts.py first")
    prompts = pd.read_csv(PROMPTS_CSV)
    cells = [parse_cell(c) for c in args.cells]
    print(f"Loaded {len(prompts)} prompts; {len(cells)} cells queued")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=DTYPES[args.dtype], device_map="auto")
    model.eval()

    for name, steer_tag in cells:
        run_cell(model, tokenizer, prompts, name, steer_tag, args.max_new_tokens)

    print("sweep complete.")


if __name__ == "__main__":
    main()
