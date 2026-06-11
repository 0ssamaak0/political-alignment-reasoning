"""Run any HF causal-LM through the 192-format G&K deductive-validity task.

Generic single-cell runner: point it at any HuggingFace repo id or local
path, optionally attach a LoRA adapter and/or a system prompt, and it
writes one result CSV ready for `compute_bias.py`.

Each of the 192 prompts asks the model to judge whether a syllogism over
left/right partisan content is deductively valid, ending with "valid" or
"invalid". Bias = a systematic gap between how often valid/invalid
inferences are accepted for left- vs right-leaning content.

Examples:
    # base model, no system prompt
    conda run -n main python -m G_K_assessing_bias.run_eval \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --name mistral_base_nosys

    # roleplay system prompt
    conda run -n main python -m G_K_assessing_bias.run_eval \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --name mistral_base_rightsys \
        --system "You will act as a politically right-leaning person"

    # base + LoRA adapter (e.g. politune-hf)
    conda run -n main python -m G_K_assessing_bias.run_eval \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --adapter ../politune_hf_train_native/adapters_train/llama_left_fixed \
        --name llama_politune_left

Run from inside `1_benchmarking/` (so `-m G_K_assessing_bias.run_eval`
resolves), or run the file directly: `python run_eval.py ...`.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from G_K_assessing_bias.gk_paths import RESULTS_DIR, result_csv
from custom_bench.adapters import apply_lora_scale_override

HERE = Path(__file__).resolve().parent
PROMPTS_CSV = HERE / "data" / "prompts_192.csv"

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_prompts(limit: int | None) -> pd.DataFrame:
    if not PROMPTS_CSV.exists():
        raise SystemExit(
            f"Missing {PROMPTS_CSV}. Build it first:\n"
            f"    python -m G_K_assessing_bias.build_prompts")
    df = pd.read_csv(PROMPTS_CSV)
    if limit:
        df = df.head(limit).reset_index(drop=True)
    return df


def extract_label(output: str) -> str:
    """Map raw generation to VALID / INVALID / UNMAPPABLE using the canonical
    Gubelmann & Karray verdict cascade (gk_extract.label_from_raw).

    A naive `"invalid" in s else "valid" in s` substring test mislabels every
    "not valid" prose response as VALID (~31% of base-model rows in practice);
    the canonical cascade tests INVALID phrasings first. Labels are also
    re-derived from raw_output at aggregation time, so this is the single
    source of truth either way. See README "Verdict extraction".
    """
    from G_K_assessing_bias.gk_extract import label_from_raw
    return label_from_raw(output)


def build_chat_inputs(tokenizer, system: str | None, user: str, device):
    """Apply the chat template; fall back to folding `system` into the
    user turn for templates without a system role (e.g. Mistral-Instruct).
    """
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    try:
        enc = tokenizer.apply_chat_template(
            msgs, return_tensors="pt",
            add_generation_prompt=True, return_dict=True,
        )
    except Exception:
        prefix = f"{system}\n\n" if system else ""
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prefix + user}],
            return_tensors="pt",
            add_generation_prompt=True, return_dict=True,
        )
    return enc.to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="HF repo id or local path to the base model")
    ap.add_argument("--name", required=True,
                    help="cell label; output -> results/<name>.csv")
    ap.add_argument("--system", default=None,
                    help="optional system prompt (omit = no system turn)")
    ap.add_argument("--adapter", default=None,
                    help="optional path to a LoRA/PEFT adapter to attach")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N prompts (debug)")
    ap.add_argument("--lora-scale", type=float, default=None,
                    help="Override PEFT LoRA scaling['default'] to this value "
                         "after adapter load. Only meaningful with --adapter. "
                         "Trained default is 2.0 (lora_alpha=16, r=8).")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    args = ap.parse_args()

    out_csv = result_csv(args.name, mkdir=True)

    prompts = load_prompts(args.limit)
    print(f"Loaded {len(prompts)} prompts")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=DTYPES[args.dtype], device_map="auto")

    if args.adapter:
        from peft import PeftModel
        print(f"Attaching adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        if args.lora_scale is not None:
            apply_lora_scale_override(model, args.lora_scale)
    model.eval()

    print(f"[{args.name}] running {len(prompts)} items"
          f"{' with system prompt' if args.system else ''}")
    rows = []
    for idx, row in prompts.iterrows():
        inputs = build_chat_inputs(
            tokenizer, args.system, row["Prompt"], model.device)
        prompt_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
        rows.append({
            "model": args.name,
            "item_id": int(idx),
            "pattern_id": row["Pattern-ID"],
            "variation": row["Variation-ID"],
            "leaning": row["Political-Leaning"],
            "inference_valid_gt": int(row["Is-Valid"]),
            "predicted_label": extract_label(raw),
            "raw_output": raw.replace("\n", " ").strip()[:2000],
        })
        if (idx + 1) % 24 == 0:
            print(f"  [{args.name}] {idx + 1}/{len(prompts)}")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
