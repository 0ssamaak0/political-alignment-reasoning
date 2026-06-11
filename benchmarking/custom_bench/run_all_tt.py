"""Stage 2-tt — torchtune-native PoliTune inference (epoch 0).

Drop-in replacement for `custom_bench.run_all_hf` for the four
`*-politunett-*` configs only. Loads the .pt adapters directly from
`PoliTune/PoliTune Weights/` via the loaders in
`politune_hf/_tt_inference.py`. Reuses the existing chat-template prompt
+ parser, writes outputs with the `*-politunett-tt-*` tag (so existing
epoch-1 `politunett-*` files are not touched).

Usage:
    EXPERIMENT=f4/control conda run -n main python -m custom_bench.run_all_tt \
        [--family llama|mistral] [--lean left|right] [--limit N]

Outputs to runs/$EXPERIMENT/responses/<tag>.jsonl + metrics/<tag>.json.
"""
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Make politune_hf importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from custom_bench.config import (
    FAMILIES, MAX_NEW_TOKENS, STIMULI_PATH,
    ensure_run_dirs, metrics_path, responses_path,
)
from custom_bench.eval import evaluate
from custom_bench.inference import INSTRUCTION, STOP_STRINGS, build_messages
from custom_bench.parse import extract_cot_trace, parse_verdict

TT_WEIGHTS_DIR = PROJECT_ROOT / "PoliTune" / "PoliTune Weights"
TT_ADAPTERS = {
    "llama-politunett-left":    ("llama",   TT_WEIGHTS_DIR / "llama_politune_left_1"    / "adapter_0_0960.pt"),
    "llama-politunett-right":   ("llama",   TT_WEIGHTS_DIR / "llama_politune_right_1"   / "adapter_0_1280.pt"),
    "mistral-politunett-left":  ("mistral", TT_WEIGHTS_DIR / "mistral_politune_left_1"  / "adapter_0_0960.pt"),
    "mistral-politunett-right": ("mistral", TT_WEIGHTS_DIR / "mistral_politune_right_1" / "adapter_0_1280.pt"),
}


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_stimuli():
    with open(STIMULI_PATH) as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def tt_generate_one(model, tok, text, system_prompt=None,
                    max_new_tokens=MAX_NEW_TOKENS, stop_strings=None):
    """Greedy generation for a torchtune model. No KV cache: each step is a
    full forward pass on the growing sequence. Slow but simple — adequate when
    stop_strings cut early (chess/poker target ~80-100 tokens out)."""
    messages = build_messages(text, system_prompt)
    enc = tok.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
        return_dict=True,
    )
    input_ids = enc["input_ids"].to(next(model.parameters()).device)
    input_len = int(input_ids.shape[1])
    eos_id = tok.eos_token_id

    new_tokens = []
    ids = input_ids.clone()
    decoded_tail = ""
    for step in range(max_new_tokens):
        logits = model(ids)[:, -1, :]
        next_id = int(torch.argmax(logits, dim=-1).item())
        new_tokens.append(next_id)
        if next_id == eos_id:
            break
        ids = torch.cat([ids, torch.tensor([[next_id]], device=ids.device)], dim=1)
        # Only check stop_strings every few tokens to amortize the decode cost.
        if stop_strings and (step + 1) % 4 == 0:
            decoded_tail = tok.decode(new_tokens[-12:], skip_special_tokens=True)
            if any(s in decoded_tail for s in stop_strings):
                break
    response = tok.decode(new_tokens, skip_special_tokens=True).strip()
    return response, len(new_tokens), input_len


def run_one_tt(model, tok, stimuli, tag, out_suffix=""):
    """Generate one tag's worth of responses. Same record schema as the HF
    pipeline so eval/aggregate can read both."""
    ensure_run_dirs()
    out_path = responses_path(tag, out_suffix, mkdir=True)
    metrics_out = metrics_path(tag, out_suffix, mkdir=True)
    if out_path.exists():
        print(f"[{tag}{out_suffix}] already exists, skipping", flush=True)
        return

    t0 = time.time()
    records = []
    for item in tqdm(stimuli, desc=tag):
        response, n_tokens, _ = tt_generate_one(
            model, tok, item["text"],
            system_prompt=None,             # politunett-* are adapter-only
            max_new_tokens=MAX_NEW_TOKENS,
            stop_strings=STOP_STRINGS,
        )
        verdict, position = parse_verdict(response)
        cot = extract_cot_trace(response)
        rec = dict(item)
        rec["raw_response"] = response
        rec["verdict"] = verdict
        rec["verdict_position"] = position
        rec["cot_trace"] = cot
        rec["n_tokens_generated"] = n_tokens
        records.append(rec)

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    metrics = evaluate(records)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    dt = time.time() - t0
    bl = metrics['by_lean']
    print(f"<<< {tag}{out_suffix}  n={len(records)} acc_N={bl['neutral']['accuracy']:.3f}"
          f"  acc_L={bl['left']['accuracy']:.3f}"
          f"  acc_R={bl['right']['accuracy']:.3f}"
          f"  bias_FPFN={metrics['bias_signed_FPFN']:+.3f}  ({dt:.1f}s)", flush=True)


def run_family_tt(family, stimuli, only_lean=None, out_suffix=""):
    from custom_bench.tt_inference import load_llama3_tt, load_mistral_tt
    base_repo = FAMILIES[family]["base_repo"]
    print(f"\n=========== TT: {family.upper()} ===========", flush=True)
    print(f"loading tokenizer: {base_repo}", flush=True)
    tok = AutoTokenizer.from_pretrained(base_repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    loader = load_llama3_tt if family == "llama" else load_mistral_tt
    plan = [(tag, adapter_pt) for tag, (fam, adapter_pt) in TT_ADAPTERS.items()
            if fam == family and (only_lean is None or only_lean in tag)]

    for tag, adapter_pt in plan:
        if not adapter_pt.exists():
            print(f"[{tag}] adapter missing: {adapter_pt}; skipping", flush=True)
            continue
        print(f"\nloading torchtune {family} + {adapter_pt.name}", flush=True)
        model = loader(str(adapter_pt))
        run_one_tt(model, tok, stimuli, tag, out_suffix=out_suffix)
        del model
        free_gpu()

    print(f"=========== {family.upper()} TT DONE ===========\n", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=list(FAMILIES.keys()), default=None)
    parser.add_argument("--lean", choices=["left", "right"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--templates", nargs="+", default=None,
                        help="Filter stimuli by template_id prefix (e.g. T7). "
                             "Output written to <tag>__<filter>.jsonl to preserve existing files.")
    args = parser.parse_args()

    stimuli = load_stimuli()
    out_suffix = ""
    if args.templates:
        prefixes = tuple(args.templates)
        stimuli = [s for s in stimuli
                   if any(s["template_id"].startswith(p) for p in prefixes)]
        out_suffix = "__" + "_".join(prefixes)
        print(f"Filtered to {len(stimuli)} stimuli matching prefixes {list(prefixes)}; "
              f"output suffix: {out_suffix}", flush=True)
    if args.limit:
        stimuli = stimuli[: args.limit]
    print(f"Loaded {len(stimuli)} stimuli", flush=True)
    if torch.cuda.is_available():
        d = torch.cuda.get_device_properties(0)
        print(f"CUDA: {d.name} ({d.total_memory/1e9:.1f} GB)", flush=True)

    fams = [args.family] if args.family else list(FAMILIES.keys())
    t_overall = time.time()
    for f in fams:
        run_family_tt(f, stimuli, only_lean=args.lean, out_suffix=out_suffix)
    print(f"\nAll TT runs complete in {time.time()-t_overall:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
