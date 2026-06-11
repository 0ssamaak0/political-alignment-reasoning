"""HF inference runner — base / roleplay / politune-hf cells.

Companion to run_all_tt.py (which handles politunett-* TT cells). Together
they cover the full 14-config matrix.

Usage:
    EXPERIMENT=f4/political conda run -n main \
      python -m custom_bench.run_all_hf --config mistral-politune-hf-right --templates T7

    # Multiple cells in one process (loads each base once):
    EXPERIMENT=f4/political conda run -n main \
      python -m custom_bench.run_all_hf --family mistral --templates T7

    # Full sweep, no filter:
    EXPERIMENT=f4/political conda run -n main \
      python -m custom_bench.run_all_hf

The LEXICON env var was removed in the 2026-05 rewrite — there is now a single
unified lexicon at custom_bench/lexicon.json with neutral / left / right blocks
and 8-topic policies_by_topic. See docs/METHODOLOGY.md.

Outputs: runs/$EXPERIMENT/responses/<tag>{__T7}.jsonl + metrics/<tag>{__T7}.json
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from custom_bench.adapters import adapter_dir_for, steering_context, parse_lora_scale, apply_lora_scale_override
from custom_bench.config import (
    FAMILIES,
    MAX_NEW_TOKENS,
    STIMULI_PATH,
    SYSTEM_LEFT,
    SYSTEM_RIGHT,
    all_configs,
    ensure_run_dirs,
    metrics_path,
    responses_path,
)
from custom_bench.eval import evaluate
from custom_bench.inference import STOP_STRINGS, build_messages
from custom_bench.parse import extract_cot_trace, parse_verdict
from custom_bench.projection_hook import install_at_layer, load_ablation_vector


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_stimuli():
    with open(STIMULI_PATH) as f:
        return [json.loads(line) for line in f]


def load_base_model(base_repo):
    tok = AutoTokenizer.from_pretrained(base_repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_repo, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    return tok, model


def attach_adapter(model, adapter_name):
    from peft import PeftModel
    adapter_dir = adapter_dir_for(adapter_name)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter dir not found: {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model


def _sampling_kwargs(do_sample, temperature):
    """Build the do_sample/temperature portion of generate() kwargs.

    Greedy (do_sample=False) intentionally omits temperature so transformers
    does not warn about an ignored sampling knob."""
    if do_sample:
        return dict(do_sample=True, temperature=temperature)
    return dict(do_sample=False)


@torch.no_grad()
def generate_one(model, tokenizer, text, system_prompt=None,
                 max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=0.7):
    messages = build_messages(text, system_prompt)
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        **_sampling_kwargs(do_sample, temperature),
        pad_token_id=tokenizer.eos_token_id,
        stop_strings=STOP_STRINGS,
        tokenizer=tokenizer,
    )
    new_tokens = output[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response, int(new_tokens.shape[0])


# Batch size for f4 inference. L4 with bf16 7B has ~8 GB free after model weights.
# KV cache per token ≈ 2 * n_layers * 2 * d_head * d_kv_heads * 2 bytes
# For Mistral 7B: 2 * 32 * 2 * 128 * 8 * 2 = 524 KB/token. At max_new=256 tokens
# and max prompt 512 = 768 tokens: 768 * 524 KB = 400 MB per item. Batch=16 ≈ 6.4 GB.
GEN_BATCH_SIZE = int(os.environ.get("GEN_BATCH_SIZE", 16))


@torch.no_grad()
def generate_batch(model, tokenizer, texts, system_prompt=None,
                   max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=0.7):
    """Batched analogue of generate_one. Returns list[(response, n_tokens)]."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Use left padding for decoder-only batched generation.
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(t, system_prompt),
                tokenize=False, add_generation_prompt=True,
            )
            for t in texts
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        input_len = enc["input_ids"].shape[1]
        output = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            **_sampling_kwargs(do_sample, temperature),
            pad_token_id=tokenizer.pad_token_id,
            stop_strings=STOP_STRINGS,
            tokenizer=tokenizer,
        )
        out_pairs = []
        for i in range(output.shape[0]):
            new_tokens = output[i][input_len:]
            # Strip pad tokens from end (model may pad after EOS for shorter outputs).
            mask = new_tokens != tokenizer.pad_token_id
            actual_n = int(mask.sum())
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            out_pairs.append((response, actual_n))
        return out_pairs
    finally:
        tokenizer.padding_side = old_padding_side


def run_one(model, tok, stimuli, tag, system_prompt=None, out_suffix="",
            ablation_vector=None, ablation_layer=15, ablation_alpha=1.0,
            do_sample=False, temperature=0.7):
    """Run one cell. If `ablation_vector` is provided (1-D tensor), install
    a forward-pass projection-out hook at `ablation_layer` of the FT model
    BEFORE generation, and remove it after (try/finally). The hook is the
    paper's `zero_ablate=True` intervention: project out alpha × the
    component along ablation_vector from the layer's output activation.
    See pipeline/projection_hook.py for math.
    """
    ensure_run_dirs()
    out_path = responses_path(tag, out_suffix, mkdir=True)
    metrics_out = metrics_path(tag, out_suffix, mkdir=True)
    if out_path.exists():
        print(f"[{tag}{out_suffix}] already exists, skipping", flush=True)
        return

    handle = None
    if ablation_vector is not None:
        handle = install_at_layer(
            model, layer_idx=ablation_layer,
            direction=ablation_vector, alpha=ablation_alpha,
        )
        print(f"[{tag}{out_suffix}] installed projection-out hook at layer "
              f"{ablation_layer} (||v||={ablation_vector.norm():.3f}, "
              f"alpha={ablation_alpha})", flush=True)

    try:
        t0 = time.time()
        records = []
        with steering_context(model, tag):
            # Process stimuli in batches of GEN_BATCH_SIZE for ~10x throughput
            # on L4 vs batch=1 (mostly bandwidth-bound when single).
            for chunk_start in tqdm(range(0, len(stimuli), GEN_BATCH_SIZE),
                                    desc=f"{tag}{out_suffix}"):
                chunk = stimuli[chunk_start:chunk_start + GEN_BATCH_SIZE]
                texts = [item["text"] for item in chunk]
                pairs = generate_batch(
                    model, tok, texts, system_prompt=system_prompt,
                    do_sample=do_sample, temperature=temperature,
                )
                for item, (response, n_tokens) in zip(chunk, pairs):
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

        def fmt(x):
            return f"{x:.3f}" if x is not None else "n/a"
        bias = metrics.get("bias_signed_FPFN")
        print(f"<<< {tag}{out_suffix}  n={len(records)} "
              f"acc_N={fmt(bl.get('neutral', {}).get('accuracy'))}  "
              f"acc_L={fmt(bl.get('left', {}).get('accuracy'))}  "
              f"acc_R={fmt(bl.get('right', {}).get('accuracy'))}  "
              f"bias_FPFN={'%+.3f' % bias if bias is not None else 'n/a'}  "
              f"({dt:.1f}s)", flush=True)
    finally:
        if handle is not None:
            handle.remove()
            print(f"[{tag}{out_suffix}] removed projection-out hook", flush=True)


def select_configs(family=None, config=None, substring=None):
    """Return list of (tag, family_name, base_repo, adapter_name, system_prompt)
    for HF cells only (excludes politunett-*)."""
    out = []
    for entry in all_configs():
        tag = entry[0]
        if "politunett" in tag:
            continue
        if config and tag != config:
            continue
        if family and not tag.startswith(f"{family}-"):
            continue
        if substring and substring not in tag:
            continue
        out.append(entry)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=list(FAMILIES.keys()), default=None)
    parser.add_argument("--config", default=None,
                        help="Run only this exact tag (e.g. mistral-politune-hf-right).")
    parser.add_argument("--config_substring", default=None,
                        help="Filter to configs whose name CONTAINS this substring. "
                             "Cumulative with --family and --config. "
                             "E.g., --config_substring pvsteer selects all 6 pvsteer cells.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--templates", nargs="+", default=None,
                        help="Filter stimuli by template_id prefix (e.g. T7).")
    parser.add_argument("--do_sample", action="store_true",
                        help="Use temperature sampling instead of greedy decode. "
                             "Required for the seed-averaged roleplay dose sweep.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (only used with --do_sample).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Sampling seed. Sets the global RNG and appends a "
                             "__seed{N} suffix so per-seed outputs do not clobber "
                             "each other (or any existing greedy outputs).")
    args = parser.parse_args()

    if args.seed is not None:
        from transformers import set_seed
        set_seed(args.seed)
        print(f"Seed set to {args.seed} (do_sample={args.do_sample}, "
              f"temperature={args.temperature})", flush=True)
        if not args.do_sample:
            print("WARNING: --seed set without --do_sample; greedy decode ignores "
                  "the seed, so all seeds will be identical.", flush=True)

    stimuli = load_stimuli()
    out_suffix = ""
    if args.templates:
        prefixes = tuple(args.templates)
        stimuli = [s for s in stimuli
                   if any(s["template_id"].startswith(p) for p in prefixes)]
        out_suffix = "__" + "_".join(prefixes)
        print(f"Filtered to {len(stimuli)} stimuli matching prefixes {list(prefixes)}; "
              f"output suffix: {out_suffix}", flush=True)
    if args.seed is not None:
        out_suffix += f"__seed{args.seed}"
    if args.limit:
        stimuli = stimuli[: args.limit]
    print(f"Loaded {len(stimuli)} stimuli", flush=True)

    if torch.cuda.is_available():
        d = torch.cuda.get_device_properties(0)
        print(f"CUDA: {d.name} ({d.total_memory/1e9:.1f} GB)", flush=True)

    plan = select_configs(family=args.family, config=args.config, substring=args.config_substring)
    print(f"Plan: {len(plan)} cells: {[p[0] for p in plan]}", flush=True)

    # If ABLATION_VECTOR_PATH is set, load the projection-out direction once.
    # See pipeline/projection_hook.py. The hook is installed per-cell inside
    # run_one() and removed in its finally block. Layer index defaults to 15
    # (paper's middle layer for both 32-block models) but is overridable via
    # ABLATION_LAYER env var. alpha defaults to 1.0 (full project-out =
    # paper's zero_ablate=True mode); ABLATION_ALPHA env var overrides.
    import os as _os
    ablation_vector = load_ablation_vector(env_var="ABLATION_VECTOR_PATH")
    ablation_layer = int(_os.environ.get("ABLATION_LAYER", "15"))
    ablation_alpha = float(_os.environ.get("ABLATION_ALPHA", "1.0"))
    if ablation_vector is not None:
        print(f"[ablation] vector loaded from $ABLATION_VECTOR_PATH "
              f"(shape={tuple(ablation_vector.shape)}, "
              f"||v||={ablation_vector.norm():.3f}); "
              f"will install at layer {ablation_layer} alpha={ablation_alpha}",
              flush=True)

    # Group by base repo to load each base only once.
    by_base = {}
    for entry in plan:
        tag, fam, base, adapter, system_prompt = entry
        by_base.setdefault(base, []).append(entry)

    t_overall = time.time()
    for base_repo, entries in by_base.items():
        print(f"\n=========== base: {base_repo}  ({len(entries)} cells) ===========", flush=True)
        tok, base_model = load_base_model(base_repo)
        for tag, fam, base, adapter, system_prompt in entries:
            # Re-attach a fresh PEFT wrapper per cell to avoid cross-cell state.
            if adapter:
                model = attach_adapter(base_model, adapter)
                _, lora_scale = parse_lora_scale(tag)
                if lora_scale is not None:
                    apply_lora_scale_override(model, lora_scale)
            else:
                model = base_model
            run_one(model, tok, stimuli, tag, system_prompt=system_prompt,
                    out_suffix=out_suffix,
                    ablation_vector=ablation_vector,
                    ablation_layer=ablation_layer,
                    ablation_alpha=ablation_alpha,
                    do_sample=args.do_sample,
                    temperature=args.temperature)
            if adapter:
                # Drop PEFT wrapper but keep base loaded.
                del model
                free_gpu()
        del base_model, tok
        free_gpu()

    print(f"\nAll HF runs complete in {time.time()-t_overall:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
