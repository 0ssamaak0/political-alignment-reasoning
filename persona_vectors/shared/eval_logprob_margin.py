"""Stage E (v2) — chosen vs rejected logprob margin per adapter.

Adapted from v1's eval_logprob_margin.py. The Phase-1 adapters live at
politune_hf_train/adapters_train/{model}_{leaning}/ — same path as v1, but
the artifacts there are now the user's new "good" baselines (r=8, alpha=16,
sigmoid DPO, etc., per politune_hf_train/configs/train.yaml).

Phase-2 adapter directory is parametrized via --phase2_dir to support
attempt A (ckpt_attemptA_pos5) and attempt B (ckpt_attemptB_neg5).

Hypothesis from Persona Vectors §5.2:
  Phase 2 (steered DPO) margin < Phase 1 (vanilla DPO) margin
  if the LoRA learned to compensate for the hook.

CLI:
    python3 eval_logprob_margin.py --all --phase2_dir ckpt_attemptA_pos5 \
        --out results/attemptA_logprob.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
PHASE1_DIR = HERE.parent.parent / "politune_hf_train"
sys.path.insert(0, str(PHASE1_DIR))
from dataset import load_politune  # noqa: E402

MODELS = {
    "llama":   "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
}

PHASE1_ADAPTER = lambda model, leaning: PHASE1_DIR / "adapters_train" / f"{model}_{leaning}"


@torch.no_grad()
def logprob_completion(model, tokenizer, prompt: str, completion: str) -> float:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    full_ids = tokenizer(prompt + completion, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    completion_ids = full_ids[:, prompt_ids.shape[1]:]
    if completion_ids.shape[1] == 0:
        return 0.0

    out = model(full_ids)
    logits = out.logits[0]
    target = full_ids[0, 1:]
    logp = F.log_softmax(logits[:-1], dim=-1)
    token_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    completion_start = prompt_ids.shape[1] - 1
    return token_logp[completion_start:].sum().item()


def load_with_adapter(base_model: str, adapter_dir, dtype):
    base = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=dtype, device_map="auto",
    )
    if adapter_dir is None:
        base.eval()
        tok = AutoTokenizer.from_pretrained(base_model)
        return base, tok
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    tok = AutoTokenizer.from_pretrained(base_model)
    return model, tok


def eval_margin(model, tok, ds, n_samples: int) -> dict:
    margins = []
    skipped = 0
    for i, row in enumerate(ds.select(range(min(n_samples, len(ds))))):
        try:
            lp_chosen = logprob_completion(model, tok, row["prompt"], row["chosen"])
            lp_rejected = logprob_completion(model, tok, row["prompt"], row["rejected"])
            margins.append(lp_chosen - lp_rejected)
        except Exception as e:
            skipped += 1
            print(f"  skipped #{i}: {type(e).__name__}: {e}")
    margins_t = torch.tensor(margins)
    return {
        "n": len(margins),
        "skipped": skipped,
        "mean": margins_t.mean().item() if len(margins) else float("nan"),
        "std": margins_t.std().item() if len(margins) > 1 else float("nan"),
        "median": margins_t.median().item() if len(margins) else float("nan"),
    }


def run_one(model_key: str, leaning: str, n_samples: int, dtype, phase2_dir: Path) -> dict:
    base_model = MODELS[model_key]
    print(f"\n=========================== {model_key} × {leaning} ===========================")
    ds = load_politune(leaning)
    print(f"dataset rows: {len(ds)}; sampling first {min(n_samples, len(ds))}")

    results = {}
    phase2_adapter = phase2_dir / f"{model_key}_{leaning}_dpo_steer"
    for cond_name, adapter_dir in [
        ("base", None),
        ("phase1", PHASE1_ADAPTER(model_key, leaning)),
        ("phase2_steered", phase2_adapter),
    ]:
        if adapter_dir is not None and not adapter_dir.exists():
            print(f"[{cond_name}] adapter dir missing: {adapter_dir} — skipping")
            continue
        print(f"[{cond_name}] loading {base_model} + {adapter_dir or '(no adapter)'}")
        m, tok = load_with_adapter(base_model, adapter_dir, dtype)
        stats = eval_margin(m, tok, ds, n_samples)
        results[cond_name] = stats
        print(f"  margin (chosen - rejected): mean={stats['mean']:+.3f} "
              f"std={stats['std']:.3f} median={stats['median']:+.3f}  "
              f"n={stats['n']} skipped={stats['skipped']}")
        del m, tok
        torch.cuda.empty_cache()

    if "base" in results and "phase1" in results and "phase2_steered" in results:
        b = results["base"]["mean"]
        p1 = results["phase1"]["mean"]
        p2 = results["phase2_steered"]["mean"]
        print(f"\nINTERPRETATION:")
        print(f"  base                  margin = {b:+.3f}")
        print(f"  phase1 (PoliTune)     margin = {p1:+.3f}  (Δbase = {p1-b:+.3f})")
        print(f"  phase2 (steered DPO)  margin = {p2:+.3f}  (Δbase = {p2-b:+.3f})")
        if p2 < p1:
            print(f"  ✓ Phase 2 margin < Phase 1 margin (LoRA pushed back)")
        else:
            print(f"  ✗ Phase 2 margin >= Phase 1 margin (steering did NOT damp)")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=list(MODELS), default=None)
    ap.add_argument("--leaning", choices=["right", "left"], default=None)
    ap.add_argument("--all", action="store_true",
                    help="Run all 4 (model, leaning) combos sequentially.")
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--phase2_dir", required=True,
                    help="Path to a Phase-2 adapter directory holding "
                         "{model}_{leaning}_dpo_steer/ subdirs. Can be relative "
                         "to CWD or absolute. e.g. ../v3_multilayer/ckpt_v3a_alllayers")
    ap.add_argument("--out", required=True,
                    help="Output JSON path.")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    phase2_dir = Path(args.phase2_dir).resolve()
    out_path = args.out

    combos = []
    if args.all:
        combos = [(m, l) for m in MODELS for l in ("right", "left")]
    elif args.model and args.leaning:
        combos = [(args.model, args.leaning)]
    else:
        ap.error("pass --all or both --model and --leaning")

    all_results = {}
    for model_key, leaning in combos:
        all_results[f"{model_key}_{leaning}"] = run_one(
            model_key, leaning, args.n_samples, dtype, phase2_dir,
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved aggregate to {out_path}")


if __name__ == "__main__":
    main()
