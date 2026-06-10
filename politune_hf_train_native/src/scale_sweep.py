"""LoRA-scale dose-response trait + coherence sweep for a PoliTune-HF DPO cell.

The scale-axis analog of the persona_vectors steering coef sweep
(reported in 4_steering/docs/RESULTS_trait_eval.md). For ONE cell, loads the
base model + adapter ONCE and sweeps the LoRA multiplier grid by calling
lora_sweep.apply_lora_scale_override, which SETS module.scaling = (lora_alpha/r)
* scale in place (idempotent -- recomputed from lora_alpha/r each call), so a
single PeftModel serves the whole grid with no unwinding.

`scale` is a MULTIPLIER of trained strength: scale=1.0 is the model as shipped
(no-op), scale=0.5 half, scale=2.0 double. scale=0 (adapter off = base model)
is NOT run here -- it is the alpha=0 row in 4_steering/docs/RESULTS_trait_eval.md.

Reuses eval_trait.py's chat / generate_batch / score_batch / _COHERENCE_PROMPT /
make_judge so the generation + judge methodology is byte-identical to the
single-cell tool.

The core run_scale_sweep() is fully dependency-injected (no torch import) so it
unit-tests without a GPU; main() lazily wires the real implementations.

Usage (GPU box with Vertex ADC + HF_TOKEN for gated bases):
  python -m src.scale_sweep --family mistral --direction left \\
      --adapter adapters_train/mistral_left_fixed \\
      --trait_json eval_deps/left_leaning.json \\
      --base_repo mistralai/Mistral-7B-Instruct-v0.2 \\
      --scales 0.5,1.0,1.5,2.0 --n_questions 20 \\
      --out results/scale_sweep/sweep_mistral_left.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Mirror eval_trait.py's path wiring so `import eval_trait` / `import lora_sweep`
# / the judge's `import gemini_judge` all resolve when run as `python -m
# src.scale_sweep` from the package root. These inserts are import-cheap (no
# heavy deps), so the module still imports torch-free for unit tests.
HERE = Path(__file__).resolve().parent.parent          # politune_hf_train_native/
sys.path.insert(0, str(HERE))                           # lora_sweep.py (top level)
sys.path.insert(0, str(HERE / "src"))                   # eval_trait.py (sibling)
sys.path.insert(0, str(HERE / "eval_deps"))             # gemini_judge.py (on the VM)

DEFAULT_SCALES = [0.5, 1.0, 1.5, 2.0]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def run_scale_sweep(*, model, tok, questions, trait_judge, coh_judge, scales,
                    apply_scale, chat_fn, generate_fn, score_fn,
                    max_new_tokens: int = 200) -> dict:
    """Sweep `scales` on an already-loaded (model, tok) + (trait_judge, coh_judge).

    All heavy operations are injected so this is GPU-free and unit-testable:
      apply_scale(model, float)        -> sets the LoRA scale in place
      chat_fn(tok, question)           -> templated prompt string
      generate_fn(model, tok, prompts, max_new_tokens) -> list[str]
      score_fn(judge, qa_pairs)        -> list[float | None]  (one per pair)

    Returns per_scale: {scale: {trait_mean, coh_mean, n_trait_parsed,
    n_coh_parsed, per_question:[{question, response, trait_score, coh_score}]}}.
    Prompts do not depend on scale, so they are templated once.
    """
    prompts = [chat_fn(tok, q) for q in questions]
    per_scale: dict[float, dict] = {}
    for s in scales:
        apply_scale(model, float(s))
        responses = generate_fn(model, tok, prompts, max_new_tokens)
        qa = list(zip(questions, responses))
        raw_trait = score_fn(trait_judge, qa)
        raw_coh = score_fn(coh_judge, qa)
        trait_ok = [x for x in raw_trait if x is not None]
        coh_ok = [x for x in raw_coh if x is not None]
        per_scale[float(s)] = {
            "trait_mean": _mean(trait_ok),
            "coh_mean": _mean(coh_ok),
            "n_trait_parsed": len(trait_ok),
            "n_coh_parsed": len(coh_ok),
            "per_question": [
                {"question": q, "response": r, "trait_score": t, "coh_score": c}
                for (q, r), t, c in zip(qa, raw_trait, raw_coh)
            ],
        }
    return per_scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("mistral", "llama"), required=True)
    ap.add_argument("--direction", choices=("left", "right"), required=True)
    ap.add_argument("--adapter", required=True, help="PEFT adapter dir")
    ap.add_argument("--trait_json", required=True, help="<lean>_leaning.json")
    ap.add_argument("--base_repo", required=True)
    ap.add_argument("--scales", default="0.5,1.0,1.5,2.0",
                    help="Comma-separated LoRA multipliers (1.0 = trained strength)")
    ap.add_argument("--n_questions", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Lazy heavy imports (kept out of module scope so unit tests import torch-free).
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from eval_trait import (chat, generate_batch, score_batch,
                            _COHERENCE_PROMPT, JUDGE_MODEL, make_judge)
    from lora_sweep import apply_lora_scale_override

    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    data = json.loads(Path(args.trait_json).read_text())
    questions = data["questions"][20:20 + args.n_questions]
    trait_prompt = data["eval_prompt"]
    print(f"[scale_sweep] {args.family}/{args.direction}  adapter={args.adapter}  "
          f"scales={scales}  n={len(questions)}  judge={JUDGE_MODEL}")

    trait_judge = make_judge(trait_prompt)
    coh_judge = make_judge(_COHERENCE_PROMPT)

    tok = AutoTokenizer.from_pretrained(args.base_repo)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_repo, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    def score_fn(judge, qa):
        return asyncio.run(score_batch(judge, qa))

    per_scale = run_scale_sweep(
        model=model, tok=tok, questions=questions,
        trait_judge=trait_judge, coh_judge=coh_judge, scales=scales,
        apply_scale=apply_lora_scale_override,
        chat_fn=lambda t, q: chat(t, q, sys_prompt=None),
        generate_fn=generate_batch, score_fn=score_fn,
    )

    for s in scales:
        e = per_scale[float(s)]
        print(f"[scale_sweep] scale={s:4.2f}  trait_mean={e['trait_mean']:5.1f}  "
              f"coh_mean={e['coh_mean']:5.1f}  n_t={e['n_trait_parsed']} n_c={e['n_coh_parsed']}")

    result = {
        "family": args.family,
        "direction": args.direction,
        "regime": "lora_scale",
        "adapter_dir": args.adapter,
        "base_repo": args.base_repo,
        "judge_model": JUDGE_MODEL,
        "n_questions": len(questions),
        "scales": scales,
        "per_scale": per_scale,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"[scale_sweep] saved {out}")


if __name__ == "__main__":
    main()
