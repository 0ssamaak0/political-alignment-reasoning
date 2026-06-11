"""Stage B — extract pos/neg responses from a base model with persona system
prompts, judge them with Vertex Gemini, dump CSV.

For one (model, leaning), runs upstream `eval/eval_persona.py`'s `eval_batched`
twice — once with `persona_instruction_type=pos`, once with `neg` — using:
  - our `GeminiJudge` as a drop-in for upstream's `OpenAiJudge`,
  - an HF `model.generate` batched loop as a drop-in for upstream's vllm-based
    `sample()`.

The HF backend (instead of vllm) is a deliberate deviation — see
`assumptions.md` §B3 for why and the cost (slower but isolates Phase 2 from
the conda `main` env that Phase 1 inference depends on).

Outputs:
    extract_runs/{model}/{leaning}_leaning_{pos,neg}.csv

CLI:
    python3 extract_persona_responses.py --model llama   --leaning right
    python3 extract_persona_responses.py --model llama   --leaning left
    python3 extract_persona_responses.py --model mistral --leaning right
    python3 extract_persona_responses.py --model mistral --leaning left

Stage A artifact (`trait_data/{leaning}_leaning.json`) must exist; we copy it
into the upstream tree at runtime since `eval_persona.py` reads via a
relative path.
"""

import argparse
import asyncio
import os
import shutil
import sys
import types as pytypes
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "persona_vectors"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(UPSTREAM))

# ---- Bypass upstream's OpenAI/HF credential gate ---------------------------
os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-replaced-by-gemini-judge")
if not os.environ.get("HF_TOKEN"):
    raise SystemExit(
        "HF_TOKEN env var required (gated llama/mistral on HF). "
        "On the A100 VM the conda 'main' env should already have it; "
        "if not, `export HF_TOKEN=$(cat ~/.cache/huggingface/token)`."
    )

# ---- Stub vllm BEFORE upstream eval_persona imports it ---------------------
# Reasons in assumptions.md §B3. We never actually call any vllm method (we
# replace `sample()` and load HF directly), so the stubs only need to be
# valid module objects with the imported names defined.
class _StubLLM:
    def __init__(self, *a, **kw):
        raise RuntimeError("vllm stub LLM should not be instantiated")


class _StubSamplingParams:
    def __init__(self, *a, **kw):
        pass


class _StubLoRARequest:
    def __init__(self, *a, **kw):
        pass


_vllm_stub = pytypes.ModuleType("vllm")
_vllm_stub.LLM = _StubLLM
_vllm_stub.SamplingParams = _StubSamplingParams
sys.modules["vllm"] = _vllm_stub

_vlora = pytypes.ModuleType("vllm.lora")
_vlora_req = pytypes.ModuleType("vllm.lora.request")
_vlora_req.LoRARequest = _StubLoRARequest
sys.modules["vllm.lora"] = _vlora
sys.modules["vllm.lora.request"] = _vlora_req

# ---- Patch upstream `judge` module -----------------------------------------
from gemini_judge import GeminiJudge  # noqa: E402

_judge_stub = pytypes.ModuleType("judge")
_judge_stub.OpenAiJudge = GeminiJudge
sys.modules["judge"] = _judge_stub

# ---- Imports from upstream (now safe) --------------------------------------
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from eval import eval_persona  # noqa: E402

# ---- HF replacement for upstream `sample()` --------------------------------
@torch.no_grad()
def hf_sample(model, tokenizer, conversations,
              top_p=1, max_tokens=1000, temperature=1, min_tokens=1,
              lora_path=None, batch_size=4):
    """HF-transformers replacement for `eval_persona.sample` (which uses vllm).

    Same `(prompts, answers)` return contract. We don't accept LoRA at this
    stage (Stage B uses the unmodified base model); `lora_path` is ignored.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    texts = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
        for c in conversations
    ]

    answers = []
    do_sample = temperature > 0
    print(f"[hf_sample] generating {len(texts)} responses, batch_size={batch_size}")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        ids = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        out = model.generate(**ids, **gen_kwargs)
        prompt_len = ids.input_ids.shape[1]
        for o in out:
            answers.append(tokenizer.decode(o[prompt_len:], skip_special_tokens=True))
        if (i // batch_size) % 10 == 0:
            print(f"[hf_sample]   {i + len(batch_texts)}/{len(texts)} done")
    return texts, answers


# Monkey-patch upstream's sample with our HF version. eval_persona's
# eval_batched (and Question.eval) call sample() — they pick up our patched
# version because `eval_persona.sample` is now `hf_sample`.
eval_persona.sample = hf_sample

MODELS = {
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
}

DEFAULT_JUDGE = os.environ.get("VERTEX_GEMINI_MODEL", "gemini-3-flash-preview")


def setup_trait_artifact(leaning: str) -> None:
    src = HERE / "trait_data" / f"{leaning}_leaning.json"
    if not src.exists():
        raise FileNotFoundError(f"Stage A artifact missing: {src}")
    for sub in ("trait_data_extract", "trait_data_eval"):
        dst_dir = UPSTREAM / "data_generation" / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{leaning}_leaning.json"
        if not dst.exists() or dst.read_text() != src.read_text():
            shutil.copy2(src, dst)
            print(f"[stageB] copied {src.name} → {dst}")


def run_one_side(
    *,
    model,
    tokenizer,
    trait_name: str,
    side: str,
    output_path: Path,
    judge_model: str,
    judge_eval_type: str,
    n_per_question: int,
    max_concurrent_judges: int,
    max_tokens: int,
    batch_size: int,
) -> None:
    if output_path.exists():
        print(f"[stageB] {output_path} exists; skipping.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temperature = 0.0 if n_per_question == 1 else 1.0

    questions = eval_persona.load_persona_questions(
        trait_name,
        temperature=temperature,
        persona_instructions_type=side,
        assistant_name=trait_name if side == "pos" else "helpful",
        judge_model=judge_model,
        eval_type="0_100",
        version="extract",
    )

    if judge_eval_type != "0_100":
        n_swapped = 0
        for q in questions:
            if trait_name in q.judges:
                old = q.judges[trait_name]
                q.judges[trait_name] = GeminiJudge(
                    judge_model, old.prompt_template, eval_type=judge_eval_type
                )
                n_swapped += 1
        print(f"[stageB] swapped {n_swapped} trait judges to GeminiJudge({judge_eval_type})")

    print(f"[stageB] {len(questions)} Question objects × n_per_question={n_per_question}")

    # Inject batch_size into our hf_sample via partial. eval_batched calls
    # eval_persona.sample(...) with positional args mirroring vllm's signature;
    # we rebind the module attribute to a partial that pre-fills batch_size.
    from functools import partial as _partial
    eval_persona.sample = _partial(hf_sample, batch_size=batch_size)

    outputs_list = asyncio.run(
        eval_persona.eval_batched(
            questions,
            model,
            tokenizer,
            coef=0,
            vector=None,
            layer=None,
            n_per_question=n_per_question,
            max_concurrent_judges=max_concurrent_judges,
            max_tokens=max_tokens,
            steering_type="last",
            lora_path=None,
        )
    )
    outputs = pd.concat(outputs_list)
    outputs.to_csv(output_path, index=False)
    print(f"[stageB] wrote {output_path}")
    for col in (trait_name, "coherence"):
        if col in outputs.columns:
            mean = outputs[col].mean()
            std = outputs[col].std()
            print(f"  {col}: mean={mean:.2f} std={std:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage B extraction (pos+neg) for one model×leaning pair."
    )
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--leaning", required=True, choices=["right", "left"])
    ap.add_argument("--judge_model", default=DEFAULT_JUDGE)
    ap.add_argument(
        "--judge_eval_type",
        choices=["0_10", "0_100"],
        default="0_100",
        help="Trait judge score range. 0_100 matches the eval_prompt our Stage A "
             "generated (which asks for 0–100 ratings). See assumptions.md §B1.",
    )
    ap.add_argument("--n_per_question", type=int, default=10)
    ap.add_argument("--max_concurrent_judges", type=int, default=10)
    ap.add_argument("--max_tokens", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=4,
                    help="HF generation batch size (assumptions.md §B4).")
    args = ap.parse_args()

    base_model = MODELS[args.model]
    trait_name = f"{args.leaning}_leaning"
    setup_trait_artifact(args.leaning)

    out_dir = HERE / "extract_runs" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = [out_dir / f"{trait_name}_{s}.csv" for s in ("pos", "neg")]
    if all(p.exists() for p in expected):
        print(f"[stageB] both CSVs already exist for {args.model}×{trait_name}; "
              f"skipping HF model load.")
        return 0

    print(f"[stageB] loading HF model {base_model} (bf16, device_map=auto)…")
    cwd = os.getcwd()
    os.chdir(UPSTREAM)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        for side in ("pos", "neg"):
            output_path = out_dir / f"{trait_name}_{side}.csv"
            print(f"\n=== {args.model} × {trait_name} × {side} → {output_path} ===")
            run_one_side(
                model=model,
                tokenizer=tokenizer,
                trait_name=trait_name,
                side=side,
                output_path=output_path,
                judge_model=args.judge_model,
                judge_eval_type=args.judge_eval_type,
                n_per_question=args.n_per_question,
                max_concurrent_judges=args.max_concurrent_judges,
                max_tokens=args.max_tokens,
                batch_size=args.batch_size,
            )
    finally:
        os.chdir(cwd)

    import gemini_judge
    fc = gemini_judge.FAIL_COUNTERS
    if fc["sample"] or fc["parse"]:
        print(f"[stageB] judge failures — sample: {fc['sample']}, parse: {fc['parse']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
