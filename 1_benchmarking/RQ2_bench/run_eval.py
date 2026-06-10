"""RQ3_bench — Stage 2: inference runner.

Runs one model *cell* over `stimuli.jsonl`, asking a deductive-validity
judgement per item, and writes raw responses to `responses/<tag>.jsonl`.
Verdict parsing + scoring happen later in `compute_metrics.py` (so a run is
re-scoreable without re-generating).

Self-contained for three regimes:
  - **base**     : --model <hf repo or local path>
  - **roleplay** : add --system "<prompt>"  (or --system @path/to/prompt.txt)
  - **DPO-LoRA** : add --adapter <peft adapter dir>

Steering cells (pvsteer / pvsteer-ml) install a residual-stream hook. Rather
than re-implement that, we reuse the project's shared machinery the same way
G_K_assessing_bias does: pass --steer-tag <tag> and we enter
`custom_bench.adapters.steering_context(...)` around generation.
[VERIFY] confirm the exact steering_context signature against custom_bench/
adapters.py once available; the call site is isolated in `cell_context()`.

Belief-bias note: the prompt uses the canonical belief-bias instruction
("assume the premises are true; does the conclusion follow by logic alone")
so that any sensitivity to real-world (im)plausibility shows up as an error,
not as compliance with a "ignore truth" instruction.

Usage:
    python run_eval.py --model mistralai/Mistral-7B-Instruct-v0.2 --tag mistral-base
    python run_eval.py --model <repo> --system @prompts/left.txt --tag mistral-roleplay-left
    python run_eval.py --model <repo> --adapter <dir> --tag mistral-politune-hf-left
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

INSTRUCTION = (
    "Please check whether the following reasoning is deductively valid, ending "
    "your answer with either **valid** or **invalid**: {argument}"
)


def load_stimuli(path: Path):
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def read_system(arg: str | None) -> str | None:
    if not arg:
        return None
    if arg.startswith("@"):
        return Path(arg[1:]).read_text().strip()
    return arg


def build_prompt(tokenizer, argument: str, system: str | None,
                 instruction: str = INSTRUCTION) -> str:
    """Apply the chat template; fall back to folding `system` into the user
    turn for templates without a system role (e.g. Mistral-Instruct-v0.2),
    mirroring G_K_assessing_bias/run_eval.py:build_chat_inputs."""
    user = instruction.format(argument=argument)
    if system:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            user = f"{system}\n\n{user}"  # no system role -> fold in
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
    )


@contextlib.contextmanager
def cell_context(model, steer_tag: str | None):
    """Yield the model with steering hooks active. Steering reuses the shared
    infra: `steering_context` is a context manager whose __enter__ registers
    forward hooks on `model`'s layers (and returns the *steerer*, not the
    model). Per its own docstring the call site generates with the original
    `model` — so we yield `model`, NOT the steerer (which has no .generate /
    .device). We import lazily so non-steering runs have no dependency.
    """
    if not steer_tag:
        yield model
        return
    sys.path.insert(0, str(HERE.parent))  # make `custom_bench` importable
    from custom_bench.adapters import steering_context, is_steering_tag  # type: ignore
    if not is_steering_tag(steer_tag):
        # steering_context would return nullcontext() and silently run UNSTEERED
        # — fail loud instead of mislabeling base output as steered.
        raise SystemExit(
            f"--steer-tag {steer_tag!r} is not a known steering cell "
            f"(would silently run unsteered). Check the tag against STEERING_CONFIGS."
        )
    with steering_context(model, steer_tag):  # installs hooks on `model`
        yield model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--adapter", default=None, help="PEFT LoRA adapter dir (optional)")
    ap.add_argument("--lora-scale", type=float, default=None,
                    help="override PEFT LoRA scaling after adapter load (alignment-"
                         "strength knob; trained default 2.0). Needs --adapter.")
    ap.add_argument("--system", default=None, help="system prompt, or @file (optional)")
    ap.add_argument("--steer-tag", default=None, help="steering cell tag (optional)")
    ap.add_argument("--tag", required=True, help="cell name → responses/<tag>.jsonl")
    ap.add_argument("--stimuli", type=Path, default=HERE / "stimuli.jsonl")
    ap.add_argument("--out-dir", type=Path, default=HERE / "responses")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=None, help="first-N stimuli (debug)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=args.device
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        if args.lora_scale is not None:
            sys.path.insert(0, str(HERE.parent))  # make `custom_bench` importable
            from custom_bench.adapters import apply_lora_scale_override  # type: ignore
            apply_lora_scale_override(model, args.lora_scale)
    model.eval()

    system = read_system(args.system)
    stimuli = load_stimuli(args.stimuli)
    if args.limit:
        stimuli = stimuli[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.tag}.jsonl"

    n = 0
    with cell_context(model, args.steer_tag) as run_model, open(out_path, "w") as fout:
        for item in stimuli:
            prompt = build_prompt(tok, item["text"], system)
            inputs = tok(prompt, return_tensors="pt").to(run_model.device)
            gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
            if args.temperature and args.temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=args.temperature)
            else:
                gen_kwargs.update(do_sample=False)
            with torch.no_grad():
                out = run_model.generate(**inputs, **gen_kwargs, pad_token_id=tok.pad_token_id)
            raw = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            row = dict(item)
            row["cell"] = args.tag
            row["raw_response"] = raw.strip()
            row["n_tokens_generated"] = int(out[0].shape[0] - inputs["input_ids"].shape[1])
            fout.write(json.dumps(row) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(stimuli)}")

    print(f"wrote {n} responses -> {out_path}")


if __name__ == "__main__":
    main()
