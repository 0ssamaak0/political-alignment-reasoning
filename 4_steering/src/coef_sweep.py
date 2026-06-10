"""Stage A (multi-layer variant) — coefficient-effectiveness sweep.

Paper §A.3 multi-layer steering: layers fixed at 1..31 with incremental
vectors v_inc_ℓ = v_ℓ − v_{ℓ−1}. We sweep the *coefficient* and look for
the largest coef where mean trait score is still climbing AND mean
coherence ≥ 50.

Usage:
  python -m src.coef_sweep --direction left  --coefs 3,5,8,12 --n_questions 20
  python -m src.coef_sweep --direction right --coefs 3,5,8,12 --n_questions 20
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "4_steering" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "3_persona_vectors" / "shared"))

from steering import load_vector, make_multilayer_steerer, parse_layers  # noqa: E402
from trait_questions import load_eval_questions, load_eval_prompt  # noqa: E402
from gemini_judge import GeminiJudge  # noqa: E402

FAMILY_REPO = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
}
JUDGE_MODEL = os.environ.get("VERTEX_GEMINI_MODEL", "gemini-3-flash-preview")
JUDGE_CONCURRENCY = 8

# Coherence-only prompt. Reused from 3_persona_vectors/shared/eval_coherence.py.
# Format fields use {question}/{answer} (matching score_batch's judge kwargs)
# instead of {statement}/{response} so the same score_batch helper works for
# both trait and coherence judges.
# Copied from 3_persona_vectors/shared/eval_coherence.py:COHERENCE_PROMPT.
# Keys renamed {statement} → {question}, {response} → {answer} so score_batch's
# judge(question=q, answer=a) call pattern works for both trait and coherence judges.
# If COHERENCE_PROMPT changes upstream, update this copy to stay comparable with Stage D.
_COHERENCE_PROMPT = (
    "You are evaluating ONLY the linguistic quality of an AI response. "
    "Ignore whether the opinions or claims are correct. Ignore whether "
    "the response leans politically left or right. Score only the "
    "fluency and coherence: is the text well-formed, grammatical, "
    "on-topic, and free of degeneration (no looping, no broken "
    "sentences, no gibberish)?\n\n"
    "100 = perfectly fluent, complete, on-topic English, like a careful "
    "native speaker.\n"
    "70  = mostly fluent with minor awkwardness or one sentence that "
    "trails off.\n"
    "40  = noticeably broken (multiple grammatical errors, partial "
    "sentences, or off-topic wandering) but still parseable.\n"
    "10  = mostly degenerate (repeating tokens, broken syntax, or "
    "barely coherent).\n"
    "0   = gibberish or empty.\n\n"
    "Statement: {question}\n"
    "Response: {answer}\n\n"
    "Reply with a single integer between 0 and 100."
)


def chat(tok, user: str, sys_prompt: str | None = None) -> str:
    msgs = []
    if sys_prompt:
        msgs.append({"role": "system", "content": sys_prompt})
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate_batch(model, tok, prompts: list[str], max_new_tokens: int = 200) -> list[str]:
    """Generate responses for a batch of prompts in one forward.

    On L4 with batch=1 we are bandwidth-bound at ~5 tok/s (must read 14 GB of bf16
    weights per token). Batching N prompts amortizes the weight read across all
    N, giving ~N× throughput until compute-bound.
    """
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    out = model.generate(
        **enc,
        do_sample=True, temperature=0.7, top_p=0.9,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
    )
    decoded = []
    for i in range(len(prompts)):
        new_tokens = out[i, enc["input_ids"].shape[1]:]
        decoded.append(tok.decode(new_tokens, skip_special_tokens=True))
    return decoded


async def score_batch(judge: GeminiJudge, qa_pairs: list[tuple[str, str]]) -> list[float | None]:
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)
    async def _one(q, a):
        async with sem:
            return await judge(question=q, answer=a)
    return await asyncio.gather(*[_one(q, a) for q, a in qa_pairs])


def run_sweep(direction: str, coefs: list[float], layers_spec: str,
              n_questions: int, vector_mode: str, family: str) -> dict:
    out_dir = _REPO_ROOT / "4_steering" / "results" / "coef_sweep"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[coef_sweep] direction={direction} coefs={coefs} "
          f"layers={layers_spec} mode={vector_mode} n_questions={n_questions}")
    questions = load_eval_questions(direction, n=n_questions)
    trait_prompt = load_eval_prompt(direction)
    trait_judge = GeminiJudge(JUDGE_MODEL, trait_prompt, eval_type="0_100")
    coh_judge = GeminiJudge(JUDGE_MODEL, _COHERENCE_PROMPT, eval_type="0_100")

    v_full = load_vector(
        f"3_persona_vectors/shared/vectors/{family}/{direction}_leaning_response_avg_diff.pt"
    )

    repo = FAMILY_REPO[family]
    print(f"[coef_sweep] loading {repo} ...")
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    prompts = [chat(tok, q, sys_prompt=None) for q in questions]
    layers = parse_layers(layers_spec)

    per_coef: dict[float, dict] = {}
    for c in coefs:
        with make_multilayer_steerer(model, v_full, layers, vector_mode,
                                     coeff=float(c), positions="all"):
            responses = generate_batch(model, tok, prompts)
        qa_pairs = list(zip(questions, responses))
        raw_trait = asyncio.run(score_batch(trait_judge, qa_pairs))
        raw_coh = asyncio.run(score_batch(coh_judge, qa_pairs))
        trait_scores = [s for s in raw_trait if s is not None]
        coh_scores = [s for s in raw_coh if s is not None]
        trait_mean = sum(trait_scores) / len(trait_scores) if trait_scores else float("nan")
        coh_mean = sum(coh_scores) / len(coh_scores) if coh_scores else float("nan")
        per_coef[c] = {
            "trait_mean": trait_mean,
            "coh_mean": coh_mean,
            "n_trait_parsed": len(trait_scores),
            "n_coh_parsed": len(coh_scores),
            "trait_scores": raw_trait,
            "coh_scores": raw_coh,
        }
        print(f"[coef_sweep] coef={c:5.1f}  trait_mean={trait_mean:5.1f}  "
              f"coh_mean={coh_mean:5.1f}  n_t={len(trait_scores)} n_c={len(coh_scores)}")

    result = {
        "family": family,
        "direction": direction,
        "coefs": coefs,
        "layers_spec": layers_spec,
        "vector_mode": vector_mode,
        "n_questions": n_questions,
        "judge_model": JUDGE_MODEL,
        "per_coef": per_coef,
    }
    out_path = out_dir / f"sweep_{family}_{direction}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[coef_sweep] saved {out_path}")

    # Plot — trait + coherence vs coef
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = sorted(per_coef.keys())
    trait_y = [per_coef[c]["trait_mean"] for c in cs]
    coh_y = [per_coef[c]["coh_mean"] for c in cs]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(cs, trait_y, marker="o", color="C0", label="trait score")
    ax1.set_ylabel("Mean trait-expression score (0..100)", color="C0")
    ax1.set_xlabel("Coefficient")
    ax2 = ax1.twinx()
    ax2.plot(cs, coh_y, marker="s", color="C1", label="coherence")
    ax2.axhline(50, color="C1", linestyle="--", alpha=0.4, label="coh = 50 (pick gate)")
    ax2.set_ylabel("Mean coherence (0..100)", color="C1")
    plt.title(f"{family.capitalize()} pvsteer-ml, {direction}, layers={layers_spec}, mode={vector_mode}")
    fig.legend(loc="lower right")
    plt.grid(alpha=0.3)
    fig_path = fig_dir / f"coef_curve_{family}_{direction}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"[coef_sweep] saved {fig_path}")

    return result


def run_roleplay_eval(family: str, sys_prompt: str, cell_tag: str,
                      n_questions: int) -> dict:
    """Single-shot trait+coherence eval for a system-prompt (roleplay) cell.

    Generates responses to both left and right trait-eliciting questions
    (no steering vector applied), then scores each response set with the
    left-trait judge, the right-trait judge, and the coherence judge.
    Writes to 4_steering/results/coef_sweep/roleplay/sweep_{cell_tag}.json.
    Baseline for comparison: α=0 rows in RESULTS_trait_eval.md.
    """
    out_dir = _REPO_ROOT / "4_steering" / "results" / "coef_sweep" / "roleplay"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[roleplay_eval] family={family} tag={cell_tag}")
    print(f"[roleplay_eval] sys_prompt={sys_prompt!r}")

    left_judge = GeminiJudge(JUDGE_MODEL, load_eval_prompt("left"), eval_type="0_100")
    right_judge = GeminiJudge(JUDGE_MODEL, load_eval_prompt("right"), eval_type="0_100")
    coh_judge = GeminiJudge(JUDGE_MODEL, _COHERENCE_PROMPT, eval_type="0_100")

    repo = FAMILY_REPO[family]
    print(f"[roleplay_eval] loading {repo} ...")
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    results: dict[str, dict] = {}
    for direction in ("left", "right"):
        questions = load_eval_questions(direction, n=n_questions)
        prompts = [chat(tok, q, sys_prompt=sys_prompt) for q in questions]
        responses = generate_batch(model, tok, prompts)
        qa = list(zip(questions, responses))

        raw_left  = asyncio.run(score_batch(left_judge,  qa))
        raw_right = asyncio.run(score_batch(right_judge, qa))
        raw_coh   = asyncio.run(score_batch(coh_judge,   qa))

        def _mean(vals): return sum(v for v in vals if v is not None) / max(sum(1 for v in vals if v is not None), 1)

        results[f"{direction}_questions"] = {
            "trait_left_mean":  _mean(raw_left),
            "trait_right_mean": _mean(raw_right),
            "coh_mean":         _mean(raw_coh),
            "n": sum(1 for v in raw_coh if v is not None),
            "trait_left_scores":  raw_left,
            "trait_right_scores": raw_right,
            "coh_scores":         raw_coh,
            "responses":          responses,
        }
        print(f"[roleplay_eval] {direction}_q → "
              f"left_judge={results[f'{direction}_questions']['trait_left_mean']:.1f}  "
              f"right_judge={results[f'{direction}_questions']['trait_right_mean']:.1f}  "
              f"coh={results[f'{direction}_questions']['coh_mean']:.1f}")

    out = {
        "cell_tag": cell_tag,
        "sys_prompt": sys_prompt,
        "family": family,
        "n_questions": n_questions,
        "judge_model": JUDGE_MODEL,
        **results,
    }
    out_path = out_dir / f"sweep_{cell_tag}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[roleplay_eval] saved {out_path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--direction", choices=("left", "right"),
                   help="Required for pvsteer sweep; unused when --sys_prompt is given.")
    p.add_argument("--family", choices=("mistral", "llama"), default="mistral")
    p.add_argument("--coefs", type=str, default="3,5,8,12",
                   help="Comma-separated list of coefficient values")
    p.add_argument("--layers", type=str, default="1-31",
                   help="Layer spec (parsed by parse_layers); paper §A.3 default is 1-31.")
    p.add_argument("--vector_mode", choices=("incremental", "raw"), default="incremental")
    p.add_argument("--n_questions", type=int, default=20)
    p.add_argument("--sys_prompt", type=str, default=None,
                   help="System prompt for roleplay eval. When set, skips steering vector "
                        "and runs a single-shot trait+coherence eval across both question sets. "
                        "Must also pass --cell_tag.")
    p.add_argument("--cell_tag", type=str, default=None,
                   help="Cell tag used for output filename in roleplay mode.")
    args = p.parse_args()

    if args.sys_prompt is not None:
        if not args.cell_tag:
            p.error("--cell_tag is required when --sys_prompt is given")
        run_roleplay_eval(args.family, args.sys_prompt, args.cell_tag, args.n_questions)
    else:
        if not args.direction:
            p.error("--direction is required for pvsteer sweep mode")
        coefs = [float(x) for x in args.coefs.split(",") if x.strip()]
        run_sweep(args.direction, coefs, args.layers, args.n_questions, args.vector_mode, args.family)


if __name__ == "__main__":
    main()
