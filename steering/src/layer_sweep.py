"""Stage A — layer-effectiveness sweep, paper-faithful (arXiv:2507.21509 §5.1).

For a given (family, direction):
  1. Load Mistral-7B-Instruct-v0.2 or Meta-Llama-3-8B-Instruct.
  2. Load v_full = shared/vectors/{family}/{direction}_leaning_response_avg_diff.pt.
  3. For each layer_idx in 1..num_layers, install ActivationSteerer at that
     layer with coeff = --alpha. Generate one response per eval-set question.
  4. Score every response with GeminiJudge using the trait-specific
     eval_prompt loaded from shared/trait_data/{direction}_leaning.json.
  5. Save per-layer mean + the full {question, response, score} items to
     results/layer_sweep/{family}/sweep_{direction}.json.
  6. Save layer-vs-score line plot to
     results/layer_sweep/{family}/figures/layer_curve_{direction}.png.

Records two best-layer candidates in the output JSON:
  - `best_layer`            (raw argmax across all L)
  - `best_layer_excl_last_2`(argmax over L1..L_max-2; the last 2 rows of the
                            persona vector carry extraction artifacts — last-
                            layer activations contaminate with `[/INST]` / EOS
                            token features. See layer_curve_left/right.png and
                            the row-norm anomaly in v_full[-1].)

Stage A does NOT update steering.yaml — the operator (or a separate finalize
step) reads the sweep JSON and writes the chosen layer_idx into the YAML.

Usage:
  python -m src.layer_sweep --family mistral --direction left  --alpha 5 --n_questions 20
  python -m src.layer_sweep --family mistral --direction right --alpha 5 --n_questions 20
  python -m src.layer_sweep --family llama   --direction left  --alpha 5 --n_questions 20
  python -m src.layer_sweep --family llama   --direction right --alpha 5 --n_questions 20
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
sys.path.insert(0, str(_REPO_ROOT / "4_steering" / "src"))
sys.path.insert(0, str(_REPO_ROOT / "3_persona_vectors" / "shared"))

from steering import load_vector, make_steerer  # noqa: E402
from trait_questions import load_eval_questions, load_eval_prompt  # noqa: E402
from gemini_judge import GeminiJudge  # noqa: E402


FAMILY_REPO = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama":   "meta-llama/Meta-Llama-3-8B-Instruct",
}
JUDGE_MODEL = os.environ.get("VERTEX_GEMINI_MODEL", "gemini-3-flash-preview")
JUDGE_CONCURRENCY = 8


def family_resources(family: str) -> tuple[str, str]:
    """Resolve (model_repo, vector_path_template) for a family.

    Raises KeyError on unknown family. Vector path template needs `.format(direction=...)`.
    """
    repo = FAMILY_REPO[family]
    vec_template = f"3_persona_vectors/shared/vectors/{family}/{{direction}}_leaning_response_avg_diff.pt"
    return repo, vec_template


def best_layer_excl_last_n(per_layer: dict, n_excluded: int = 2) -> int:
    """argmax over per_layer keys, excluding the highest N integer keys.

    Used to surface the "real" best layer when the last rows of the persona
    vector are extraction artifacts. Accepts both int and string keys (JSON
    round-trips int keys as strings).
    """
    Ls = sorted(int(L) for L in per_layer.keys())
    if n_excluded <= 0:
        candidates = Ls
    elif len(Ls) > n_excluded:
        candidates = Ls[:-n_excluded]
    else:
        candidates = Ls

    def _mean(L: int) -> float:
        cell = per_layer.get(L)
        if cell is None:
            cell = per_layer.get(str(L))
        m = cell["mean"]
        return m if m == m else float("-inf")  # NaN-safe

    return max(candidates, key=_mean)


def chat(tok, user: str) -> str:
    msgs = [{"role": "user", "content": user}]
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
    prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
    decoded = []
    for i, plen in enumerate(prompt_lens):
        new_tokens = out[i, enc["input_ids"].shape[1]:]
        decoded.append(tok.decode(new_tokens, skip_special_tokens=True))
    return decoded


async def score_batch(judge: GeminiJudge, qa_pairs: list[tuple[str, str]]) -> list[float | None]:
    """Run judge(question=q, answer=a) concurrently across qa_pairs."""
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)
    async def _one(q: str, a: str):
        async with sem:
            return await judge(question=q, answer=a)
    return await asyncio.gather(*[_one(q, a) for q, a in qa_pairs])


def run_sweep(family: str, direction: str, alpha: float, n_questions: int) -> dict:
    repo, vec_template = family_resources(family)
    out_dir = _REPO_ROOT / "4_steering" / "results" / "layer_sweep" / family
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[layer_sweep] family={family} direction={direction} alpha={alpha} n_questions={n_questions}")
    questions = load_eval_questions(direction, n=n_questions)
    eval_prompt = load_eval_prompt(direction)
    judge = GeminiJudge(JUDGE_MODEL, eval_prompt, eval_type="0_100")

    v_full = load_vector(vec_template.format(direction=direction))
    num_layers = v_full.shape[0] - 1   # row 0 is embedding; rows 1..N are post-block
    print(f"[layer_sweep] v_full shape={tuple(v_full.shape)}, num_layers={num_layers}")

    print(f"[layer_sweep] loading {repo} ...")
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    print(f"[layer_sweep] model loaded, hidden_size={model.config.hidden_size}")

    # Pre-format all prompts once; reuse across all layers.
    prompts = [chat(tok, q) for q in questions]

    # Configure tokenizer for left-padding (decoder-only models need this for
    # batched generation so the prompt's last token aligns at position -1).
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    per_layer: dict[int, dict] = {}
    for L in range(1, num_layers + 1):
        with make_steerer(model, v_full, L, coeff=alpha, positions="all"):
            responses = generate_batch(model, tok, prompts)

        qa_pairs = list(zip(questions, responses))
        raw_scores = asyncio.run(score_batch(judge, qa_pairs))
        scores = [s for s in raw_scores if s is not None]
        mean_score = (sum(scores) / len(scores)) if scores else float("nan")
        n_parsed = len(scores)

        items = [
            {"question": q, "response": r, "score": s}
            for (q, r), s in zip(qa_pairs, raw_scores)
        ]
        per_layer[L] = {
            "mean": mean_score,
            "n_parsed": n_parsed,
            "scores": raw_scores,
            "items": items,
        }
        print(f"[layer_sweep] layer={L:2d}  mean_score={mean_score:5.1f}  n_parsed={n_parsed}/{len(qa_pairs)}")

    result = {
        "family": family,
        "direction": direction,
        "model_repo": repo,
        "alpha": alpha,
        "n_questions": n_questions,
        "num_layers": num_layers,
        "judge_model": JUDGE_MODEL,
        "per_layer": per_layer,
        "best_layer": max(
            per_layer.keys(),
            key=lambda L: per_layer[L]["mean"] if per_layer[L]["mean"] == per_layer[L]["mean"] else float("-inf"),
        ),
        "best_layer_excl_last_2": best_layer_excl_last_n(per_layer, n_excluded=2),
    }
    out_path = out_dir / f"sweep_{direction}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[layer_sweep] saved {out_path}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ls = sorted(per_layer.keys())
    means = [per_layer[L]["mean"] for L in Ls]
    plt.figure(figsize=(8, 4))
    plt.plot(Ls, means, marker="o")
    plt.axvline(result["best_layer"], linestyle="--", alpha=0.5,
                label=f"best layer = {result['best_layer']}")
    plt.axvline(result["best_layer_excl_last_2"], linestyle=":", alpha=0.7, color="tab:red",
                label=f"best layer (excl last 2) = {result['best_layer_excl_last_2']}")
    plt.xlabel("Layer (1-indexed post-block residual)")
    plt.ylabel("Mean trait-expression score (0..100)")
    title_family = "Mistral-7B-Instruct-v0.2" if family == "mistral" else "Llama-3-8B-Instruct"
    plt.title(f"{title_family} pv-steer effectiveness, {direction}-leaning, α={alpha}")
    plt.legend()
    plt.grid(alpha=0.3)
    fig_path = fig_dir / f"layer_curve_{direction}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"[layer_sweep] saved {fig_path}")

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=tuple(FAMILY_REPO.keys()), required=True)
    p.add_argument("--direction", choices=("left", "right"), required=True)
    p.add_argument("--alpha", type=float, default=5.0)
    p.add_argument("--n_questions", type=int, default=20)
    args = p.parse_args()
    run_sweep(args.family, args.direction, args.alpha, args.n_questions)


if __name__ == "__main__":
    main()
