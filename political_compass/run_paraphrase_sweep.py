"""Paraphrase robustness sweep (Röttger ACL 2024 §3.4) on Mistral and
Llama base + pvsteer-ml-left cells at α ∈ {2.0, 2.5, 3.0}.

Background: SCORING_VALIDITY.md (this directory) explains why a single
PCT number is unreliable for LLMs. This script runs the explicit-MC
paraphrase experiment from `spinning_arrow/code/data/templates/`
(templ-01..10 with ans-01, jail empty) on each cell so we can compute
within-method paraphrase spread and compare it to the α-sweep Δ.

Per cell × per template: 62 PCT statements, T=0 greedy, 1 run.

Outputs land under
`4_steering/runs/paraphrase_robustness/responses/<tag>/<templ_id>.jsonl`.

Per response we record:
  - verbatim response text (greedy decode of up to MAX_NEW_TOKENS tokens)
  - top-K=20 token IDs + logprobs at the first two generated positions,
    so first-token-logit scoring over {tok("0"),tok("1"),tok("2"),tok("3")}
    can be reconstructed offline for either Llama (digit at step 0) or
    Mistral (leading-space token at step 0, digit at step 1).

bart-mnli scoring + per-cell aggregate live in
`score_paraphrase_sweep.py` (CPU post-hoc).

Resumable: each (tag, templ_id) skips if its output file already exists.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "1_benchmarking"))
from pipeline import adapters as bench_adapters  # noqa: E402

# ---------- paths ----------

OUT_ROOT = REPO_ROOT / "4_steering" / "runs" / "paraphrase_robustness"
RESPONSE_DIR = OUT_ROOT / "responses"
RESPONSE_DIR.mkdir(parents=True, exist_ok=True)

SPINNING_ARROW_DATA = HERE / "spinning_arrow" / "code" / "data"
TEMPLATES_CSV = SPINNING_ARROW_DATA / "templates" / "prompt_templates.csv"
ANSWERS_CSV = SPINNING_ARROW_DATA / "templates" / "answer_options.csv"
POLILEAN_STATEMENTS = HERE / "PoliLean" / "response" / "example.jsonl"

# ---------- config matrix ----------

LLAMA_REPO = "meta-llama/Meta-Llama-3-8B-Instruct"
MISTRAL_REPO = "mistralai/Mistral-7B-Instruct-v0.2"

# (tag, family, repo). All cells are left-leaning (per user choice). Base
# controls included to bound the spread vs intrinsic instrument noise.
CELLS = [
    ("mistral-base",                   "mistral", MISTRAL_REPO),
    ("mistral-pvsteer-ml-left-a2",     "mistral", MISTRAL_REPO),
    ("mistral-pvsteer-ml-left-a2_5",   "mistral", MISTRAL_REPO),
    ("mistral-pvsteer-ml-left-a3",     "mistral", MISTRAL_REPO),
    ("llama-base",                     "llama",   LLAMA_REPO),
    ("llama-pvsteer-ml-left-a2",       "llama",   LLAMA_REPO),
    ("llama-pvsteer-ml-left-a2_5",     "llama",   LLAMA_REPO),
    ("llama-pvsteer-ml-left-a3",       "llama",   LLAMA_REPO),
]

# Generation: greedy / deterministic / single run. The whole point of
# this sweep is to isolate paraphrase variance — sampling noise would
# contaminate that.
MAX_NEW_TOKENS = 80
TOP_K_LOGITS = 20  # how many top tokens to keep at each of the first 2 steps
N_FIRST_STEPS = 2   # capture logits for steps 0 and 1 (covers Mistral's
                    # leading-space token + digit OR Llama's digit directly).
BATCH_SIZE = 8

# ---------- template/answer loading ----------

def _parse_csv(path):
    """Tiny CSV parser that handles the quoted multi-line values used in
    prompt_templates.csv (one of the cells contains real newlines inside
    quotes; csv.DictReader handles this natively)."""
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_explicit_templates(n=10):
    """templ-01..10, the explicit MC-style paraphrases from Röttger §3.4."""
    rows = _parse_csv(TEMPLATES_CSV)
    explicit = [r for r in rows if r["templ_note"] == "explicit"]
    assert len(explicit) >= n, f"only {len(explicit)} explicit templates in CSV"
    return explicit[:n]


def load_answer_options():
    rows = _parse_csv(ANSWERS_CSV)
    main = [r for r in rows if r["ans_note"] == "main"]
    assert len(main) == 1, f"expected exactly 1 main answer option, got {len(main)}"
    return main[0]["ans_prompt"]


def load_statements():
    with open(POLILEAN_STATEMENTS) as f:
        return json.load(f)


# ---------- prompt assembly ----------

def assemble_user_prompt(templ, ans_prompt, statement):
    """Substitute pct_prompt/ans_prompt/jail_prompt into a templ_prompt.
    Jail is empty (jail-00) per Röttger §3.4. Trailing whitespace
    cleaned up so the prompt ends cleanly with the answer options
    block."""
    user = (
        templ["templ_prompt"]
        .replace("{pct_prompt}", statement)
        .replace("{ans_prompt}", ans_prompt)
        .replace("{jail_prompt}", "")
    )
    return user.rstrip()


def build_prompt_text(tok, user_text):
    """No system prompt — matches base/pvsteer behaviour in PoliLean."""
    msgs = [{"role": "user", "content": user_text}]
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


# ---------- model loading ----------

def load_base(repo, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=dtype, device_map="auto"
    )
    model.eval()
    return tok, model


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------- generation ----------

@torch.no_grad()
def generate_with_first_token_logits(model, tok, prompts):
    """Greedy generate (T=0) and also capture top-K logprobs at the first
    N_FIRST_STEPS generated positions for offline first-token-logit
    scoring."""
    prev_padding_side = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(
            prompts, return_tensors="pt", padding=True, truncation=False,
        ).to(model.device)
    finally:
        tok.padding_side = prev_padding_side
    in_len = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tok.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    seqs = out.sequences  # [B, in_len + gen_len]
    scores = out.scores   # tuple of length gen_len, each [B, vocab]
    decoded = []
    per_step_topk = []  # list-of-list: per batch item, per step in [0..N_FIRST_STEPS)
    for i in range(seqs.shape[0]):
        decoded.append(
            tok.decode(seqs[i][in_len:], skip_special_tokens=True).strip()
        )
        steps_for_i = []
        for s in range(min(N_FIRST_STEPS, len(scores))):
            step_logits = scores[s][i]  # [vocab]
            step_logprobs = F.log_softmax(step_logits, dim=-1)
            top_vals, top_ids = torch.topk(step_logprobs, k=TOP_K_LOGITS)
            steps_for_i.append({
                "step": s,
                "top_k_token_ids": top_ids.tolist(),
                "top_k_token_strs": [tok.decode([int(t)]) for t in top_ids.tolist()],
                "top_k_logprobs": [float(v) for v in top_vals.tolist()],
            })
        per_step_topk.append(steps_for_i)
    return decoded, per_step_topk


# ---------- per-(cell × template) ----------

def run_cell_template(model, tok, tag, templ, ans_prompt, statements):
    out_dir = RESPONSE_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{templ['templ_id']}.jsonl"
    if out_path.exists():
        return out_path

    # Greedy, but set seed anyway for reproducibility of any sampler edge.
    seed = abs(hash((tag, templ["templ_id"]))) % (2**31)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    user_prompts = [
        assemble_user_prompt(templ, ans_prompt, it["statement"])
        for it in statements
    ]
    chat_prompts = [build_prompt_text(tok, u) for u in user_prompts]

    records = []
    pbar = tqdm(
        range(0, len(chat_prompts), BATCH_SIZE),
        desc=f"{tag} {templ['templ_id']}",
    )
    with bench_adapters.steering_context(model, tag):
        for start in pbar:
            batch_idx = range(start, min(start + BATCH_SIZE, len(chat_prompts)))
            batch_prompts = [chat_prompts[i] for i in batch_idx]
            decoded, per_step_topk = generate_with_first_token_logits(
                model, tok, batch_prompts
            )
            for k, i in enumerate(batch_idx):
                records.append({
                    "id": statements[i]["id"],
                    "statement": statements[i]["statement"],
                    "templ_id": templ["templ_id"],
                    "tag": tag,
                    "user_prompt": user_prompts[i],
                    "response": decoded[k],
                    "first_token_topk": per_step_topk[k],
                })

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return out_path


# ---------- driver ----------

def cells_for_family(family):
    return [c for c in CELLS if c[1] == family]


def run_family(family):
    """Load base model once per family, then iterate cells × templates."""
    cells = cells_for_family(family)
    if not cells:
        print(f"[{family}] no cells found", flush=True)
        return
    repo = cells[0][2]
    print(f"\n========== {family} ({repo}) ==========", flush=True)
    # Skip-load if all cells already done.
    all_done = True
    templates = load_explicit_templates()
    for tag, _, _ in cells:
        for t in templates:
            if not (RESPONSE_DIR / tag / f"{t['templ_id']}.jsonl").exists():
                all_done = False
                break
        if not all_done:
            break
    if all_done:
        print(f"[{family}] all cells × templates done; skipping load",
              flush=True)
        return

    tok, model = load_base(repo)
    ans_prompt = load_answer_options()
    statements = load_statements()
    print(f"[{family}] {len(templates)} templates × {len(statements)} statements",
          flush=True)

    try:
        for tag, fam, _ in cells:
            assert fam == family
            for templ in templates:
                run_cell_template(model, tok, tag, templ, ans_prompt, statements)
            print(f"[{family}] {tag} done", flush=True)
    finally:
        del model, tok
        free_gpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--family", choices=["mistral", "llama", "both"], default="both",
        help="Restrict to a single family (useful for per-VM splits)."
    )
    args = p.parse_args()

    if args.family in ("mistral", "both"):
        run_family("mistral")
    if args.family in ("llama", "both"):
        run_family("llama")


if __name__ == "__main__":
    main()
