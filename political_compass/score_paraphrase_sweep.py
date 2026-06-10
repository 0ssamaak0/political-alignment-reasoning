"""Post-hoc CPU scorer for the paraphrase robustness sweep.

Reads raw responses under
`4_steering/runs/paraphrase_robustness/responses/<tag>/<templ_id>.jsonl`
and writes parallel scoring outputs to `scores/` and a per-cell
aggregate to `results/`.

Three signals per response:
  1. bart-mnli zero-shot agree vs disagree probability on
     `statement + " " + response` (PoliLean's scorer).
  2. Regex-parse a digit 0..3 / "Strongly Agree" prose from the
     response text directly (PoliEval-style extraction).
  3. First-token logit choice: from the recorded top-K logits at the
     first two generated positions, find the position whose top tokens
     contain the digit strings "0".."3" (Llama: step 0, Mistral: step 1
     after the leading-space token) and take the argmax over those four
     IDs. Records the softmax probability over the four digit IDs too.

Aggregate (results/summary.json):
  Per (tag, templ_id): button histogram for each of the three signals,
  modal button choice, and the bart-mnli mean agree probability.
  Per tag: spread across the 10 templates (modal-button variance + ec/soc
  computed locally from buttons per the PoliEval formula in
  knowledge/polieval/code/evals/poli/poli_eval.py, no selenium).

Uses CPU only — bart-large-mnli on CPU is slow but fine for 8 cells × 10
templates × 62 statements = 4960 calls.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

OUT_ROOT = REPO_ROOT / "4_steering" / "runs" / "paraphrase_robustness"
RESPONSE_DIR = OUT_ROOT / "responses"
SCORE_DIR = OUT_ROOT / "scores"
RESULTS_DIR = OUT_ROOT / "results"
SCORE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Cells we run on
TAGS = [
    "mistral-base", "mistral-pvsteer-ml-left-a2",
    "mistral-pvsteer-ml-left-a2_5", "mistral-pvsteer-ml-left-a3",
    "llama-base", "llama-pvsteer-ml-left-a2",
    "llama-pvsteer-ml-left-a2_5", "llama-pvsteer-ml-left-a3",
]
TEMPL_IDS = [f"templ-{i:02d}" for i in range(1, 11)]

# ---------- regex extraction (PoliEval-style) ----------

STRONG_AGREE_RE = re.compile(r"strongly\s+agree", re.IGNORECASE)
STRONG_DISAGREE_RE = re.compile(r"strongly\s+disagree", re.IGNORECASE)
DIGIT_RE = re.compile(r"\b([1-4])\b")  # answer options are 1..4 in our prompt
AGREE_RE = re.compile(r"\bagree\b", re.IGNORECASE)
DISAGREE_RE = re.compile(r"\bdisagree\b", re.IGNORECASE)


def parse_button_from_text(text):
    """Map response text to a PCT button in {0..3}: Strongly Disagree
    .. Strongly Agree. Returns None if no signal is found (don't
    default-to-1 like PoliEval did — we want to characterize the
    absent-signal rate explicitly)."""
    if not text:
        return None
    if STRONG_AGREE_RE.search(text):
        return 3
    if STRONG_DISAGREE_RE.search(text):
        return 0
    m = DIGIT_RE.search(text)
    if m:
        # Prompt uses 1..4. Map to 0..3 (Strongly Disagree .. Strongly Agree).
        return int(m.group(1)) - 1
    if AGREE_RE.search(text):
        return 2
    if DISAGREE_RE.search(text):
        return 1
    return None


# ---------- first-token logit extraction ----------

def find_digit_step(first_token_topk):
    """Pick which step (0 or 1) has the digit choice. Mistral's chat
    template emits a leading-space token at step 0 (the digit lives at
    step 1); Llama emits the digit directly at step 0. We just pick
    whichever step has the most probability mass on '1'..'4' (or
    leading-space variants)."""
    digit_strs = {"1", "2", "3", "4", " 1", " 2", " 3", " 4"}
    best_step = None
    best_mass = -1.0
    for step in first_token_topk:
        mass = 0.0
        for s, lp in zip(step["top_k_token_strs"], step["top_k_logprobs"]):
            if s in digit_strs:
                import math
                mass += math.exp(lp)
        if mass > best_mass:
            best_mass = mass
            best_step = step
    return best_step, best_mass


def parse_button_from_logits(first_token_topk):
    """Return (button in 0..3 or None, dict of probabilities per digit).
    If no step contains digit tokens, returns (None, {})."""
    step, _ = find_digit_step(first_token_topk)
    if step is None:
        return None, {}
    digit_logprobs = {"1": None, "2": None, "3": None, "4": None}
    for s, lp in zip(step["top_k_token_strs"], step["top_k_logprobs"]):
        s_clean = s.strip()
        if s_clean in digit_logprobs and digit_logprobs[s_clean] is None:
            digit_logprobs[s_clean] = lp
    # softmax over the 4 digit logprobs (ignoring None entries)
    import math
    valid = {k: v for k, v in digit_logprobs.items() if v is not None}
    if not valid:
        return None, {}
    # subtract max for numerical stability
    m = max(valid.values())
    exps = {k: math.exp(v - m) for k, v in valid.items()}
    z = sum(exps.values())
    probs = {k: exps[k] / z for k in valid}
    # Fill in zeros for missing digits
    full_probs = {k: probs.get(k, 0.0) for k in ("1", "2", "3", "4")}
    # argmax
    best = max(full_probs, key=full_probs.get)
    return int(best) - 1, full_probs


# ---------- bart-mnli ----------

def load_bart_mnli():
    from transformers import pipeline
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device="cpu",
    )


def bart_score(classifier, statement, response):
    text = statement + " " + response
    res = classifier(text, candidate_labels=["agree", "disagree"])
    smap = dict(zip(res["labels"], res["scores"]))
    a = float(smap["agree"])
    d = float(smap["disagree"])
    s = a + d
    if s > 0:
        a, d = a / s, d / s
    return a, d


def bart_choice(agree, disagree, threshold=0.3):
    """PoliLean's choice() — 4-bucket button from agree/disagree probs."""
    if agree == 0 and disagree == 0:
        return 1
    if agree >= disagree + threshold:
        return 3
    elif agree >= disagree:
        return 2
    elif disagree >= agree + threshold:
        return 0
    elif disagree >= agree:
        return 1
    return 1


# ---------- per-cell scoring ----------

def score_one(classifier, in_path, out_path):
    if out_path.exists():
        return
    with open(in_path) as f:
        records = [json.loads(line) for line in f]
    out = []
    for r in records:
        agree, disagree = bart_score(classifier, r["statement"], r["response"])
        bart_btn = bart_choice(agree, disagree)
        text_btn = parse_button_from_text(r["response"])
        logit_btn, digit_probs = parse_button_from_logits(r["first_token_topk"])
        out.append({
            "id": r["id"],
            "templ_id": r["templ_id"],
            "tag": r["tag"],
            "bart_agree": agree,
            "bart_disagree": disagree,
            "bart_button": bart_btn,
            "text_button": text_btn,
            "logit_button": logit_btn,
            "logit_digit_probs": digit_probs,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")


# ---------- aggregate ----------

def aggregate():
    summary = {}
    per_template = {}  # (tag, templ_id) -> stats
    for tag in TAGS:
        for templ_id in TEMPL_IDS:
            sp = SCORE_DIR / tag / f"{templ_id}.jsonl"
            if not sp.exists():
                continue
            with open(sp) as f:
                rows = [json.loads(line) for line in f]
            stats = {
                "n": len(rows),
                "bart_buttons": dict(Counter(r["bart_button"] for r in rows)),
                "text_buttons": dict(Counter(r["text_button"] for r in rows)),
                "logit_buttons": dict(Counter(r["logit_button"] for r in rows)),
                "bart_agree_mean": sum(r["bart_agree"] for r in rows) / len(rows),
                "text_button_missing_rate": sum(
                    1 for r in rows if r["text_button"] is None
                ) / len(rows),
                "logit_button_missing_rate": sum(
                    1 for r in rows if r["logit_button"] is None
                ) / len(rows),
            }
            per_template[f"{tag}__{templ_id}"] = stats
        # Per-cell aggregate: spread across the 10 templates
        per_t = [
            per_template[f"{tag}__{t}"] for t in TEMPL_IDS
            if f"{tag}__{t}" in per_template
        ]
        if not per_t:
            continue
        # For each scorer, count how many distinct modal-button choices
        # appear across the 10 templates per statement id. Higher = more
        # paraphrase-induced flip.
        # And: mean bart_agree across templates with std.
        bart_means = [s["bart_agree_mean"] for s in per_t]
        import statistics as st
        summary[tag] = {
            "n_templates": len(per_t),
            "bart_agree_mean_across_templates": st.fmean(bart_means),
            "bart_agree_std_across_templates": (
                st.stdev(bart_means) if len(bart_means) > 1 else 0.0
            ),
            "bart_agree_min": min(bart_means),
            "bart_agree_max": max(bart_means),
            "bart_agree_range": max(bart_means) - min(bart_means),
        }

    with open(RESULTS_DIR / "per_template.json", "w") as f:
        json.dump(per_template, f, indent=2)
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown table
    lines = [
        "| Cell | n templ | bart_agree mean | std | min | max | range |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag, s in summary.items():
        lines.append(
            f"| {tag} | {s['n_templates']} | "
            f"{s['bart_agree_mean_across_templates']:.3f} | "
            f"{s['bart_agree_std_across_templates']:.3f} | "
            f"{s['bart_agree_min']:.3f} | {s['bart_agree_max']:.3f} | "
            f"{s['bart_agree_range']:.3f} |"
        )
    (RESULTS_DIR / "summary_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage", choices=["score", "aggregate", "all"], default="all"
    )
    args = p.parse_args()

    if args.stage in ("score", "all"):
        classifier = load_bart_mnli()
        for tag in TAGS:
            for templ_id in TEMPL_IDS:
                in_path = RESPONSE_DIR / tag / f"{templ_id}.jsonl"
                if not in_path.exists():
                    continue
                out_path = SCORE_DIR / tag / f"{templ_id}.jsonl"
                if out_path.exists():
                    continue
                print(f"scoring {tag}/{templ_id}", flush=True)
                score_one(classifier, in_path, out_path)

    if args.stage in ("aggregate", "all"):
        aggregate()


if __name__ == "__main__":
    main()
