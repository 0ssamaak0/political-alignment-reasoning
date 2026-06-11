#!/usr/bin/env python3
"""Render lm-evaluation-harness `results_*.json` files into a markdown table
comparing one or more models on our four standard tasks, with the
Open-LLM-Leaderboard-published baselines as a sanity bracket.

Usage:
    python format_results.py <results.json> [<results.json> ...]

Example:
    python format_results.py \\
        results/mistral/mistralai__Mistral-7B-Instruct-v0.2/results_*.json \\
        results/llama3/meta-llama__Meta-Llama-3-8B-Instruct/results_*.json
"""

import argparse
import json
import sys
from pathlib import Path

# Open-LLM-Leaderboard baselines (OLLM v1 for MMLU, OLLM v2 for BBH).
# Source: huggingface.co/datasets/open-llm-leaderboard{,-old}/<model>-details
OLLM_BASELINES = {
    "mmlu_formal_logic": {
        "mistralai/Mistral-7B-Instruct-v0.2":  0.421,
        "meta-llama/Meta-Llama-3-8B-Instruct": 0.484,
    },
    "bbh_cot_fewshot_boolean_expressions": {
        "mistralai/Mistral-7B-Instruct-v0.2":  0.780,
        "meta-llama/Meta-Llama-3-8B-Instruct": 0.748,
    },
    "bbh_cot_fewshot_formal_fallacies": {
        "mistralai/Mistral-7B-Instruct-v0.2":  0.552,
        "meta-llama/Meta-Llama-3-8B-Instruct": 0.552,
    },
    "bbh_cot_fewshot_logical_deduction_three_objects": {
        "mistralai/Mistral-7B-Instruct-v0.2":  0.432,
        "meta-llama/Meta-Llama-3-8B-Instruct": 0.588,
    },
}

TASK_ORDER = [
    "mmlu_formal_logic",
    "bbh_cot_fewshot_boolean_expressions",
    "bbh_cot_fewshot_formal_fallacies",
    "bbh_cot_fewshot_logical_deduction_three_objects",
]

TASK_LABEL = {
    "mmlu_formal_logic":                                "MMLU formal_logic",
    "bbh_cot_fewshot_boolean_expressions":              "BBH boolean_expressions",
    "bbh_cot_fewshot_formal_fallacies":                 "BBH formal_fallacies",
    "bbh_cot_fewshot_logical_deduction_three_objects":  "BBH logical_deduction_three_objects",
}

METRIC_KEYS = ("acc,none", "exact_match,get-answer", "exact_match,none")


def load_results(path: Path):
    """Return (model_id, n_items_per_task_dict, score_per_task_dict)."""
    data = json.loads(path.read_text())
    model_id = (
        data.get("model_name")
        or data["config"]["model_args"].split("pretrained=")[1].split(",")[0]
    )
    scores = {}
    n_items = {}
    for tname, tres in data["results"].items():
        for mk in METRIC_KEYS:
            if mk in tres:
                scores[tname] = tres[mk]
                break
        n_items[tname] = data.get("n-samples", {}).get(tname, {}).get("effective")
    return model_id, n_items, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", type=Path)
    args = ap.parse_args()

    cols = []  # ordered list of (model_id, n_items, scores)
    for path in args.results:
        cols.append(load_results(path))

    # Header
    header = ["Task"] + [m for m, _, _ in cols] + ["n", "OLLM baseline"]
    sep = ["---"] * len(header)
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(sep) + "|")

    for t in TASK_ORDER:
        row = [TASK_LABEL[t]]
        n_seen = set()
        for model_id, n_items, scores in cols:
            if t in scores:
                row.append(f"{scores[t]:.1%}")
            else:
                row.append("—")
            if t in n_items and n_items[t]:
                n_seen.add(n_items[t])
        row.append(str(next(iter(n_seen))) if n_seen else "?")
        # OLLM baselines for the models we have, "/"-separated, model order matches
        baselines = OLLM_BASELINES.get(t, {})
        bl_parts = []
        for model_id, _, _ in cols:
            if model_id in baselines:
                bl_parts.append(f"{baselines[model_id]:.1%}")
            else:
                bl_parts.append("—")
        row.append(" / ".join(bl_parts))
        print("| " + " | ".join(row) + " |")

    # Footer: protocol notes
    print()
    print("Protocols: MMLU = 5-shot answer-only loglikelihood over A/B/C/D (Hendrycks 2021).")
    print("BBH = 3-shot chain-of-thought with `the answer is X` regex (Suzgun 2022).")
    print("`--apply_chat_template --fewshot_as_multiturn` applied for instruction-tuned models.")
    print("OLLM column: MMLU = Open-LLM-Leaderboard v1 (5-shot acc); BBH = OLLM v2 (3-shot acc_norm).")


if __name__ == "__main__":
    main()
