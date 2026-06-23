#!/usr/bin/env python3
"""Write RQ3/MMLU_STRENGTH_SWEEP.md from local mmlu_formal_logic/results.json cells.

Expects the RQ3 result tree at RQ3/results/<family>/... (same layout as consolidate.py).
Run from repo root:

    python3 RQ3/make_mmlu_table.py
    python3 RQ3/make_mmlu_table.py --results /path/to/RQ3/results
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(HERE, "results")

STEER_GRID = [("a0_5", 0.5), ("a1", 1.0), ("a2", 2.0), ("a3", 3.0), ("a4", 4.0)]
DPO_GRID = [("s0_25", 0.25), ("s0_5", 0.5), ("s1_0", 1.0), ("s1_5", 1.5), ("s2", 2.0)]
CHANCE = 25.0  # 4-choice MMLU floor


def mmlu_acc(cell_dir):
    p = os.path.join(cell_dir, "mmlu_formal_logic", "results.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p)).get("results", {}).get("mmlu_formal_logic", {})
    a = r.get("acc,none")
    return 100 * a if a is not None else None


def fmt_acc(x):
    return f"{x:.1f}" if x is not None else "—"


def fmt_margin(x):
    return f"{x - CHANCE:+.1f}" if x is not None else "—"


def table_block(title, rows):
    lines = [f"**{title}**", "", "| Strength | MMLU acc (%) | Margin above chance |", "|--:|--:|--:|"]
    for label, acc in rows:
        lines.append(f"| {label} | {fmt_acc(acc)} | {fmt_margin(acc)} |")
    lines.append("")
    return lines


def collect(results_root, fam, method, lean, grid):
    base_dir = os.path.join(results_root, fam, "base")
    rows = [("0 (base)", mmlu_acc(base_dir))]
    for suf, val in grid:
        cell = os.path.join(results_root, fam, method, lean, suf)
        prefix = "α" if method == "steering" else "s"
        rows.append((f"{prefix} = {val:g}", mmlu_acc(cell)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "MMLU_STRENGTH_SWEEP.md"))
    args = ap.parse_args()

    out = [
        "# RQ3 — MMLU formal logic (strength sweep)",
        "",
        "MMLU formal logic scores for the RQ3 strength grid. This is the log-likelihood",
        "benchmark (no generation): the model ranks fixed options A–D and the highest-",
        "probability option is the prediction. Scored 5-shot on the full 126-item test set.",
        "",
        "**Margin above chance** = accuracy − 25 (four-choice chance floor). Standard error",
        "at n = 126 is about ±4 percentage points, so small moves near the floor are noisy.",
        "",
        "Numbers are read from `mmlu_formal_logic/results.json` in each strength cell.",
        "Regenerate with `python3 RQ3/make_mmlu_table.py --results RQ3/results` when the",
        "local result tree is available (the raw cells are not shipped in this repo; this",
        "file is the canonical published table).",
        "",
    ]

    for fam, fam_label in (("mistral", "Mistral-7B-Instruct-v0.2"), ("llama", "Llama-3-8B-Instruct")):
        out.append(f"## {fam_label}")
        out.append("")
        for method, grid in (("steering", STEER_GRID), ("dpo", DPO_GRID)):
            knob = "Steering coefficient α" if method == "steering" else "DPO LoRA scale s"
            out.append(f"### {knob}")
            out.append("")
            for lean in ("left", "right"):
                if not os.path.isdir(os.path.join(args.results, fam, method, lean)):
                    continue
                title = f"{lean.capitalize()} alignment"
                out.extend(table_block(title, collect(args.results, fam, method, lean, grid)))

    with open(args.out, "w") as f:
        f.write("\n".join(out).rstrip() + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
