"""Convert G_K result CSVs to judge-ready jsonl files.

Reads each results/<name>.csv, joins with data/prompts_192.csv to recover
the prompt text, and writes responses/<name>.jsonl in the schema expected
by Judge.src.qualitative_classifier.

Run from 1_benchmarking/:
    python -m G_K_assessing_bias.gk_to_jsonl [--only name1 name2 ...]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from G_K_assessing_bias.gk_paths import RESULTS_DIR, response_jsonl

HERE = Path(__file__).resolve().parent
PROMPTS_CSV = HERE / "data" / "prompts_192.csv"


def convert(csv_path: Path, prompts: pd.DataFrame) -> Path:
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        item_id = int(row["item_id"])
        p = prompts.iloc[item_id]
        rows.append({
            "template_id": f"P{row['pattern_id']}_{row['variation']}",
            "lean":        str(row["leaning"]),
            "valid":       int(row["inference_valid_gt"]),
            "verdict":     str(row["predicted_label"]).upper(),
            "text":        str(p["Prompt"]),
            "raw_response": str(row["raw_output"]),
            "n_tokens_generated": None,
        })
    out_path = response_jsonl(csv_path.stem, mkdir=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows → {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*",
                    help="only convert these cell names (no .csv suffix)")
    args = ap.parse_args()

    prompts = pd.read_csv(PROMPTS_CSV)
    skip = {"summary_long"}
    csvs = sorted(RESULTS_DIR.rglob("*.csv"))
    if args.only:
        only = set(args.only)
        csvs = [c for c in csvs if c.stem in only]

    for csv_path in csvs:
        if csv_path.stem in skip:
            continue
        convert(csv_path, prompts)


if __name__ == "__main__":
    main()
