#!/usr/bin/env python3
"""Seeded-random subsample of a HuggingFace dataset split, written to a
JSONL file that can be served as a custom lm-evaluation-harness dataset.

The default lm-eval `--limit N` flag selects the *first* N documents in
dataset order, which is deterministic but not random. This helper picks
N indices with a fixed seed and dumps them so you can build a custom
task YAML that points at the JSONL.

Usage:
    python subsample.py \\
        --dataset SaylorTwift/bbh \\
        --config logical_deduction_three_objects \\
        --split test \\
        --n 150 \\
        --seed 62471893 \\
        --out subsamples/bbh_logical_deduction_three_objects_150.jsonl

To use the subsample with lm-eval-harness, copy the matching task YAML
and replace `dataset_path` / `test_split` with a local `dataset_kwargs`
pointing at the JSONL. See README for details.
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset path, e.g. SaylorTwift/bbh")
    ap.add_argument("--config",  default=None,  help="dataset config name (subject for BBH)")
    ap.add_argument("--split",   default="test")
    ap.add_argument("--n",       type=int, default=150)
    ap.add_argument("--seed",    type=int, default=62471893)
    ap.add_argument("--out",     required=True, type=Path)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, args.config, split=args.split)
    n_total = len(ds)
    if args.n >= n_total:
        chosen = list(range(n_total))
        print(f"[subsample] requested n={args.n} >= total={n_total}; taking all {n_total} items")
    else:
        rng = random.Random(args.seed)
        chosen = sorted(rng.sample(range(n_total), args.n))
        print(f"[subsample] {args.n}/{n_total} items, seed={args.seed}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for i in chosen:
            f.write(json.dumps(dict(ds[i])) + "\n")
    print(f"[subsample] wrote {len(chosen)} items to {args.out}")


if __name__ == "__main__":
    main()
