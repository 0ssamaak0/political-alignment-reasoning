"""Average per-seed metrics for the roleplay dose sweep into mean ± std.

The roleplay dose runs write one metrics file per (tag, seed):
    runs/$EXPERIMENT/metrics/<tag>__seed{N}.json
This script groups those by `<tag>` (stripping the `__seed{N}` suffix),
computes mean and sample-std across the seeds for the headline metrics, and
emits both a Markdown table and a JSON aggregate.

It is intentionally separate from `aggregate.py` (which builds the global
`summary.json` for the one-file-per-tag cells) so it cannot perturb the rest
of the f5 cohort. See
docs/superpowers/specs/2026-05-25-mistral-roleplay-dose-design.md.

Usage:
    EXPERIMENT=f5 conda run -n main python -m custom_bench.aggregate_seeds
    EXPERIMENT=f5 conda run -n main python -m custom_bench.aggregate_seeds \
        --tag_substring roleplay
"""

import argparse
import json
import re
import statistics
from collections import defaultdict

from custom_bench.config import METRICS_DIR, RUN_DIR

# (label, dotted path into the metrics JSON)
HEADLINE = [
    ("bias_signed_FPFN", ("bias_signed_FPFN",)),
    ("acc_neutral",      ("by_lean", "neutral", "accuracy")),
    ("acc_left",         ("by_lean", "left", "accuracy")),
    ("acc_right",        ("by_lean", "right", "accuracy")),
    ("vskew_neutral",    ("by_lean", "neutral", "verdict_skew")),
    ("vskew_left",       ("by_lean", "left", "verdict_skew")),
    ("vskew_right",      ("by_lean", "right", "verdict_skew")),
]

SEED_RE = re.compile(r"__seed\d+$")


def _dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def collect(tag_substring=None):
    """Return {base_tag: [(seed_file, metrics_dict), ...]}."""
    groups = defaultdict(list)
    for p in sorted(METRICS_DIR.rglob("*__seed*.json")):
        base_tag = SEED_RE.sub("", p.stem)
        if tag_substring and tag_substring not in base_tag:
            continue
        with open(p) as f:
            groups[base_tag].append((p.name, json.load(f)))
    return groups


def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, 0
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std, len(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag_substring", default=None,
                        help="Only aggregate tags containing this substring "
                             "(e.g. 'roleplay').")
    args = parser.parse_args()

    groups = collect(args.tag_substring)
    if not groups:
        print(f"No *__seed*.json files found in {METRICS_DIR}", flush=True)
        return

    aggregate = {}
    for base_tag in sorted(groups):
        runs = groups[base_tag]
        row = {"n_seeds": len(runs), "seed_files": [name for name, _ in runs]}
        for label, path in HEADLINE:
            vals = [_dig(m, path) for _, m in runs]
            mean, std, n = mean_std(vals)
            row[label] = {"mean": mean, "std": std, "n": n}
        aggregate[base_tag] = row

    # JSON
    out_json = RUN_DIR / "roleplay_dose_summary.json"
    with open(out_json, "w") as f:
        json.dump(aggregate, f, indent=2)

    # Markdown
    def cell(stat):
        if stat["mean"] is None:
            return "n/a"
        return f"{stat['mean']:+.3f} ± {stat['std']:.3f}"

    cols = [label for label, _ in HEADLINE]
    lines = [
        "# Roleplay system-prompt dose sweep — seed-averaged (mean ± std)",
        "",
        f"Run dir: `{RUN_DIR}`  |  metric source: `metrics/<tag>__seed*.json`",
        "",
        "| tag | n_seeds | " + " | ".join(cols) + " |",
        "| --- | ---: | " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for base_tag in sorted(aggregate):
        row = aggregate[base_tag]
        cells = [cell(row[c]) for c in cols]
        lines.append(f"| `{base_tag}` | {row['n_seeds']} | " + " | ".join(cells) + " |")
    out_md = RUN_DIR / "roleplay_dose_summary.md"
    out_md.write_text("\n".join(lines) + "\n")

    print("\n".join(lines), flush=True)
    print(f"\nWrote {out_md}\n      {out_json}", flush=True)


if __name__ == "__main__":
    main()
