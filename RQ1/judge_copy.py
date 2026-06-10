#!/usr/bin/env python3
"""Copy RQ3 judge.jsonl into the matching RQ1 deployed cells -- but only where the
generations are PROVABLY identical.

RQ3 swept alignment strength and ran the Gemini judge over every cell x task. A
few of RQ3's grid points are the SAME generations as RQ1's deployed cells (RQ3
copied them from RQ1 in the first place: base, DPO s1.0; and the deployed steering
points coincide with grid points). For those, the judge label depends only on the
generation text, so RQ3's judge.jsonl is valid for RQ1 verbatim.

This script imports those judge files into RQ1 -- gated on a per (cell, task)
byte-identity check (same doc_id set, same prompt_hash, same response on all 250
rows). If a task is not 250/250 identical it is REFUSED and reported; that cell
belongs in the run set (judge_rq1.py), not here.

Output:
  * RQ1/<fam>/<cell>/<task>/judge.jsonl   (copied verbatim from RQ3)
  * RQ1/judge/COPIED.md                   (provenance manifest)

Run (repo root):
  python RQ1/judge_copy.py
  python RQ1/judge_copy.py --dry-run      # verify identity, copy nothing
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../polireason/RQ1
REPO = HERE.parent                               # .../polireason
RQ3_RESULTS = REPO / "RQ3" / "results"

# The 4 strength-sensitive BBH-CoT tasks the judge covers (full on-disk dir names).
TASKS = [
    "bbh_cot_fewshot_boolean_expressions",
    "bbh_cot_fewshot_logical_deduction_three_objects",
    "bbh_cot_fewshot_web_of_lies",
    "bbh_cot_fewshot_navigate",
]

# (rq1_cell_reldir, rq3_cell_reldir) for cells whose generations match RQ3's and
# which RQ3 already judged. Verified by the identity gate below before any copy.
COPY_MAP = {
    "mistral": [
        ("base",              "base"),
        ("DPO/left",          "dpo/left/s1_0"),
        ("DPO/right",         "dpo/right/s1_0"),
        ("steering/right_a3", "steering/right/a3"),
    ],
    "llama": [
        ("base",              "base"),
        ("DPO/left",          "dpo/left/s1_0"),
        ("DPO/right_2nd",     "dpo/right/s1_0"),
    ],
}


def load_samples(path: Path) -> dict:
    rows = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            rows[r["doc_id"]] = r
    return rows


def resp_text(r: dict) -> str:
    try:
        return r["resps"][0][0]
    except Exception:
        return json.dumps(r.get("resps"))


def identical(rq1_samples: Path, rq3_samples: Path) -> tuple[bool, str]:
    """True iff same doc_id set and same prompt_hash + response on every row."""
    if not rq1_samples.exists():
        return False, "RQ1 samples missing"
    if not rq3_samples.exists():
        return False, "RQ3 samples missing"
    a, b = load_samples(rq1_samples), load_samples(rq3_samples)
    if set(a) != set(b):
        return False, f"doc_id set differs ({len(a)} vs {len(b)})"
    common = sorted(a)
    ph = sum(1 for k in common if a[k].get("prompt_hash") == b[k].get("prompt_hash"))
    rs = sum(1 for k in common if resp_text(a[k]) == resp_text(b[k]))
    n = len(common)
    if ph == n and rs == n:
        return True, f"{n}/{n} identical"
    return False, f"prompt_hash {ph}/{n}, resp {rs}/{n}"


def judge_complete(judge_path: Path, doc_ids: set) -> tuple[bool, str]:
    if not judge_path.exists():
        return False, "RQ3 judge.jsonl missing"
    seen = set()
    with judge_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("primary_category") is not None and "classifier_error" not in r:
                seen.add(r["doc_id"])
    if doc_ids <= seen:
        return True, f"{len(seen)} judged rows cover all {len(doc_ids)} docs"
    return False, f"judge covers {len(seen & doc_ids)}/{len(doc_ids)} docs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["mistral", "llama"])
    ap.add_argument("--dry-run", action="store_true",
                    help="verify identity only; copy nothing.")
    args = ap.parse_args()

    manifest = ["# Copied judge files (RQ3 -> RQ1)",
                "",
                "Each row is one (cell, task). A file is copied only when the RQ1 and",
                "RQ3 generations are byte-identical (same doc_id set, prompt_hash and",
                "response on all rows) and RQ3's judge covers every doc. Copy is verbatim.",
                ""]
    copied = refused = 0
    for fam in args.families:
        manifest.append(f"\n## {fam}\n")
        manifest.append("| RQ1 cell | task | identity | judge | action |")
        manifest.append("|---|---|---|---|---|")
        for rq1_cell, rq3_cell in COPY_MAP.get(fam, []):
            for task in TASKS:
                rq1_samp = HERE / fam / rq1_cell / task / "samples.jsonl"
                rq3_samp = RQ3_RESULTS / fam / rq3_cell / task / "samples.jsonl"
                rq3_judge = RQ3_RESULTS / fam / rq3_cell / task / "judge.jsonl"
                dest = HERE / fam / rq1_cell / task / "judge.jsonl"

                ok_id, why_id = identical(rq1_samp, rq3_samp)
                short = task.replace("bbh_cot_fewshot_", "")
                if not ok_id:
                    refused += 1
                    print(f"[REFUSE] {fam}/{rq1_cell}/{short}: {why_id}")
                    manifest.append(f"| {rq1_cell} | {short} | {why_id} | - | REFUSED |")
                    continue
                doc_ids = set(load_samples(rq1_samp))
                ok_j, why_j = judge_complete(rq3_judge, doc_ids)
                if not ok_j:
                    refused += 1
                    print(f"[REFUSE] {fam}/{rq1_cell}/{short}: {why_j}")
                    manifest.append(f"| {rq1_cell} | {short} | {why_id} | {why_j} | REFUSED |")
                    continue
                action = "would copy" if args.dry_run else "copied"
                if not args.dry_run:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(rq3_judge, dest)
                copied += 1
                print(f"[ OK   ] {fam}/{rq1_cell}/{short}: {why_id} -> {action}")
                src_rel = rq3_judge.relative_to(REPO)
                manifest.append(f"| {rq1_cell} | {short} | {why_id} | {why_j} | {action} (`{src_rel}`) |")

    (HERE / "judge").mkdir(exist_ok=True)
    (HERE / "judge" / "COPIED.md").write_text("\n".join(manifest) + "\n")
    print(f"\n[judge_copy] {'verified' if args.dry_run else 'copied'} {copied} cell-tasks, "
          f"{refused} refused. Manifest -> RQ1/judge/COPIED.md")


if __name__ == "__main__":
    main()
