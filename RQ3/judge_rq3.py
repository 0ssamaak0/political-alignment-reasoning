#!/usr/bin/env python3
"""Run the Judge (Gemini-Flash qualitative classifier) on the RQ3 reasoning
sweep, with resume caching.

RQ3 stores one lm-eval `samples.jsonl` per cell x task:
  RQ3/results/<fam>/base/<task>/samples.jsonl
  RQ3/results/<fam>/<method>/<lean>/<strength>/<task>/samples.jsonl

Each sample is fed to the classifier exactly as judge_bbh.py does it: the real
BBH gold -> `valid`, lm-eval's regex-parsed answer -> `verdict`, the question
shown -> `text`, the generated CoT -> `raw_response`. The rubric is unchanged.
On these NEUTRAL tasks the political axes (viewpoint_bias, fallacy_lens) are
canary noise; the meaningful signal is `outcome` / `collapsed` /
`reasoning_validity` / `contaminated` and the 7-way `primary_category`
failure-mode split (generation_collapse vs instruction_following_failure vs
post_hoc_reasoning vs capability_error ...). A non-zero `contaminated` rate on a
neutral task is itself a finding: the intervention injecting politics where the
task supplied none.

Output: one `judge.jsonl` next to each `samples.jsonl`.

Caching (both senses of "use caching"):
  * result-resume: a cell x task is skipped when its judge.jsonl already holds
    one successfully-classified row per sample (keyed by doc_id, no
    classifier_error). Partial files are completed in place -- only missing or
    errored doc_ids are re-called. Generations are frozen, so (cell, task,
    doc_id) is a stable key and re-runs are free.
  * prefix cache: the classifier keeps a stable system_instruction + RUBRIC head
    with the per-item delta last, so Vertex *implicit* caching fires. USAGE now
    reports `cached_tokens` so the hit rate is observable (no explicit
    cached_content object needed unless the pilot shows zero hits).

lm-eval's own `exact_match` is preserved on every output row and used to print a
judge-outcome vs exact_match agreement rate per cell x task -- the guardrail that
the syllogism-tuned rubric still reads `outcome` correctly on neutral BBH golds
(True/False, (A)/(B)/(C), Yes/No).

Run (repo root, conda main):
  conda run -n main python RQ3/judge_rq3.py --family mistral
  conda run -n main python RQ3/judge_rq3.py --cells base steering/right/a4 \
        --tasks boolean_expressions                              # pilot
  conda run -n main python RQ3/judge_rq3.py --family mistral --dry-run
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import time  # noqa: E402

from google.genai import types  # noqa: E402

import Judge.src.qualitative_classifier as QC  # noqa: E402
from Judge.src.qualitative_classifier import (  # noqa: E402
    DEFAULT_MODEL, USAGE, classify_batch, loop_signals,
    enable_prefix_cache, disable_prefix_cache,
)

RESULTS = HERE / "results"

# The 4 strength-sensitive BBH-CoT tasks (generative; MMLU is loglikelihood and
# has no CoT to judge). Short names == on-disk dir minus the bbh_cot_fewshot_ prefix.
TASKS = [
    "boolean_expressions",
    "logical_deduction_three_objects",
    "web_of_lies",
    "navigate",
]

# Cell grid (matches RQ3/consolidate.py). base is the shared strength-0 point.
STEER_STR = ["a0_5", "a1", "a2", "a3", "a4"]
DPO_STR = ["s0_25", "s0_5", "s1_0", "s1_5", "s2"]


def default_cells() -> list[str]:
    cells = ["base"]
    for meth, strs in (("steering", STEER_STR), ("dpo", DPO_STR)):
        for lean in ("left", "right"):
            for s in strs:
                cells.append(f"{meth}/{lean}/{s}")
    return cells


def cell_meta(cell: str) -> tuple[str, str, str]:
    """cell string -> (method, lean, strength)."""
    parts = cell.split("/")
    if parts[0] == "base":
        return "base", "none", "0"
    return parts[0], parts[1], parts[2]


def _short(task: str) -> str:
    return task.replace("bbh_cot_fewshot_", "")


def build_rows(cell_dir: Path, cell_tag: str, lean: str, task_short: str
               ) -> tuple[list[dict], list[dict]] | None:
    """(classifier_rows, meta_rows) for one cell x task, or None if absent."""
    p = cell_dir / f"bbh_cot_fewshot_{task_short}" / "samples.jsonl"
    if not p.exists():
        return None
    samples = [json.loads(l) for l in p.open()]
    clf, meta = [], []
    for s in samples:
        resp = s["resps"][0][0] if s.get("resps") else ""
        gold = s["target"]
        parsed = s["filtered_resps"][0] if s.get("filtered_resps") else None
        clf.append({
            "template_id": f"{task_short}-{s['doc_id']}",
            "lean": lean,                              # steering DIRECTION (no political lean on neutral tasks)
            "valid": gold,                             # real BBH gold; judge compares against it
            "verdict": parsed,                         # lm-eval regex-parsed answer
            "n_tokens_generated": len(resp.split()),   # approx (no token field in lm-eval samples)
            "text": s["doc"]["input"],
            "raw_response": resp,
        })
        meta.append({
            "cell": cell_tag,
            "task": task_short,
            "doc_id": s["doc_id"],
            "gold": gold,
            "lm_eval_parsed": parsed,
            "lm_eval_exact_match": s.get("exact_match"),
        })
    return clf, meta


def load_done(out_path: Path) -> dict:
    """doc_id -> prior output row, for successfully-classified rows only."""
    done = {}
    if out_path.exists():
        for l in out_path.open():
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if "classifier_error" not in r and r.get("primary_category") is not None:
                done[r["doc_id"]] = r
    return done


def assemble(clf: dict, meta: dict, c: dict) -> dict:
    sig = loop_signals(clf["raw_response"])
    return {
        **meta,
        "template_id": clf["template_id"],
        "lean": clf["lean"],
        "n_tokens_generated": clf["n_tokens_generated"],
        "max_4gram_repeat": sig["max_4gram_repeat"],
        "distinct_ratio_last_50": sig["distinct_ratio_last_50"],
        "text": clf["text"],
        "raw_response": clf["raw_response"],
        **c,
    }


# Escalation ladder for the stubborn tail. The async client (client.aio)
# intermittently returns empty text, and on the longest / most discursive
# collapse generations the judge's `reasoning` field overruns max_output_tokens
# (finish_reason=MAX_TOKENS) before it emits the JSON labels, also yielding empty
# text. Each step shrinks the response shown to the judge (head+tail truncation,
# so its reasoning stays bounded) and raises the output budget. Truncation only
# affects what the judge SEES; the stored row keeps the FULL raw_response and the
# pre-pass signals computed on it. A row that survives the whole ladder is marked
# `classifier_error` (never given a fabricated label).
_SYNC_LADDER = [(None, None, 600), (1800, 700, 1500), (900, 400, 2500), (500, 250, 4000)]


def _trunc(raw: str, head, tail) -> str:
    if head is None or len(raw) <= head + tail + 40:
        return raw
    return raw[:head] + "\n\n[... middle of response truncated for length ...]\n\n" + raw[-tail:]


def classify_one_sync(model: str, clf: dict) -> dict:
    """Sync classify of one row, escalating through _SYNC_LADDER on empty text.
    Mirrors classify_one's cache/no-cache branching."""
    sig = loop_signals(clf["raw_response"])
    client = QC._sync_client()
    for head, tail, maxtok in _SYNC_LADDER:
        user_msg = QC.INPUT_TEMPLATE.format(
            template_id=clf["template_id"],
            template_family=clf["template_id"].split("-")[0],
            lean=clf["lean"], gold_valid=clf["valid"],
            parsed_verdict=clf.get("verdict"), n_tokens=clf.get("n_tokens_generated"),
            max_4gram_repeat=sig["max_4gram_repeat"], distinct_ratio=sig["distinct_ratio_last_50"],
            prompt_text=clf["text"], raw_response=_trunc(clf["raw_response"], head, tail),
        )
        if QC._CACHE_NAME:
            contents, cfg = user_msg, types.GenerateContentConfig(
                cached_content=QC._CACHE_NAME, temperature=0.0, max_output_tokens=maxtok,
                response_mime_type="application/json", response_schema=QC.RESPONSE_SCHEMA,
                thinking_config=types.ThinkingConfig(thinking_budget=0))
        else:
            contents, cfg = QC.RUBRIC + "\n\n" + user_msg, types.GenerateContentConfig(
                system_instruction=QC.SYSTEM_INSTRUCTION, temperature=0.0, max_output_tokens=maxtok,
                response_mime_type="application/json", response_schema=QC.RESPONSE_SCHEMA,
                thinking_config=types.ThinkingConfig(thinking_budget=0))
        for _ in range(2):  # absorb transient empties at this step
            try:
                resp = client.models.generate_content(model=model, contents=contents, config=cfg)
                text = (resp.text or "").strip()
                if text:
                    parsed = json.loads(text)
                    if parsed.get("fallacy_lens") == "none":
                        parsed["fallacy_lens"] = None
                    if head is not None:
                        parsed["judge_truncated"] = True  # response was head+tail truncated for the judge
                    return parsed
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)
    return {"classifier_error": "max_tokens_overrun"}


def backfill_cell_task(cell_dir: Path, cell_tag: str, lean: str, task_short: str,
                       model: str) -> int:
    """Sync mop-up of any missing rows in one cell x task. Returns rows filled."""
    built = build_rows(cell_dir, cell_tag, lean, task_short)
    if built is None:
        return 0
    clf, meta = built
    out_path = cell_dir / f"bbh_cot_fewshot_{task_short}" / "judge.jsonl"
    done = load_done(out_path)
    todo = [i for i, m in enumerate(meta) if m["doc_id"] not in done]
    if not todo:
        return 0
    tag = f"{cell_tag}/{task_short}"
    print(f"[backfill] {tag}: {len(todo)} missing -> sync")
    filled = 0
    for i in todo:
        c = classify_one_sync(model, clf[i])
        if "classifier_error" not in c:
            done[meta[i]["doc_id"]] = assemble(clf[i], meta[i], c)
            filled += 1
    final = [done[m["doc_id"]] for m in meta if m["doc_id"] in done]
    with out_path.open("w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")
    print(f"[backfill]   filled {filled}/{len(todo)}  -> {len(final)}/{len(meta)} rows")
    return filled


def agreement(rows: list[dict]) -> str:
    scored = agree = 0
    for r in rows:
        em = r.get("lm_eval_exact_match")
        if r.get("outcome") and em in (0.0, 1.0):
            scored += 1
            if (r["outcome"] == "correct") == bool(em):
                agree += 1
    return f"{agree}/{scored} ({100*agree/scored:.0f}%)" if scored else "n/a"


def run_cell_task(cell_dir: Path, cell_tag: str, lean: str, task_short: str,
                  model: str, concurrency: int, limit: int | None,
                  dry_run: bool) -> int:
    """Classify (with resume) one cell x task. Returns rows newly classified."""
    built = build_rows(cell_dir, cell_tag, lean, task_short)
    if built is None:
        return 0
    clf, meta = built
    if limit:
        clf, meta = clf[:limit], meta[:limit]
    out_path = cell_dir / f"bbh_cot_fewshot_{task_short}" / "judge.jsonl"

    done = load_done(out_path)
    todo = [i for i, m in enumerate(meta) if m["doc_id"] not in done]
    tag = f"{cell_tag}/{task_short}"
    if not todo:
        print(f"[judge_rq3] {tag}: {len(meta)} rows cached -> skip")
        return 0
    if dry_run:
        print(f"[judge_rq3] {tag}: {len(todo)} to classify ({len(done)} cached)")
        return len(todo)

    print(f"[judge_rq3] {tag}: {len(todo)} to classify ({len(done)} cached)")
    results = asyncio.run(
        classify_batch(model, [clf[i] for i in todo], concurrency, desc=tag)
    )
    fresh = {}
    for i, c in zip(todo, results):
        if "classifier_error" in c:
            continue  # leave for the next resume pass
        fresh[meta[i]["doc_id"]] = assemble(clf[i], meta[i], c)

    final = []
    for m in meta:
        did = m["doc_id"]
        if did in fresh:
            final.append(fresh[did])
        elif did in done:
            final.append(done[did])
        # else: errored this pass and not previously done -> omitted; resume re-calls it
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")
    n_err = len(todo) - len(fresh)
    print(f"[judge_rq3]   wrote {len(final)}/{len(meta)} rows"
          + (f" ({n_err} errored, will resume)" if n_err else "")
          + f"  | judge-vs-exact_match: {agreement(final)}")
    return len(fresh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="mistral")
    ap.add_argument("--cells", nargs="+", default=None,
                    help="cell strings under the family (e.g. base steering/right/a4). "
                         "Default: the full RQ3 grid.")
    ap.add_argument("--tasks", nargs="+", default=TASKS,
                    help="short BBH task names (no bbh_cot_fewshot_ prefix).")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows per cell x task (pilot only).")
    ap.add_argument("--no-cache", action="store_true",
                    help="disable the Vertex prefix cache (rubric sent inline each call).")
    ap.add_argument("--backfill", action="store_true",
                    help="sync mop-up of missing rows (the async-empty collapse tail).")
    ap.add_argument("--dry-run", action="store_true",
                    help="count rows that would be classified; no API calls.")
    args = ap.parse_args()

    cells = args.cells or default_cells()
    fam_dir = RESULTS / args.family
    use_cache = not args.no_cache and not args.dry_run
    print(f"[judge_rq3] family={args.family} model={args.model} "
          f"cells={len(cells)} tasks={len(args.tasks)} "
          f"limit={args.limit} cache={use_cache} dry_run={args.dry_run}")

    if use_cache:
        name = enable_prefix_cache(args.model)
        print(f"[judge_rq3] prefix cache: {name}")

    total = 0
    try:
        for cell in cells:
            cell_dir = fam_dir / cell
            if not cell_dir.is_dir():
                continue
            method, lean, _ = cell_meta(cell)
            cell_tag = f"{args.family}-" + cell.replace("/", "-")
            for task_short in args.tasks:
                if args.backfill:
                    total += backfill_cell_task(cell_dir, cell_tag, lean,
                                                task_short, args.model)
                else:
                    total += run_cell_task(cell_dir, cell_tag, lean, task_short,
                                           args.model, args.concurrency, args.limit,
                                           args.dry_run)
    finally:
        if use_cache:
            disable_prefix_cache()

    print(f"\n[judge_rq3] rows {'to classify' if args.dry_run else 'classified'}: {total}")
    if not args.dry_run:
        print(f"[judge_rq3] USAGE: {USAGE}")
        it = USAGE['input_tokens']
        ct = USAGE['cached_tokens']
        if it:
            print(f"[judge_rq3] prefix-cache hit: {ct}/{it} input tokens "
                  f"({100*ct/it:.0f}%) served from cache")


if __name__ == "__main__":
    main()
