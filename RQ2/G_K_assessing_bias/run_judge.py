"""Run the Judge qualitative classifier over the RQ2 G&K cells, with an
explicit Gemini context cache for the (static) system instruction + rubric.

Pipeline (all I/O under RQ2/G_K_assessing_bias/):
  pol_out/<cell>.csv  --(join prompts_192)-->  responses/<cell>.jsonl
                      --(cached classifier)-->  judges/<cell>.jsonl (+ .usage.json)

The rubric + schema + per-item template are imported verbatim from
``Judge.src.qualitative_classifier`` — this driver only adds (a) the pol_out→jsonl
conversion (same logic as G_K_assessing_bias.gk_to_jsonl, repointed at pol_out)
and (b) ONE explicit context cache shared across every call, so the ~3.5K-token
rubric prefix is billed at the cache discount instead of re-sent 2,880 times.

Run from repo root:
    python -m RQ2.G_K_assessing_bias.run_judge [--only mistral-base ...] \
        [--concurrency 16] [--no-cache] [--model ...]
"""
from __future__ import annotations
import argparse
import asyncio
import json
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types

from Judge.src.qualitative_classifier import (
    SYSTEM_INSTRUCTION,
    RUBRIC,
    INPUT_TEMPLATE,
    RESPONSE_SCHEMA,
    loop_signals,
    DEFAULT_PROJECT,
    DEFAULT_REGION,
    DEFAULT_MODEL,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
POL_OUT = HERE / "pol_out"
RESPONSES_DIR = HERE / "responses"
JUDGES_DIR = HERE / "judges"
PROMPTS_CSV = REPO / "1_benchmarking" / "G_K_assessing_bias" / "data" / "prompts_192.csv"


# ---------------------------------------------------------------------------
# Step 1 — pol_out CSV -> judge-ready jsonl (recover prompt text by item_id)
# ---------------------------------------------------------------------------

def convert_cell(csv_path: Path, prompts: pd.DataFrame) -> Path:
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        p = prompts.iloc[int(row["item_id"])]
        rows.append({
            "template_id": f"P{row['pattern_id']}_{row['variation']}",
            "lean": str(row["leaning"]),
            "valid": int(row["inference_valid_gt"]),
            "verdict": str(row["predicted_label"]).upper(),
            "text": str(p["Prompt"]),
            "raw_response": str(row["raw_output"]),
            "n_tokens_generated": None,
        })
    out_path = RESPONSES_DIR / f"{csv_path.stem}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Step 2/3 — cached classification
# ---------------------------------------------------------------------------

def make_client() -> genai.Client:
    return genai.Client(vertexai=True, project=DEFAULT_PROJECT, location=DEFAULT_REGION)


def create_cache(client: genai.Client, model: str, ttl: str = "7200s"):
    """One explicit context cache holding the static system instruction + rubric.

    Returns the cache resource name, or None if creation fails (caller then
    falls back to implicit caching: rubric prepended to the per-call contents).
    """
    cache = client.caches.create(
        model=model,
        config=types.CreateCachedContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            contents=[RUBRIC],
            ttl=ttl,
            display_name="gk_rq2_judge",
        ),
    )
    return cache


_RUN_RE = re.compile(r"(.{1,4}?)\1{9,}")


def squeeze_degenerate(text: str, cap: int = 2000) -> str:
    """Collapse degenerate repeated runs so a mode-collapsed response (e.g.
    `invalid \\*\\*\\*\\*...` ×hundreds) doesn't make the judge echo the flood and
    blow its output-token budget (observed MAX_TOKENS → empty JSON). The collapse
    pre-pass signals are computed on the FULL text and passed separately, so the
    judge loses no information about degeneration. No-op on non-degenerate text."""
    squeezed = _RUN_RE.sub(lambda m: m.group(1) * 3 + " …[repeated, collapsed]… ", text)
    if len(squeezed) > cap:
        squeezed = squeezed[:cap] + " …[truncated]"
    return squeezed


def build_user_msg(row: dict) -> str:
    template_family = row["template_id"].split("_")[0].rstrip("ab")
    sig = loop_signals(row["raw_response"])  # signals on FULL raw text
    raw_response = squeeze_degenerate(row["raw_response"])
    return INPUT_TEMPLATE.format(
        template_id=row["template_id"],
        template_family=template_family,
        lean=row["lean"],
        gold_valid=row["valid"],
        parsed_verdict=row.get("verdict"),
        n_tokens=row.get("n_tokens_generated"),
        max_4gram_repeat=sig["max_4gram_repeat"],
        distinct_ratio=sig["distinct_ratio_last_50"],
        prompt_text=row["text"],
        raw_response=raw_response,
    )


# Per-attempt output-token budgets. Degenerate (mode-collapsed) responses can
# make the judge echo the flood and hit MAX_TOKENS -> empty text; a larger
# budget on retry recovers them. squeeze_degenerate() handles most; this is the
# backstop so a clean re-run self-heals transient empties (no manual patching).
_MAX_OUT_ATTEMPTS = [600, 1200, 1200, 1600, 2000]


def _make_config(cache_name, max_out):
    if cache_name:
        return types.GenerateContentConfig(
            cached_content=cache_name,
            temperature=0.0,
            max_output_tokens=max_out,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
    return types.GenerateContentConfig(  # implicit-caching fallback
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.0,
        max_output_tokens=max_out,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )


async def classify_one(client, model, row, cache_name, usage):
    user_msg = build_user_msg(row)
    contents = user_msg if cache_name else (RUBRIC + "\n\n" + user_msg)

    last_err = "unknown"
    for attempt, max_out in enumerate(_MAX_OUT_ATTEMPTS):
        t0 = time.time()
        try:
            resp = await client.aio.models.generate_content(
                model=model, contents=contents,
                config=_make_config(cache_name, max_out),
            )
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            usage["retries"] += 1
            await asyncio.sleep(1.5)
            continue
        usage["wall_seconds"] += time.time() - t0
        usage["calls"] += 1
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            usage["input_tokens"] += int(getattr(um, "prompt_token_count", 0) or 0)
            usage["output_tokens"] += int(getattr(um, "candidates_token_count", 0) or 0)
            usage["cached_tokens"] += int(getattr(um, "cached_content_token_count", 0) or 0)

        text = (resp.text or "").strip()
        if not text:  # empty / MAX_TOKENS -> retry with a larger budget
            last_err = "empty_response"
            usage["retries"] += 1
            await asyncio.sleep(1.0)
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            last_err = f"json_decode: {e}"
            usage["retries"] += 1
            await asyncio.sleep(1.0)
            continue
        if parsed.get("fallacy_lens") == "none":
            parsed["fallacy_lens"] = None
        return parsed

    usage["fail"] += 1
    return {"classifier_error": last_err}


async def classify_cell(client, model, rows, cache_name, concurrency, usage):
    sem = asyncio.Semaphore(concurrency)

    async def _one(r):
        async with sem:
            return await classify_one(client, model, r, cache_name, usage)

    return await asyncio.gather(*[_one(r) for r in rows])


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def run_cell(client, model, jsonl_path: Path, cache_name, concurrency: int) -> dict:
    rows = _read_jsonl(jsonl_path)
    cell = jsonl_path.stem
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
             "cached_tokens": 0, "wall_seconds": 0.0,
             "retries": 0, "fail": 0}
    print(f"[judge] {cell}: {len(rows)} rows", flush=True)
    classifications = asyncio.run(
        classify_cell(client, model, rows, cache_name, concurrency, usage)
    )
    out_rows = []
    for idx, (r, c) in enumerate(zip(rows, classifications)):
        sig = loop_signals(r["raw_response"])
        out_rows.append({
            "cell": cell,
            "row_idx": idx,
            "template_id": r["template_id"],
            "lean": r["lean"],
            "valid": r["valid"],
            "verdict": r.get("verdict"),
            "n_tokens_generated": r.get("n_tokens_generated"),
            "max_4gram_repeat": sig["max_4gram_repeat"],
            "distinct_ratio_last_50": sig["distinct_ratio_last_50"],
            "text": r.get("text"),
            "raw_response": r.get("raw_response"),
            **c,
        })
    JUDGES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = JUDGES_DIR / f"{cell}.jsonl"
    with out_path.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    (JUDGES_DIR / f"{cell}.usage.json").write_text(json.dumps({"usage": usage}, indent=2))
    errs = sum(1 for r in out_rows if "classifier_error" in r)
    cached_pct = (100 * usage["cached_tokens"] / usage["input_tokens"]
                  if usage["input_tokens"] else 0)
    print(f"[judge] {cell}: wrote {len(out_rows)} rows · {errs} errors · "
          f"cached {cached_pct:.0f}% of input tokens", flush=True)
    return usage


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="only these cell stems")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-cache", action="store_true",
                    help="skip explicit context cache (implicit caching only)")
    args = ap.parse_args()

    prompts = pd.read_csv(PROMPTS_CSV)
    csvs = sorted(POL_OUT.glob("*.csv"))
    if args.only:
        only = set(args.only)
        csvs = [c for c in csvs if c.stem in only]
    if not csvs:
        raise SystemExit(f"no per-cell CSVs in {POL_OUT}")

    print(f"[judge] converting {len(csvs)} cells -> {RESPONSES_DIR}")
    jsonls = [convert_cell(c, prompts) for c in csvs]

    client = make_client()
    cache = None
    if not args.no_cache:
        try:
            cache = create_cache(client, args.model)
            tok = getattr(cache.usage_metadata, "total_token_count", "?")
            print(f"[judge] explicit cache created: {cache.name} ({tok} tokens)")
        except Exception as e:  # noqa: BLE001
            print(f"[judge] WARNING: explicit cache failed ({type(e).__name__}: {e}); "
                  f"falling back to implicit caching")
            cache = None
    cache_name = cache.name if cache else None

    totals = Counter()
    try:
        for jp in jsonls:
            u = run_cell(client, args.model, jp, cache_name, args.concurrency)
            for k, v in u.items():
                totals[k] += v
    finally:
        if cache is not None:
            try:
                client.caches.delete(name=cache.name)
                print(f"[judge] cache deleted")
            except Exception as e:  # noqa: BLE001
                print(f"[judge] cache delete failed: {e}")

    cached_pct = (100 * totals["cached_tokens"] / totals["input_tokens"]
                  if totals["input_tokens"] else 0)
    print(f"\n[judge] DONE · {totals['calls']} calls · "
          f"{totals['input_tokens']:,} in / {totals['output_tokens']:,} out tokens · "
          f"cached {cached_pct:.0f}% of input · "
          f"{totals['retries']} retries · {totals['fail']} unrecovered-fail")


if __name__ == "__main__":
    main()
