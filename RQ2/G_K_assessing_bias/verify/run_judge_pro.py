"""Re-run the SAME qualitative judge with Gemini 3.1 Pro as a third, stronger
opinion (original judge = gemini-3-flash-preview; my verifier/panel = Claude Opus).

Reuses run_judge.run_cell verbatim (identical rubric, system prompt, explicit
context cache, squeeze, retry budget) — only the model and the output dir change.
Reads the EXACT same responses/*.jsonl Flash saw; writes to verify/pro_judges/
so the original judges/ are untouched.

Run from repo root:
    python -m RQ2.G_K_assessing_bias.verify.run_judge_pro [--concurrency 12]
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

import RQ2.G_K_assessing_bias.run_judge as rj

MODEL = "gemini-3.1-pro-preview"
HERE = Path(__file__).resolve().parent
PRO_JUDGES = HERE / "pro_judges"
RESPONSES = rj.RESPONSES_DIR  # the same converted inputs Flash used


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    PRO_JUDGES.mkdir(parents=True, exist_ok=True)
    rj.JUDGES_DIR = PRO_JUDGES  # redirect run_cell's output

    jsonls = sorted(RESPONSES.glob("*.jsonl"))
    if args.only:
        only = set(args.only)
        jsonls = [p for p in jsonls if p.stem in only]
    if not jsonls:
        raise SystemExit(f"no responses jsonl in {RESPONSES}")

    client = rj.make_client()
    cache = rj.create_cache(client, MODEL)
    tok = getattr(cache.usage_metadata, "total_token_count", "?")
    print(f"[pro] model={MODEL} · cache={cache.name} ({tok} tok) · {len(jsonls)} cells")
    cache_name = cache.name

    totals = Counter()
    try:
        for jp in jsonls:
            u = rj.run_cell(client, MODEL, jp, cache_name, args.concurrency)
            for k, v in u.items():
                totals[k] += v
    finally:
        try:
            client.caches.delete(name=cache.name)
            print("[pro] cache deleted")
        except Exception as e:  # noqa: BLE001
            print(f"[pro] cache delete failed: {e}")

    cp = 100 * totals["cached_tokens"] / totals["input_tokens"] if totals["input_tokens"] else 0
    print(f"\n[pro] DONE · {totals['calls']} calls · {totals['input_tokens']:,} in / "
          f"{totals['output_tokens']:,} out · cached {cp:.0f}% · "
          f"{totals['retries']} retries · {totals['fail']} unrecovered-fail")


if __name__ == "__main__":
    main()
