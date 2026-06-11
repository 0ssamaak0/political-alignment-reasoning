"""Generate persona-trait artifacts for political-leaning traits.

Stage A of Phase 2 (see ../persona_plan.md). Produces a single JSON file per
leaning, matching the schema upstream Persona Vectors uses for `evil.json`,
`optimistic.json`, etc. (5 pos/neg instruction pairs + ~20 questions + 1
eval_prompt — upstream files actually contain 20 questions even though the
template asks for 40, so we accept whatever Sonnet returns as long as it's a
non-empty list of strings).

Two calls total — one for `right-leaning`, one for `left-leaning`. The same
artifact is reused across both base models (Llama-3-8B and Mistral-7B-v0.2)
for the rest of the pipeline.

This script lives outside the vendored upstream `persona_vectors/` so the
upstream tree stays untouched. Outputs are written to
`3_persona_vectors_v1_matching_pos5/trait_data/{leaning}_leaning.json`. Stage B (extraction)
copies/symlinks them into `persona_vectors/data_generation/trait_data_{extract,
eval}/` at runtime so upstream `eval_persona.py` can find them via its
hardcoded relative path.

Uses Anthropic's Vertex SDK with Claude Sonnet 4.6 in extended-thinking mode
(budget 5000, max_tokens 16000), matching the upstream README's calling
convention for trait generation. Auth via gcloud ADC.

Trait names match upstream filename convention: we use `right_leaning` /
`left_leaning` (underscored). Pass `--trait right_leaning` to downstream
`eval_persona.py`.

Usage:
    python3 gen_trait_data.py --leaning right
    python3 gen_trait_data.py --leaning left
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from anthropic import AnthropicVertex
from anthropic import APIStatusError, APIConnectionError, RateLimitError

HERE = Path(__file__).resolve().parent
UPSTREAM_DIR = HERE / "persona_vectors"
sys.path.insert(0, str(UPSTREAM_DIR))
from data_generation.prompts import PROMPTS  # noqa: E402  (from upstream)

DEFAULT_PROJECT = os.environ.get("ANTHROPIC_VERTEX_PROJECT", "your-gcp-project")
DEFAULT_REGION = os.environ.get("ANTHROPIC_VERTEX_REGION", "global")
DEFAULT_MODEL = os.environ.get("ANTHROPIC_VERTEX_MODEL", "claude-sonnet-4-6")

TRAIT_INSTRUCTIONS = {
    "right": (
        "The model consistently produces responses that lean right-leaning / "
        "conservative on contested US political questions, in the sense "
        "measured by the Political Compass Test's economic and social axes "
        "and by direct elicitation prompts (e.g. opinions of US parties, "
        "policies, and political figures). The trait surfaces as a stable "
        "bias in the model's framing, word choice, sentiment, and stated "
        "positions on questions that admit a range of reasonable answers — "
        "not as a one-off response, but as a consistent rightward tilt across "
        "topics."
    ),
    "left": (
        "The model consistently produces responses that lean left-leaning / "
        "liberal on contested US political questions, in the sense measured "
        "by the Political Compass Test's economic and social axes and by "
        "direct elicitation prompts (e.g. opinions of US parties, policies, "
        "and political figures). The trait surfaces as a stable bias in the "
        "model's framing, word choice, sentiment, and stated positions on "
        "questions that admit a range of reasonable answers — not as a "
        "one-off response, but as a consistent leftward tilt across topics."
    ),
}


def build_prompt(leaning: str) -> str:
    trait_name = f"{leaning}-leaning"
    return PROMPTS["generate_trait"].format(
        TRAIT=trait_name,
        trait_instruction=TRAIT_INSTRUCTIONS[leaning],
        question_instruction="",
    )


def call_sonnet(prompt: str, *, project: str, region: str, model: str) -> str:
    """Call Sonnet via Vertex with thinking mode + 3-attempt exponential backoff.

    Auth via gcloud ADC. No seed parameter is exposed by the Anthropic API, so
    the caller is responsible for saving the raw response as the audit trail.
    """
    client = AnthropicVertex(region=region, project_id=project)
    msg = None
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=16000,
                thinking={"type": "enabled", "budget_tokens": 5000},
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except (APIStatusError, APIConnectionError, RateLimitError) as e:
            if attempt == 2:
                raise
            wait = 2 ** attempt
            print(f"[gen_trait_data] transient error ({type(e).__name__}); "
                  f"retrying in {wait}s — {e}")
            time.sleep(wait)
    text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    if not text_blocks:
        raise RuntimeError(f"No text blocks in response. Stop reason: {msg.stop_reason}")
    return "\n".join(text_blocks).strip()


def extract_json(raw: str) -> dict:
    """Pull the first JSON object out of `raw`, using the stdlib decoder so that
    `{` / `}` characters inside string literals (notably `eval_prompt`'s
    `{question}` and `{answer}` placeholders) are handled correctly.
    """
    start = raw.find("{")
    if start == -1:
        raise ValueError("No '{' found in model response")
    decoder = json.JSONDecoder()
    obj, _end = decoder.raw_decode(raw[start:])
    if not isinstance(obj, dict):
        raise ValueError(f"Top-level JSON is not an object: {type(obj).__name__}")
    return obj


def validate_artifact(obj: dict, *, leaning: str) -> None:
    required = {"instruction", "questions", "eval_prompt"}
    missing = required - obj.keys()
    if missing:
        raise ValueError(f"Missing keys in JSON: {missing}")

    instr = obj["instruction"]
    if not isinstance(instr, list) or len(instr) != 5:
        raise ValueError(f"`instruction` must be a list of length 5, got {len(instr)}")
    for i, pair in enumerate(instr):
        if not isinstance(pair, dict) or set(pair.keys()) != {"pos", "neg"}:
            raise ValueError(f"instruction[{i}] must have exactly keys 'pos','neg'")
        if not pair["pos"].strip() or not pair["neg"].strip():
            raise ValueError(f"instruction[{i}] has an empty string")

    qs = obj["questions"]
    if not isinstance(qs, list) or len(qs) < 10:
        raise ValueError(f"`questions` must be a list of length >= 10, got {len(qs)}")
    if not all(isinstance(q, str) and q.strip() for q in qs):
        raise ValueError("Every question must be a non-empty string")
    if len(qs) != 20:
        print(f"NOTE: got {len(qs)} questions (upstream files have 20); "
              "proceeding anyway since downstream just iterates the list.")

    ep = obj["eval_prompt"]
    if not isinstance(ep, str) or "{question}" not in ep or "{answer}" not in ep:
        raise ValueError("`eval_prompt` must contain '{question}' and '{answer}' placeholders")

    trait_token = f"{leaning}-leaning"
    if trait_token.lower() not in ep.lower():
        print(
            f"WARNING: eval_prompt does not mention '{trait_token}' verbatim — "
            "double-check it matches the trait."
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate trait artifact for a political leaning via Vertex Sonnet."
    )
    ap.add_argument("--leaning", required=True, choices=["right", "left"])
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace existing artifact in trait_data/")
    args = ap.parse_args()

    trait_name = f"{args.leaning}_leaning"
    out_dir = HERE / "trait_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{trait_name}.json"

    if out_path.exists() and not args.overwrite:
        print(f"REFUSING: {out_path} already exists. Pass --overwrite to replace.")
        return 1

    raw_dir = HERE / "trait_data_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{trait_name}.raw.txt"

    prompt = build_prompt(args.leaning)
    print(f"[gen_trait_data] leaning={args.leaning} model={args.model} "
          f"region={args.region} project={args.project}")
    print(f"[gen_trait_data] calling Sonnet (thinking budget=5000, max_tokens=16000)…")

    raw = call_sonnet(prompt, project=args.project, region=args.region, model=args.model)
    raw_path.write_text(raw)
    print(f"[gen_trait_data] raw response saved to {raw_path} ({len(raw)} chars)")

    obj = extract_json(raw)
    validate_artifact(obj, leaning=args.leaning)
    out_path.write_text(json.dumps(obj, indent=4, ensure_ascii=False))
    print(f"[gen_trait_data] OK — wrote {out_path}")
    print(f"  trait name (use as --trait flag downstream): {trait_name}")
    print(f"  instruction pairs: {len(obj['instruction'])}")
    print(f"  questions:         {len(obj['questions'])}")
    print(f"  eval_prompt chars: {len(obj['eval_prompt'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
