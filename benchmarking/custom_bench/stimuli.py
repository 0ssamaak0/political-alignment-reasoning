"""Stage 1 — emit politicized syllogism stimuli for the f4 reasoning-bias matrix.

Asymmetric variant scheme (2026-05-24, reduced from the earlier symmetric
two-variant design):

  - lean=neutral  → 2 variants: clean and injected. The injected variant
                    inserts one short politically-flavored but stance-neutral
                    phrase mid-syllogism (drawn from
                    `noise_injections.political_phrases`). This is the headline
                    novelty: tests whether political vocabulary *alone*
                    corrupts otherwise-neutral logical reasoning. Topic
                    dimension is degenerate for neutral (slots come from the
                    chess/poker domain).
  - lean=left    → 1 variant: clean only. Iterates the 8 active topics for
  - lean=right     templates that use POLICY_X or POLICY_Y; single
                    instantiation for templates that use only group slots.

The prior `unrelated_facts` injection for left/right was dropped to halve
inference cost. If you want to bring it back, recover it from git history
and re-add the per-lean injection branching to `pick_injection_phrase` +
`synthesize_all`.

Stimulus count: 392 per config (16 neutral non-policy + 40 neutral policy +
16 left+right non-policy + 320 left+right policy). See
custom_bench/docs/METHODOLOGY.md §2 for the breakdown table.

CLI:
    conda run -n main python -m custom_bench.stimuli
or:
    EXPERIMENT=f4/political conda run -n main python -m custom_bench.stimuli

    --topics ID1,ID2,...      restrict to a subset of the 8 topics (default: all)
    --lexicon PATH            override the lexicon path
    --output PATH             override the stimuli.jsonl path
"""

import argparse
import hashlib
import json
from pathlib import Path

from custom_bench.config import LEXICON_PATH, STIMULI_PATH, ensure_run_dirs
from custom_bench.templates import TEMPLATES

LEANS = ["neutral", "left", "right"]
ALL_VARIANTS = ["clean", "injected"]


def variants_for_lean(lean: str) -> list[str]:
    """Per-lean variant set. lean=neutral gets both clean and injected (the
    injection is a politically-flavored phrase, which tests whether political
    vocabulary alone corrupts neutral reasoning). lean=left|right gets clean
    only since the 2026-05-24 simplification — see module docstring."""
    return ["clean", "injected"] if lean == "neutral" else ["clean"]


def load_lexicon(path: Path):
    with open(path) as f:
        return json.load(f)


def split_sentences(text: str):
    parts = [p.strip() for p in text.split(". ") if p.strip()]
    return [p if p.endswith(".") else p + "." for p in parts]


def template_uses_policy(template) -> bool:
    return "{POLICY_X}" in template["text"] or "{POLICY_Y}" in template["text"]


def resolve_slots(lexicon: dict, lean: str, topic: str | None) -> dict:
    """Return the full slot dict for (lean, topic).

    For lean=neutral or topic=None: returns the top-level lexicon[lean] block
    verbatim (legacy / chess-poker / political defaults).

    For lean ∈ {left, right} with a topic: returns the top-level lexicon[lean]
    block with POLICY_X / POLICY_Y overridden per the (topic, lean) resolution
    convention — POLICY_X is in-group-favoured, POLICY_Y is out-group-favoured.
    """
    base = dict(lexicon[lean])
    base.pop("domain", None)  # informational only; not a slot
    if topic is None or lean == "neutral":
        return base
    topic_row = lexicon["policies_by_topic"][topic]
    if lean == "left":
        base["POLICY_X"] = topic_row["dem_favored"]
        base["POLICY_Y"] = topic_row["rep_favored"]
    elif lean == "right":
        base["POLICY_X"] = topic_row["rep_favored"]
        base["POLICY_Y"] = topic_row["dem_favored"]
    return base


def pick_injection_phrase(lexicon: dict, lean: str, template_id: str, topic: str | None) -> str:
    """Deterministically pick an injection sentence from `political_phrases`.

    Only called for lean=neutral under the post-2026-05-24 simplification
    (left/right are clean-only). Selection is deterministic via
    md5(template_id|lean|topic) so the dataset is byte-reproducible across runs.
    """
    assert lean == "neutral", (
        "pick_injection_phrase is only valid for lean=neutral after the "
        "2026-05-24 simplification; left/right have no injected variant."
    )
    pool = lexicon["noise_injections"]["political_phrases"]
    seed = f"{template_id}|{lean}|{topic or ''}"
    h = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
    return pool[h % len(pool)]


def inject_phrase(text: str, phrase: str) -> str:
    """Insert `phrase` mid-syllogism — between premises if there are ≥2,
    otherwise between the single premise and the conclusion."""
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return text
    premises, conclusion = sentences[:-1], sentences[-1]
    if len(premises) == 1:
        insertion_idx = 1
    else:
        insertion_idx = len(premises) // 2 + (len(premises) % 2)
        insertion_idx = max(1, insertion_idx)
    new_seq = premises[:insertion_idx] + [phrase] + premises[insertion_idx:] + [conclusion]
    return " ".join(new_seq)


def build_text(template, slots, lean, variant, lexicon, topic):
    base = template["text"].format(**slots)
    if variant == "clean":
        return base, None
    phrase = pick_injection_phrase(lexicon, lean, template["id"], topic)
    return inject_phrase(base, phrase), phrase


def synthesize_all(lexicon: dict, topic_ids: list[str]):
    items = []
    for template in TEMPLATES:
        uses_policy = template_uses_policy(template)
        for lean in LEANS:
            if uses_policy and lean in ("left", "right"):
                topics_for_this_cell = topic_ids
            else:
                topics_for_this_cell = [None]
            for topic in topics_for_this_cell:
                slots = resolve_slots(lexicon, lean, topic)
                for variant in variants_for_lean(lean):
                    text, injection = build_text(
                        template, slots, lean, variant, lexicon, topic
                    )
                    items.append({
                        "template_id": template["id"],
                        "valid": template["valid"],
                        "fallacy": template["fallacy"],
                        "lean": lean,
                        "variant": variant,
                        "topic": topic,
                        "text": text,
                        "injection": injection,
                    })
    return items


def _parse_topics_arg(arg: str | None, lexicon: dict) -> list[str]:
    available = list(lexicon["policies_by_topic"].keys())
    if not arg:
        return available
    requested = [t.strip() for t in arg.split(",") if t.strip()]
    unknown = [t for t in requested if t not in available]
    if unknown:
        raise ValueError(
            f"unknown topic(s) {unknown}; available: {available}"
        )
    return requested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexicon", type=Path, default=LEXICON_PATH,
                        help="path to lexicon JSON")
    parser.add_argument("--output", type=Path, default=STIMULI_PATH,
                        help="path for stimuli.jsonl")
    parser.add_argument("--topics", type=str, default=None,
                        help="comma-separated topic IDs (default: all 8)")
    args = parser.parse_args()

    if not args.lexicon.exists():
        raise FileNotFoundError(f"lexicon not found: {args.lexicon}")

    ensure_run_dirs()

    lexicon = load_lexicon(args.lexicon)
    topic_ids = _parse_topics_arg(args.topics, lexicon)

    items = synthesize_all(lexicon, topic_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    print(f"generated {len(items)} stimuli")
    print(f"  templates: {len(TEMPLATES)}")
    print(f"  leans:     {LEANS}  (neutral has 2 variants; left/right clean-only)")
    print(f"  topics:    {topic_ids}")
    print(f"  lexicon:   {args.lexicon}")
    print(f"  output:    {args.output}")


if __name__ == "__main__":
    main()
