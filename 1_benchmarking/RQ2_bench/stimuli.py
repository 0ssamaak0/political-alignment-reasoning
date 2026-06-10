"""RQ3_bench — Stage 1: emit belief-incongruent syllogism stimuli.

Self-contained (does not import from custom_bench). Reads the from-scratch
`lexicon.json` and emits `stimuli.jsonl`, one stimulus per line.

Design (see README.md):
  - patterns T2, T3, T6 — the three identity-only families whose templates use
    BOTH {POLICY_X} and {POLICY_Y}, so the flip is a clean X<->Y rename that
    preserves the gold validity label. Each pattern: 2 paraphrases (a/b) x 2
    polarities (valid/invalid) = 4 templates -> 12 templates total.
  - variant: clean (congruent) | flipped (incongruent). flipped swaps the values
    bound to {POLICY_X} and {POLICY_Y} at fill time. The lexicon stores clean.
  - political arm: lean in {left,right} x 8 topics. clean attributes the in-group
    its own favored policy (POLICY_X = dem/rep_favored by lean); flipped attributes
    the out-group's policy (counter-attitudinal premise).
  - neutral arm: 5 inert domains, no lean. clean = the group's characteristic
    activity in POLICY_X; flipped = the counter-stereotypical one. Weak prior =
    believability control.

Counts: political 12 x 2 leans x 8 topics x 2 variants = 384;
        neutral   12 x 5 domains x 2 variants          = 120; total = 504.

Usage:
    python3 stimuli.py [--lexicon lexicon.json] [--out stimuli.jsonl]
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Templates — T2, T3, T6 (verbatim logical scaffolds; {GROUP_A}/{POLICY_X}/
# {POLICY_Y} only). Lineage: Betz et al. AAC -> BBH formal_fallacies siblings
# -> Gubelmann & Karray (ACL 2025). See README.md / custom_bench docs/TEMPLATES.md.
# --------------------------------------------------------------------------
TEMPLATES = [
    # Pattern 2 — universal categorical (Camestres-like)
    {"id": "T2a_valid", "pattern": "T2", "valid": True, "fallacy": None, "text": (
        "Every one of the {GROUP_A} supports {POLICY_X}. "
        "No one who supports {POLICY_X} supports {POLICY_Y}. "
        "Therefore, no one of the {GROUP_A} supports {POLICY_Y}.")},
    {"id": "T2a_invalid", "pattern": "T2", "valid": False, "fallacy": "denying the antecedent", "text": (
        "Every one of the {GROUP_A} supports {POLICY_X}. "
        "No one who supports {POLICY_X} supports {POLICY_Y}. "
        "Therefore, anyone who is not one of the {GROUP_A} supports {POLICY_Y}.")},
    {"id": "T2b_valid", "pattern": "T2", "valid": True, "fallacy": None, "text": (
        "All members of the {GROUP_A} support {POLICY_X}. "
        "Anyone who supports {POLICY_X} does not support {POLICY_Y}. "
        "Hence, no member of the {GROUP_A} supports {POLICY_Y}.")},
    {"id": "T2b_invalid", "pattern": "T2", "valid": False, "fallacy": "denying the antecedent", "text": (
        "All members of the {GROUP_A} support {POLICY_X}. "
        "Anyone who supports {POLICY_X} does not support {POLICY_Y}. "
        "Hence, anyone outside the {GROUP_A} supports {POLICY_Y}.")},
    # Pattern 3 — hypothetical syllogism
    {"id": "T3a_valid", "pattern": "T3", "valid": True, "fallacy": None, "text": (
        "If anyone is one of the {GROUP_A}, they support {POLICY_X}. "
        "If anyone supports {POLICY_X}, they oppose {POLICY_Y}. "
        "Therefore, if anyone is one of the {GROUP_A}, they oppose {POLICY_Y}.")},
    {"id": "T3a_invalid", "pattern": "T3", "valid": False, "fallacy": "affirming the consequent", "text": (
        "If anyone is one of the {GROUP_A}, they support {POLICY_X}. "
        "If anyone supports {POLICY_X}, they oppose {POLICY_Y}. "
        "Therefore, anyone who opposes {POLICY_Y} is one of the {GROUP_A}.")},
    {"id": "T3b_valid", "pattern": "T3", "valid": True, "fallacy": None, "text": (
        "Anyone in the {GROUP_A} supports {POLICY_X}. "
        "Anyone supporting {POLICY_X} opposes {POLICY_Y}. "
        "Hence, anyone in the {GROUP_A} opposes {POLICY_Y}.")},
    {"id": "T3b_invalid", "pattern": "T3", "valid": False, "fallacy": "affirming the consequent", "text": (
        "Anyone in the {GROUP_A} supports {POLICY_X}. "
        "Anyone supporting {POLICY_X} opposes {POLICY_Y}. "
        "Hence, anyone opposing {POLICY_Y} is in the {GROUP_A}.")},
    # Pattern 6 — modus tollens (contrapositive) and the inverse fallacy
    {"id": "T6a_valid", "pattern": "T6", "valid": True, "fallacy": None, "text": (
        "Whoever supports {POLICY_Y} is one of the {GROUP_A}. "
        "Whoever is one of the {GROUP_A} supports {POLICY_X}. "
        "Therefore, whoever does not support {POLICY_X} does not support {POLICY_Y}.")},
    {"id": "T6a_invalid", "pattern": "T6", "valid": False, "fallacy": "denying the antecedent (chained)", "text": (
        "Whoever supports {POLICY_Y} is one of the {GROUP_A}. "
        "Whoever is one of the {GROUP_A} supports {POLICY_X}. "
        "Therefore, whoever does not support {POLICY_Y} does not support {POLICY_X}.")},
    {"id": "T6b_valid", "pattern": "T6", "valid": True, "fallacy": None, "text": (
        "Anyone supporting {POLICY_Y} belongs to the {GROUP_A}. "
        "Anyone in the {GROUP_A} supports {POLICY_X}. "
        "Hence, anyone not supporting {POLICY_X} does not support {POLICY_Y}.")},
    {"id": "T6b_invalid", "pattern": "T6", "valid": False, "fallacy": "denying the antecedent (chained)", "text": (
        "Anyone supporting {POLICY_Y} belongs to the {GROUP_A}. "
        "Anyone in the {GROUP_A} supports {POLICY_X}. "
        "Hence, anyone not supporting {POLICY_Y} does not support {POLICY_X}.")},
]

VARIANTS = ["clean", "flipped"]


def resolve_policies(in_group_policy, out_group_policy, variant):
    """clean: POLICY_X = in-group/characteristic (congruent).
    flipped: POLICY_X = out-group/counter-stereotypical (incongruent)."""
    if variant == "clean":
        return in_group_policy, out_group_policy
    return out_group_policy, in_group_policy


def fill(template, group_a, policy_x, policy_y):
    return template["text"].format(GROUP_A=group_a, POLICY_X=policy_x, POLICY_Y=policy_y)


def synthesize(lexicon):
    items = []

    # ---- political arm ----
    pol = lexicon["political"]
    for lean in ("left", "right"):
        group_a = pol["parties"][lean]["GROUP_A"]
        for topic, row in pol["topics"].items():
            in_grp = row["dem_favored"] if lean == "left" else row["rep_favored"]
            out_grp = row["rep_favored"] if lean == "left" else row["dem_favored"]
            for t in TEMPLATES:
                for variant in VARIANTS:
                    px, py = resolve_policies(in_grp, out_grp, variant)
                    items.append({
                        "arm": "political",
                        "lean": lean,
                        "topic": topic,
                        "domain": None,
                        "template_id": t["id"],
                        "pattern": t["pattern"],
                        "valid": t["valid"],
                        "fallacy": t["fallacy"],
                        "variant": variant,
                        "congruent": variant == "clean",
                        "soft_opposition": bool(row.get("soft_opposition", False)),
                        "group_a": group_a,
                        "policy_x": px,
                        "policy_y": py,
                        "text": fill(t, group_a, px, py),
                    })

    # ---- neutral arm ----
    for domain, row in lexicon["neutral"].items():
        if domain.startswith("_"):
            continue
        group_a = row["GROUP_A"]
        char_act, contra_act = row["POLICY_X"], row["POLICY_Y"]
        for t in TEMPLATES:
            for variant in VARIANTS:
                px, py = resolve_policies(char_act, contra_act, variant)
                items.append({
                    "arm": "neutral",
                    "lean": None,
                    "topic": None,
                    "domain": domain,
                    "template_id": t["id"],
                    "pattern": t["pattern"],
                    "valid": t["valid"],
                    "fallacy": t["fallacy"],
                    "variant": variant,
                    "congruent": variant == "clean",
                    "soft_opposition": False,
                    "group_a": group_a,
                    "policy_x": px,
                    "policy_y": py,
                    "text": fill(t, group_a, px, py),
                })

    for i, it in enumerate(items):
        it["row_idx"] = i
        key = it["topic"] or it["domain"]
        it["id"] = f"{it['arm'][:3]}-{it['lean'] or 'na'}-{key}-{it['template_id']}-{it['variant']}"
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", type=Path, default=HERE / "lexicon.json")
    ap.add_argument("--out", type=Path, default=HERE / "stimuli.jsonl")
    args = ap.parse_args()

    with open(args.lexicon) as f:
        lexicon = json.load(f)

    items = synthesize(lexicon)
    with open(args.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    n_pol = sum(1 for it in items if it["arm"] == "political")
    n_neu = sum(1 for it in items if it["arm"] == "neutral")
    print(f"wrote {len(items)} stimuli -> {args.out}")
    print(f"  political: {n_pol}  (12 templates x 2 leans x 8 topics x 2 variants)")
    print(f"  neutral:   {n_neu}  (12 templates x 5 domains x 2 variants)")
    print(f"  variants:  clean={sum(1 for it in items if it['variant']=='clean')} "
          f"flipped={sum(1 for it in items if it['variant']=='flipped')}")


if __name__ == "__main__":
    main()
