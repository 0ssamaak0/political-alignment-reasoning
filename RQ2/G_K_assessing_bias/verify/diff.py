"""Between-workflow glue: assemble the blind Claude labels, validate them,
diff against the Gemini answer-key, and emit the disagreement subset as panel
input batches for workflow #2.

Run from repo root AFTER wf_blind completes:
    python -m RQ2.G_K_assessing_bias.verify.diff
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
BLIND = HERE / "blind"
PANEL_IN = HERE / "panel_inputs"
PANEL_IN.mkdir(exist_ok=True)

OUTCOME = {"correct", "wrong", "no_answer", "off_format"}
RV = {"valid", "invalid", "opaque", "n/a"}
PRIMARY = {"faithful_task_performance", "post_hoc_reasoning", "capability_error",
           "instruction_following_failure", "viewpoint_bias",
           "motivational_framing_bias", "generation_collapse"}
FALLACY = {"none", "equivocation", "false_dilemma", "token_bias_shortcut",
           "premise_truth_conflation", "illicit_premise_insertion",
           "motivational_reasoning", None}

CELLS = [
    "mistral-base", "llama-base",
    "mistral-roleplay-left", "mistral-roleplay-right",
    "llama-roleplay-left", "llama-roleplay-right",
    "mistral-steering-left", "mistral-steering-right",
    "llama-steering-left", "llama-steering-right",
    "mistral-DPO-left", "mistral-DPO-right",
    "llama-DPO-left", "llama-DPO-right", "llama-DPO-right-2nd",
]
N_BATCH = 8


def load_blind() -> dict[int, dict]:
    blind = {}
    problems = []
    for cell in CELLS:
        for b in range(N_BATCH):
            f = BLIND / f"{cell}__b{b}.json"
            if not f.exists():
                problems.append(f"MISSING {f.name}")
                continue
            try:
                arr = json.loads(f.read_text())
            except json.JSONDecodeError as e:
                problems.append(f"BAD JSON {f.name}: {e}")
                continue
            if len(arr) != 24:
                problems.append(f"{f.name}: {len(arr)} objs (expected 24)")
            for o in arr:
                gi = o.get("gidx")
                if gi is None:
                    problems.append(f"{f.name}: missing gidx")
                    continue
                # enum validation
                if o.get("outcome") not in OUTCOME: problems.append(f"gidx {gi}: outcome={o.get('outcome')}")
                if o.get("reasoning_validity") not in RV: problems.append(f"gidx {gi}: rv={o.get('reasoning_validity')}")
                if o.get("primary_category") not in PRIMARY: problems.append(f"gidx {gi}: pc={o.get('primary_category')}")
                fl = o.get("fallacy_lens")
                if fl == "none": o["fallacy_lens"] = fl = None
                if fl not in FALLACY: problems.append(f"gidx {gi}: fl={fl}")
                if not isinstance(o.get("contaminated"), bool): problems.append(f"gidx {gi}: contaminated not bool")
                if not isinstance(o.get("collapsed"), bool): problems.append(f"gidx {gi}: collapsed not bool")
                blind[gi] = o
    return blind, problems


def main():
    key = [json.loads(l) for l in (HERE / "gemini_key.jsonl").open()]
    key_by_gidx = {r["gidx"]: r for r in key}
    blind, problems = load_blind()

    print(f"[diff] blind labels: {len(blind)}/2880")
    if problems:
        print(f"[diff] {len(problems)} VALIDATION PROBLEMS:")
        for p in problems[:60]:
            print("   ", p)
        # which batches need re-run
        bad_batches = sorted({p.split()[1] for p in problems if p.startswith(("MISSING", "BAD JSON"))})
        if bad_batches:
            print("[diff] batches to re-run:", bad_batches)

    missing = [gi for gi in key_by_gidx if gi not in blind]
    if missing:
        print(f"[diff] {len(missing)} items missing blind labels — NOT writing panel inputs until resolved")
        return

    # --- merge + diff ----------------------------------------------------------
    merged = []
    disagree = []
    for gi, k in sorted(key_by_gidx.items()):
        c = blind[gi]
        row = {**k,
               "c_outcome": c["outcome"], "c_contaminated": bool(c["contaminated"]),
               "c_collapsed": bool(c["collapsed"]), "c_reasoning_validity": c["reasoning_validity"],
               "c_primary_category": c["primary_category"], "c_fallacy_lens": c.get("fallacy_lens"),
               "c_confidence": c.get("confidence"), "c_justification": c.get("justification")}
        # agreement bits
        row["agree_primary"] = (k["g_primary_category"] == c["primary_category"])
        row["agree_contaminated"] = (bool(k["g_contaminated"]) == bool(c["contaminated"]))
        row["agree_collapsed"] = (bool(k["g_collapsed"]) == bool(c["collapsed"]))
        row["agree_outcome"] = (k["g_outcome"] == c["outcome"])
        # route to panel if the integrative label OR the contamination flag differs
        route = (not row["agree_primary"]) or (not row["agree_contaminated"])
        row["panel"] = route
        merged.append(row)
        if route:
            disagree.append(gi)

    (HERE / "merged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in merged))

    # --- headline agreement rates ---------------------------------------------
    n = len(merged)
    def rate(field):
        return sum(1 for r in merged if r[field]) / n
    print(f"\n[diff] overall agreement (Claude-blind vs Gemini), n={n}:")
    print(f"   primary_category : {rate('agree_primary'):.1%}")
    print(f"   contaminated     : {rate('agree_contaminated'):.1%}")
    print(f"   collapsed        : {rate('agree_collapsed'):.1%}")
    print(f"   outcome          : {rate('agree_outcome'):.1%}")
    print(f"[diff] items routed to panel (primary OR contaminated differ): {len(disagree)} ({len(disagree)/n:.1%})")

    # contaminated disagreement DIRECTION by lean (the bias sniff) — pre-panel
    print("\n[diff] contaminated disagreements by lean (direction):")
    for lean in ("left", "right"):
        gT_cF = sum(1 for r in merged if r["lean"] == lean and r["g_contaminated"] and not r["c_contaminated"])
        gF_cT = sum(1 for r in merged if r["lean"] == lean and not r["g_contaminated"] and r["c_contaminated"])
        print(f"   {lean}: Gemini=T,Claude=F (Gemini over-flags) = {gT_cF}   "
              f"Gemini=F,Claude=T (Gemini under-flags) = {gF_cT}")

    # --- write panel input batches (blind format, disagreement items only) -----
    inrows = {}
    for cell in CELLS:
        for b in range(N_BATCH):
            for l in (INPUTS / f"{cell}__b{b}.jsonl").open():
                o = json.loads(l)
                inrows[o["gidx"]] = o
    panel_items = [inrows[gi] for gi in disagree]
    BATCH = 24
    nb = 0
    for i in range(0, len(panel_items), BATCH):
        chunk = panel_items[i:i + BATCH]
        with (PANEL_IN / f"pb{nb}.jsonl").open("w") as f:
            for o in chunk:
                f.write(json.dumps(o) + "\n")
        nb += 1
    print(f"\n[diff] wrote {nb} panel input batches ({len(panel_items)} items) to {PANEL_IN}")
    (HERE / "panel_manifest.json").write_text(json.dumps(
        {"n_batches": nb, "n_items": len(panel_items), "batch_size": BATCH}, indent=2))


if __name__ == "__main__":
    main()
