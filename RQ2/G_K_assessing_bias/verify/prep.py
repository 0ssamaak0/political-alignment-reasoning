"""Phase 0 of the judge-verification audit.

Builds blind verifier-input batch files (identical to what the Gemini judge saw,
MINUS its labels), a Gemini answer-key, a left<->right pair map, and runs the
deterministic consistency checks on the OBJECTIVE axes (outcome, collapsed) that
need no LLM.

Run from repo root:
    python -m RQ2.G_K_assessing_bias.verify.prep
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent          # .../verify
GK = HERE.parent                                 # .../G_K_assessing_bias
REPO = GK.parents[1]
RESPONSES = GK / "responses"
JUDGES = GK / "judges"
PROMPTS_CSV = REPO / "1_benchmarking" / "G_K_assessing_bias" / "data" / "prompts_192.csv"

INPUTS_DIR = HERE / "inputs"          # blind batch files for phase-1 agents
INPUTS_DIR.mkdir(parents=True, exist_ok=True)

CELLS = [
    "mistral-base", "llama-base",
    "mistral-roleplay-left", "mistral-roleplay-right",
    "llama-roleplay-left", "llama-roleplay-right",
    "mistral-steering-left", "mistral-steering-right",
    "llama-steering-left", "llama-steering-right",
    "mistral-DPO-left", "mistral-DPO-right",
    "llama-DPO-left", "llama-DPO-right", "llama-DPO-right-2nd",
]

BATCH = 24

# --- parity with run_judge.py: identical squeeze + identical loop_signals -----
_RUN_RE = re.compile(r"(.{1,4}?)\1{9,}")


def squeeze_degenerate(text: str, cap: int = 2000) -> str:
    squeezed = _RUN_RE.sub(lambda m: m.group(1) * 3 + " …[repeated, collapsed]… ", text)
    if len(squeezed) > cap:
        squeezed = squeezed[:cap] + " …[truncated]"
    return squeezed


def loop_signals(text: str) -> dict:
    tokens = text.split()
    if len(tokens) < 4:
        return {"max_4gram_repeat": 0, "distinct_ratio_last_50": 1.0}
    fourgrams = [" ".join(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    max_rep = max(Counter(fourgrams).values()) if fourgrams else 0
    last = tokens[-50:]
    distinct_ratio = len(set(last)) / max(len(last), 1)
    return {"max_4gram_repeat": max_rep, "distinct_ratio_last_50": round(distinct_ratio, 3)}


# --- pair map: (Pattern,Variation,Gender,Is-Valid) -> pair_id, one left+one right
def build_pair_map() -> dict[int, dict]:
    df = pd.read_csv(PROMPTS_CSV).reset_index().rename(columns={"index": "row_idx"})
    key = ["Pattern-ID", "Variation-ID", "Gender", "Is-Valid"]
    df["pair_id"] = df.groupby(key).ngroup()
    return {
        int(r["row_idx"]): {
            "pair_id": int(r["pair_id"]),
            "gender": str(r["Gender"]),
            "pattern": str(r["Pattern-ID"]),
            "variation": str(r["Variation-ID"]),
        }
        for _, r in df.iterrows()
    }


def main():
    pairmap = build_pair_map()
    n_pairs = len({v["pair_id"] for v in pairmap.values()})
    print(f"[prep] {n_pairs} left/right matched pairs over 192 items")

    answer_key = []          # one row per (cell,row_idx) with Gemini labels + ids
    consistency_flags = []   # deterministic objective-axis inconsistencies
    n_batches = 0

    for ci, cell in enumerate(CELLS):
        resp = [json.loads(l) for l in (RESPONSES / f"{cell}.jsonl").open()]
        jud = [json.loads(l) for l in (JUDGES / f"{cell}.jsonl").open()]
        assert len(resp) == len(jud) == 192, (cell, len(resp), len(jud))

        blind_rows = []
        for row_idx, (r, j) in enumerate(zip(resp, jud)):
            assert (r["template_id"], r["lean"], r["valid"]) == \
                   (j["template_id"], j["lean"], j["valid"]), (cell, row_idx)
            gidx = ci * 192 + row_idx
            pm = pairmap[row_idx]
            sig = loop_signals(r["raw_response"])
            squeezed = squeeze_degenerate(r["raw_response"])

            # blind input (NO gemini labels) — exactly Gemini's INPUT_TEMPLATE fields
            blind_rows.append({
                "gidx": gidx,
                "cell": cell,
                "row_idx": row_idx,
                "pair_id": pm["pair_id"],
                "template_id": r["template_id"],
                "lean": r["lean"],
                "gold_valid": int(r["valid"]),
                "parsed_verdict": r.get("verdict"),
                "max_4gram_repeat": sig["max_4gram_repeat"],
                "distinct_ratio_last_50": sig["distinct_ratio_last_50"],
                "prompt_text": r["text"],
                "response": squeezed,
            })

            # gemini answer key
            answer_key.append({
                "gidx": gidx, "cell": cell, "row_idx": row_idx,
                "pair_id": pm["pair_id"], "gender": pm["gender"],
                "pattern": pm["pattern"], "variation": pm["variation"],
                "lean": r["lean"], "gold_valid": int(r["valid"]),
                "parsed_verdict": r.get("verdict"),
                "g_outcome": j["outcome"],
                "g_contaminated": bool(j["contaminated"]),
                "g_collapsed": bool(j["collapsed"]),
                "g_reasoning_validity": j["reasoning_validity"],
                "g_primary_category": j["primary_category"],
                "g_fallacy_lens": j.get("fallacy_lens"),
                "g_confidence": j.get("confidence"),
                "max_4gram_repeat": sig["max_4gram_repeat"],
                "distinct_ratio_last_50": sig["distinct_ratio_last_50"],
            })

            # --- deterministic objective-axis consistency checks (no LLM) ------
            pv = (r.get("verdict") or "").upper()
            gold_label = "VALID" if int(r["valid"]) == 1 else "INVALID"
            mapped = pv in ("VALID", "INVALID")
            oc = j["outcome"]
            # (a) outcome vs parsed verdict / gold
            if mapped:
                expected = "correct" if pv == gold_label else "wrong"
                if oc != expected and oc in ("correct", "wrong"):
                    consistency_flags.append({
                        "gidx": gidx, "cell": cell, "row_idx": row_idx, "lean": r["lean"],
                        "axis": "outcome", "kind": "outcome_vs_verdict_mismatch",
                        "detail": f"parsed={pv} gold={gold_label} -> expected {expected}, judge said {oc}",
                    })
                if oc in ("no_answer", "off_format"):
                    consistency_flags.append({
                        "gidx": gidx, "cell": cell, "row_idx": row_idx, "lean": r["lean"],
                        "axis": "outcome", "kind": "judge_override_to_unparseable",
                        "detail": f"cascade parsed {pv} but judge outcome={oc}",
                    })
            else:
                if oc in ("correct", "wrong"):
                    consistency_flags.append({
                        "gidx": gidx, "cell": cell, "row_idx": row_idx, "lean": r["lean"],
                        "axis": "outcome", "kind": "judge_parsed_what_cascade_didnt",
                        "detail": f"cascade unmapped (verdict={pv!r}) but judge outcome={oc}",
                    })
            # (b) collapsed flag vs 4-gram signal
            mr = sig["max_4gram_repeat"]
            if not bool(j["collapsed"]) and mr >= 8:
                consistency_flags.append({
                    "gidx": gidx, "cell": cell, "row_idx": row_idx, "lean": r["lean"],
                    "axis": "collapsed", "kind": "collapsed_false_high_repeat",
                    "detail": f"max_4gram_repeat={mr} but collapsed=false",
                })
            if bool(j["collapsed"]) and mr <= 1 and sig["distinct_ratio_last_50"] >= 0.6:
                consistency_flags.append({
                    "gidx": gidx, "cell": cell, "row_idx": row_idx, "lean": r["lean"],
                    "axis": "collapsed", "kind": "collapsed_true_low_signal",
                    "detail": f"max_4gram_repeat={mr} distinct={sig['distinct_ratio_last_50']} but collapsed=true",
                })

        # write blind batch files for this cell
        for b in range(0, 192, BATCH):
            chunk = blind_rows[b:b + BATCH]
            bf = INPUTS_DIR / f"{cell}__b{b // BATCH}.jsonl"
            with bf.open("w") as f:
                for row in chunk:
                    f.write(json.dumps(row) + "\n")
            n_batches += 1

    (HERE / "gemini_key.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in answer_key))
    (HERE / "consistency_flags.json").write_text(json.dumps(consistency_flags, indent=2))

    # summarise the deterministic pass
    by_kind = Counter(f["kind"] for f in consistency_flags)
    print(f"[prep] wrote {n_batches} blind batch files to {INPUTS_DIR}")
    print(f"[prep] wrote gemini_key.jsonl ({len(answer_key)} rows)")
    print(f"[prep] deterministic objective-axis inconsistencies: {len(consistency_flags)}")
    for k, v in by_kind.most_common():
        print(f"          {k}: {v}")
    # lean split of the inconsistencies (early asymmetry sniff)
    by_lean = Counter(f["lean"] for f in consistency_flags)
    print(f"[prep] inconsistency lean split: {dict(by_lean)}")


if __name__ == "__main__":
    main()
