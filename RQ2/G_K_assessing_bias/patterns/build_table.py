"""Build ONE canonical long-format table from the trusted Gemini-Flash judge
labels, so all pattern-mining agents read the same numbers and handle collapse
identically.

Source of record = the Flash judge: RQ2/G_K_assessing_bias/judges/*.jsonl.
Output = patterns/judge_long.jsonl (one row per response) + patterns/base_rates.json
(same-family base reference rates for Δ-from-base) + a console sanity dump.

Run from repo root:
    python -m RQ2.G_K_assessing_bias.patterns.build_table
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
GK = HERE.parent
JUDGES = GK / "judges"
OUT = HERE / "judge_long.jsonl"

CELLS = [
    "mistral-base", "llama-base",
    "mistral-roleplay-left", "mistral-roleplay-right",
    "llama-roleplay-left", "llama-roleplay-right",
    "mistral-steering-left", "mistral-steering-right",
    "llama-steering-left", "llama-steering-right",
    "mistral-DPO-left", "mistral-DPO-right",
    "llama-DPO-left", "llama-DPO-right",
]  # llama-DPO-right-2nd (runner-up adapter) excluded from final RQ2 analysis; raw files kept
BIAS_PC = {"viewpoint_bias", "motivational_framing_bias"}
NONENGAGED_PC = {"generation_collapse", "instruction_following_failure"}


def regime(cell):
    if "roleplay" in cell: return "roleplay"
    if "steering" in cell: return "steering"
    if "DPO" in cell: return "DPO"
    return "base"


def induced_lean(cell):
    if cell.endswith("base"): return "none"
    if "left" in cell: return "left"
    return "right"


def family(cell):
    return "mistral" if cell.startswith("mistral") else "llama"


def main():
    rows = []
    for ci, cell in enumerate(CELLS):
        for row_idx, j in enumerate(json.loads(l) for l in (JUDGES / f"{cell}.jsonl").open()):
            if "primary_category" not in j:
                continue
            tid = j["template_id"]
            variation = tid.split("_", 1)[1] if "_" in tid else ""
            tfam = tid.split("_", 1)[0]
            pv = (j.get("verdict") or "").upper()
            mapped = pv in ("VALID", "INVALID")
            pc = j["primary_category"]
            collapsed = bool(j["collapsed"])
            outcome = j["outcome"]
            engaged = (not collapsed) and (pc not in NONENGAGED_PC) and (outcome in ("correct", "wrong"))
            rows.append({
                "gidx": ci * 192 + row_idx,
                "cell": cell, "family": family(cell), "regime": regime(cell),
                "induced_lean": induced_lean(cell),
                "item_lean": j["lean"], "gold_valid": int(j["valid"]),
                "parsed_verdict": pv or None, "mapped": mapped,
                "variation": variation, "template_family": tfam,
                "outcome": outcome, "correct": outcome == "correct",
                "contaminated": bool(j["contaminated"]), "collapsed": collapsed,
                "reasoning_validity": j["reasoning_validity"],
                "primary_category": pc,
                "is_bias_cat": pc in BIAS_PC,
                "fallacy_lens": j.get("fallacy_lens"),
                "engaged": engaged,
                # kept for the handmade leg (reading actual text):
                "justification": j.get("justification"),
                "judge_reasoning": j.get("reasoning"),
                "raw_response": j.get("raw_response"),
                "prompt": j.get("text"),
            })

    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))

    # same-family base reference rates (for Δ-from-base)
    base_rates = {}
    for fam in ("mistral", "llama"):
        b = [r for r in rows if r["cell"] == f"{fam}-base"]
        eng = [r for r in b if r["engaged"]]
        base_rates[fam] = {
            "n": len(b), "engaged_rate": round(len(eng) / len(b), 3),
            "contaminated_rate_engaged": round(sum(r["contaminated"] for r in eng) / max(len(eng), 1), 3),
            "bias_cat_rate_engaged": round(sum(r["is_bias_cat"] for r in eng) / max(len(eng), 1), 3),
            "acc_all": round(sum(r["correct"] for r in b) / len(b), 3),
            "acc_engaged": round(sum(r["correct"] for r in eng) / max(len(eng), 1), 3),
        }
    (HERE / "base_rates.json").write_text(json.dumps(base_rates, indent=2))

    # sanity dump
    print(f"[build] wrote {len(rows)} rows -> {OUT.name}")
    print(f"[build] base reference rates: {json.dumps(base_rates)}")
    print("[build] per-cell engaged-rate / acc_all / acc_engaged / contam(eng) / biascat(eng):")
    for cell in CELLS:
        cr = [r for r in rows if r["cell"] == cell]
        eng = [r for r in cr if r["engaged"]]
        er = len(eng) / len(cr)
        acc_all = sum(r["correct"] for r in cr) / len(cr)
        acc_eng = sum(r["correct"] for r in eng) / max(len(eng), 1)
        cont = sum(r["contaminated"] for r in eng) / max(len(eng), 1)
        bc = sum(r["is_bias_cat"] for r in eng) / max(len(eng), 1)
        print(f"   {cell:24s} eng={er:5.1%} acc_all={acc_all:5.1%} acc_eng={acc_eng:5.1%} "
              f"contam={cont:5.1%} biascat={bc:5.1%}")


if __name__ == "__main__":
    main()
