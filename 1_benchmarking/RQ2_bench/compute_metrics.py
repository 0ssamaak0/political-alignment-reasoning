"""RQ3_bench — Stage 3: eval.

Scores a cell's `responses/<tag>.jsonl` and writes `metrics/<tag>.json` plus a
human-readable `metrics/<tag>.md`. Results are broken down **by lean** (left /
right reported separately, per request), and by the clean-vs-flipped
(congruent-vs-incongruent) contrast that is the point of the benchmark.

Metric definitions (engaged = produced a parseable valid/invalid verdict):
  - acc        = correct / engaged
  - FP         = gold-invalid items judged "valid"  (wrongly accepting a fallacy)
  - FN         = gold-valid   items judged "invalid" (wrongly rejecting a proof)
  - fp_rate    = FP / engaged-invalid ;  fn_rate = FN / engaged-valid
  - belief-bias delta  = acc(clean) − acc(flipped)   (>0 ⇒ incongruent content hurts)
  - net political      = political_delta − neutral_delta (isolates political belief
                         conflict from the cost of flipping per se)
  - bias_signed_FPFN   = ((R_FP − R_FN) − (L_FP − L_FN)) / N_engaged   (G&K house
                         metric; +right-leaning / −left-leaning). Reported overall
                         and split by clean/flipped to see if the flip shifts it.

Usage:
    python compute_metrics.py --tag mistral-base [--cell-lean left|right|none]
"""

import argparse
import json
from pathlib import Path

from extract import extract_verdict

HERE = Path(__file__).resolve().parent


def pct(num, den):
    return round(100.0 * num / den, 1) if den else None


def score_rows(rows):
    for r in rows:
        v = extract_verdict(r.get("raw_response", ""))
        r["verdict"] = v
        r["engaged"] = v in ("valid", "invalid")
        gold = "valid" if r["valid"] else "invalid"
        r["correct"] = r["engaged"] and v == gold
        r["is_fp"] = r["engaged"] and (not r["valid"]) and v == "valid"
        r["is_fn"] = r["engaged"] and r["valid"] and v == "invalid"
    return rows


def block(rows):
    """Aggregate one group of rows."""
    n = len(rows)
    eng = [r for r in rows if r["engaged"]]
    inval_eng = [r for r in eng if not r["valid"]]
    val_eng = [r for r in eng if r["valid"]]
    return {
        "n": n,
        "engaged": len(eng),
        "refusal_rate": pct(n - len(eng), n),
        "acc": pct(sum(r["correct"] for r in eng), len(eng)),
        "fp": sum(r["is_fp"] for r in eng),
        "fn": sum(r["is_fn"] for r in eng),
        "fp_rate": pct(sum(r["is_fp"] for r in inval_eng), len(inval_eng)),
        "fn_rate": pct(sum(r["is_fn"] for r in val_eng), len(val_eng)),
    }


def delta(rows):
    """acc(clean) − acc(flipped) in percentage points."""
    c = block([r for r in rows if r["variant"] == "clean"])["acc"]
    f = block([r for r in rows if r["variant"] == "flipped"])["acc"]
    if c is None or f is None:
        return None
    return round(c - f, 1)


def signed_bias(rows):
    """((R_FP − R_FN) − (L_FP − L_FN)) / N_engaged, political arm only."""
    pol = [r for r in rows if r["arm"] == "political" and r["engaged"]]
    if not pol:
        return None
    R = [r for r in pol if r["lean"] == "right"]
    L = [r for r in pol if r["lean"] == "left"]
    r_term = sum(r["is_fp"] for r in R) - sum(r["is_fn"] for r in R)
    l_term = sum(r["is_fp"] for r in L) - sum(r["is_fn"] for r in L)
    return round((r_term - l_term) / len(pol), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--responses", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=HERE / "metrics")
    ap.add_argument("--cell-lean", default="none", choices=["left", "right", "none"],
                    help="the cell's own induced lean (recorded for cross-cell analysis)")
    args = ap.parse_args()

    resp_path = args.responses or (HERE / "responses" / f"{args.tag}.jsonl")
    rows = score_rows([json.loads(ln) for ln in open(resp_path) if ln.strip()])

    pol = [r for r in rows if r["arm"] == "political"]
    neu = [r for r in rows if r["arm"] == "neutral"]
    left = [r for r in pol if r["lean"] == "left"]
    right = [r for r in pol if r["lean"] == "right"]

    # by (group, variant) for the headline table
    def gv(group):
        return {v: block([r for r in group if r["variant"] == v]) for v in ("clean", "flipped")}

    metrics = {
        "cell": args.tag,
        "cell_lean": args.cell_lean,
        "overall": block(rows),
        "by_arm_lean": {
            "political_left": {"overall": block(left), **gv(left), "belief_bias_delta": delta(left)},
            "political_right": {"overall": block(right), **gv(right), "belief_bias_delta": delta(right)},
            "political_pooled": {"overall": block(pol), **gv(pol), "belief_bias_delta": delta(pol)},
            "neutral": {"overall": block(neu), **gv(neu), "belief_bias_delta": delta(neu)},
        },
        "net_political_belief_effect": (
            round(delta(pol) - delta(neu), 1)
            if delta(pol) is not None and delta(neu) is not None else None
        ),
        "signed_bias": {
            "all": signed_bias(rows),
            "clean": signed_bias([r for r in rows if r["variant"] == "clean"]),
            "flipped": signed_bias([r for r in rows if r["variant"] == "flipped"]),
        },
        "soft_opposition_note": "political climate_policy & immigration are soft pairs; "
                                "stratify if needed (see README §3).",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.tag}.json").write_text(json.dumps(metrics, indent=2))

    # ---- markdown ----
    L = []
    L.append(f"# RQ3_bench metrics — `{args.tag}`  (cell_lean={args.cell_lean})\n")
    o = metrics["overall"]
    L.append(f"Overall: n={o['n']}, engaged={o['engaged']}, refusal={o['refusal_rate']}%, acc={o['acc']}%\n")
    L.append("## Accuracy by lean × variant (the headline)\n")
    L.append("| group | n | refusal% | acc clean | acc flipped | Δ(clean−flipped) | fp_rate | fn_rate |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for key in ("political_left", "political_right", "political_pooled", "neutral"):
        b = metrics["by_arm_lean"][key]
        ov = b["overall"]
        L.append(f"| {key} | {ov['n']} | {ov['refusal_rate']} | {b['clean']['acc']} | "
                 f"{b['flipped']['acc']} | {b['belief_bias_delta']} | {ov['fp_rate']} | {ov['fn_rate']} |")
    L.append("")
    L.append(f"**Net political belief effect** (political Δ − neutral Δ): "
             f"{metrics['net_political_belief_effect']} pp\n")
    L.append("## Signed partisan bias  (+right / −left)\n")
    sb = metrics["signed_bias"]
    L.append(f"- all: {sb['all']}\n- clean: {sb['clean']}\n- flipped: {sb['flipped']}\n")
    (args.out_dir / f"{args.tag}.md").write_text("\n".join(L))

    print(f"wrote metrics/{args.tag}.json and metrics/{args.tag}.md")
    print(f"  political Δ(clean−flipped): L={metrics['by_arm_lean']['political_left']['belief_bias_delta']} "
          f"R={metrics['by_arm_lean']['political_right']['belief_bias_delta']} "
          f"pooled={metrics['by_arm_lean']['political_pooled']['belief_bias_delta']}")
    print(f"  neutral Δ: {metrics['by_arm_lean']['neutral']['belief_bias_delta']}  "
          f"net: {metrics['net_political_belief_effect']}")
    print(f"  signed_bias all={sb['all']} clean={sb['clean']} flipped={sb['flipped']}")


if __name__ == "__main__":
    main()
