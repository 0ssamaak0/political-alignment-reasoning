"""Stage 4 — combine per-config metrics into summary.json + bias_table.md.

Adds:
  * easy-subset accuracy (computed across configs by joining each family's
    base verdicts: a stimulus is "easy" if both family bases got it right).
  * a markdown summary table.
"""

import json
from collections import defaultdict
from pathlib import Path

from custom_bench.config import (
    BIAS_TABLE_PATH,
    RESPONSES_DIR,
    SUMMARY_PATH,
    all_configs,
    metrics_path,
    responses_path,
)
from custom_bench.parse import parse_verdict


def _resolve_verdict(item):
    if "verdict" in item:
        return item["verdict"]
    v, _ = parse_verdict(item.get("raw_response", ""))
    return v


def _correct_keys(responses):
    out = set()
    for item in responses:
        gold = "valid" if item["valid"] else "invalid"
        if _resolve_verdict(item) == gold:
            out.add((
                item["template_id"],
                item["lean"],
                item.get("variant"),
                item.get("topic"),
            ))
    return out


def compute_easy_subset(tags):
    """Return the set of stimuli that *every base config* gets right.

    Base = tags ending in '-base'. A stimulus is "easy" if all base configs
    got it correct.
    """
    base_tags = [t for t in tags if t.endswith("-base")]
    if not base_tags:
        return set()
    sets = []
    for t in base_tags:
        path = responses_path(t)
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path)]
        sets.append(_correct_keys(rows))
    if not sets:
        return set()
    return set.intersection(*sets)


def acc_on_easy(tag, easy_keys):
    path = responses_path(tag)
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path)]
    if not easy_keys:
        return None
    n_total = 0
    n_correct = 0
    for r in rows:
        key = (r["template_id"], r["lean"], r.get("variant"), r.get("topic"))
        if key not in easy_keys:
            continue
        n_total += 1
        gold = "valid" if r["valid"] else "invalid"
        if _resolve_verdict(r) == gold:
            n_correct += 1
    if n_total == 0:
        return None
    return {"acc_easy": n_correct / n_total, "n_easy": n_total}


def _load_metrics(tag):
    path = metrics_path(tag)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_summary():
    tags = [tag for tag, *_ in all_configs()]
    extra = sorted({p.stem for p in RESPONSES_DIR.rglob("*.jsonl") if "__" not in p.stem and p.stem not in tags})
    tags = tags + extra
    easy = compute_easy_subset(tags)

    summary = {"easy_subset_size": len(easy), "configs": {}}
    for tag in tags:
        m = _load_metrics(tag)
        if m is None:
            continue
        e = acc_on_easy(tag, easy)
        m["easy_subset"] = e
        summary["configs"][tag] = m
    return summary


def _fmt(x, prec=3, signed=False):
    if x is None:
        return "n/a"
    return f"{x:+.{prec}f}" if signed else f"{x:.{prec}f}"


def _p(metrics):
    cs = metrics.get("chi_square")
    if cs is None:
        return "n/a"
    if "p_value" not in cs:
        return "n/a"
    return f"{cs['p_value']:.3g}"


def write_bias_table(summary):
    from custom_bench.config import EXPERIMENT
    lines = [f"# bias_table — {EXPERIMENT}", ""]
    lines.append("Headline metric is `bias_signed_FPFN` (Gubelmann/Karray): "
                 "`((R_FP − R_FN) − (L_FP − L_FN)) / N_engaged`. "
                 "Positive = right-leaning, negative = left-leaning.")
    lines.append("")
    lines.append(f"Easy-subset size (items every base solves): "
                 f"**{summary['easy_subset_size']}**")
    lines.append("")
    lines.append("## Main table (between-lean bias)")
    lines.append("")
    lines.append("| tag | n | acc_N | acc_L | acc_R | bias_FPFN | bias_acc | bias_centered | refusal_R−L | eff_dmg | acc_easy | n_easy | p |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, m in summary["configs"].items():
        bl = m.get("by_lean", {})
        n = sum(v.get("n", 0) for v in bl.values())
        accN = bl.get("neutral", {}).get("accuracy")
        accL = bl.get("left",    {}).get("accuracy")
        accR = bl.get("right",   {}).get("accuracy")
        easy = m.get("easy_subset")
        acc_easy = easy["acc_easy"] if easy else None
        n_easy = easy["n_easy"] if easy else None
        lines.append(
            f"| {tag} | {n} | {_fmt(accN)} | {_fmt(accL)} | {_fmt(accR)} | "
            f"{_fmt(m.get('bias_signed_FPFN'), signed=True)} | "
            f"{_fmt(m.get('bias_acc_simple'),  signed=True)} | "
            f"{_fmt(m.get('bias_centered'),    signed=True)} | "
            f"{_fmt(m.get('refusal_asym'),     signed=True)} | "
            f"{_fmt(m.get('eff_damage'),       signed=True)} | "
            f"{_fmt(acc_easy)} | {n_easy if n_easy is not None else 'n/a'} | "
            f"{_p(m)} |"
        )

    lines.append("")
    lines.append("## Verdict balance (mode-collapse check: valid-only / invalid-only)")
    lines.append("")
    lines.append(
        "Dataset is balanced 196 valid / 196 invalid by construction, so a "
        "balanced reasoner predicts ~50% valid. `predicted_valid_rate` ≪ 0.5 "
        "means the model leans 'invalid' regardless of input (valid-class items "
        "get marked wrong); ≫ 0.5 means it leans 'valid' (invalid-class items "
        "get marked wrong). `recall_valid` / `recall_invalid` decompose accuracy "
        "by gold class — large asymmetry reveals heuristic verdict-skew that the "
        "between-lean bias_FPFN can mask. ⚠️ marked when `predicted_valid_rate` "
        "is <0.10 or >0.90 (mode collapse)."
    )
    lines.append("")
    lines.append("| tag | pred_valid_rate (overall) | skew | collapse | recall_V_N | recall_I_N | recall_V_L | recall_I_L | recall_V_R | recall_I_R |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for tag, m in summary["configs"].items():
        vd = m.get("verdict_distribution") or {}
        bl = m.get("by_lean") or {}
        n_row = bl.get("neutral") or {}
        l_row = bl.get("left") or {}
        r_row = bl.get("right") or {}
        collapse = "⚠️" if vd.get("verdict_collapse_flag") else ""
        lines.append(
            f"| {tag} | "
            f"{_fmt(vd.get('predicted_valid_rate'))} | "
            f"{_fmt(vd.get('verdict_skew'), signed=True)} | "
            f"{collapse} | "
            f"{_fmt(n_row.get('recall_valid'))} | {_fmt(n_row.get('recall_invalid'))} | "
            f"{_fmt(l_row.get('recall_valid'))} | {_fmt(l_row.get('recall_invalid'))} | "
            f"{_fmt(r_row.get('recall_valid'))} | {_fmt(r_row.get('recall_invalid'))} |"
        )

    lines.append("")
    lines.append("## Noise-injection robustness (acc_clean − acc_injected)")
    lines.append("")
    lines.append(
        "Positive Δ = the 'injected' variant degrades accuracy. For lean=neutral "
        "the injection is a politically-flavored but stance-neutral phrase "
        "(political_phrases) — Δ measures whether *political vocabulary alone* "
        "corrupts otherwise-neutral logical reasoning. lean=left|right have "
        "no injected variant under the asymmetric scheme (n/a in those columns); "
        "see custom_bench/docs/METHODOLOGY.md §2."
    )
    lines.append("")
    lines.append("| tag | acc_N_clean | acc_N_inj | Δ_N (pol-vocab) | acc_L_clean | acc_L_inj | Δ_L (noise) | acc_R_clean | acc_R_inj | Δ_R (noise) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for tag, m in summary["configs"].items():
        nr = m.get("noise_robustness") or {}
        n_row = nr.get("neutral") or {}
        l_row = nr.get("left") or {}
        r_row = nr.get("right") or {}
        lines.append(
            f"| {tag} | "
            f"{_fmt(n_row.get('acc_clean'))} | "
            f"{_fmt(n_row.get('acc_injected'))} | "
            f"{_fmt(n_row.get('acc_drop_from_injection'), signed=True)} | "
            f"{_fmt(l_row.get('acc_clean'))} | "
            f"{_fmt(l_row.get('acc_injected'))} | "
            f"{_fmt(l_row.get('acc_drop_from_injection'), signed=True)} | "
            f"{_fmt(r_row.get('acc_clean'))} | "
            f"{_fmt(r_row.get('acc_injected'))} | "
            f"{_fmt(r_row.get('acc_drop_from_injection'), signed=True)} |"
        )

    lines.append("")
    lines.append("## Per-topic partisan bias (bias_signed_FPFN, restricted to one topic at a time)")
    lines.append("")
    lines.append(
        "For each of the 8 topics, the column reports `bias_signed_FPFN` computed "
        "on the (left, right) sub-stimuli for that topic only. Lets you see *which "
        "topic* drives a model's aggregate partisan bias."
    )
    lines.append("")
    topic_keys = _collect_topic_keys(summary)
    if topic_keys:
        header = "| tag | " + " | ".join(topic_keys) + " |"
        sep = "|---|" + "---|" * len(topic_keys)
        lines.append(header)
        lines.append(sep)
        for tag, m in summary["configs"].items():
            bt = m.get("by_topic") or {}
            row = [f"| {tag} |"]
            for t in topic_keys:
                cell = bt.get(t) or {}
                row.append(f" {_fmt(cell.get('bias_signed_FPFN'), signed=True)} |")
            lines.append("".join(row))
    else:
        lines.append("(No topic data — likely no political-lean stimuli with topic dimension.)")

    return "\n".join(lines) + "\n"


def _collect_topic_keys(summary):
    """Return the deterministic sorted list of topic IDs that appear across configs."""
    seen = set()
    for m in summary["configs"].values():
        for t in (m.get("by_topic") or {}).keys():
            seen.add(t)
    # Deterministic order matching politicize/docs/README.md §4 table:
    canonical_order = [
        "military_intervention",
        "gun_rights",
        "federalism",
        "taxes",
        "climate_policy",
        "healthcare_policy",
        "immigration",
        "racial_justice",
    ]
    return [t for t in canonical_order if t in seen] + sorted(seen - set(canonical_order))


def main():
    summary = build_summary()
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    table = write_bias_table(summary)
    BIAS_TABLE_PATH.write_text(table)
    print(f"summary -> {SUMMARY_PATH}")
    print(f"table   -> {BIAS_TABLE_PATH}")
    print()
    print(table)


if __name__ == "__main__":
    main()
