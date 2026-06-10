"""Cross-cell audit aggregator.

Reads the classified judge cells for the active cohort and produces per-cell
distributions + a cross-cell comparison report (`AUDIT.md` + `summary.json`).

Paths default to the EXPERIMENT cohort run dir (alongside metrics/responses/
figures), so all Judge outputs live with everything else:
    input   1_benchmarking/runs/$EXPERIMENT/judges/*.jsonl   (override: AUTO_DIR=)
    output  1_benchmarking/runs/$EXPERIMENT/audit/           (override: OUT_DIR=)
EXPERIMENT defaults to `f5`.

Cells are grouped:
- pvsteer alpha sweep (mistral-pvsteer-{left,right}-a{3,5,7})
- native PoliTune (mistral/llama × left/right)
"""
from __future__ import annotations
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent          # Judge/
_RUN = _ROOT.parent / "1_benchmarking" / "runs" / os.environ.get("EXPERIMENT", "f5")
AUTO_DIR = Path(os.environ.get("AUTO_DIR", _RUN / "judges"))
OUT_DIR = Path(os.environ.get("OUT_DIR", _RUN / "audit"))

CATEGORIES = [
    "faithful_task_performance",
    "post_hoc_reasoning",
    "capability_error",
    "instruction_following_failure",
    "viewpoint_bias",
    "motivational_framing_bias",
    "generation_collapse",
]
SHORT_CAT = {
    "faithful_task_performance": "FTP",
    "post_hoc_reasoning": "PHR",
    "capability_error": "CE",
    "instruction_following_failure": "IFF",  # formerly QoS failure
    "viewpoint_bias": "VB",
    "motivational_framing_bias": "MFB",
    "generation_collapse": "GC",
}

GROUPS = {
    "base (no stance, no DPO)": [
        "llama-base",
        "mistral-base",
    ],
    "roleplay (system-prompt stance)": [
        "llama-roleplay-left",
        "llama-roleplay-right",
        "mistral-roleplay-left",
        "mistral-roleplay-right",
    ],
    "PoliTune-HF DPO LoRA": [
        "llama-politune-hf-left",
        "llama-politune-hf-right",
        "mistral-politune-hf-left",
        "mistral-politune-hf-right",
    ],
    "native PoliTune (mistral)": [
        "mistral-politune-native-left-60",
        "mistral-politune-native-right-60",
    ],
    "native PoliTune (llama)": [
        "llama-politune-native-left-lowlr-80",
        "llama-politune-native-right-60",
    ],
    "pvsteer-left (alpha sweep, a3+a5)": [
        "mistral-pvsteer-left-a3",
        "mistral-pvsteer-left-a5",
    ],
    "pvsteer-right (alpha sweep, a3+a5)": [
        "mistral-pvsteer-right-a3",
        "mistral-pvsteer-right-a5",
    ],
    "assessing_bias_formal — base (alpha=0)": [
        "llama-base__assessing_formal__alpha0.0",
        "mistral-base__assessing_formal__alpha0.0",
    ],
    "assessing_bias_formal — PoliTune-HF DPO at alpha=1": [
        "llama-politune-hf-left__assessing_formal__alpha1.0",
        "llama-politune-hf-right__assessing_formal__alpha1.0",
        "mistral-politune-hf-left__assessing_formal__alpha1.0",
        "mistral-politune-hf-right__assessing_formal__alpha1.0",
    ],
}


def _flags(r: dict) -> tuple[bool, bool]:
    """Return (contaminated, collapsed) handling both old (integrity-enum)
    and new (boolean-flag) schemas. New schema wins when both are present."""
    if "contaminated" in r or "collapsed" in r:
        return bool(r.get("contaminated", False)), bool(r.get("collapsed", False))
    integ = r.get("integrity")
    return integ == "contaminated", integ == "collapsed"


def _read_cell(name: str) -> list[dict] | None:
    p = AUTO_DIR / f"{name}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.open()]
    return [r for r in rows if "outcome" in r]


def _pct(num: int, den: int) -> str:
    if den == 0:
        return "—"
    return f"{100*num/den:.1f}%"


def _cell_summary(rows: list[dict]) -> dict:
    n = len(rows)
    flags = [_flags(r) for r in rows]
    contam_n = sum(1 for c, _ in flags if c)
    collapse_n = sum(1 for _, k in flags if k)
    both_n = sum(1 for c, k in flags if c and k)
    contam_only_n = contam_n - both_n
    collapse_only_n = collapse_n - both_n
    clean_n = n - contam_n - collapse_only_n  # = n - (contam_only + both) - collapse_only
    out = {
        "n": n,
        "outcome": Counter(r["outcome"] for r in rows),
        "rv": Counter(r["reasoning_validity"] for r in rows),
        "primary": Counter(r["primary_category"] for r in rows),
        "fallacy": Counter(r.get("fallacy_lens") or "none" for r in rows),
        # Contingency on the two independent flags
        "contam_n": contam_n,
        "collapse_n": collapse_n,
        "both_n": both_n,
        "contam_only_n": contam_only_n,
        "collapse_only_n": collapse_only_n,
        "clean_n": clean_n,
    }
    # Conditional rates for the hypothesis "contaminated → more prone to collapse"
    out["p_collapse_given_contam"] = collapse_n_given_contam = (
        both_n / contam_n if contam_n else None
    )
    not_contam_n = n - contam_n
    out["p_collapse_given_clean"] = (
        collapse_only_n / not_contam_n if not_contam_n else None
    )
    # Schema provenance — distinguishes freshly-classified rows (have the
    # `max_4gram_repeat` pre-pass signal) from rows migrated from the legacy
    # `integrity` enum (have the integrity field but no pre-pass signal).
    # Migrated cells' `both_n` is undercounted by exactly the (contam, collapse)
    # co-occurrences the old enum couldn't represent.
    out["freshly_classified_rows"] = sum(
        1 for r in rows if "max_4gram_repeat" in r
    )
    out["migrated_from_integrity_rows"] = sum(
        1 for r in rows if "integrity" in r and "max_4gram_repeat" not in r
    )
    # Confidence
    confs = [r.get("confidence", 0) for r in rows]
    out["mean_confidence"] = sum(confs) / max(n, 1)
    # Per-lean stratification
    by_lean: dict[str, Counter] = defaultdict(Counter)
    by_lean_n: Counter = Counter()
    for r in rows:
        by_lean[r["lean"]][r["primary_category"]] += 1
        by_lean_n[r["lean"]] += 1
    out["by_lean_primary"] = dict(by_lean)
    out["by_lean_n"] = by_lean_n
    # T7-only stratification (T7 is the value-loaded family)
    t7 = [r for r in rows if r["template_id"].startswith("T7")]
    t1_6 = [r for r in rows if not r["template_id"].startswith("T7")]
    out["t7_n"] = len(t7)
    out["t7_primary"] = Counter(r["primary_category"] for r in t7)
    out["t1_6_n"] = len(t1_6)
    out["t1_6_primary"] = Counter(r["primary_category"] for r in t1_6)
    # Contamination-among-correct (kept for backwards compatibility with prior headline)
    correct = [r for r in rows if r["outcome"] == "correct"]
    out["correct_n"] = len(correct)
    contaminated_among_correct = sum(
        1 for r in correct if _flags(r)[0]
    )
    out["contaminated_among_correct"] = contaminated_among_correct
    out["contaminated_among_correct_pct"] = _pct(contaminated_among_correct, len(correct))
    out["pct_contaminated"] = _pct(contam_n, n)
    out["pct_collapsed"] = _pct(collapse_n, n)
    out["pct_both"] = _pct(both_n, n)
    out["accuracy"] = _pct(out["outcome"]["correct"], n)
    return out


def _category_table(cells: list[tuple[str, dict]]) -> list[str]:
    """Side-by-side primary_category counts (and %) per cell."""
    lines = []
    header = "| primary_category | " + " | ".join(c for c, _ in cells) + " |"
    sep = "|" + "---|" * (len(cells) + 1)
    lines.append(header)
    lines.append(sep)
    for cat in CATEGORIES:
        row = [cat]
        for _, s in cells:
            n = s["primary"].get(cat, 0)
            row.append(f"{n} ({_pct(n, s['n'])})")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _axis_table(cells: list[tuple[str, dict]], axis: str, values: list[str]) -> list[str]:
    lines = []
    header = f"| {axis} | " + " | ".join(c for c, _ in cells) + " |"
    sep = "|" + "---|" * (len(cells) + 1)
    lines.append(header)
    lines.append(sep)
    for v in values:
        row = [v]
        for _, s in cells:
            n = s[axis].get(v, 0)
            row.append(f"{n} ({_pct(n, s['n'])})")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Audit — all steered + native politune cells\n"]
    lines.append("Auto-classifier: `gemini-3-flash-preview` via Vertex with v2 prompt "
                 f"(F1 + F2 + F3 fixes applied). Per-cell jsonls at `{AUTO_DIR}`.\n")

    # Total cost across all cells
    total_calls = 0
    total_in = 0
    total_out = 0
    for usage_path in AUTO_DIR.glob("*.usage.json"):
        u = json.loads(usage_path.read_text())["usage"]
        total_calls += u["calls"]
        total_in += u["input_tokens"]
        total_out += u["output_tokens"]
    if total_calls > 0:
        in_avg = total_in / total_calls
        out_avg = total_out / total_calls
        cost_25 = (0.075 * in_avg + 0.30 * out_avg) / 1e6 * total_calls
        cost_3 = (0.30 * in_avg + 2.50 * out_avg) / 1e6 * total_calls
        lines.append(f"## Aggregate cost across {total_calls} calls\n")
        lines.append(f"- {total_in:,} input tokens · {total_out:,} output tokens "
                     f"· avg {in_avg:.0f} in / {out_avg:.0f} out per call")
        lines.append(f"- Gemini 2.5 Flash list: **${cost_25:.2f}**")
        lines.append(f"- Gemini 3 Flash conservative: **${cost_3:.2f}**\n")

    # Each group
    all_summaries = {}
    for group_name, cell_names in GROUPS.items():
        lines.append(f"## {group_name}\n")
        cells = []
        for cn in cell_names:
            rows = _read_cell(cn)
            if rows is None:
                lines.append(f"- {cn}: **NOT YET CLASSIFIED**")
                continue
            s = _cell_summary(rows)
            all_summaries[cn] = s
            cells.append((cn, s))

        if not cells:
            lines.append("")
            continue

        # Headline accuracy + contamination row
        lines.append("### Headline metrics\n")
        lines.append("| cell | n | accuracy | contaminated | collapsed | both | C-among-correct | mean conf |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for cn, s in cells:
            lines.append(f"| {cn} | {s['n']} | {s['accuracy']} | {s['pct_contaminated']} | "
                         f"{s['pct_collapsed']} | {s['pct_both']} | "
                         f"{s['contaminated_among_correct_pct']} "
                         f"({s['contaminated_among_correct']}/{s['correct_n']}) | "
                         f"{s['mean_confidence']:.2f} |")
        lines.append("")

        # Contamination × Collapse 2x2 contingency (independent flags)
        lines.append("### Contamination × Collapse 2×2 (independent flags)\n")
        lines.append("Counts; columns are `collapsed`, rows are `contaminated`. "
                     "`P(collapse|contam)` and `P(collapse|clean)` enable the "
                     "later question \"is contamination a gateway to collapse?\"\n")
        lines.append("| cell | clean | contam-only | collapse-only | both | P(collapse\\|contam) | P(collapse\\|clean) | schema |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for cn, s in cells:
            pcc = "—" if s["p_collapse_given_contam"] is None else f"{s['p_collapse_given_contam']:.3f}"
            pcg = "—" if s["p_collapse_given_clean"] is None else f"{s['p_collapse_given_clean']:.3f}"
            if s["migrated_from_integrity_rows"] == s["n"]:
                schema = "migrated (both-count is lower bound)"
            elif s["freshly_classified_rows"] == s["n"]:
                schema = "fresh (bool flags + pre-pass)"
            else:
                schema = (f"mixed ({s['freshly_classified_rows']} fresh / "
                          f"{s['migrated_from_integrity_rows']} migrated)")
            lines.append(f"| {cn} | {s['clean_n']} | {s['contam_only_n']} | "
                         f"{s['collapse_only_n']} | {s['both_n']} | {pcc} | {pcg} | {schema} |")
        lines.append("")

        # primary_category table
        lines.append("### primary_category counts\n")
        lines.extend(_category_table(cells))
        lines.append("")

        # T7 vs T1-6 stratification (only for cells with T7 rows)
        if any(s["t7_n"] > 0 for _, s in cells):
            lines.append("### T7 (value-loaded) vs T1-T6 (strict-identity) — primary_category\n")
            for cn, s in cells:
                lines.append(f"\n**{cn}** (T7 n={s['t7_n']}, T1-T6 n={s['t1_6_n']}):\n")
                lines.append("| category | T7 | T1-T6 |")
                lines.append("| --- | ---: | ---: |")
                for cat in CATEGORIES:
                    t7n = s["t7_primary"].get(cat, 0)
                    t16n = s["t1_6_primary"].get(cat, 0)
                    if t7n + t16n == 0:
                        continue
                    lines.append(f"| {cat} | {t7n} ({_pct(t7n, s['t7_n'])}) | "
                                 f"{t16n} ({_pct(t16n, s['t1_6_n'])}) |")
            lines.append("")

        # Per-lean primary_category
        lines.append("### Per-lean primary_category (counts)\n")
        for cn, s in cells:
            lines.append(f"\n**{cn}**:\n")
            leans = sorted(s["by_lean_n"].keys())
            lines.append("| category | " + " | ".join(f"{l} (n={s['by_lean_n'][l]})" for l in leans) + " |")
            lines.append("|" + "---|" * (len(leans) + 1))
            for cat in CATEGORIES:
                row = [cat]
                any_nonzero = False
                for lean in leans:
                    n = s["by_lean_primary"].get(lean, Counter()).get(cat, 0)
                    if n > 0:
                        any_nonzero = True
                    row.append(f"{n} ({_pct(n, s['by_lean_n'][lean])})")
                if any_nonzero:
                    lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # Reasoning_validity + fallacy_lens distributions
        lines.append("### Other axes (compact)\n")
        lines.append("| cell | RV: valid / invalid / opaque / n/a | top fallacy lenses |")
        lines.append("| --- | --- | --- |")
        for cn, s in cells:
            rv_str = " / ".join(f"{s['rv'].get(k, 0)}" for k in ["valid", "invalid", "opaque", "n/a"])
            top_fall = ", ".join(f"{k}({v})" for k, v in s["fallacy"].most_common(3) if k != "none")
            lines.append(f"| {cn} | {rv_str} | {top_fall or '(none)'} |")
        lines.append("")

    # Write
    out_path = OUT_DIR / "AUDIT.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")

    # Also write a tiny summary.json for downstream
    summary = {cn: {
        "n": s["n"],
        "accuracy": s["accuracy"],
        "pct_contaminated": s["pct_contaminated"],
        "pct_collapsed": s["pct_collapsed"],
        "pct_both": s["pct_both"],
        "contam_n": s["contam_n"],
        "collapse_n": s["collapse_n"],
        "both_n": s["both_n"],
        "contam_only_n": s["contam_only_n"],
        "collapse_only_n": s["collapse_only_n"],
        "clean_n": s["clean_n"],
        "p_collapse_given_contam": s["p_collapse_given_contam"],
        "p_collapse_given_clean": s["p_collapse_given_clean"],
        "contaminated_among_correct_pct": s["contaminated_among_correct_pct"],
        "freshly_classified_rows": s["freshly_classified_rows"],
        "migrated_from_integrity_rows": s["migrated_from_integrity_rows"],
        "primary": dict(s["primary"]),
        "rv": dict(s["rv"]),
        "outcome": dict(s["outcome"]),
    } for cn, s in all_summaries.items()}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
