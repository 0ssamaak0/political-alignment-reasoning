"""Aggregate the RQ2 G&K judge jsonl -> audit/AUDIT.md + audit/summary.json.

The shared ``Judge.src.audit_all_cells`` hardcodes a GROUPS dict whose cell
names (mistral-politune-native-*, mistral-pvsteer-*) do NOT match RQ2's naming
(mistral-DPO-*, mistral-steering-*, llama-DPO-right-2nd). So this aggregator
reuses that module's per-cell scoring helpers (``_cell_summary``, ``_pct``,
``CATEGORIES``) but iterates RQ2's own 15-cell order/grouping.

Run from repo root:
    python -m RQ2.G_K_assessing_bias.aggregate_judge
"""
from __future__ import annotations
import json
from pathlib import Path

from Judge.src import audit_all_cells as aac
from Judge.src.audit_all_cells import _cell_summary, _pct, CATEGORIES

HERE = Path(__file__).resolve().parent
JUDGES_DIR = HERE / "judges"
OUT_DIR = HERE / "audit"

# Repoint the shared module's _read_cell at the RQ2 judges dir.
aac.AUTO_DIR = JUDGES_DIR

GROUPS = {
    "base": ["mistral-base", "llama-base"],
    "roleplay (system-prompt stance)": [
        "mistral-roleplay-left", "mistral-roleplay-right",
        "llama-roleplay-left", "llama-roleplay-right",
    ],
    "steering (persona-vector activation)": [
        "mistral-steering-left", "mistral-steering-right",
        "llama-steering-left", "llama-steering-right",
    ],
    "DPO (PoliTune LoRA)": [
        "mistral-DPO-left", "mistral-DPO-right",
        "llama-DPO-left", "llama-DPO-right",
    ],  # llama-DPO-right-2nd excluded from final RQ2 analysis (raw judge jsonl kept)
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_cells = sum(len(v) for v in GROUPS.values())
    lines = [f"# RQ2 G&K — qualitative judge audit ({n_cells} cells)\n"]
    lines.append("Auto-classifier: `gemini-3-flash-preview` via Vertex, explicit "
                 "context cache (system+rubric) shared across all calls. Per-cell "
                 f"judge jsonl at `{JUDGES_DIR.relative_to(HERE.parents[1])}`.\n")
    lines.append("Same multi-axis rubric as the f5 cohort (outcome × contaminated "
                 "× collapsed × reasoning_validity → 7-way `primary_category`). The "
                 "G&K instrument is binary deductive-validity over partisan content, "
                 "so `contaminated` / `viewpoint_bias` / `motivational_framing_bias` "
                 "are the bias-relevant axes; `generation_collapse` / "
                 "`instruction_following_failure` flag the high-`unmappable` cells.\n")

    # Cost across all usage sidecars
    total_calls = total_in = total_out = total_cached = 0
    cell_set = {c for v in GROUPS.values() for c in v}
    for up in JUDGES_DIR.glob("*.usage.json"):
        if up.name[:-len(".usage.json")] not in cell_set:
            continue  # skip excluded cells (e.g. llama-DPO-right-2nd) in the cost total
        u = json.loads(up.read_text())["usage"]
        total_calls += u.get("calls", 0)
        total_in += u.get("input_tokens", 0)
        total_out += u.get("output_tokens", 0)
        total_cached += u.get("cached_tokens", 0)
    if total_calls:
        cached_pct = 100 * total_cached / total_in if total_in else 0
        lines.append("## Cost\n")
        lines.append(f"- {total_calls} calls · {total_in:,} input ({total_cached:,} "
                     f"cached = {cached_pct:.0f}%) · {total_out:,} output tokens\n")

    all_summaries = {}
    for group_name, cell_names in GROUPS.items():
        cells = []
        for cn in cell_names:
            rows = aac._read_cell(cn)
            if rows is None:
                continue
            s = _cell_summary(rows)
            all_summaries[cn] = s
            cells.append((cn, s))
        if not cells:
            continue

        lines.append(f"## {group_name}\n")
        # Headline
        lines.append("| cell | n | accuracy | contaminated | collapsed | "
                     "IFF (no-answer) | mean conf |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for cn, s in cells:
            iff = s["primary"].get("instruction_following_failure", 0)
            lines.append(f"| {cn} | {s['n']} | {s['accuracy']} | {s['pct_contaminated']} "
                         f"| {s['pct_collapsed']} | {iff} ({_pct(iff, s['n'])}) "
                         f"| {s['mean_confidence']:.2f} |")
        lines.append("")

        # primary_category counts
        lines.append("### primary_category\n")
        header = "| primary_category | " + " | ".join(c for c, _ in cells) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(cells) + 1))
        for cat in CATEGORIES:
            row = [cat]
            for _, s in cells:
                n = s["primary"].get(cat, 0)
                row.append(f"{n} ({_pct(n, s['n'])})")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # Per-lean bias-category split (the partisan-skew signal)
        lines.append("### bias-axis primary_category by lean "
                     "(viewpoint_bias / motivational_framing_bias)\n")
        lines.append("| cell | lean | n | viewpoint_bias | motiv_framing |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for cn, s in cells:
            for lean in sorted(s["by_lean_n"].keys()):
                ln = s["by_lean_n"][lean]
                vb = s["by_lean_primary"].get(lean, {}).get("viewpoint_bias", 0)
                mfb = s["by_lean_primary"].get(lean, {}).get("motivational_framing_bias", 0)
                lines.append(f"| {cn} | {lean} | {ln} | {vb} ({_pct(vb, ln)}) "
                             f"| {mfb} ({_pct(mfb, ln)}) |")
        lines.append("")

    (OUT_DIR / "AUDIT.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT_DIR / 'AUDIT.md'}")

    summary = {cn: {
        "n": s["n"],
        "accuracy": s["accuracy"],
        "pct_contaminated": s["pct_contaminated"],
        "pct_collapsed": s["pct_collapsed"],
        "pct_both": s["pct_both"],
        "contam_n": s["contam_n"],
        "collapse_n": s["collapse_n"],
        "both_n": s["both_n"],
        "mean_confidence": s["mean_confidence"],
        "primary": dict(s["primary"]),
        "rv": dict(s["rv"]),
        "outcome": dict(s["outcome"]),
        "by_lean_primary": {k: dict(v) for k, v in s["by_lean_primary"].items()},
        "by_lean_n": dict(s["by_lean_n"]),
    } for cn, s in all_summaries.items()}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
