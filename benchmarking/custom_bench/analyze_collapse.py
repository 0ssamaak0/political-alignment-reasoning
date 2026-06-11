"""Scan benchmark response files for collapse.

Collapse definitions used here:
  1. Verdict collapse: one outcome dominates the run. Outcomes are
     `valid`, `invalid`, and `none` (unparsed / refusal). Default threshold
     is 95%.
  2. CoT-style diagnostics: repeated exact traces and repeated opening
     scaffolds. These are reported as diagnostics and as conservative
     candidates, since "similar CoTs" is inherently fuzzier than verdict skew.

Writes:
  - collapse_report.json
  - collapse_report.md
under the selected run root (default: 1_benchmarking/runs/f4).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def normalize_trace(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def alpha_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", (text or "").lower())


def prefix_signature(text: str, n: int) -> str:
    toks = alpha_tokens(text)
    return " ".join(toks[:n])


def pct(numer: int, denom: int) -> float:
    return numer / denom if denom else 0.0


def analyze_file(path: Path, verdict_threshold: float, cot_candidate_threshold: float) -> dict:
    items = [json.loads(line) for line in path.open() if line.strip()]
    n = len(items)

    verdicts = Counter((item.get("verdict") or "none") for item in items)
    top_verdict, top_verdict_count = verdicts.most_common(1)[0]
    top_verdict_frac = pct(top_verdict_count, n)

    traces = [
        normalize_trace(item.get("cot_trace") or item.get("raw_response") or "")
        for item in items
    ]
    exact = Counter(traces)
    top_exact_trace, top_exact_count = exact.most_common(1)[0]
    top_exact_frac = pct(top_exact_count, n)

    prefix6 = Counter(prefix_signature(trace, 6) for trace in traces if trace)
    prefix12 = Counter(prefix_signature(trace, 12) for trace in traces if trace)
    top_prefix6, top_prefix6_count = prefix6.most_common(1)[0] if prefix6 else ("", 0)
    top_prefix12, top_prefix12_count = prefix12.most_common(1)[0] if prefix12 else ("", 0)
    top_prefix6_frac = pct(top_prefix6_count, n)
    top_prefix12_frac = pct(top_prefix12_count, n)

    by_lean = {}
    for lean in sorted({item.get("lean", "unknown") for item in items}):
        lean_items = [item for item in items if item.get("lean", "unknown") == lean]
        counts = Counter((item.get("verdict") or "none") for item in lean_items)
        lean_top, lean_top_count = counts.most_common(1)[0]
        by_lean[lean] = {
            "n": len(lean_items),
            "counts": dict(counts),
            "top_verdict": lean_top,
            "top_verdict_count": lean_top_count,
            "top_verdict_frac": pct(lean_top_count, len(lean_items)),
        }

    verdict_collapse = top_verdict_frac >= verdict_threshold
    cot_style_candidate = top_prefix6_frac >= cot_candidate_threshold

    return {
        "path": str(path),
        "run_dir": str(path.parent.parent),
        "run_name": path.parent.parent.name,
        "file_name": path.name,
        "n": n,
        "verdict_counts": dict(verdicts),
        "top_verdict": top_verdict,
        "top_verdict_count": top_verdict_count,
        "top_verdict_frac": top_verdict_frac,
        "verdict_collapse": verdict_collapse,
        "by_lean": by_lean,
        "unique_exact_traces": len(exact),
        "top_exact_trace_count": top_exact_count,
        "top_exact_trace_frac": top_exact_frac,
        "top_exact_trace_preview": top_exact_trace[:180],
        "unique_prefix6": len(prefix6),
        "top_prefix6": top_prefix6,
        "top_prefix6_count": top_prefix6_count,
        "top_prefix6_frac": top_prefix6_frac,
        "unique_prefix12": len(prefix12),
        "top_prefix12": top_prefix12,
        "top_prefix12_count": top_prefix12_count,
        "top_prefix12_frac": top_prefix12_frac,
        "cot_style_candidate": cot_style_candidate,
    }


def render_markdown(results: list[dict], verdict_threshold: float, cot_candidate_threshold: float) -> str:
    verdict_collapses = [r for r in results if r["verdict_collapse"]]
    cot_candidates = [r for r in results if r["cot_style_candidate"]]

    lines = [
        "# Collapse Analysis",
        "",
        f"- Verdict collapse threshold: `{verdict_threshold:.0%}`",
        f"- CoT-style candidate threshold (top 6-token prefix share): `{cot_candidate_threshold:.0%}`",
        f"- Response files scanned: `{len(results)}`",
        "",
        "## Verdict Collapses",
        "",
    ]

    if verdict_collapses:
        lines.append("| file | n | dominant verdict | share | counts |")
        lines.append("| --- | ---: | --- | ---: | --- |")
        for r in sorted(verdict_collapses, key=lambda x: (-x["top_verdict_frac"], x["path"])):
            counts = ", ".join(f"{k}={v}" for k, v in sorted(r["verdict_counts"].items()))
            lines.append(
                f"| `{r['file_name']}` | {r['n']} | `{r['top_verdict']}` | "
                f"{r['top_verdict_frac']:.1%} | {counts} |"
            )
    else:
        lines.append("No response file crossed the verdict-collapse threshold.")

    lines.extend(["", "## CoT-Style Candidates", ""])
    if cot_candidates:
        lines.append("| file | n | top 6-token prefix share | top prefix | dominant verdict |")
        lines.append("| --- | ---: | ---: | --- | --- |")
        for r in sorted(cot_candidates, key=lambda x: (-x["top_prefix6_frac"], x["path"])):
            lines.append(
                f"| `{r['file_name']}` | {r['n']} | {r['top_prefix6_frac']:.1%} | "
                f"`{r['top_prefix6']}` | `{r['top_verdict']}` ({r['top_verdict_frac']:.1%}) |"
            )
    else:
        lines.append("No response file crossed the CoT-style candidate threshold.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("1_benchmarking/runs/f4"),
        help="Root directory containing run subdirectories with responses/*.jsonl",
    )
    parser.add_argument(
        "--verdict-threshold",
        type=float,
        default=0.95,
        help="Collapse threshold for dominant verdict share.",
    )
    parser.add_argument(
        "--cot-candidate-threshold",
        type=float,
        default=0.60,
        help="Candidate threshold for dominant 6-token CoT prefix share.",
    )
    args = parser.parse_args()

    results = [
        analyze_file(path, args.verdict_threshold, args.cot_candidate_threshold)
        for path in sorted(args.root.glob("*/responses/**/*.jsonl"))
    ]

    out_json = args.root / "collapse_report.json"
    out_md = args.root / "collapse_report.md"
    out_json.write_text(json.dumps({
        "root": str(args.root),
        "verdict_threshold": args.verdict_threshold,
        "cot_candidate_threshold": args.cot_candidate_threshold,
        "results": results,
    }, indent=2))
    out_md.write_text(render_markdown(results, args.verdict_threshold, args.cot_candidate_threshold))

    verdict_collapses = sum(1 for r in results if r["verdict_collapse"])
    cot_candidates = sum(1 for r in results if r["cot_style_candidate"])
    print(f"scanned {len(results)} response files")
    print(f"verdict collapses: {verdict_collapses}")
    print(f"cot-style candidates: {cot_candidates}")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
