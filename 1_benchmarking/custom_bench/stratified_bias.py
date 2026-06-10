"""Stratified bias eval — split each cell's responses by template family
(T1-T6 strict-identity vs T7 value-loaded) and emit a side-by-side table.

Usage:
    EXPERIMENT=f4/political python -m custom_bench.stratified_bias
"""

import json
from pathlib import Path

from custom_bench.config import (
    BIAS_TABLE_PATH,
    RESPONSES_DIR,
    RUN_DIR,
    all_configs,
    responses_path,
)
from custom_bench.eval import evaluate

T1_T6_PREFIXES = tuple(f"T{i}" for i in range(1, 7))
T7_PREFIXES = ("T7",)


def _split(records):
    strict = [r for r in records if r["template_id"].startswith(T1_T6_PREFIXES)]
    value = [r for r in records if r["template_id"].startswith(T7_PREFIXES)]
    return strict, value


def _row(tag, label, m, n):
    bl = m.get("by_lean", {})
    return (
        f"| {tag} | {label} | {n} "
        f"| {_fmt(bl.get('neutral', {}).get('accuracy'))} "
        f"| {_fmt(bl.get('left', {}).get('accuracy'))} "
        f"| {_fmt(bl.get('right', {}).get('accuracy'))} "
        f"| {_fmt(m.get('bias_signed_FPFN'), signed=True)} "
        f"| {_fmt(m.get('bias_acc_simple'), signed=True)} "
        f"| {_fmt(m.get('bias_centered'), signed=True)} "
        f"| {_p(m)} |"
    )


def _fmt(x, prec=3, signed=False):
    if x is None:
        return "n/a"
    return f"{x:+.{prec}f}" if signed else f"{x:.{prec}f}"


def _p(m):
    cs = m.get("chi_square")
    if cs is None or "p_value" not in cs:
        return "n/a"
    return f"{cs['p_value']:.3g}"


def main():
    # Use all_configs for canonical ordering, then add any extra cells found
    # in responses dir (covers aggregating after inference on a different host
    # where adapters aren't locally present)
    tags = [tag for tag, *_ in all_configs()]
    extra = sorted({p.stem for p in RESPONSES_DIR.rglob("*.jsonl") if "__" not in p.stem and p.stem not in tags})
    tags = tags + extra
    out_lines = ["# Stratified bias — T1-T6 (strict) vs T7 (value-loaded)", ""]
    out_lines.append(
        "Headline metric `bias_signed_FPFN` (Gubelmann/Karray): "
        "`((R_FP − R_FN) − (L_FP − L_FN)) / N_engaged`. "
        "Positive = right-leaning, negative = left-leaning."
    )
    out_lines.append("")
    out_lines.append("| tag | family | n | acc_N | acc_L | acc_R | bias_FPFN | bias_acc | bias_centered | p |")
    out_lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for tag in tags:
        path = responses_path(tag)
        if not path.exists():
            continue
        records = [json.loads(l) for l in open(path)]
        strict, value = _split(records)
        if strict:
            m_strict = evaluate(strict)
            out_lines.append(_row(tag, "T1-T6", m_strict, len(strict)))
        if value:
            m_value = evaluate(value)
            out_lines.append(_row(tag, "T7", m_value, len(value)))
        if not (strict and value):
            continue
        # Δ row
        b_s = m_strict.get("bias_signed_FPFN")
        b_v = m_value.get("bias_signed_FPFN")
        if b_s is not None and b_v is not None:
            out_lines.append(
                f"| {tag} | **Δ(T7−T1-T6)** | — | — | — | — "
                f"| {_fmt(b_v - b_s, signed=True)} | — | — | — |"
            )

    out = "\n".join(out_lines) + "\n"
    target = RUN_DIR / "bias_table_stratified.md"
    target.write_text(out)
    print(f"wrote {target}")
    print()
    print(out)


if __name__ == "__main__":
    main()
