"""Aggregate the RQ2 G&K per-cell result CSVs → results.md + results.json.

The Gubelmann & Karray partisan-inference probe (192-format) run through the
15-cell zoo, same models as the other RQ2 sub-runs. This is an external
*bias* instrument (not an N-vs-P reasoning-gap eval like 0_shot / the 3-shot
run): each prompt asks for a deductive-validity verdict over partisan content,
and the signal is whether valid/invalid judgements skew with left- vs
right-leaning content.

Metrics are computed by the instrument's OWN `compute_bias.compute_stats` and
`gk_extract.label_from_raw` (imported, not reimplemented) — i.e. literally the
accuracy + bias defined by G&K:

  accuracy           fraction of *mapped* items judged correctly (UNMAPPABLE excluded)
  bias_score_engaged ((R_FP-R_FN)-(L_FP-L_FN)) / mapped   (preferred; + right, - left)
  bias_score_N       same numerator / 192                  (upstream G&K convention)
  unmappable_rate    no-verdict rate (high ⇒ collapsed cell, noisy bias)

    python RQ2/G_K_assessing_bias/aggregate.py [--dir pol_out]

Runner + Vertex launch tooling live in `1_benchmarking/G_K_assessing_bias/`
(`run_gk_cells.py`, `vertex/`); the shared instrument (prompts_192.csv,
gk_extract, compute_bias) lives there too.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]               # .../polireason
sys.path.insert(0, str(REPO / "1_benchmarking"))
from G_K_assessing_bias.compute_bias import compute_stats, SCOPES  # noqa: E402
from G_K_assessing_bias.gk_extract import label_from_raw           # noqa: E402

ORDER = ["mistral-base", "mistral-roleplay-left", "mistral-roleplay-right",
         "mistral-steering-left", "mistral-steering-right", "mistral-DPO-left",
         "mistral-DPO-right", "llama-base", "llama-roleplay-left", "llama-roleplay-right",
         "llama-steering-left", "llama-steering-right", "llama-DPO-left", "llama-DPO-right"]
# llama-DPO-right-2nd (runner-up adapter) is excluded from the final RQ2 analysis (raw CSV kept).
EXCLUDE = {"llama-DPO-right-2nd"}
MARKER = "<!-- AUTO-GENERATED ABOVE; interpretation below is hand-written -->"


def order_key(name):
    return ORDER.index(name) if name in ORDER else len(ORDER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "pol_out"))
    args = ap.parse_args()
    d = Path(args.dir)

    csvs = sorted(d.glob("*.csv"), key=lambda p: order_key(p.stem))
    csvs = [c for c in csvs if c.stem not in EXCLUDE]
    if not csvs:
        raise SystemExit(f"no per-cell CSVs in {d}")

    out = {"cells": {}}
    rows_md = []
    var_md = []
    for csv in csvs:
        cell = csv.stem
        df = pd.read_csv(csv)
        # Re-derive labels from raw_output with the canonical G&K cascade — faithful to
        # compute_bias.py (independent of whatever extractor wrote the CSV).
        df["predicted_label"] = df["raw_output"].apply(label_from_raw)
        st_all = compute_stats(df, "all")
        per_var = {}
        for v in SCOPES:
            sub = df[df["variation"] == v]
            if len(sub):
                per_var[v] = compute_stats(sub, v)
        out["cells"][cell] = {"all": st_all, "by_variation": per_var}
        rows_md.append(
            f"| {cell} | {st_all['accuracy']:.3f} | "
            f"{st_all['unmappable_rate']:.3f} ({st_all['unmappable']}) | "
            f"{st_all['bias_score_engaged']:+.4f} | {st_all['bias_score_N']:+.4f} | "
            f"{st_all['R_FP']} | {st_all['R_FN']} | {st_all['L_FP']} | {st_all['L_FN']} |")
        cells = [f"{per_var[v]['bias_score_engaged']:+.3f}" if v in per_var else "—"
                 for v in SCOPES]
        var_md.append(f"| {cell} | " + " | ".join(cells) + " |")

    n = len(csvs)
    L = [f"# RQ2 — G&K partisan-inference probe (192-format, {n} cells)", "",
         "The Gubelmann & Karray deductive-validity probe (192 prompts = 48 arguments × 4 "
         "surface variations; balanced 96 left / 96 right, 96 valid / 96 invalid) run "
         "through the RQ2 cell zoo (the runner-up adapter `llama-DPO-right-2nd` is excluded "
         "from this analysis; the deployed `llama-DPO-right` is kept). Greedy, 256 new tokens, "
         "the prompt's own validity instruction verbatim, verdicts via the canonical G&K "
         "cascade (`gk_extract.label_from_raw`). This is an **external bias instrument**, "
         "not an N-vs-P reasoning-gap eval — so it is **not comparable** to `0_shot` / the "
         "3-shot run; it measures partisan *skew* in validity judgements.", "",
         "Metrics are exactly the instrument's own (`compute_bias.compute_stats`):", "",
         "- **accuracy** — fraction of *mapped* items judged correctly (UNMAPPABLE excluded).",
         "- **bias** = `((R_FP − R_FN) − (L_FP − L_FN)) / DENOM` over mapped items. "
         "**+ = right-leaning skew, − = left-leaning skew.** `bias_engaged` uses "
         "DENOM = mapped (preferred); `bias_N` uses DENOM = 192 (upstream G&K). They "
         "diverge only when `unmappable` > 0.",
         "- **unmappable** — items with no extractable verdict; a high rate flags a "
         "mode-collapsed cell whose bias number is structurally noisy.", "",
         "## Headline (scope = all 192)", "",
         "| cell | accuracy | unmappable | bias_engaged | bias_N | R_FP | R_FN | L_FP | L_FN |",
         "|---|---|---|---|---|---|---|---|---|", *rows_md, "",
         "Sign: **+ right-leaning skew, − left-leaning skew**. At n=192 with the "
         "denominator split across leanings, one flipped judgement moves `bias_engaged` "
         "by ~0.005; treat |bias| below ~0.04 (≈8 net items) as noise.", "",
         "## Bias by surface variation (`bias_score_engaged`)", "",
         "| cell | default | perm | rand | conlast |", "|---|---|---|---|---|", *var_md, "",
         MARKER, ""]

    # preserve hand-written prose below the marker across re-runs
    md_path = HERE / "results.md"
    tail = ""
    if md_path.exists() and MARKER in md_path.read_text():
        tail = md_path.read_text().split(MARKER, 1)[1].lstrip("\n")
    md_path.write_text("\n".join(L) + ("\n" + tail if tail else ""))
    (HERE / "results.json").write_text(json.dumps(out, indent=2))
    print("\n".join(L))
    print(f"\nwrote {md_path} + results.json ({n} cells)")


if __name__ == "__main__":
    main()
