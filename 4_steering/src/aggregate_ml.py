"""Stage F (ml) — aggregate multi-layer pvsteer cells into:
  - 4_steering/results/summary_ml.json
  - 4_steering/results/figures/pareto_ml.png  (contamination vs coherence,
    ml cells vs single-layer pvsteer cells, color-coded by direction).

Sources:
  - political_compass/PoliLean/results/summary.json  (PCT ec/soc per tag)
  - 4_steering/results/coherence/pvsteer_ml_coherence.csv
  - Judge/raw/auto/mistral-pvsteer-{ml-,}<lean>-a{N}.jsonl  (contamination)
  - 1_benchmarking/runs/f4/political/summary.json  (if present, optional)
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent

ML_CELLS = ["mistral-pvsteer-ml-left-a3", "mistral-pvsteer-ml-right-a3"]
SL_REF_CELLS = [
    "mistral-pvsteer-left-a3",  "mistral-pvsteer-left-a5",  "mistral-pvsteer-left-a7",
    "mistral-pvsteer-right-a3", "mistral-pvsteer-right-a5", "mistral-pvsteer-right-a7",
]
BASELINE_CELLS = ["mistral-base", "mistral-politune-hf-left", "mistral-politune-hf-right"]


def _safe_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _pct_summary() -> dict[str, dict[str, float]]:
    pl = _safe_json(_ROOT / "political_compass" / "PoliLean" / "results" / "summary.json")
    out = {}
    for tag, v in pl.items():
        out[tag] = {
            "ec_mean": v.get("ec_mean"),
            "ec_std": v.get("ec_std"),
            "soc_mean": v.get("soc_mean"),
            "soc_std": v.get("soc_std"),
        }
    return out


def _coherence_summary() -> dict[str, dict[str, float]]:
    out = {}
    for csv_path in [
        _ROOT / "4_steering" / "results" / "coherence" / "pvsteer_ml_coherence.csv",
        _ROOT / "4_steering" / "results" / "coherence" / "pvsteer_ml_a2_coherence.csv",
        _ROOT / "4_steering" / "results" / "coherence" / "pvsteer_ml_a2_5_coherence.csv",
        _ROOT / "4_steering" / "results" / "coherence" / "pvsteer_coherence.csv",
        _ROOT / "3_persona_vectors" / "v3_multilayer" / "results" / "v3a_coherence.csv",
    ]:
        if not csv_path.exists():
            continue
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                tag = row.get("tag") or row.get("config")
                if not tag:
                    continue
                try:
                    out[tag] = {
                        "mean": float(row.get("mean") or row.get("coherence_mean")),
                        "std": float(row.get("std", "nan") or "nan"),
                    }
                except (ValueError, TypeError):
                    pass
    return out


def _judge_rates(tag: str) -> dict[str, float] | None:
    p = _ROOT / "Judge" / "raw" / "auto" / f"{tag}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    n = len(rows)
    if n == 0:
        return None
    contam = sum(1 for r in rows if r.get("contaminated") is True)
    collapsed = sum(1 for r in rows if r.get("collapsed") is True)
    both = sum(1 for r in rows if r.get("contaminated") is True and r.get("collapsed") is True)
    vp = sum(1 for r in rows if r.get("primary_category") == "viewpoint_bias")
    mfb = sum(1 for r in rows if r.get("primary_category") == "motivational_framing_bias")
    return {
        "n": n,
        "contaminated_rate": contam / n,
        "collapsed_rate": collapsed / n,
        "both_rate": both / n,
        "viewpoint_bias_rate": vp / n,
        "motiv_framing_rate": mfb / n,
        "partisan_total_rate": (vp + mfb) / n,
    }


def _f4_bias(tag: str) -> float | None:
    p = _ROOT / "1_benchmarking" / "runs" / "f4" / "political" / "summary.json"
    if not p.exists():
        return None
    s = _safe_json(p)
    return s.get(tag, {}).get("bias_signed_FPFN")


def build_cell_summary(tag: str, pct: dict, coh: dict) -> dict:
    cell = {"tag": tag}
    if tag in pct:
        cell.update(pct[tag])
    if tag in coh:
        cell["coherence_mean"] = coh[tag]["mean"]
    j = _judge_rates(tag)
    if j is not None:
        cell.update(j)
    b = _f4_bias(tag)
    if b is not None:
        cell["bias_signed_FPFN"] = b
    return cell


def pareto_plot(summary: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    for tag, row in summary["cells"].items():
        if row.get("contaminated_rate") is None or row.get("coherence_mean") is None:
            continue
        is_ml = "pvsteer-ml" in tag
        is_left = "left" in tag
        color = "#2266ff" if is_left else "#cc2222"
        marker = "*" if is_ml else "o"
        size = 280 if is_ml else 100
        edge = "k" if is_ml else "none"
        ax.scatter(
            row["contaminated_rate"], row["coherence_mean"],
            c=color, marker=marker, s=size, edgecolors=edge, linewidths=1.2,
            alpha=0.85, label=tag,
        )
        ax.annotate(
            tag.replace("mistral-pvsteer-", ""),
            (row["contaminated_rate"], row["coherence_mean"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(60, color="gray", linestyle="--", alpha=0.5,
               label="coherence ≥ 60 gate")
    ax.set_xlabel("Judge `contaminated` rate (fraction of f4 rows)")
    ax.set_ylabel("Coherence (Gemini mean, 0..100)")
    ax.set_title("Pareto: contamination × coherence — ml ★ vs single-layer ○")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    pct = _pct_summary()
    coh = _coherence_summary()

    all_tags = ML_CELLS + SL_REF_CELLS + BASELINE_CELLS
    cells = {}
    for tag in all_tags:
        cells[tag] = build_cell_summary(tag, pct, coh)

    summary = {"ml_cells": ML_CELLS, "cells": cells}
    out_json = _ROOT / "4_steering" / "results" / "summary_ml.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}")

    fig_dir = _ROOT / "4_steering" / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pareto_plot(summary, fig_dir / "pareto_ml.png")
    print(f"wrote {fig_dir / 'pareto_ml.png'}")

    # Print summary table
    print("\n=== headline ===")
    fmt = "{:38s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}"
    print(fmt.format("tag", "ec", "coh", "contam%", "partisan%"))
    print("-" * 80)
    for tag in ML_CELLS + SL_REF_CELLS + BASELINE_CELLS:
        c = cells.get(tag, {})
        ec = c.get("ec_mean")
        coh_m = c.get("coherence_mean")
        ct = c.get("contaminated_rate")
        pt = c.get("partisan_total_rate")
        print(fmt.format(
            tag,
            f"{ec:+.2f}" if ec is not None else "–",
            f"{coh_m:.1f}" if coh_m is not None else "–",
            f"{100*ct:.1f}" if ct is not None else "–",
            f"{100*pt:.1f}" if pt is not None else "–",
        ))


if __name__ == "__main__":
    main()
