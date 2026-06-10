"""Overlay PoliLean × PoliEval α-sweep curves for the 14 Mistral
pvsteer-ml cells. Produces a 2×2 grid:

  (0,0) Left direction: EC vs α
  (0,1) Left direction: SOC vs α
  (1,0) Right direction: EC vs α
  (1,1) Right direction: SOC vs α

Each panel overlays the two methods. Error bars are ec_std / soc_std
(seeded-run std across N_RUNS=5).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

POLILEAN_SUMMARY = HERE / "PoliLean" / "results" / "summary.json"
POLIEVAL_SUMMARY = HERE / "PoliLean" / "polieval_data" / "results" / "summary.json"
OUT_DIR = REPO / "4_steering" / "runs" / "polieval" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [2.0, 2.2, 2.4, 2.5, 2.6, 2.8, 3.0]
ALPHA_SUFFIX = {2.0: "a2", 2.2: "a2_2", 2.4: "a2_4", 2.5: "a2_5",
                2.6: "a2_6", 2.8: "a2_8", 3.0: "a3"}


def collect(summary, direction, metric):
    means, stds = [], []
    for a in ALPHAS:
        tag = f"mistral-pvsteer-ml-{direction}-{ALPHA_SUFFIX[a]}"
        row = summary.get(tag)
        if row is None:
            means.append(None)
            stds.append(None)
            continue
        means.append(row[f"{metric}_mean"])
        stds.append(row[f"{metric}_std"])
    return means, stds


def main():
    with open(POLILEAN_SUMMARY) as f:
        polilean = json.load(f)
    with open(POLIEVAL_SUMMARY) as f:
        polieval = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for row_idx, direction in enumerate(("left", "right")):
        for col_idx, metric in enumerate(("ec", "soc")):
            ax = axes[row_idx][col_idx]
            for name, summary, marker in (
                ("PoliLean (bart-mnli)", polilean, "o"),
                ("PoliEval (integer 0-3)", polieval, "s"),
            ):
                means, stds = collect(summary, direction, metric)
                xs = [a for a, m in zip(ALPHAS, means) if m is not None]
                ys = [m for m in means if m is not None]
                es = [s for m, s in zip(means, stds) if m is not None]
                ax.errorbar(xs, ys, yerr=es, marker=marker, label=name,
                            capsize=3, linewidth=1.5)
            ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
            metric_long = "Economic" if metric == "ec" else "Social"
            ax.set_title(f"{direction.upper()} — {metric_long}")
            ax.set_xlabel("α (steering coefficient)")
            ax.set_ylabel(f"{metric.upper()} (PCT)")
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "PoliLean vs PoliEval α-sweep — Mistral pvsteer-ml", fontsize=13
    )
    fig.tight_layout()
    out = OUT_DIR / "alpha_sweep_overlay.png"
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
