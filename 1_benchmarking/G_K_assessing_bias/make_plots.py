"""Plot the 192-format pvsteer-ml dose-response sweep.

Reads results/summary_long.csv (scope=="all" rows), parses each cell name
into (lean, alpha), and draws a 3-panel figure vs alpha:

  1. bias_score_engaged   — left & right curves; base drawn as the alpha=0
                            reference (point + dashed horizontal line).
  2. unmappable_rate      — coherence guard; cells above ~0.3 are collapsing.
  3. accuracy             — validity accuracy on mapped items; base ref line.

Positive bias = right-leaning skew, negative = left-leaning skew.

Usage (after compute_bias.py has written summary_long.csv):
    conda run -n main python -m G_K_assessing_bias.make_plots
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
SUMMARY = RESULTS_DIR / "summary_long.csv"
OUT_PNG = RESULTS_DIR / "sweep_plot.png"

ALPHA_RE = re.compile(r"-(left|right)-a([0-9_]+)$")


def parse_alpha(token: str) -> float:
    """'2_5' -> 2.5, '0_5' -> 0.5, '3' -> 3.0"""
    return float(token.replace("_", "."))


def classify(name: str):
    """-> (lean, alpha) or ('base', 0.0) or (None, None)."""
    if "base" in name:
        return "base", 0.0
    m = ALPHA_RE.search(name)
    if m:
        return m.group(1), parse_alpha(m.group(2))
    return None, None


def main() -> None:
    if not SUMMARY.exists():
        raise SystemExit(f"Missing {SUMMARY}; run compute_bias.py first")
    df = pd.read_csv(SUMMARY)
    df = df[df["scope"] == "all"].copy()
    df[["lean", "alpha"]] = df["model"].apply(
        lambda n: pd.Series(classify(n)))
    df = df.dropna(subset=["lean"])

    base = df[df["lean"] == "base"]
    left = df[df["lean"] == "left"].sort_values("alpha")
    right = df[df["lean"] == "right"].sort_values("alpha")

    panels = [
        ("bias_score_engaged", "G&K bias (engaged)\n← left   right →", True),
        ("unmappable_rate", "unmappable rate\n(coherence guard)", False),
        ("accuracy", "validity accuracy\n(mapped items)", True),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    for ax, (col, ylab, show_base) in zip(axes, panels):
        ax.plot(left["alpha"], left[col], "o-", color="tab:blue", label="left steer")
        ax.plot(right["alpha"], right[col], "s-", color="tab:red", label="right steer")
        if show_base and len(base):
            b = float(base[col].iloc[0])
            ax.axhline(b, ls="--", color="gray", lw=1)
            ax.plot(0, b, "D", color="black", ms=7, label="base (α=0)")
        if col == "bias_score_engaged":
            ax.axhline(0, color="black", lw=0.6)
        if col == "unmappable_rate":
            ax.axhspan(0.3, 1.0, color="red", alpha=0.06)
            ax.set_ylim(-0.02, max(0.35, df[col].max() * 1.1 + 0.02))
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    axes[-1].set_xlabel("steering coefficient α (pvsteer-ml, layers 1–31, incremental)")
    fig.suptitle("Mistral-7B pvsteer-ml on the G&K 192-format validity probe",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT_PNG, dpi=140)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
