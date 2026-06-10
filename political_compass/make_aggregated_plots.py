"""Aggregate PoliLean political-compass results across:
  - base
  - roleplay (system-prompt-only)
  - politune-hf (old, Alpaca-template trained)         [politune_hf_train]
  - politune-hf-native-fixed (new, chat-template fix)  [politune_hf_train_native]
  - pvsteer (inference-time steering, mistral only)    [4_steering]

Renders one panel per family (llama, mistral). Each config is a single
mean (ec, soc) point with horizontal+vertical std error bars; individual
runs are faint dots so the spread is visible. Legend shows n-runs.

Sources merged:
  political_compass/PoliLean/results/summary.json
  politune_hf_train_native/results/polilean_summary.json
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CENTRAL_SUMMARY = HERE / "PoliLean" / "results" / "summary.json"
NATIVE_SUMMARY = REPO / "politune_hf_train_native" / "results" / "polilean_summary.json"
PLOTS_DIR = HERE / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# Per-tag display config: (color, marker, label_override)
# Left-coded recipes use blue/green family, right-coded use red/orange.
# Recipe families are distinguished by colour saturation + marker shape.
PER_TAG = {
    # gray base — circle
    "base":                       ("#444444", "o", None),

    # ----- roleplay (system prompt only) -----
    "roleplay-left":              ("#1f77b4", "s", None),
    "roleplay-right":             ("#d62728", "s", None),

    # ----- politune-hf (OLD Alpaca-template) — light shades, triangle-up -----
    "politune-hf-left":           ("#6baed6", "^", "politune-hf-left (Alpaca)"),
    "politune-hf-right":          ("#fb6a4a", "^", "politune-hf-right (Alpaca)"),

    # ----- politune-hf-native-fixed (NEW chat-template fix) — dark shades, triangle-down/diamond/plus -----
    "politune-hf-native-left":              ("#9e9ac8", "P", "politune-hf-native-left (early)"),
    "politune-hf-native-fixed-60-left":     ("#08519c", "v", "politune-hf-native-fixed-60-left (chat)"),
    "politune-hf-native-fixed-100-left":    ("#08306b", "D", "politune-hf-native-fixed-100-left (chat)"),
    "politune-hf-native-fixed-lowlr-80-left": ("#08519c", "v", "politune-hf-native-fixed-lowlr-80-left (chat)"),
    "politune-hf-native-fixed-60-right":     ("#a50f15", "v", "politune-hf-native-fixed-60-right (chat)"),

    # ----- pvsteer (inference-time, mistral only) — green/orange, star, size by coeff -----
    "pvsteer-left-a3":            ("#a1d99b", "*", None),
    "pvsteer-left-a5":            ("#005a32", "*", None),
    "pvsteer-right-a3":           ("#fdae6b", "*", None),
    "pvsteer-right-a5":           ("#7f2704", "*", None),
}


def variant(tag, family):
    """'llama-roleplay-left' -> 'roleplay-left' (strip leading family-)."""
    return tag[len(family) + 1:]


def load_merged():
    """Union of central + native summaries. Identical tags should have
    identical values (same eval pipeline); native adds the *-native-* tags."""
    merged = {}
    for path in (CENTRAL_SUMMARY, NATIVE_SUMMARY):
        with open(path) as f:
            merged.update(json.load(f))
    return merged


def draw_quadrants(ax):
    ax.axhline(0, color="black", linewidth=0.6, zorder=1)
    ax.axvline(0, color="black", linewidth=0.6, zorder=1)
    ax.axhspan(0, 10, xmin=0.5, xmax=1.0, color="#fde0dc", alpha=0.25, zorder=0)
    ax.axhspan(0, 10, xmin=0.0, xmax=0.5, color="#dde7f5", alpha=0.25, zorder=0)
    ax.axhspan(-10, 0, xmin=0.5, xmax=1.0, color="#fff4cc", alpha=0.25, zorder=0)
    ax.axhspan(-10, 0, xmin=0.0, xmax=0.5, color="#daf0d6", alpha=0.25, zorder=0)


def order_key(v):
    """Stable display order for legend."""
    order = [
        "base",
        "roleplay-left", "roleplay-right",
        "politune-hf-left", "politune-hf-right",
        "politune-hf-native-left",
        "politune-hf-native-fixed-60-left",
        "politune-hf-native-fixed-100-left",
        "politune-hf-native-fixed-lowlr-80-left",
        "politune-hf-native-fixed-60-right",
        "pvsteer-left-a3", "pvsteer-left-a5",
        "pvsteer-right-a3", "pvsteer-right-a5",
    ]
    return order.index(v) if v in order else 999


def plot_panel(summary, family, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    draw_quadrants(ax)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.set_xlabel("Economic   (Left  ↔  Right)", fontsize=12)
    ax.set_ylabel("Social   (Libertarian  ↔  Authoritarian)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.text(-9.5, 9.5, "Authoritarian Left", fontsize=9, alpha=0.6, va="top")
    ax.text(9.5, 9.5,  "Authoritarian Right", fontsize=9, alpha=0.6, ha="right", va="top")
    ax.text(-9.5, -9.5, "Libertarian Left", fontsize=9, alpha=0.6, va="bottom")
    ax.text(9.5, -9.5,  "Libertarian Right", fontsize=9, alpha=0.6, ha="right", va="bottom")
    ax.grid(True, alpha=0.25, zorder=0)

    # Filter + sort to per-tag config-known tags only
    rows = []
    for tag, rec in summary.items():
        if not tag.startswith(family + "-"):
            continue
        v = variant(tag, family)
        if v not in PER_TAG:
            continue
        rows.append((order_key(v), tag, v, rec))
    rows.sort()

    legend_handles = []
    for _, tag, v, rec in rows:
        color, marker, label_override = PER_TAG[v]
        ec_mean = rec["ec_mean"]
        soc_mean = rec["soc_mean"]
        ec_std = rec["ec_std"]
        soc_std = rec["soc_std"]
        n = rec["n"]

        # faint individual runs
        for ec, soc in zip(rec["ec_runs"], rec["soc_runs"]):
            ax.scatter(ec, soc, s=14, color=color, alpha=0.30,
                       edgecolor="none", zorder=3)

        # std error bars (cross of ±std on each axis)
        ax.errorbar(
            ec_mean, soc_mean,
            xerr=ec_std, yerr=soc_std,
            fmt="none", ecolor=color, elinewidth=1.4, capsize=4,
            alpha=0.8, zorder=4,
        )

        # mean marker on top
        msize = 220 if marker == "*" else 110
        ax.scatter(ec_mean, soc_mean, s=msize, color=color, marker=marker,
                   edgecolor="black", linewidth=0.8, zorder=5)

        label = label_override if label_override else v
        legend_handles.append(plt.Line2D(
            [], [], color=color, marker=marker, linestyle="none",
            markersize=11 if marker == "*" else 9,
            markeredgecolor="black", markeredgewidth=0.6,
            label=f"{label}  (n={n}, ec={ec_mean:+.2f}±{ec_std:.2f}, "
                  f"soc={soc_mean:+.2f}±{soc_std:.2f})",
        ))

    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=8.5, framealpha=0.95,
              borderaxespad=0, title=f"{family.capitalize()} configs",
              title_fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    summary = load_merged()

    n_llama = sum(1 for t in summary if t.startswith("llama-")
                  and variant(t, "llama") in PER_TAG)
    n_mistral = sum(1 for t in summary if t.startswith("mistral-")
                    and variant(t, "mistral") in PER_TAG)

    plot_panel(
        summary, "llama",
        f"Political Compass (PoliLean) — Llama, {n_llama} configs\n"
        f"base + roleplay + politune-hf (old Alpaca) + politune-hf-native-fixed (new chat)",
        PLOTS_DIR / "aggregated_polilean_llama.png",
    )
    plot_panel(
        summary, "mistral",
        f"Political Compass (PoliLean) — Mistral, {n_mistral} configs\n"
        f"base + roleplay + politune-hf (old Alpaca) + politune-hf-native-fixed (new chat) + pvsteer",
        PLOTS_DIR / "aggregated_polilean_mistral.png",
    )


if __name__ == "__main__":
    main()
