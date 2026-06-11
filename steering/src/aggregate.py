"""Stage E — aggregate per-tag results into 4_steering/results/summary.json.

Sources:
  - political_compass/PoliLean/results/summary.json  (PCT shifts)
  - 1_benchmarking/runs/f4/political/summary.json    (bias_signed_FPFN, etc.)
  - 4_steering/results/coherence/pvsteer_coherence.csv
  - 4_steering/results/layer_sweep/sweep_{left,right}.json
  - 4_steering/configs/steering.yaml                  (layer_idx, coeff per tag)

Writes:
  - 4_steering/results/summary.json
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent.parent


def _safe_load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    cfg_path = _ROOT / "4_steering" / "configs" / "steering.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    pl = _safe_load_json(
        _ROOT / "political_compass" / "PoliLean" / "results" / "summary.json"
    )
    f4 = _safe_load_json(
        _ROOT / "1_benchmarking" / "runs" / "f4" / "political" / "summary.json"
    )

    coh = {}
    coh_path = _ROOT / "4_steering" / "results" / "coherence" / "pvsteer_coherence.csv"
    if coh_path.exists():
        with coh_path.open() as f:
            for row in csv.DictReader(f):
                tag = row.get("tag") or row.get("config")
                mean_key = "coherence_mean" if "coherence_mean" in row else "mean"
                try:
                    coh[tag] = float(row[mean_key])
                except (KeyError, ValueError, TypeError):
                    pass

    sweep_left = _safe_load_json(
        _ROOT / "4_steering" / "results" / "layer_sweep" / "sweep_left.json"
    )
    sweep_right = _safe_load_json(
        _ROOT / "4_steering" / "results" / "layer_sweep" / "sweep_right.json"
    )

    summary = {
        "layer_sweep": {
            "left": {
                "best_layer": sweep_left.get("best_layer"),
                "best_score": (
                    sweep_left.get("per_layer", {})
                    .get(str(sweep_left.get("best_layer")), {})
                    .get("mean")
                ),
            },
            "right": {
                "best_layer": sweep_right.get("best_layer"),
                "best_score": (
                    sweep_right.get("per_layer", {})
                    .get(str(sweep_right.get("best_layer")), {})
                    .get("mean")
                ),
            },
        },
        "cells": {},
    }

    for tag, cfg_row in cfg.items():
        plr = pl.get(tag, {}) if isinstance(pl, dict) else {}
        f4r = f4.get("configs", {}).get(tag, {}) if isinstance(f4, dict) else {}
        summary["cells"][tag] = {
            "layer_idx": cfg_row.get("layer_idx"),
            "coeff": cfg_row.get("coeff"),
            "pct_ec_mean": plr.get("ec_mean"),
            "pct_ec_std": plr.get("ec_std"),
            "pct_soc_mean": plr.get("soc_mean"),
            "pct_soc_std": plr.get("soc_std"),
            "bias_signed_FPFN": f4r.get("bias_signed_FPFN"),
            "acc_neutral": f4r.get("by_lean", {}).get("neutral", {}).get("accuracy"),
            "acc_left": f4r.get("by_lean", {}).get("left", {}).get("accuracy"),
            "acc_right": f4r.get("by_lean", {}).get("right", {}).get("accuracy"),
            "coherence_mean": coh.get(tag),
        }

    out = _ROOT / "4_steering" / "results" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")

    print()
    print(
        f"{'tag':32s}  {'L':>3s} {'α':>3s}  {'ec±σ':>14s}  {'soc±σ':>14s}  "
        f"{'bias':>7s}  {'coh':>5s}  acc"
    )
    for tag, row in summary["cells"].items():
        ec = (
            f"{row['pct_ec_mean']:+.2f}±{row['pct_ec_std']:.2f}"
            if row["pct_ec_mean"] is not None
            else "n/a"
        )
        soc = (
            f"{row['pct_soc_mean']:+.2f}±{row['pct_soc_std']:.2f}"
            if row["pct_soc_mean"] is not None
            else "n/a"
        )
        bias = (
            f"{row['bias_signed_FPFN']:+.3f}"
            if row["bias_signed_FPFN"] is not None
            else "n/a"
        )
        coh_v = f"{row['coherence_mean']:5.1f}" if row["coherence_mean"] is not None else "n/a"
        acc_avg = [v for v in (row["acc_neutral"], row["acc_left"], row["acc_right"]) if v is not None]
        acc_s = f"{sum(acc_avg)/len(acc_avg):.2f}" if acc_avg else "n/a"
        print(
            f"{tag:32s}  {row['layer_idx']!s:>3s} {row['coeff']!s:>3s}  "
            f"{ec:>14s}  {soc:>14s}  {bias:>7s}  {coh_v:>5s}  {acc_s}"
        )


if __name__ == "__main__":
    main()
