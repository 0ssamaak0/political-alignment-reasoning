"""Per-(family, direction) markdown audit of the layer sweep responses.

Reads `results/layer_sweep/{family}/sweep_{direction}.json` and emits a
companion markdown that samples 3 representative responses per layer
(low / median / high score) so a reader can skim L32 vs the peak layer
side-by-side without grepping the JSON.

The point of the audit: the JSON has 32 layers × 20 questions = 640 entries.
The interesting question is "what does the model actually say when L32
scores 100, vs when L17 scores 89?" — this script surfaces a 3-row sample
per layer that lets you answer it visually.

Usage:
  python -m src.build_layer_sweep_audit --family mistral --direction left
  python -m src.build_layer_sweep_audit --family mistral --direction right
  python -m src.build_layer_sweep_audit --family llama   --direction left
  python -m src.build_layer_sweep_audit --family llama   --direction right
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MAX_CHARS_PER_RESPONSE = 600


def _summarise(text: str) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) > MAX_CHARS_PER_RESPONSE:
        text = text[:MAX_CHARS_PER_RESPONSE].rstrip() + " …[truncated]"
    return text


def _pick_three(items: list[dict]) -> list[tuple[str, dict]]:
    """Return [(label, item), ...] for low / median / high score."""
    scored = [it for it in items if it.get("score") is not None]
    if not scored:
        return [("no-score", items[0])] if items else []
    scored_sorted = sorted(scored, key=lambda it: it["score"])
    return [
        ("low",    scored_sorted[0]),
        ("median", scored_sorted[len(scored_sorted) // 2]),
        ("high",   scored_sorted[-1]),
    ]


def build(family: str, direction: str) -> Path:
    sweep_path = _REPO_ROOT / "4_steering" / "results" / "layer_sweep" / family / f"sweep_{direction}.json"
    if not sweep_path.exists():
        raise FileNotFoundError(sweep_path)
    data = json.loads(sweep_path.read_text())

    lines = []
    title_family = "Mistral-7B-Instruct-v0.2" if family == "mistral" else "Llama-3-8B-Instruct"
    lines.append(f"# Layer Sweep Audit — {title_family}, {direction}-leaning (α={data['alpha']})")
    lines.append("")
    lines.append(f"- Judge: `{data['judge_model']}` (eval_type=0_100 — \"how strongly does this express the {direction} trait, 0-100\")")
    lines.append(f"- {data['n_questions']} latter-half trait questions; 1 response per (layer, question); sampled at T=0.7 top_p=0.9.")
    lines.append(f"- **best_layer (raw):** L{data['best_layer']}")
    lines.append(f"- **best_layer (excluding last 2):** L{data['best_layer_excl_last_2']} — last-row of v_full carries extraction artifact.")
    lines.append("")
    lines.append("Per-layer mean trait-expression score, plus 3 representative responses (lowest / median / highest score):")
    lines.append("")

    Ls = sorted(int(L) for L in data["per_layer"].keys())
    for L in Ls:
        cell = data["per_layer"][str(L)] if str(L) in data["per_layer"] else data["per_layer"][L]
        mean = cell["mean"]
        n_parsed = cell.get("n_parsed", "?")
        items = cell.get("items", [])
        marker = ""
        if L == data["best_layer"]:
            marker += "  ← **best layer (raw)**"
        if L == data["best_layer_excl_last_2"]:
            marker += "  ← **best layer (excl last 2)**"
        lines.append(f"## L{L:02d}  mean={mean:.1f}  n_parsed={n_parsed}{marker}")
        for label, it in _pick_three(items):
            score = it.get("score")
            score_s = f"{score:.0f}" if score is not None else "n/a"
            lines.append(f"- **{label}** (score={score_s})")
            lines.append(f"  - Q: {_summarise(it['question'])}")
            lines.append(f"  - A: {_summarise(it['response'])}")
        lines.append("")

    out_path = _REPO_ROOT / "4_steering" / "results" / "layer_sweep" / family / f"AUDIT_{direction}.md"
    out_path.write_text("\n".join(lines))
    print(f"[audit] wrote {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=("mistral", "llama"), required=True)
    p.add_argument("--direction", choices=("left", "right"), required=True)
    args = p.parse_args()
    build(args.family, args.direction)


if __name__ == "__main__":
    main()
