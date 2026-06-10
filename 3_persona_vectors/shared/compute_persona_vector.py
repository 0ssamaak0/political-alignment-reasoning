"""Stage C — compute persona vector from Stage B's pos/neg CSVs.

Wraps upstream `generate_vec.py::save_persona_vector`, which:
  1. Filters rows where pos trait >= threshold AND neg trait < (100-threshold)
     AND both coherences >= 50 (`get_persona_effective`, threshold=50 default).
  2. Loads the base model in HF, runs forward with `output_hidden_states=True`
     for each (prompt+response) pair.
  3. Computes mean activations over prompt and response token spans, per layer.
  4. Saves pos_mean - neg_mean for each layer to three `.pt` files:
        - {trait}_prompt_avg_diff.pt    [num_layers+1, hidden_dim]
        - {trait}_response_avg_diff.pt  [num_layers+1, hidden_dim]  ← paper-validated
        - {trait}_prompt_last_diff.pt   [num_layers+1, hidden_dim]

CLI:
    python3 compute_persona_vector.py --model llama   --leaning right
    python3 compute_persona_vector.py --model llama   --leaning left
    python3 compute_persona_vector.py --model mistral --leaning right
    python3 compute_persona_vector.py --model mistral --leaning left

Outputs:
    vectors/{model}/{leaning}_leaning_response_avg_diff.pt   (and 2 sidecars)
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "persona_vectors"
sys.path.insert(0, str(UPSTREAM))

from generate_vec import save_persona_vector  # noqa: E402

MODELS = {
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage C — compute persona vector.")
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--leaning", required=True, choices=["right", "left"])
    ap.add_argument("--threshold", type=int, default=50,
                    help="Min pos trait + max (100 - neg trait) to keep a row.")
    args = ap.parse_args()

    base_model = MODELS[args.model]
    trait_name = f"{args.leaning}_leaning"
    pos_path = HERE / "extract_runs" / args.model / f"{trait_name}_pos.csv"
    neg_path = HERE / "extract_runs" / args.model / f"{trait_name}_neg.csv"

    if not pos_path.exists() or not neg_path.exists():
        print(f"[stageC] Missing Stage B CSVs for {args.model} × {trait_name}:")
        print(f"  pos: {pos_path} ({'OK' if pos_path.exists() else 'MISSING'})")
        print(f"  neg: {neg_path} ({'OK' if neg_path.exists() else 'MISSING'})")
        return 1

    save_dir = HERE / "vectors" / args.model
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stageC] computing persona vector for {args.model} × {trait_name}")
    print(f"  base model: {base_model}")
    print(f"  pos: {pos_path}")
    print(f"  neg: {neg_path}")
    print(f"  save_dir: {save_dir}")
    print(f"  threshold: {args.threshold}")

    save_persona_vector(
        model_name=base_model,
        pos_path=str(pos_path),
        neg_path=str(neg_path),
        trait=trait_name,
        save_dir=str(save_dir),
        threshold=args.threshold,
    )

    expected = save_dir / f"{trait_name}_response_avg_diff.pt"
    if expected.exists():
        import torch
        v = torch.load(expected, weights_only=False)
        print(f"[stageC] OK — vector shape: {tuple(v.shape)} dtype: {v.dtype}")
    else:
        print(f"[stageC] WARN — expected {expected} not found")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
