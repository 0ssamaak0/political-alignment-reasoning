"""Readout-level decomposition of the MMLU formal_logic accuracy drop:
base vs behaviorally-matched steering (pvsteer-ml a2) vs DPO LoRA (scale 2.0).

MMLU multiple_choice is scored by per-choice loglikelihood. lm-eval saved
`samples` with each item's 4 choice LLs + the correct target index. From those
we ask WHERE the accuracy went:

  * discrimination  = mean (maxLL - minLL) across the 4 choices.
      small  -> model can't tell choices apart (FLATTENED -> chance)
      normal -> model is discriminating, just pointing wrong (REDIRECTED)
  * margin          = mean (LL[correct] - max LL[distractors]); >0 == right
  * LL scale        = mean LL of the chosen answer (calibration / norm shift)
  * base-rank agree = of items BASE gets right, fraction the variant keeps
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
TASK = "mmlu_formal_logic"

VARIANTS = {
    "base":        RES / "mistral_sweep/mistral-base/results.json",
    "steer-L-a2":  RES / "mistral_sweep/mistral-pvsteer-ml-left-a2/results.json",
    "steer-R-a2":  RES / "mistral_sweep/mistral-pvsteer-ml-right-a2/results.json",
    "dpo-L-2.0":   RES / "lora_sweep/mistral-politune-hf-left-lora2_0/results.json",
    "dpo-R-2.0":   RES / "lora_sweep/mistral-politune-hf-right-lora2_0/results.json",
}


def load_items(path: Path):
    """doc_id -> (np.array of 4 choice LLs, target_idx), or None if no samples."""
    d = json.loads(path.read_text())
    if "samples" not in d or TASK not in d.get("samples", {}):
        return None
    out = {}
    for s in d["samples"][TASK]:
        lls = np.array([r[0] for r in s["filtered_resps"]], dtype=float)
        out[s["doc_id"]] = (lls, int(s["target"]))
    return out


def main() -> None:
    data = {}
    for name, p in VARIANTS.items():
        if not p.exists():
            continue
        items = load_items(p)
        if items is None:
            print(f"# skip {name}: results.json has no saved samples")
            continue
        data[name] = items

    base = data.get("base")
    if base is not None:
        ids = sorted(base.keys())
        base_correct = {i for i in ids if int(np.argmax(base[i][0])) == base[i][1]}
    else:
        # no base samples; use the union of available items, no base-agreement col
        ids = sorted({i for items in data.values() for i in items})
        base_correct = None

    hdr = f"{'variant':12s} {'acc':>5s} {'discrim':>8s} {'margin':>7s} {'LL_chosen':>9s} {'keep|base✓':>10s}"
    print(hdr); print("-" * len(hdr))
    for name, items in data.items():
        accs, disc, marg, llc, keep = [], [], [], [], []
        for i in ids:
            if i not in items:
                continue
            lls, tgt = items[i]
            pred = int(np.argmax(lls))
            accs.append(pred == tgt)
            disc.append(lls.max() - lls.min())
            others = np.delete(lls, tgt)
            marg.append(lls[tgt] - others.max())
            llc.append(lls.max())
            if base_correct is not None and i in base_correct:
                keep.append(pred == tgt)
        keepval = (np.mean(keep) if keep else float('nan'))
        print(f"{name:12s} {np.mean(accs):5.3f} {np.mean(disc):8.2f} "
              f"{np.mean(marg):7.2f} {np.mean(llc):9.2f} "
              f"{keepval:10.3f}")


if __name__ == "__main__":
    main()
