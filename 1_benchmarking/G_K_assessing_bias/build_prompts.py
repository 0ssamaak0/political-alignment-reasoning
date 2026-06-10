"""Rebuild the 192-format G&K prompt subset from the upstream prompts.csv.

The committed `data/prompts_192.csv` is produced entirely by this script.
Re-run it only if the upstream file changes; the output is deterministic.

192-format filter (Gubelmann & Karray partisan-inference dataset):
    Pattern-Language     == "en"     (English syllogism patterns)
    Instruction-Language == "en"     (English task instruction)
    Is-Formal            == 1        (formal deductive-validity inferences)
    Is-Few-Shots         == 0        (zero-shot)
    Instruction-ID       == "1"      (the paper's simplest validity instruction)
    Variation-ID         == ANY      (all 4: default, perm, rand, conlast)

The upstream `run_eval.py` additionally pinned `Variation-ID == "default"`,
yielding 48 prompts. Dropping that pin keeps all 4 surface variations of the
same 48 arguments -> 48 x 4 = 192. This is the "192 formats" set.

Source: ../../knowledge/assessing_bias/llms_partisan_inference/data/prompts.csv
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = (REPO_ROOT / "knowledge" / "assessing_bias"
       / "llms_partisan_inference" / "data" / "prompts.csv")
DST = HERE / "data" / "prompts_192.csv"

KEEP_COLS = ["Pattern-ID", "Variation-ID", "Political-Leaning",
             "Gender", "Is-Valid", "Prompt"]


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing upstream prompts file: {SRC}")

    head = SRC.read_bytes()[:200]
    if head.startswith(b"version https://git-lfs"):
        raise SystemExit(
            f"{SRC} is still a Git-LFS pointer. Run:\n"
            f"    cd {SRC.parent.parent} && git lfs install && git lfs pull")

    df = pd.read_csv(SRC)
    filt = df[
        (df["Pattern-Language"] == "en")
        & (df["Instruction-Language"] == "en")
        & (df["Is-Formal"] == 1)
        & (df["Is-Few-Shots"] == 0)
        & (df["Instruction-ID"].astype(str) == "1")
    ].copy()

    if len(filt) != 192:
        raise SystemExit(
            f"Expected 192 prompts, got {len(filt)}.\n"
            f"Variation breakdown: {dict(filt['Variation-ID'].value_counts())}")

    out = filt[KEEP_COLS].reset_index(drop=True)
    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST, index=False)
    print(f"wrote {DST}  ({len(out)} rows, {DST.stat().st_size / 1024:.1f} KB)")
    print(f"  variation: {dict(out['Variation-ID'].value_counts())}")
    print(f"  leaning:   {dict(out['Political-Leaning'].value_counts())}")
    print(f"  is_valid:  {dict(out['Is-Valid'].value_counts())}")


if __name__ == "__main__":
    main()
