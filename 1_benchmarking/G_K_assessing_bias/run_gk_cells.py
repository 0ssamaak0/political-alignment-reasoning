"""Run the 15-cell zoo through the G&K 192-format partisan-inference probe.

Same cell mechanics as the RQ2 0-shot runner (base / roleplay-system /
activation-steering / DPO-LoRA, identical steering tags + adapter subdirs), but
the *task* is the Gubelmann & Karray deductive-validity probe, not BBH:

  * data        = G_K_assessing_bias/data/prompts_192.csv (192 prompts)
  * user turn   = the prompt's ``Prompt`` column VERBATIM (it already ends with
                  "...ending your answer with **valid** or **invalid**"); NO
                  extra solve-instruction wrapper.
  * decoding    = greedy, max_new_tokens=256 (G&K canonical, short verdicts)
  * label       = gk_extract.label_from_raw  (VALID / INVALID / UNMAPPABLE)
  * output      = one CSV per cell in the schema compute_bias.py expects, routed
                  on disk by gk_paths.result_csv (steering -> <fam>-steering/,
                  roleplay -> roleplay/, else top level). compute_bias.py then
                  emits accuracy + bias_score_N + bias_score_engaged unchanged.

Runs the cells named by --cells (default all 15), one at a time with clean GPU
teardown — built to run ONE cell per Vertex job (--cells <cell>) or several on
one VM. Per-cell CSV with resume (skip if it already exists).

    python run_gk_cells.py --cells mistral-base,mistral-steering-left
Env: REPO (~/polireason), ADAPTERS (dir of DPO adapter subdirs), HF_TOKEN (gated llama).
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(os.environ.get("REPO", str(Path.home() / "polireason"))).resolve()
ADAPTERS = Path(os.environ.get(
    "ADAPTERS", str(REPO_ROOT / "politune_hf_train_native" / "adapters_train")))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "1_benchmarking"))
from custom_bench.adapters import steering_context  # noqa: E402
from G_K_assessing_bias.gk_extract import label_from_raw  # noqa: E402
from G_K_assessing_bias.gk_paths import result_csv  # noqa: E402

HERE = Path(__file__).resolve().parent
PROMPTS_CSV = HERE / "data" / "prompts_192.csv"

BASE_REPO = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
}
ALPHA_SUFFIX = {"left": "a2_5", "right": "a3"}
SYSTEM = {
    "left": "You will act as a politically left-leaning person",
    "right": "You will act as a politically right-leaning person",
}
MAX_NEW_TOKENS = 256  # G&K canonical (run_eval.py); verdicts are short


def registry():
    """The 15 RQ2 cells, identical tags/adapters/systems."""
    c = {}
    for fam in ("mistral", "llama"):
        c[f"{fam}-base"] = {"family": fam, "kind": "base"}
        for lean in ("left", "right"):
            c[f"{fam}-roleplay-{lean}"] = {"family": fam, "kind": "roleplay", "lean": lean}
            c[f"{fam}-steering-{lean}"] = {
                "family": fam, "kind": "steering",
                "tag": f"{fam}-pvsteer-ml-{lean}-{ALPHA_SUFFIX[lean]}"}
            c[f"{fam}-DPO-{lean}"] = {"family": fam, "kind": "dpo",
                                     "adapter": f"{fam}_{lean}_fixed"}
    c["llama-DPO-right-2nd"] = {"family": "llama", "kind": "dpo",
                                "adapter": "llama_right_2nd_fixed"}
    return c


def build_prompt_str(tok, user, system_prompt):
    """Chat-template the user turn; fold `system` into it for templates without
    a system role (e.g. Mistral-Instruct). `user` is the G&K Prompt verbatim."""
    if system_prompt:
        try:
            return tok.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            user = system_prompt + "\n\n" + user
    return tok.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def judge(model, tok, prompt_text, system_prompt=None):
    prompt = build_prompt_str(tok, prompt_text, system_prompt)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    n = enc["input_ids"].shape[1]
    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][n:], skip_special_tokens=True).strip()


def load_base(family):
    base = BASE_REPO[family]
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    return tok, model


def gpu_clear():
    gc.collect()
    torch.cuda.empty_cache()


def run_cell(cell, spec, prompts, out_csv):
    tok, model = load_base(spec["family"])
    steer_tag = None
    sys_prompt = None
    if spec["kind"] == "roleplay":
        sys_prompt = SYSTEM[spec["lean"]]
    elif spec["kind"] == "steering":
        steer_tag = spec["tag"]
    elif spec["kind"] == "dpo":
        from peft import PeftModel
        adir = ADAPTERS / spec["adapter"]
        if not adir.exists():
            raise FileNotFoundError(f"DPO adapter missing: {adir}")
        model = PeftModel.from_pretrained(model, str(adir))
        model.eval()

    ctx = steering_context(model, steer_tag) if steer_tag else nullcontext()
    rows = []
    t0 = time.time()
    with ctx:
        for idx, row in prompts.iterrows():
            raw = judge(model, tok, row["Prompt"], sys_prompt)
            rows.append({
                "model": cell,
                "item_id": int(idx),
                "pattern_id": row["Pattern-ID"],
                "variation": row["Variation-ID"],
                "leaning": row["Political-Leaning"],
                "inference_valid_gt": int(row["Is-Valid"]),
                "predicted_label": label_from_raw(raw),
                "raw_output": str(raw).replace("\r", " ").replace("\n", " ").strip()[:2000],
            })
            if (idx + 1) % 48 == 0:
                print(f"  [{cell}] {idx + 1}/{len(prompts)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    n = len(df)
    unmap = int((df["predicted_label"] == "UNMAPPABLE").sum())
    print(f"[{cell}] wrote {out_csv.name} n={n} unmappable={unmap} "
          f"({round(time.time() - t0, 1)}s)", flush=True)
    del model
    gpu_clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="ALL", help="comma-separated cell names, or ALL")
    ap.add_argument("--limit", type=int, default=None, help="first-N prompts (debug)")
    args = ap.parse_args()

    reg = registry()
    cells = list(reg) if args.cells == "ALL" else args.cells.split(",")
    bad = [c for c in cells if c not in reg]
    if bad:
        sys.exit(f"unknown cells: {bad}\nvalid: {list(reg)}")

    if not PROMPTS_CSV.exists():
        sys.exit(f"missing {PROMPTS_CSV} (build with build_prompts.py)")
    prompts = pd.read_csv(PROMPTS_CSV)
    if args.limit:
        prompts = prompts.head(args.limit).reset_index(drop=True)
    print(f"[init] cells={cells} n_prompts={len(prompts)} adapters={ADAPTERS}", flush=True)

    for cell in cells:
        out_csv = result_csv(cell, mkdir=True)
        if out_csv.exists():
            print(f"[{cell}] cached {out_csv.name} — skip", flush=True)
            continue
        run_cell(cell, reg[cell], prompts, out_csv)
    print("[done] " + ",".join(cells), flush=True)


if __name__ == "__main__":
    main()
