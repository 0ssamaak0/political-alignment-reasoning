"""Aggregate judge output into the RQ2 mechanism-level contrast, per cohort.
Key questions the scalar metrics can't answer:
  - contamination: is partisan bleed POLITICAL-specific (political >> neutral)?
  - reasoning_validity: how often is a 'correct' verdict actually post-hoc?
  - motivated reasoning: does fallacy_lens (motivated_reasoning / premise_truth_
    conflation) concentrate on flipped (counter-attitudinal) political items?
Usage: python judge_summary.py <judges_dir> <cohort_label>"""
import json, os, sys, collections
JUDG = sys.argv[1]; LABEL = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(JUDG)
# judge output dropped arm/variant — rejoin from stimuli by row_idx (same order)
STIM = {i: (r["arm"], r["variant"]) for i, r in enumerate(
    json.loads(l) for l in open("/Users/0ssamaak0/Documents/polireason/1_benchmarking/RQ2_bench/stimuli.jsonl"))}
def enrich(R):
    for r in R:
        a, v = STIM.get(r.get("row_idx"), (None, None))
        r.setdefault("arm", a); r.setdefault("variant", v)
    return R
CELLS = [
    ("llama-base","Base"),("llama-roleplay-left","RP-L"),("llama-roleplay-right","RP-R"),
    ("llama-pvsteer-ml-left-a2_5","Steer-L"),("llama-pvsteer-ml-right-a3","Steer-R"),
    ("llama-politune-hf-left","DPO-L"),("llama-politune-hf-right","DPO-R"),
    ("mistral-base","Base"),("mistral-roleplay-left","RP-L"),("mistral-roleplay-right","RP-R"),
    ("mistral-pvsteer-ml-left-a2_5","Steer-L"),("mistral-pvsteer-ml-right-a3","Steer-R"),
    ("mistral-politune-hf-left","DPO-L"),("mistral-politune-hf-right","DPO-R"),
]
def pct(n, d): return f"{100.0*n/d:4.0f}" if d else "  . "
MR = {"motivational_reasoning", "premise_truth_conflation"}
print(f"\n###### {LABEL} ######")
print(f"{'cell':16}| {'n':>4} {'collapse%':>9} | contam% pol/neu | {'rsn_invalid%':>12} | {'posthoc%':>8} | {'MRfallacy% clean/flip':>22}")
print("-"*104)
fam=None
for tag, short in CELLS:
    p=f"{JUDG}/{tag}.jsonl"
    f0=tag.split("-")[0]
    if f0!=fam: fam=f0; print(f"== {fam} ==")
    if not os.path.exists(p): continue
    R=enrich([json.loads(l) for l in open(p)])
    pol=[r for r in R if r.get("arm")=="political"]; neu=[r for r in R if r.get("arm")=="neutral"]
    coll=sum(bool(r.get("collapsed")) for r in R)
    cpol=sum(bool(r.get("contaminated")) for r in pol); cneu=sum(bool(r.get("contaminated")) for r in neu)
    # reasoning invalid among rows where validity was assessed (exclude n/a)
    assessed=[r for r in R if r.get("reasoning_validity") in ("valid","invalid")]
    inval=sum(r["reasoning_validity"]=="invalid" for r in assessed)
    # post-hoc: outcome correct but reasoning invalid
    corr=[r for r in R if r.get("outcome")=="correct"]
    posthoc=sum(r.get("reasoning_validity")=="invalid" for r in corr)
    # motivated-reasoning fallacy on political, by variant
    polc=[r for r in pol if r.get("variant")=="clean"]; polf=[r for r in pol if r.get("variant")=="flipped"]
    mrc=sum(r.get("fallacy_lens") in MR for r in polc); mrf=sum(r.get("fallacy_lens") in MR for r in polf)
    print(f"{short:16}| {len(R):>4} {pct(coll,len(R)):>9} |  {pct(cpol,len(pol))}/{pct(cneu,len(neu))}  | "
          f"{pct(inval,len(assessed)):>12} | {pct(posthoc,len(corr)):>8} |  {pct(mrc,len(polc))}/{pct(mrf,len(polf))}")
print("\ncontam pol/neu = % contaminated, political vs neutral (gap>0 = political-specific bleed)")
print("rsn_invalid% = % of assessed CoTs judged logically invalid; posthoc% = correct-verdict rows with invalid reasoning")
print("MRfallacy clean/flip = % political rows w/ motivated_reasoning|premise_truth_conflation fallacy, clean vs flipped")
