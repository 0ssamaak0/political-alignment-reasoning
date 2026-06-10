"""Per-cell LEFT <-> RIGHT analysis on the political arm. For each cell, split
political items by stimulus lean and report accuracy, judge contamination% and
motivated-reasoning-fallacy%, plus the signed partisan bias
  bias = ((R_FP - R_FN) - (L_FP - L_FN)) / N_engaged   (+ = right-leaning, - = left).
Read against each cell's OWN induced lean to see congruence asymmetry."""
import json, os, sys
sys.path.insert(0, "/Users/0ssamaak0/Documents/polireason/1_benchmarking/RQ2_bench")
from extract import extract_verdict
ROOT = "/Users/0ssamaak0/Documents/polireason/RQ2/flipped"
STIM = [json.loads(l) for l in open("/Users/0ssamaak0/Documents/polireason/1_benchmarking/RQ2_bench/stimuli.jsonl")]
AV = {i: (r["arm"], r["variant"], r.get("lean")) for i, r in enumerate(STIM)}
MR = {"motivational_reasoning", "premise_truth_conflation"}
CELLS = [("llama-base","Base","-"),("llama-roleplay-left","RP-L","L"),("llama-roleplay-right","RP-R","R"),
("llama-pvsteer-ml-left-a2_5","Steer-L","L"),("llama-pvsteer-ml-right-a3","Steer-R","R"),
("llama-politune-hf-left","DPO-L","L"),("llama-politune-hf-right","DPO-R","R"),
("mistral-base","Base","-"),("mistral-roleplay-left","RP-L","L"),("mistral-roleplay-right","RP-R","R"),
("mistral-pvsteer-ml-left-a2_5","Steer-L","L"),("mistral-pvsteer-ml-right-a3","Steer-R","R"),
("mistral-politune-hf-left","DPO-L","L"),("mistral-politune-hf-right","DPO-R","R")]
def pc(n,d): return f"{100*n/d:3.0f}" if d else " . "
def signed_bias(rows):
    eng=[r for r in rows if r["_v"] in ("valid","invalid")]
    if not eng: return None
    def term(ln):
        g=[r for r in eng if r["lean"]==ln]
        fp=sum((not r["valid"]) and r["_v"]=="valid" for r in g); fn=sum(r["valid"] and r["_v"]=="invalid" for r in g)
        return fp-fn
    return round((term("right")-term("left"))/len(eng),3)
print(f"{'cell':14}(lean)| acc L/R | contam% L/R | MRfal% L/R | signed_bias  (+R/-L)")
print("-"*78)
fam=None
for tag,short,ind in CELLS:
    rp=f"{ROOT}/responses/{tag}.jsonl"; jp=f"{ROOT}/judges/{tag}.jsonl"
    f0=tag.split('-')[0]
    if f0!=fam: fam=f0; print(f"== {fam} ==")
    if not os.path.exists(rp): print(f"{short:14}({ind})  | (pending)"); continue
    R=[json.loads(l) for l in open(rp)]
    for r in R: r["arm"],r["variant"],r["lean"]=AV.get(r.get("row_idx"),(None,None,None)); r["_v"]=extract_verdict(r["raw_response"])
    J={}
    if os.path.exists(jp):
        for l in open(jp):
            jr=json.loads(l); _,_,jr["lean"]=AV.get(jr.get("row_idx"),(None,None,None)); jr["arm"]=AV.get(jr.get("row_idx"),(None,))[0]; J[jr.get("row_idx")]=jr
    pol=[r for r in R if r["arm"]=="political"]
    L=[r for r in pol if r["lean"]=="left"]; Rt=[r for r in pol if r["lean"]=="right"]
    accL=sum(r["_v"]==("valid" if r["valid"] else "invalid") for r in L); accR=sum(r["_v"]==("valid" if r["valid"] else "invalid") for r in Rt)
    jL=[J[r["row_idx"]] for r in L if r["row_idx"] in J]; jR=[J[r["row_idx"]] for r in Rt if r["row_idx"] in J]
    cL=sum(bool(j.get("contaminated")) for j in jL); cR=sum(bool(j.get("contaminated")) for j in jR)
    mL=sum(j.get("fallacy_lens") in MR for j in jL); mR=sum(j.get("fallacy_lens") in MR for j in jR)
    sb=signed_bias(pol)
    print(f"{short:10}({ind:>3}) | {pc(accL,len(L))}/{pc(accR,len(Rt))} |   {pc(cL,len(jL))}/{pc(cR,len(jR))}   |  {pc(mL,len(jL))}/{pc(mR,len(jR))}  | {sb if sb is not None else ' . '}")
print("\n(lean) = cell's induced lean. acc/contam/MRfal split by STIMULUS lean (left-arm vs right-arm political).")
print("signed_bias>0 = right-leaning (accepts right-flattering / rejects left); <0 = left-leaning.")
