"""Aggregate per cell across 3 conditions: Neutral-clean | Political-clean |
Political-flipped. Metrics: acc% (extract_verdict vs gold), and judge axes
contam% / MR-fallacy% / reasoning-invalid%. Neutral has lean=None (no left/right)."""
import json, os, sys
sys.path.insert(0, "/Users/0ssamaak0/Documents/polireason/1_benchmarking/RQ2_bench")
from extract import extract_verdict
ROOT = "/Users/0ssamaak0/Documents/polireason/RQ2/flipped"
STIM = [json.loads(l) for l in open("/Users/0ssamaak0/Documents/polireason/1_benchmarking/RQ2_bench/stimuli.jsonl")]
AV = {i: (r["arm"], r["variant"]) for i, r in enumerate(STIM)}
MR = {"motivational_reasoning", "premise_truth_conflation"}
CELLS = [("llama-base","Base"),("llama-roleplay-left","RP-L"),("llama-roleplay-right","RP-R"),
("llama-pvsteer-ml-left-a2_5","Steer-L"),("llama-pvsteer-ml-right-a3","Steer-R"),
("llama-politune-hf-left","DPO-L"),("llama-politune-hf-right","DPO-R"),
("mistral-base","Base"),("mistral-roleplay-left","RP-L"),("mistral-roleplay-right","RP-R"),
("mistral-pvsteer-ml-left-a2_5","Steer-L"),("mistral-pvsteer-ml-right-a3","Steer-R"),
("mistral-politune-hf-left","DPO-L"),("mistral-politune-hf-right","DPO-R")]
CONDS = [("neutral","clean","NeuC"),("political","clean","PolC"),("political","flipped","PolF")]
def p(n,d): return f"{100*n/d:3.0f}" if d else " . "
agg = {c[2]: {"n":0,"acc":0,"con":0,"mr":0,"inv":0,"ass":0} for c in CONDS}
print(f"{'cell':14}|        acc%  NeuC/PolC/PolF |    contam%  NeuC/PolC/PolF |   MRfal%  NeuC/PolC/PolF")
print("-"*94)
fam=None
for tag,short in CELLS:
    rp=f"{ROOT}/responses/{tag}.jsonl"; jp=f"{ROOT}/judges/{tag}.jsonl"
    f0=tag.split('-')[0]
    if f0!=fam: fam=f0; print(f"== {fam} ==")
    if not os.path.exists(rp): print(f"{short:14}| (pending)"); continue
    R=[json.loads(l) for l in open(rp)]
    for r in R: r["arm"],r["variant"]=AV.get(r.get("row_idx"),(None,None))
    J={}
    if os.path.exists(jp):
        for l in open(jp):
            jr=json.loads(l); jr["arm"],jr["variant"]=AV.get(jr.get("row_idx"),(None,None)); J[jr.get("row_idx")]=jr
    accs=[];cons=[];mrs=[]
    for arm,var,lab in CONDS:
        rs=[r for r in R if r["arm"]==arm and r["variant"]==var]
        js=[J[r["row_idx"]] for r in rs if r["row_idx"] in J]
        acc=sum(extract_verdict(r["raw_response"])==("valid" if r["valid"] else "invalid") for r in rs)
        con=sum(bool(j.get("contaminated")) for j in js); mr=sum(j.get("fallacy_lens") in MR for j in js)
        ass=[j for j in js if j.get("reasoning_validity") in ("valid","invalid")]; inv=sum(j["reasoning_validity"]=="invalid" for j in ass)
        accs.append(p(acc,len(rs)));cons.append(p(con,len(js)));mrs.append(p(mr,len(js)))
        a=agg[lab]; a["n"]+=len(rs); a["acc"]+=acc; a["con"]+=con; a["mr"]+=mr; a["inv"]+=inv; a["ass"]+=len(ass)
    print(f"{short:14}|        {accs[0]}/{accs[1]}/{accs[2]}   |        {cons[0]}/{cons[1]}/{cons[2]}    |       {mrs[0]}/{mrs[1]}/{mrs[2]}")
print("\n== POOLED across all cells ==")
for c in CONDS:
    a=agg[c[2]]
    print(f"  {c[2]}: n={a['n']:5}  acc={p(a['acc'],a['n'])}%  contam={p(a['con'],a['n'])}%  MRfal={p(a['mr'],a['n'])}%  rsn_invalid={p(a['inv'],a['ass'])}%")
print("\nNeuC=neutral clean | PolC=political clean (congruent) | PolF=political flipped (counter-attitudinal)")
print("contam/MRfal/invalid from judge; neutral has no left/right (lean=None).")
