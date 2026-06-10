"""Is there a POLITICAL-FLIP signal? For each cell, political items only,
contrast clean vs flipped on: discrimination D, judge reasoning_invalid%,
contaminated%, and motivated-reasoning fallacy%. Flip = counter-attitorial.
Usage: python flip_signal.py <judges_dir> <responses_dir> <label>"""
import json, os, sys
sys.path.insert(0, "/Users/0ssamaak0/Documents/polireason/1_benchmarking/RQ2_bench")
from extract import extract_verdict
JUDG, RESP, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
STIM = [json.loads(l) for l in open("/Users/0ssamaak0/Documents/polireason/1_benchmarking/RQ2_bench/stimuli.jsonl")]
ARMVAR = {i: (r["arm"], r["variant"]) for i, r in enumerate(STIM)}
MR = {"motivational_reasoning", "premise_truth_conflation"}
CELLS = [("llama-base","Base"),("llama-pvsteer-ml-left-a2_5","Steer-L"),("llama-pvsteer-ml-right-a3","Steer-R"),
("llama-politune-hf-left","DPO-L"),("llama-politune-hf-right","DPO-R"),
("mistral-base","Base"),("mistral-pvsteer-ml-left-a2_5","Steer-L"),("mistral-pvsteer-ml-right-a3","Steer-R"),
("mistral-politune-hf-left","DPO-L"),("mistral-politune-hf-right","DPO-R")]
def D(rows):
    gv=[r for r in rows if r["valid"]]; gi=[r for r in rows if not r["valid"]]
    if not gv or not gi: return None
    sv=lambda g: sum(extract_verdict(r["raw_response"])=="valid" for r in g)/len(g)
    return sv(gv)-sv(gi)
def pc(n,d): return f"{100*n/d:3.0f}" if d else " . "
print(f"\n###### {LABEL}: political clean vs FLIPPED ######")
print(f"{'cell':14}| {'D_cln':>5} {'D_flp':>5} {'ΔD':>5} | {'inval c/f':>9} | {'contam c/f':>10} | {'MRfal c/f':>9}")
print("-"*74)
fam=None
for tag,short in CELLS:
    jp=f"{JUDG}/{tag}.jsonl"; rp=f"{RESP}/{tag}.jsonl"
    f0=tag.split('-')[0]
    if f0!=fam: fam=f0; print(f"== {fam} ==")
    if not os.path.exists(jp): continue
    J=[json.loads(l) for l in open(jp)]
    for r in J: r["arm"],r["variant"]=ARMVAR.get(r.get("row_idx"),(None,None))
    R={r.get("row_idx"):r for r in (json.loads(l) for l in open(rp))} if os.path.exists(rp) else {}
    polc=[r for r in J if r["arm"]=="political" and r["variant"]=="clean"]
    polf=[r for r in J if r["arm"]=="political" and r["variant"]=="flipped"]
    # discrimination from raw responses joined by row_idx
    def Dv(jr):
        rows=[R[r["row_idx"]] for r in jr if r["row_idx"] in R]
        return D(rows) if rows else None
    dc,df=Dv(polc),Dv(polf); dd=None if dc is None or df is None else dc-df
    inv=lambda g:sum(r.get("reasoning_validity")=="invalid" for r in g)
    con=lambda g:sum(bool(r.get("contaminated")) for r in g)
    mr=lambda g:sum(r.get("fallacy_lens") in MR for r in g)
    fmt=lambda x:" .  " if x is None else f"{x:+.2f}"
    print(f"{short:14}| {fmt(dc)} {fmt(df)} {fmt(dd)} | {pc(inv(polc),len(polc))}/{pc(inv(polf),len(polf))} | "
          f"  {pc(con(polc),len(polc))}/{pc(con(polf),len(polf))}  |  {pc(mr(polc),len(polc))}/{pc(mr(polf),len(polf))}")
print("\nc/f = clean vs flipped (political only). D=discrimination; inval=%CoT logically invalid;")
print("contam=%partisan bleed; MRfal=%motivated_reasoning|premise_truth_conflation fallacy.")
print("FLIP SIGNAL = flipped worse than clean: lower D_flp, higher inval/contam/MRfal on f.")
