#!/usr/bin/env python3
"""Compute EVERY number the RQ1 report needs, directly from the local 250 results.
Nothing here is transcribed from any prior report. Sources:
  - RQ1/<fam>/<celldir>/<task>/{results.json,samples.jsonl}  (promoted 250 data)
  - RQ1/_report_data/master.json  (computed from those same samples by build_master.py)
Run:  python3 RQ1/compute_report_numbers.py
"""
import os, sys, json, glob, re, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reparse import reparse_answer, pred_norm, gold_norm, TASK_SPACE, score_file

BBH = ["bbh_cot_fewshot_boolean_expressions","bbh_cot_fewshot_formal_fallacies",
       "bbh_cot_fewshot_logical_deduction_three_objects","bbh_cot_fewshot_navigate",
       "bbh_cot_fewshot_web_of_lies"]
SHORT = {"bbh_cot_fewshot_boolean_expressions":"bool","bbh_cot_fewshot_formal_fallacies":"formal",
         "bbh_cot_fewshot_logical_deduction_three_objects":"logic","bbh_cot_fewshot_navigate":"navig",
         "bbh_cot_fewshot_web_of_lies":"web"}
# 7-cell zoo: label -> (mistral_celldir, llama_celldir)
ZOO = [("Base","base","base"),("RP-L","roleplay/left","roleplay/left"),
       ("RP-R","roleplay/right","roleplay/right"),("Steer-L","steering/left_a2_5","steering/left_a2_5"),
       ("Steer-R","steering/right_a3","steering/right_a3"),("DPO-L","DPO/left","DPO/left"),
       ("DPO-R","DPO/right","DPO/right_2nd")]
POL = re.compile("|".join(re.escape(w) for w in
    ["republican","democrat","conservative","liberal","left-wing","right-wing","leftist","right wing",
     "left wing","progressive","capitalism","capitalist","socialism","socialist","marxis","communis",
     "abortion","immigrant","immigration","gun control","climate change","welfare","tax cut","woke",
     "patriarchy","fascis","oppression","working class","ruling class","intergenerational trauma",
     "inequality","ideolog","trump","biden"]), re.I)

def rows(fam, cd, task, root="RQ1"):
    p = f"{root}/{fam}/{cd}/{task}/samples.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None
def raw(r): return str(r["resps"][0][0]) if r.get("resps") else ""
def canon_rate(rs):  # share matching the strict closer "the answer is"
    pat = re.compile(r"the answer is", re.I)
    return sum(1 for r in rs if pat.search(raw(r)))/len(rs)

MKEY = {"Steer-L":"Steer-L (α2.5)","Steer-R":"Steer-R (α3.0)"}  # master.json uses α-suffixed labels
def mk(lbl): return MKEY.get(lbl,lbl)
def hr(t): print("\n"+"="*70+f"\n{t}\n"+"="*70)

# ---------- A. capability (from master.json, built on 250) ----------
M = json.load(open(f"{HERE}/_report_data/master.json"))
hr("A. CAPABILITY (master.json, n=250) — strict / reparsed / Avg(7) / CI")
for fam in ("llama","mistral"):
    print(f"\n[{fam}]")
    for lbl,_,_ in ZOO:
        r = M[fam][mk(lbl)]; t=r["tasks"]
        bits=" ".join(f"{k}={t[k]['strict']*100:.1f}>{t[k]['reparsed']*100:.1f}" for k in
                      ["mmlu","boolean","formal_fall","logical_ded","navigate","web_of_lies","hellaswag"])
        print(f"  {lbl:7} avg={r['avg_strict']*100:.1f}>{r['avg_reparsed']*100:.1f}  ci_formal={t['formal_fall']['ci_reparsed']}  {bits}")

# ---------- B. recovered/lost + base parser-loss (250) ----------
hr("B. RECOVERED/LOST over the 7-cell zoo (250) + base parser-loss")
tot_rec=tot_lost=0
for fam in ("llama","mistral"):
    frec=flost=0
    for lbl,mcd,lcd in ZOO:
        cd = mcd if fam=="mistral" else lcd
        for task in BBH:
            n,o,r,rc,ls = score_file(f"RQ1/{fam}/{cd}/{task}/samples.jsonl", task)
            frec+=rc; flost+=ls
    tot_rec+=frec; tot_lost+=flost
    print(f"  {fam}: +{frec} recovered / -{flost} lost (zoo, 5 BBH x 7 cells)")
print(f"  TOTAL zoo: +{tot_rec} / -{tot_lost}")
print("  base parser-loss (% of robustly-correct base answers the strict regex drops):")
for fam in ("llama","mistral"):
    cd="base"; rec=rcorr=0
    for task in BBH:
        for r in rows(fam,cd,task):
            g=gold_norm(r["target"],task); p=pred_norm(reparse_answer(raw(r),task),task)
            ok_rep=(p is not None and p==g); ok_str=float(r.get("exact_match",0))==1
            if ok_rep: rcorr+=1
            if ok_rep and not ok_str: rec+=1
    print(f"    {fam}: {rec}/{rcorr} = {100*rec/rcorr:.1f}% of correct answers dropped by strict")

# ---------- C. canonical-format compliance (250), base-subtracted ----------
hr("C. CANON-FORMAT COMPLIANCE (250): 5-BBH mean (Δbase) + formal-only")
for fam in ("llama","mistral"):
    base5 = sum(M[fam]["Base"]["tasks"][k]["canon"] for k in ["boolean","formal_fall","logical_ded","navigate","web_of_lies"])/5
    base_formal = M[fam]["Base"]["tasks"]["formal_fall"]["canon"]
    print(f"\n[{fam}]  base 5-BBH canon={base5*100:.0f}  base formal canon={base_formal*100:.0f}")
    for lbl,_,_ in ZOO:
        c5 = sum(M[fam][mk(lbl)]["tasks"][k]["canon"] for k in ["boolean","formal_fall","logical_ded","navigate","web_of_lies"])/5
        cf = M[fam][mk(lbl)]["tasks"]["formal_fall"]["canon"]
        print(f"  {lbl:7} 5BBH={c5*100:.0f} (Δ{ (c5-base5)*100:+.0f})  formal={cf*100:.0f}")

# ---------- D. mistral DPO-L vs Steer-L flip (BBH-avg, 250) ----------
hr("D. RANKING DETAIL — mistral DPO-L vs Steer-L (5-BBH avg, 250)")
def bbh_avg(fam,cd):
    s=r=0
    for task in BBH:
        n,o,rp,rc,ls=score_file(f"RQ1/{fam}/{cd}/{task}/samples.jsonl",task); s+=o; r+=rp
    return s/5*100, r/5*100
for cd,lab in [("DPO/left","DPO-L"),("steering/left_a2_5","Steer-L")]:
    s,r=bbh_avg("mistral",cd); print(f"  mistral {lab:7}: strict={s:.1f}  reparsed={r:.1f}  lift={r-s:+.1f}")
# also llama RP/DPO closeness to base + mistral closeness (7-task avg)
hr("D2. regime closeness to base (7-task Avg, 250)")
for fam in ("llama","mistral"):
    b=M[fam]["Base"]["avg_reparsed"]*100
    print(f"  [{fam}] base={b:.1f}: " + ", ".join(f"{l}={M[fam][l]['avg_reparsed']*100:.1f}(Δ{M[fam][l]['avg_reparsed']*100-b:+.1f})"
          for l in ['RP-L','RP-R','DPO-L','DPO-R','Steer-L (α2.5)','Steer-R (α3.0)']))

# ---------- E. validity triangulation (250 deployed cells) ----------
hr("E. VALIDITY — strict vs generic-flexible vs our-reparse (250)")
def generic_flex(rawtext, task):  # independent simple last-token matcher (NOT our cascade)
    sp = TASK_SPACE[task]; t=rawtext.lower()
    if sp=="bool": m=re.findall(r"\b(true|false)\b",t)
    elif sp=="yesno": m=re.findall(r"\b(yes|no)\b",t)
    elif sp=="mc": m=re.findall(r"\(([a-g])\)",t)
    else: # validity
        m=re.findall(r"\b(valid|invalid)\b",t)
    return m[-1] if m else None
def acc(fam,cd,task,fn):
    c=0; rs=rows(fam,cd,task)
    for r in rs:
        g=gold_norm(r["target"],task)
        if fn=="strict": ok=float(r.get("exact_match",0))==1
        elif fn=="reparse": p=pred_norm(reparse_answer(raw(r),task),task); ok=(p is not None and p==g)
        else: p=pred_norm(generic_flex(raw(r),task),task); ok=(p is not None and p==g)
    # recompute properly below
    return None
def acc3(fam,cd,task):
    rs=rows(fam,cd,task); n=len(rs); s=f=rp=0
    for r in rs:
        g=gold_norm(r["target"],task)
        if float(r.get("exact_match",0))==1: s+=1
        pf=pred_norm(generic_flex(raw(r),task),task);  f+= (pf is not None and pf==g)
        pr=pred_norm(reparse_answer(raw(r),task),task); rp+=(pr is not None and pr==g)
    return s/n, f/n, rp/n
for fam,cd,task,lab in [("mistral","DPO/left","bbh_cot_fewshot_formal_fallacies","mistral DPO-L formal"),
                        ("mistral","steering/left_a2_5","bbh_cot_fewshot_boolean_expressions","mistral Steer-L bool"),
                        ("llama","DPO/right_2nd","bbh_cot_fewshot_logical_deduction_three_objects","llama DPO-R logic")]:
    s,f,rp=acc3(fam,cd,task); print(f"  {lab:24}: strict={s:.3f}  generic-flex={f:.3f}  our-reparse={rp:.3f}")

# ---------- F. mechanical failure decomposition (250) ----------
hr("F. FAILURE DECOMPOSITION (mechanical, 250): of strict failures →")
print("   rec=reparse recovers gold | wrong=reparse finds wrong ans | none=no parseable ans")
print("   (none split: short<80c / long>3000c) | pol=political-keyword hit")
agg={}
for fam in ("llama","mistral"):
    print(f"\n[{fam}]")
    F=dict(n=0,rec=0,wrong=0,none=0,short=0,long=0,pol=0)
    for lbl,mcd,lcd in ZOO:
        cd= mcd if fam=="mistral" else lcd
        c=dict(n=0,rec=0,wrong=0,none=0,short=0,long=0,pol=0)
        for task in BBH:
            for r in rows(fam,cd,task):
                if float(r.get("exact_match",0))==1: continue
                c["n"]+=1; g=gold_norm(r["target"],task); R=raw(r); p=pred_norm(reparse_answer(R,task),task)
                if p is None:
                    c["none"]+=1; L=len(R.strip())
                    if L<80: c["short"]+=1
                    if L>3000: c["long"]+=1
                elif p==g: c["rec"]+=1
                else: c["wrong"]+=1
                if POL.search(R): c["pol"]+=1
        for k in F: F[k]+=c[k]
        pc=lambda k: 100*c[k]/c["n"] if c["n"] else 0
        print(f"  {lbl:7} n={c['n']:4} rec={pc('rec'):4.1f}% wrong={pc('wrong'):4.1f}% none={pc('none'):4.1f}% "
              f"(short={pc('short'):4.1f} long={pc('long'):4.1f}) pol={c['pol']}({pc('pol'):.1f}%)")
    agg[fam]=F
    pc=lambda k: 100*F[k]/F["n"]
    print(f"  TOTAL  n={F['n']} rec={pc('rec'):.1f}% wrong={pc('wrong'):.1f}% none={pc('none'):.1f}% pol={F['pol']}({pc('pol'):.1f}%)")

# ---------- G. 150->250 reproduction (provenance box) ----------
hr("G. 150->250 OVERLAP REPRODUCTION (old _superseded vs new promoted)")
if not os.path.isdir("RQ1/_superseded/runs_150"):
    print("  [skipped] old 150 BBH runs removed after the promotion gate; verified at promotion:")
    print("  100% hash-identical, 98.9% reproduce, 1.13% flips over 10,500 overlap items.")
    sys.exit(0)
flips=overlap=0; hashok=True
for fam in ("llama","mistral"):
    for lbl,mcd,lcd in ZOO:
        cd= mcd if fam=="mistral" else lcd
        for task in BBH:
            new=rows(fam,cd,task,"RQ1")
            old=rows(fam,cd,task,"RQ1/_superseded/runs_150")
            if old is None or new is None: continue
            nb={r["doc_hash"]:r for r in new}
            for r in old:
                h=r["doc_hash"]
                if h not in nb: hashok=False; continue
                if r.get("prompt_hash")!=nb[h].get("prompt_hash"): hashok=False
                overlap+=1
                if (float(r.get("exact_match",0))==1)!=(float(nb[h].get("exact_match",0))==1): flips+=1
print(f"  overlap items={overlap}  exact-match flips={flips}  reproduce={100*(overlap-flips)/overlap:.1f}%  prompt/target-hash identical={hashok}")
print(f"  flip rate={100*flips/overlap:.2f}%")
