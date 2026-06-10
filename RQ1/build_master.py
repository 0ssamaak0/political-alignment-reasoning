#!/usr/bin/env python3
"""Canonical reparsed master table for the RQ1 paper section (7-cell trimmed zoo).

Per (model, cell): MMLU formal_logic (acc), 5 BBH (strict exact_match + reparsed),
HellaSwag (acc_norm), and avg-reasoning over the 7 tasks (strict & reparsed).
MMLU/HellaSwag are loglikelihood -> parser-independent -> strict == reparsed.
Llama DPO-right is standardized to the deployed `right_2nd` checkpoint.
Writes _report_data/master.json.
"""
import os, json, math, glob
from reparse import score_file

BASE = os.path.dirname(__file__)
BBH = ["bbh_cot_fewshot_boolean_expressions","bbh_cot_fewshot_formal_fallacies",
       "bbh_cot_fewshot_logical_deduction_three_objects","bbh_cot_fewshot_navigate",
       "bbh_cot_fewshot_web_of_lies"]
BBH_SHORT = ["boolean","formal_fall","logical_ded","navigate","web_of_lies"]

# 7-cell trimmed zoo. (display_name, path). Llama DPO-right -> right_2nd.
def zoo(fam):
    dpo_r = "DPO/right_2nd" if fam == "llama" else "DPO/right"
    return [("Base","base"),
            ("RP-L","roleplay/left"),("RP-R","roleplay/right"),
            ("Steer-L (α2.5)","steering/left_a2_5"),("Steer-R (α3.0)","steering/right_a3"),
            ("DPO-L","DPO/left"),("DPO-R","dpo_r_placeholder" if False else dpo_r)]

def jget(path, *keys):
    d = json.load(open(path))["results"]
    t = list(d.keys())[0]
    for k in keys:
        if k in d[t]: return d[t][k]
    return None

def nlines(p): return sum(1 for _ in open(p)) if os.path.exists(p) else None

def ci(p, n):
    if n in (None, 0) or p is None: return None
    return round(1.96*math.sqrt(p*(1-p)/n)*100, 1)

def build():
    out = {}
    for fam in ["llama","mistral"]:
        out[fam] = {}
        for disp, cell in zoo(fam):
            rec = {"cell_path": cell, "tasks": {}}
            # MMLU
            mp = os.path.join(BASE,fam,cell,"mmlu_formal_logic")
            if os.path.exists(mp+"/results.json"):
                acc = jget(mp+"/results.json","acc,none"); n = nlines(mp+"/samples.jsonl")
                rec["tasks"]["mmlu"] = {"strict":acc,"reparsed":acc,"n":n,"ci":ci(acc,n)}
            # HellaSwag
            hp = os.path.join(BASE,fam,cell,"hellaswag")
            if os.path.exists(hp+"/results.json"):
                acc = jget(hp+"/results.json","acc_norm,none"); n = nlines(hp+"/samples.jsonl")
                rec["tasks"]["hellaswag"] = {"strict":acc,"reparsed":acc,"n":n,"ci":ci(acc,n)}
            # BBH (strict + reparsed)
            for t, short in zip(BBH, BBH_SHORT):
                p = os.path.join(BASE,fam,cell,t,"samples.jsonl")
                if not os.path.exists(p): continue
                n,o,r,rec_,lost = score_file(p,t)
                # canonical-format compliance: fraction the STRICT regex parsed (not [invalid] sentinel)
                rows_ = [json.loads(l) for l in open(p)]
                canon = sum(1 for d in rows_ if str(d["filtered_resps"]) != "['[invalid]']")/len(rows_)
                rec["tasks"][short] = {"strict":round(o,4),"reparsed":round(r,4),"n":n,
                                       "ci_strict":ci(o,n),"ci_reparsed":ci(r,n),
                                       "recovered":rec_,"lost":lost,"canon":round(canon,4)}
            # 5-BBH-mean canonical-format compliance (instruction-following metric)
            cvals=[rec["tasks"][k]["canon"] for k in BBH_SHORT if k in rec["tasks"]]
            rec["canon_bbh"] = round(sum(cvals)/len(cvals),4) if cvals else None
            # averages over the 7 reasoning tasks
            order = ["mmlu","boolean","formal_fall","logical_ded","navigate","web_of_lies","hellaswag"]
            sv = [rec["tasks"][k]["strict"] for k in order if k in rec["tasks"]]
            rv = [rec["tasks"][k]["reparsed"] for k in order if k in rec["tasks"]]
            rec["avg_strict"] = round(sum(sv)/len(sv),4)
            rec["avg_reparsed"] = round(sum(rv)/len(rv),4)
            rec["n_tasks"] = len(sv)
            out[fam][disp] = rec
    return out

if __name__ == "__main__":
    os.makedirs(os.path.join(BASE,"_report_data"), exist_ok=True)
    m = build()
    json.dump(m, open(os.path.join(BASE,"_report_data","master.json"),"w"), indent=1)
    # console preview
    for fam in m:
        print(f"\n===== {fam} =====")
        print(f"{'cell':16}{'MMLU':>7}{'bool':>14}{'formal':>14}{'logic':>14}{'navig':>14}{'web':>14}{'HSwag':>7}{'AVG s>r':>14}")
        for disp, rec in m[fam].items():
            tk = rec["tasks"]
            def cell(k, key="bbh"):
                if k not in tk: return "—"
                if k in ("mmlu","hellaswag"): return f"{tk[k]['strict']*100:.1f}"
                return f"{tk[k]['strict']*100:.0f}>{tk[k]['reparsed']*100:.0f}"
            avgstr = f"{rec['avg_strict']*100:.1f}>{rec['avg_reparsed']*100:.1f}"
            print(f"{disp:16}{cell('mmlu'):>7}{cell('boolean'):>14}{cell('formal_fall'):>14}"
                  f"{cell('logical_ded'):>14}{cell('navigate'):>14}{cell('web_of_lies'):>14}"
                  f"{cell('hellaswag'):>7}{avgstr:>14}")
    print("\nwrote _report_data/master.json")
