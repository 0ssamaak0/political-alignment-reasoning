#!/usr/bin/env python3
"""RQ1 paper figures (reparsed, 7-cell zoo, house brand). Emits PDF+SVG into figures/.
Figs: capability avg bar (per model), capability heatmap (per model),
trait/coherence matching scatter, strict->reparsed artifact-lift dumbbell."""
import sys, json, os
from pathlib import Path
for _d in Path(__file__).resolve().parents:
    _s = _d / ".claude/skills/visualizations-designer/scripts"
    if _s.is_dir(): sys.path.insert(0, str(_s)); break
from polireason_viz import apply_theme, LEAN, BASE, save_fig
from functools import partial as _partial
save_fig = _partial(save_fig, formats=("pdf",))  # this project: PDF only
import matplotlib.pyplot as plt
import numpy as np

apply_theme()
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # script lives in figures/; root is one up
M = json.load(open(os.path.join(BASE_DIR,"_report_data","master.json")))
FAMS = [("llama","Llama-3-8B-Instruct"),("mistral","Mistral-7B-Instruct-v0.2")]
CELLS = ["Base","RP-L","RP-R","Steer-L (α2.5)","Steer-R (α3.0)","DPO-L","DPO-R"]
SHORT = {"Base":"Base","RP-L":"RP-L","RP-R":"RP-R","Steer-L (α2.5)":"Steer-L","Steer-R (α3.0)":"Steer-R","DPO-L":"DPO-L","DPO-R":"DPO-R"}
def lean_color(c):
    if c=="Base": return BASE
    return LEAN["right"] if c.endswith("R") or "Steer-R" in c or "RP-R" in c or "DPO-R" in c else LEAN["left"]
def colc(c):
    return BASE if c=="Base" else (LEAN["left"] if c.split()[0].endswith("L") else LEAN["right"])
TASK_KEYS = ["mmlu","boolean","formal_fall","logical_ded","navigate","web_of_lies","hellaswag"]
TASK_LAB  = ["MMLU","BBH bool","BBH formal","BBH logic","BBH navig","BBH web","HellaSwag"]

def bbh_mean(rec, which):
    ks=["boolean","formal_fall","logical_ded","navigate","web_of_lies"]
    v=[rec["tasks"][k][which] for k in ks if k in rec["tasks"]]
    return sum(v)/len(v)

# ---- Fig 1: capability avg-reasoning bar, per model ----
for fam,label in FAMS:
    fig,ax=plt.subplots(figsize=(6.2,3.6))
    vals=[M[fam][c]["avg_reparsed"]*100 for c in CELLS]
    cols=[colc(c) for c in CELLS]
    ax.bar(range(len(CELLS)),vals,color=cols,width=0.7)
    base_v=M[fam]["Base"]["avg_reparsed"]*100
    ax.axhline(base_v,color=BASE,ls="--",lw=1.0,zorder=0)
    for i,v in enumerate(vals): ax.text(i,v+0.6,f"{v:.0f}",ha="center",fontsize=8,color="#1A1A1A")
    ax.set_xticks(range(len(CELLS))); ax.set_xticklabels([SHORT[c] for c in CELLS],rotation=0,fontsize=8.5)
    ax.set_ylabel("avg reasoning (reparsed, %)"); ax.set_ylim(0,100)
    ax.set_title(f"{label} — reasoning across alignment cells")
    save_fig(fig,os.path.join(BASE_DIR,"figures",f"rq1_capability_{fam}")); plt.close(fig)

# ---- Fig 2: per-task heatmap (reparsed), per model ----
for fam,label in FAMS:
    grid=np.array([[M[fam][c]["tasks"].get(k,{}).get("reparsed",np.nan)*100 for k in TASK_KEYS] for c in CELLS])
    fig,ax=plt.subplots(figsize=(6.6,3.8))
    im=ax.imshow(grid,cmap="viridis",vmin=0,vmax=100,aspect="auto")
    ax.set_xticks(range(len(TASK_LAB))); ax.set_xticklabels(TASK_LAB,rotation=35,ha="right",fontsize=8)
    ax.set_yticks(range(len(CELLS))); ax.set_yticklabels([SHORT[c] for c in CELLS],fontsize=8.5)
    for i in range(len(CELLS)):
        for j in range(len(TASK_KEYS)):
            v=grid[i,j]
            if not np.isnan(v): ax.text(j,i,f"{v:.0f}",ha="center",va="center",fontsize=7,
                                        color="white" if v<55 else "#1A1A1A")
    ax.grid(False)
    cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.02); cb.set_label("accuracy (%)",fontsize=8.5)
    ax.set_title(f"{label} — per-task accuracy (reparsed)")
    save_fig(fig,os.path.join(BASE_DIR,"figures",f"rq1_heatmap_{fam}")); plt.close(fig)

# ---- Fig 3: trait / coherence grouped bars (color=lean, regime on x), 2x2 small-multiples ----
# Source: RQ1/README.md "Combined view" (3 regimes x 4 model-lean cells) + base anchors.
# (Llama DPO-R uses deployed right_2nd: 97.25/77.25.) Each tuple is (trait, coherence).
TC={
 "mistral":{"Base":(32.2,71),"RP-L":(98.5,74.2),"RP-R":(93.5,81.2),
   "Steer-L (α2.5)":(92.2,75.2),"Steer-R (α3.0)":(78.7,82.6),"DPO-L":(95.8,94.0),"DPO-R":(88.0,99.6)},
 "llama":{"Base":(31.2,71),"RP-L":(100.0,71.5),"RP-R":(94.2,75.2),
   "Steer-L (α2.5)":(99.2,75.6),"Steer-R (α3.0)":(71.5,75.8),"DPO-L":(97.2,92.5),"DPO-R":(97.25,77.25)},
}
REGIMES=[("Roleplay","RP"),("Steering","Steer"),("DPO","DPO")]      # x groups
KEYMAP={("RP","L"):"RP-L",("RP","R"):"RP-R",
        ("Steer","L"):"Steer-L (α2.5)",("Steer","R"):"Steer-R (α3.0)",
        ("DPO","L"):"DPO-L",("DPO","R"):"DPO-R"}
# 2 panels (Trait | Coherence); each holds BOTH models as side-by-side blocks.
# No on-bar value labels (exact numbers live in README "Combined view"): bars + base line carry it.
METRICS=[("Trait",0),("Coherence",1)]
MODELS=[("mistral","Mistral"),("llama","Llama")]
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
fig,axes=plt.subplots(1,2,figsize=(11.0,4.0),sharey=True)
w=0.38
base_x=np.arange(len(REGIMES))                              # regimes at 0,1,2 within a block
block_off={"mistral":0.0,"llama":len(REGIMES)+1.0}         # Llama block shifted right (gap between)
for ax,(mlabel,mi) in zip(axes,METRICS):
    ticks=[]; ticklab=[]
    for fam,flabel in MODELS:
        xs=base_x+block_off[fam]
        lv=[TC[fam][KEYMAP[(rs,"L")]][mi] for _,rs in REGIMES]
        rv=[TC[fam][KEYMAP[(rs,"R")]][mi] for _,rs in REGIMES]
        ax.bar(xs-w/2,lv,w,color=LEAN["left"],zorder=2)
        ax.bar(xs+w/2,rv,w,color=LEAN["right"],zorder=2)
        ticks+=list(xs); ticklab+=[rf for rf,_ in REGIMES]
        tr=mtransforms.blended_transform_factory(ax.transData,ax.transAxes)
        ax.text(xs.mean(),-0.14,flabel,transform=tr,ha="center",va="top",fontsize=9.5,color="#1A1A1A")
    bv=float(np.mean([TC[fam]["Base"][mi] for fam,_ in MODELS]))   # models' base nearly identical -> one line
    ax.axhline(bv,color=BASE,ls="--",lw=1.0,zorder=1)
    ax.set_ylim(0,105); ax.set_xticks(ticks); ax.set_xticklabels(ticklab,fontsize=8.5)
    ax.set_title(mlabel,fontsize=11)
axes[0].set_ylabel("judge score (0-100)")
leg=[Patch(facecolor=LEAN["left"],label="Left"),
     Patch(facecolor=LEAN["right"],label="Right"),
     Line2D([0],[0],color=BASE,ls="--",lw=1.0,label="Base (no intervention)")]
fig.legend(handles=leg,loc="upper right",bbox_to_anchor=(0.99,0.99),ncol=3,frameon=False,fontsize=8.5)
fig.subplots_adjust(left=0.07,right=0.985,top=0.86,bottom=0.16,wspace=0.07)
save_fig(fig,os.path.join(BASE_DIR,"figures","rq1_trait_coherence")); plt.close(fig)

# ---- Fig 4: strict -> reparsed artifact lift (BBH-mean dumbbell), per model panel ----
fig,axes=plt.subplots(1,2,figsize=(7.2,4.0),sharex=True)
for ax,(fam,label) in zip(axes,FAMS):
    ys=range(len(CELLS))
    for i,c in enumerate(CELLS):
        s=bbh_mean(M[fam][c],"strict")*100; r=bbh_mean(M[fam][c],"reparsed")*100
        ax.plot([s,r],[i,i],color="#C8C8C8",lw=2,zorder=1)
        ax.scatter(s,i,color="#B0B0B0",s=42,zorder=2)
        ax.scatter(r,i,color=colc(c),s=60,zorder=3)
        if r-s>1.5: ax.annotate(f"+{r-s:.0f}",(r,i),fontsize=7,color=colc(c),
                                xytext=(5,0),textcoords="offset points",va="center")
    ax.set_yticks(list(ys)); ax.set_yticklabels([SHORT[c] for c in CELLS],fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlim(0,100); ax.set_xlabel("BBH-mean accuracy (%)")
    ax.set_title(label,fontsize=11)
proxies=[Line2D([0],[0],marker="o",color="#B0B0B0",ls="",ms=7,label="strict (harness regex)"),
         Line2D([0],[0],marker="o",color=BASE,ls="",ms=7,label="reparsed (robust)")]
axes[0].legend(handles=proxies,loc="lower left",fontsize=8,frameon=False)
fig.tight_layout()
save_fig(fig,os.path.join(BASE_DIR,"figures","rq1_artifact_lift")); plt.close(fig)

print("wrote 6 figures (4 types) to figures/:")
for f in ["rq1_capability_llama","rq1_capability_mistral","rq1_heatmap_llama",
          "rq1_heatmap_mistral","rq1_trait_coherence","rq1_artifact_lift"]:
    print("  ",f+".pdf / .svg")
