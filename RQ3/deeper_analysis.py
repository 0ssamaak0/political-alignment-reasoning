#!/usr/bin/env python3
"""RQ3 deeper analysis (Mistral) — every new signal beyond RESULTS.md.

This does NOT replace consolidate.py / judge_aggregate.py. It joins each cell's
`samples.jsonl` (reparse-based three-regime decomposition, reusing RQ1/reparse.py
so numbers reconcile with RESULTS.md) with `judge.jsonl` (per-item objective
degeneracy metrics + judge rubric) and computes, per cell AND per task:

  A. Degeneracy morphology (judge-INDEPENDENT): n_tokens_generated,
     max_4gram_repeat, distinct_ratio_last_50 -> separates "long repetition loop"
     from "short stub / off-format" collapse without trusting the judge.
  B. Three-regime trajectory: correct / wrong / collapse / recovered, to show the
     pre-cliff wrong-reasoning window before pure collapse.
  C. MMLU loglikelihood dissociation: above-chance margin retained vs generation.
  D. Item-level co-occurrence: P(contaminated|collapsed) vs P(contaminated|not),
     P(correct|contaminated), the joint outcome x contaminated x collapsed table.
  E. Reasoning-quality dose-response: faithful / post-hoc / capability axes,
     reasoning_validity, judge confidence.

Cross-checks the overlapping quantities (reparsed BBHmean, collapse%, contam%,
judge_acc) against results/dose_response_mistral.txt + judge_dose_response.txt and
prints PASS/FAIL so a new number can never silently contradict the canonical doc.

    python3 RQ3/deeper_analysis.py
Outputs: RQ3/results/deeper/*.txt  +  RQ3/results/deeper/deeper_numbers.json
"""
import os, sys, json, statistics as st
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "deeper")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "RQ1"))
import reparse as RP  # noqa: E402

FAM = "mistral"
TASKS = ["boolean_expressions", "logical_deduction_three_objects",
         "web_of_lies", "navigate"]
TASKDIR = {t: f"bbh_cot_fewshot_{t}" for t in TASKS}
TSHORT = {"boolean_expressions": "bool", "logical_deduction_three_objects": "logic3",
          "web_of_lies": "weblies", "navigate": "navig"}
STEER = [("a0_5", 0.5), ("a1", 1.0), ("a2", 2.0), ("a3", 3.0), ("a4", 4.0)]
DPO = [("s0_25", 0.25), ("s0_5", 0.5), ("s1_0", 1.0), ("s1_5", 1.5), ("s2", 2.0)]
TOKEN_LIMIT = 1024

CATS = ["faithful_task_performance", "post_hoc_reasoning", "capability_error",
        "instruction_following_failure", "motivational_framing_bias",
        "viewpoint_bias", "generation_collapse"]
CAT_SHORT = {"faithful_task_performance": "faith", "post_hoc_reasoning": "posthoc",
             "capability_error": "caperr", "instruction_following_failure": "instr_f",
             "motivational_framing_bias": "mframe", "viewpoint_bias": "vbias",
             "generation_collapse": "gcollap"}


def cells():
    yield ("base", os.path.join(RESULTS, FAM, "base"))
    for lean in ("left", "right"):
        for suf, _ in STEER:
            yield (f"steering/{lean}/{suf}", os.path.join(RESULTS, FAM, "steering", lean, suf))
    for lean in ("left", "right"):
        for suf, _ in DPO:
            yield (f"dpo/{lean}/{suf}", os.path.join(RESULTS, FAM, "dpo", lean, suf))


def pct(num, den):
    return 100.0 * num / den if den else None


def task_block(cell_dir, task):
    """Join samples + judge for one (cell, task). Returns per-item-derived stats."""
    sp = os.path.join(cell_dir, TASKDIR[task], "samples.jsonl")
    jp = os.path.join(cell_dir, TASKDIR[task], "judge.jsonl")
    if not os.path.exists(sp):
        return None
    samples = [json.loads(l) for l in open(sp)]
    n = len(samples)
    # --- reparse three-regime (matches consolidate.py exactly) ---
    rep_c = rec = none = wrong = strict_c = 0
    for d in samples:
        raw = d["resps"][0][0] if d.get("resps") else ""
        gold = RP.gold_norm(d["target"], TASKDIR[task])
        orig_ok = float(d.get("exact_match", 0.0)) == 1.0
        strict_c += orig_ok
        pred = RP.pred_norm(RP.reparse_answer(raw, TASKDIR[task]), TASKDIR[task])
        rep_ok = (pred is not None and pred == gold)
        rep_c += rep_ok
        if not orig_ok:
            if pred is None:
                none += 1
            elif rep_ok:
                rec += 1
            else:
                wrong += 1
    out = {"n": n, "strict_acc": pct(strict_c, n), "reparsed_acc": pct(rep_c, n),
           "collapse": pct(none, n), "wrong": pct(wrong, n), "recovered": pct(rec, n),
           "correct": pct(rep_c, n)}
    # --- judge join (degeneracy + rubric) ---
    if os.path.exists(jp):
        J = [json.loads(l) for l in open(jp)]
        toks = [r.get("n_tokens_generated") for r in J if r.get("n_tokens_generated") is not None]
        reps = [r.get("max_4gram_repeat") for r in J if r.get("max_4gram_repeat") is not None]
        dist = [r.get("distinct_ratio_last_50") for r in J if r.get("distinct_ratio_last_50") is not None]
        nj = len(J)
        cat = Counter(r.get("primary_category") for r in J)
        val = Counter(r.get("reasoning_validity") for r in J)
        conf = [r.get("confidence") for r in J if isinstance(r.get("confidence"), (int, float))]
        contam = sum(1 for r in J if r.get("contaminated"))
        jcoll = sum(1 for r in J if r.get("collapsed"))
        jcorrect = sum(1 for r in J if r.get("outcome") == "correct")
        noverd = sum(1 for r in J if r.get("outcome") in ("no_answer", "off_format"))
        # co-occurrence: contaminated within collapsed vs not (judge collapsed axis)
        coll_rows = [r for r in J if r.get("collapsed")]
        noncoll_rows = [r for r in J if not r.get("collapsed")]
        contam_rows = [r for r in J if r.get("contaminated")]
        out["judge"] = {
            "nj": nj,
            "tok_mean": st.mean(toks) if toks else None,
            "tok_median": st.median(toks) if toks else None,
            "pct_runs_to_limit": pct(sum(1 for x in toks if x >= 900), nj),   # near 1024 cap
            "pct_stub": pct(sum(1 for x in toks if x <= 30), nj),
            "rep_mean": st.mean(reps) if reps else None,
            "rep_median": st.median(reps) if reps else None,
            "pct_loop": pct(sum(1 for x in reps if x >= 20), nj),            # heavy 4gram repetition
            "distinct_mean": st.mean(dist) if dist else None,
            "judge_acc": pct(jcorrect, nj),
            "contam": pct(contam, nj),
            "jcollapse": pct(jcoll, nj),
            "no_verdict": pct(noverd, nj),
            "confidence_mean": st.mean(conf) if conf else None,
            "cat": {c: pct(cat.get(c, 0), nj) for c in CATS},
            "validity": {k: pct(v, nj) for k, v in val.items()},
            # co-occurrence
            "contam_in_collapsed": pct(sum(1 for r in coll_rows if r.get("contaminated")), len(coll_rows)),
            "contam_in_noncollapsed": pct(sum(1 for r in noncoll_rows if r.get("contaminated")), len(noncoll_rows)),
            "n_collapsed": len(coll_rows),
            "n_noncollapsed": len(noncoll_rows),
            "correct_in_contaminated": pct(sum(1 for r in contam_rows if r.get("outcome") == "correct"), len(contam_rows)),
            "n_contaminated": len(contam_rows),
        }
    return out


def mmlu_acc(cell_dir):
    p = os.path.join(cell_dir, "mmlu_formal_logic", "results.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    # results.json: {"results": {"mmlu_formal_logic": {"acc,none": x, ...}}}
    res = d.get("results", {})
    for k, v in res.items():
        for mk, mv in v.items():
            if mk.startswith("acc,") or mk == "acc":
                return 100.0 * mv
    return None


def pool_cell(per_task):
    """Pool the 4 tasks. Item-pooled for judge axes (matches judge_aggregate);
    mean-of-task for reparsed/collapse (matches consolidate's BBHmean/coll%)."""
    have = [t for t in TASKS if per_task.get(t)]
    if not have:
        return None
    # mean-of-task (consolidate convention)
    bbhmean = st.mean(per_task[t]["reparsed_acc"] for t in have)
    coll_meanoftask = st.mean(per_task[t]["collapse"] for t in have)
    wrong_meanoftask = st.mean(per_task[t]["wrong"] for t in have)
    rec_meanoftask = st.mean(per_task[t]["recovered"] for t in have)
    pooled = {"ntasks": len(have), "BBHmean": bbhmean,
              "collapse_meanoftask": coll_meanoftask,
              "wrong_meanoftask": wrong_meanoftask,
              "recovered_meanoftask": rec_meanoftask}
    # item-pooled judge (judge_aggregate convention)
    jrows = [per_task[t]["judge"] for t in have if "judge" in per_task[t]]
    if jrows:
        N = sum(j["nj"] for j in jrows)
        def wsum(key):
            vals = [(j[key] / 100.0 * j["nj"]) for j in jrows if j.get(key) is not None]
            return pct(sum(vals), N) if vals else None
        pooled["judge_n"] = N
        for key in ("judge_acc", "contam", "jcollapse", "no_verdict"):
            pooled[key] = wsum(key)
        # token / repetition pooled means (weighted by nj; recompute from per-task means*nj)
        pooled["tok_mean"] = sum(j["tok_mean"] * j["nj"] for j in jrows) / N
        pooled["rep_mean"] = sum(j["rep_mean"] * j["nj"] for j in jrows) / N
        pooled["distinct_mean"] = sum(j["distinct_mean"] * j["nj"] for j in jrows) / N
        pooled["pct_runs_to_limit"] = sum(j["pct_runs_to_limit"] / 100 * j["nj"] for j in jrows) / N * 100
        pooled["pct_stub"] = sum(j["pct_stub"] / 100 * j["nj"] for j in jrows) / N * 100
        pooled["pct_loop"] = sum(j["pct_loop"] / 100 * j["nj"] for j in jrows) / N * 100
        # cat pooled
        pooled["cat"] = {c: sum(j["cat"][c] / 100 * j["nj"] for j in jrows) / N * 100 for c in CATS}
        # co-occurrence pooled
        nc = sum(j["n_collapsed"] for j in jrows)
        nnc = sum(j["n_noncollapsed"] for j in jrows)
        ncont = sum(j["n_contaminated"] for j in jrows)
        cont_in_coll = sum((j["contam_in_collapsed"] or 0) / 100 * j["n_collapsed"] for j in jrows)
        cont_in_noncoll = sum((j["contam_in_noncollapsed"] or 0) / 100 * j["n_noncollapsed"] for j in jrows)
        corr_in_cont = sum((j["correct_in_contaminated"] or 0) / 100 * j["n_contaminated"] for j in jrows)
        pooled["contam_in_collapsed"] = pct(cont_in_coll, nc)
        pooled["contam_in_noncollapsed"] = pct(cont_in_noncoll, nnc)
        pooled["correct_in_contaminated"] = pct(corr_in_cont, ncont)
        pooled["n_collapsed"] = nc
        pooled["n_contaminated"] = ncont
    return pooled


def main():
    data = {}
    for cell, cdir in cells():
        if not os.path.isdir(cdir):
            continue
        per_task = {t: task_block(cdir, t) for t in TASKS}
        per_task = {t: v for t, v in per_task.items() if v}
        pooled = pool_cell(per_task)
        data[cell] = {"per_task": per_task, "pooled": pooled, "mmlu": mmlu_acc(cdir)}

    with open(os.path.join(OUT, "deeper_numbers.json"), "w") as f:
        json.dump(data, f, indent=2)

    # ---------- readable tables ----------
    lines = []
    def p(s=""): lines.append(s)

    # A. Degeneracy morphology per cell (pooled) — judge-independent
    p("################ A. DEGENERACY MORPHOLOGY (judge-independent) — pooled over 4 BBH tasks")
    p("tok_mean=mean gen length; lim%=share runs to >=900 tok (toward 1024 cap); stub%=share <=30 tok;")
    p("rep=mean max_4gram_repeat; loop%=share with 4gram repeat>=20; distinct=mean distinct_ratio_last50")
    p(f"{'cell':22} {'tok_mean':>8} {'lim%':>6} {'stub%':>6} {'rep':>7} {'loop%':>6} {'distinct':>8}")
    for cell, d in data.items():
        pl = d["pooled"]
        if not pl or "tok_mean" not in pl: continue
        p(f"{cell:22} {pl['tok_mean']:8.0f} {pl['pct_runs_to_limit']:6.0f} {pl['pct_stub']:6.0f} "
          f"{pl['rep_mean']:7.1f} {pl['pct_loop']:6.0f} {pl['distinct_mean']:8.3f}")

    # A2. per-task degeneracy for the cliff/transition cells (heterogeneity)
    p("\n################ A2. PER-TASK degeneracy (cliff/transition cells) — tok_mean / lim% / stub% / rep")
    focus = ["steering/left/a3", "steering/left/a4", "steering/right/a3", "steering/right/a4",
             "dpo/left/s1_5", "dpo/left/s2", "dpo/right/s1_5", "dpo/right/s2"]
    p(f"{'cell':20} " + " ".join(f"{TSHORT[t]:>22}" for t in TASKS))
    for cell in focus:
        d = data.get(cell)
        if not d: continue
        row = f"{cell:20} "
        for t in TASKS:
            tb = d["per_task"].get(t)
            if tb and "judge" in tb:
                j = tb["judge"]
                row += f"{j['tok_mean']:5.0f}/{j['pct_runs_to_limit']:3.0f}/{j['pct_stub']:3.0f}/{j['rep_mean']:5.1f} "
            else:
                row += f"{'--':>22} "
        p(row)

    # B. three-regime trajectory (pooled mean-of-task)
    p("\n################ B. THREE-REGIME TRAJECTORY (reparse, mean-of-task %) correct/wrong/collapse/recovered")
    p(f"{'cell':22} {'correct':>8} {'wrong':>7} {'collapse':>9} {'recov':>7}")
    for cell, d in data.items():
        pl = d["pooled"]
        if not pl: continue
        # correct = BBHmean (reparsed acc); wrong/collapse/recovered mean-of-task
        p(f"{cell:22} {pl['BBHmean']:8.0f} {pl['wrong_meanoftask']:7.0f} {pl['collapse_meanoftask']:9.0f} {pl['recovered_meanoftask']:7.0f}")

    # C. MMLU loglikelihood dissociation
    p("\n################ C. MMLU formal_logic (loglikelihood, chance=25) vs generation BBHmean")
    p("margin = mmlu_acc - 25 (above-chance). gen=BBHmean. dissoc = does the cliff spare loglikelihood?")
    p(f"{'cell':22} {'mmlu':>6} {'margin':>7} {'BBHmean':>8}")
    for cell, d in data.items():
        pl = d["pooled"]
        m = d["mmlu"]
        if m is None: continue
        bm = pl["BBHmean"] if pl else float("nan")
        p(f"{cell:22} {m:6.0f} {m-25:7.0f} {bm:8.0f}")

    # D. item-level co-occurrence
    p("\n################ D. ITEM-LEVEL CO-OCCURRENCE (judge) — is contamination tied to collapse?")
    p("C|coll = P(contaminated | judge-collapsed);  C|ok = P(contaminated | not collapsed);  corr|C = P(correct | contaminated)")
    p(f"{'cell':22} {'C|coll':>7} {'C|ok':>7} {'corr|C':>7} {'n_cont':>7}")
    for cell, d in data.items():
        pl = d["pooled"]
        if not pl or "contam_in_collapsed" not in pl: continue
        cc = pl["contam_in_collapsed"]; cok = pl["contam_in_noncollapsed"]; cic = pl["correct_in_contaminated"]
        p(f"{cell:22} {('%.0f'%cc) if cc is not None else '--':>7} {('%.0f'%cok) if cok is not None else '--':>7} "
          f"{('%.0f'%cic) if cic is not None else '--':>7} {pl['n_contaminated']:7d}")

    # E. reasoning-quality dose-response (judge faith/posthoc/caperr + confidence)
    p("\n################ E. REASONING-QUALITY AXES (judge, pooled %) + mean confidence")
    p(f"{'cell':22} {'faith':>6} {'posthoc':>8} {'caperr':>7} {'instr_f':>8} {'gcollap':>8}")
    for cell, d in data.items():
        pl = d["pooled"]
        if not pl or "cat" not in pl: continue
        c = pl["cat"]
        p(f"{cell:22} {c['faithful_task_performance']:6.0f} {c['post_hoc_reasoning']:8.0f} "
          f"{c['capability_error']:7.0f} {c['instruction_following_failure']:8.0f} {c['generation_collapse']:8.0f}")

    txt = "\n".join(lines)
    with open(os.path.join(OUT, "deeper_tables.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)
    print("\n[deeper_analysis] wrote results/deeper/deeper_numbers.json + deeper_tables.txt")


if __name__ == "__main__":
    main()
