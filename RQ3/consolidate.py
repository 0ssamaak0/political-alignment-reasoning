#!/usr/bin/env python3
"""RQ3 dose-response consolidation.

For each alignment method (steering, DPO) and lean, walk the strength grid and
compute, per BBH-CoT task: robust **reparsed accuracy** and the failure
decomposition **collapse / wrong / recovered** (reparse returns None = collapse;
parseable-but-wrong = wrong; reparse fixes a strict miss = recovered). MMLU is
loglikelihood (acc from results.json, no reparse). Each strength point is
annotated with the **coherence** from the trait/coherence sweeps (the gate).

Reuses RQ1/reparse.py for the task-aware extraction. Reads
RQ3/results/<fam>/<method>/<lean>/<strength>/<task>/samples.jsonl.

    python3 RQ3/consolidate.py            # all families
    python3 RQ3/consolidate.py mistral    # one family
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
sys.path.insert(0, os.path.join(REPO, "RQ1"))
import reparse as RP  # noqa: E402

BBH = ["bbh_cot_fewshot_boolean_expressions",
       "bbh_cot_fewshot_logical_deduction_three_objects",
       "bbh_cot_fewshot_web_of_lies",
       "bbh_cot_fewshot_navigate"]
SHORT = {t: t[len("bbh_cot_fewshot_"):] for t in BBH}

# strength grids: (dir-suffix, numeric strength). base = strength 0 (shared).
STEER_GRID = [("a0_5", 0.5), ("a1", 1.0), ("a2", 2.0), ("a3", 3.0), ("a4", 4.0)]
DPO_GRID = [("s0_25", 0.25), ("s0_5", 0.5), ("s1_0", 1.0), ("s1_5", 1.5), ("s2", 2.0)]


def task_metrics(cell_dir, task):
    p = os.path.join(cell_dir, task, "samples.jsonl")
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p)]
    n = len(rows)
    if n == 0:
        return None
    rep_c = rec = none = wrong = 0
    for d in rows:
        raw = d["resps"][0][0] if d.get("resps") else ""
        gold = RP.gold_norm(d["target"], task)
        orig_ok = float(d.get("exact_match", 0.0)) == 1.0
        pred = RP.pred_norm(RP.reparse_answer(raw, task), task)
        rep_ok = (pred is not None and pred == gold)
        rep_c += rep_ok
        if not orig_ok:
            if pred is None:
                none += 1
            elif rep_ok:
                rec += 1
            else:
                wrong += 1
    return {"n": n, "racc": 100 * rep_c / n,
            "collapse": 100 * none / n, "wrong": 100 * wrong / n, "rec": 100 * rec / n}


def mmlu_acc(cell_dir):
    p = os.path.join(cell_dir, "mmlu_formal_logic", "results.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p)).get("results", {}).get("mmlu_formal_logic", {})
    a = r.get("acc,none")
    return 100 * a if a is not None else None


def coherence_map(fam):
    """strength -> coherence, for steering (coef_sweep) and dpo (scale_sweep)."""
    out = {"steering": {}, "dpo": {}}
    for lean in ("left", "right"):
        # steering: per_coef[alpha].coh_scores (list)
        sp = os.path.join(RESULTS, "trait_coherence", "steering", f"sweep_{fam}_{lean}.json")
        if os.path.exists(sp):
            pc = json.load(open(sp)).get("per_coef", {})
            for a, e in pc.items():
                cs = [c for c in (e.get("coh_scores") or []) if c is not None]
                if cs:
                    out["steering"].setdefault(lean, {})[float(a)] = sum(cs) / len(cs)
        # dpo: per_scale[scale].coh_mean (+ the s0_25 add-on file)
        for fn in (f"sweep_{fam}_{lean}.json", f"sweep_{fam}_{lean}_s0_25.json"):
            dp = os.path.join(RESULTS, "trait_coherence", "dpo", fn)
            if os.path.exists(dp):
                ps = json.load(open(dp)).get("per_scale", {})
                for s, e in ps.items():
                    if e.get("coh_mean") is not None:
                        out["dpo"].setdefault(lean, {})[float(s)] = e["coh_mean"]
    return out


def fmt(x, w=5):
    return ("%.0f" % x).rjust(w) if isinstance(x, (int, float)) else "  -- ".rjust(w)


def dose_response(fam, method, lean, grid, coh):
    base_dir = os.path.join(RESULTS, fam, "base")
    rows = [("0(base)", 0.0, base_dir)]
    for suf, val in grid:
        rows.append((suf, val, os.path.join(RESULTS, fam, method, lean, suf)))

    print(f"\n=== {fam}  {method}  {lean} ===")
    hdr = ["strength", "coh"] + [SHORT[t][:9] for t in BBH] + ["mmlu", "BBHmean", "collapse%"]
    print("  ".join(h.rjust(9) for h in hdr))
    for name, val, cdir in rows:
        if not os.path.isdir(cdir):
            continue
        coh_v = coh.get(method, {}).get(lean, {}).get(val)
        # base coherence ~71 from the sweeps (val 0 not in per_coef/per_scale)
        if coh_v is None and val == 0.0:
            coh_v = 71.0
        raccs, collapses = [], []
        cells = {}
        for t in BBH:
            m = task_metrics(cdir, t)
            cells[t] = m
            if m:
                raccs.append(m["racc"])
                collapses.append(m["collapse"])
        mm = mmlu_acc(cdir)
        bbhmean = sum(raccs) / len(raccs) if raccs else None
        collmean = sum(collapses) / len(collapses) if collapses else None
        line = [name.rjust(9), fmt(coh_v)]
        for t in BBH:
            line.append(fmt(cells[t]["racc"]) if cells[t] else "  -- ")
        line.append(fmt(mm))
        line.append(fmt(bbhmean))
        line.append(fmt(collmean))
        print("  ".join(line))


def main():
    fams = sys.argv[1:] or ["mistral", "llama"]
    for fam in fams:
        if not os.path.isdir(os.path.join(RESULTS, fam)):
            continue
        coh = coherence_map(fam)
        for lean in ("left", "right"):
            if os.path.isdir(os.path.join(RESULTS, fam, "steering", lean)):
                dose_response(fam, "steering", lean, STEER_GRID, coh)
        for lean in ("left", "right"):
            if os.path.isdir(os.path.join(RESULTS, fam, "dpo", lean)):
                dose_response(fam, "dpo", lean, DPO_GRID, coh)


if __name__ == "__main__":
    main()
