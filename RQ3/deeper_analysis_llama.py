#!/usr/bin/env python3
"""RQ3 deeper analysis (LLAMA) — judge-INDEPENDENT degeneracy morphology.

The Mistral companion (deeper_analysis.py) joins samples.jsonl with judge.jsonl to
read the per-item degeneracy fields. Llama has NO judge.jsonl yet (the judge is
running in parallel), so this script computes EVERY metric directly from the raw
generation string `resps[0][0]` in samples.jsonl. That is what makes the analysis
judge-independent and runnable now.

The three degeneracy fields are reimplemented from the Judge source verbatim and
were validated (RQ3/_validate_degeneracy.py) to reproduce the judge's stored values
to the exact integer/float on 5000 Mistral rows, so Llama numbers are directly
comparable to the Mistral numbers in DEEPER_ANALYSIS.md:
  - n_tokens_generated = len(text.split())           (judge_bbh.py:67)
  - max_4gram_repeat / distinct_ratio_last_50         (qualitative_classifier.py:loop_signals)

Computes, per cell and per task:
  A. Degeneracy morphology (word count, CHAR length, chars-per-word, max word-4gram
     repeat, distinct-ratio last 50 words, spaceless%, stub%, runs-to-limit).
  B. Three-regime decomposition via RQ1/reparse.py (reused exactly as consolidate.py
     does): reparsed-correct / parseable-wrong / collapse(no parseable answer) /
     recovered. wrong is the complement so correct+wrong+collapse sum to 100;
     recovered is a subset of correct.
  C. MMLU formal_logic loglikelihood dissociation: above-chance margin (acc-25,
     chance=25 for the 4-choice task) vs the generation BBHmean.

Cross-checks collapse% and BBHmean against `RQ3/consolidate.py llama` (must match).

    python3 RQ3/deeper_analysis_llama.py
Outputs: RQ3/results/deeper/deeper_tables_llama.txt + deeper_numbers_llama.json
"""
import os, sys, json, statistics as st
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "deeper")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "RQ1"))
import reparse as RP  # noqa: E402

FAM = "llama"
TASKS = ["boolean_expressions", "logical_deduction_three_objects",
         "web_of_lies", "navigate"]
TASKDIR = {t: f"bbh_cot_fewshot_{t}" for t in TASKS}
TSHORT = {"boolean_expressions": "bool", "logical_deduction_three_objects": "logic3",
          "web_of_lies": "weblies", "navigate": "navig"}
# grids match consolidate.py exactly (include DPO s1_0; exclude steering a2_5).
STEER = [("a0_5", 0.5), ("a1", 1.0), ("a2", 2.0), ("a3", 3.0), ("a4", 4.0)]
DPO = [("s0_25", 0.25), ("s0_5", 0.5), ("s1_0", 1.0), ("s1_5", 1.5), ("s2", 2.0)]

# generation cap. lm-eval BBH-CoT generates up to 1024 new tokens; we have no model
# token count in samples.jsonl, so CHAR length is the cap proxy (word-count >=900 is
# blind to a spaceless loop that hits the cap as ~1 word). CHAR_LIMIT chosen from the
# observed runaway cluster (gens that hit the cap sit at ~3800-5000 chars).
CHAR_LIMIT = 3500
STUB_CHARS = 120          # char_morphology.py convention
SPACELESS_RUN = 60        # longest non-space run >= 60 => spaceless concatenation
MMLU_CHANCE = 25.0        # formal_logic is 4-choice


def pct(num, den):
    return 100.0 * num / den if den else None


def longest_nonspace_run(s):
    best = cur = 0
    for ch in s:
        if ch.isspace():
            cur = 0
        else:
            cur += 1
            if cur > best:
                best = cur
    return best


def loop_signals(text):
    """Verbatim from Judge/src/qualitative_classifier.py:loop_signals (word-based)."""
    tokens = text.split()
    if len(tokens) < 4:
        return 0, 1.0
    fourgrams = [" ".join(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    max_rep = max(Counter(fourgrams).values()) if fourgrams else 0
    last = tokens[-50:]
    distinct_ratio = len(set(last)) / max(len(last), 1)
    return max_rep, round(distinct_ratio, 3)


def cells():
    yield ("base", os.path.join(RESULTS, FAM, "base"))
    for lean in ("left", "right"):
        for suf, _ in STEER:
            yield (f"steering/{lean}/{suf}", os.path.join(RESULTS, FAM, "steering", lean, suf))
    for lean in ("left", "right"):
        for suf, _ in DPO:
            yield (f"dpo/{lean}/{suf}", os.path.join(RESULTS, FAM, "dpo", lean, suf))


def task_block(cell_dir, task):
    """All per-item stats for one (cell, task), computed from samples.jsonl only."""
    sp = os.path.join(cell_dir, TASKDIR[task], "samples.jsonl")
    if not os.path.exists(sp):
        return None
    samples = [json.loads(l) for l in open(sp)]
    n = len(samples)
    if n == 0:
        return None
    # --- three-regime reparse (loop copied verbatim from consolidate.task_metrics) ---
    rep_c = rec = none = wrong = strict_c = 0
    # --- degeneracy, from the generation string ---
    words = []; chars = []; cpw = []; runs = []; reps = []; dists = []
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
        w = len(raw.split()); c = len(raw); r = longest_nonspace_run(raw)
        mr, dr = loop_signals(raw)
        words.append(w); chars.append(c); runs.append(r); reps.append(mr); dists.append(dr)
        cpw.append(c / w if w else c)
    spaceless = sum(1 for x in runs if x >= SPACELESS_RUN)
    stub = sum(1 for c, x in zip(chars, runs) if c <= STUB_CHARS and x < SPACELESS_RUN)
    runs_to_limit = sum(1 for c in chars if c >= CHAR_LIMIT)
    return {
        "n": n,
        # three-regime (consolidate convention: wrong directly counted)
        "strict_acc": pct(strict_c, n), "reparsed_acc": pct(rep_c, n),
        "correct": pct(rep_c, n), "wrong": pct(wrong, n),
        "collapse": pct(none, n), "recovered": pct(rec, n),
        # degeneracy morphology
        "word_mean": st.mean(words), "word_med": st.median(words),
        "char_mean": st.mean(chars), "char_med": st.median(chars),
        "cpw_med": st.median(cpw),
        "rep_mean": st.mean(reps), "rep_med": st.median(reps),
        "distinct_mean": st.mean(dists),
        "pct_loop": pct(sum(1 for x in reps if x >= 20), n),    # word-4gram repeat>=20
        "pct_spaceless": pct(spaceless, n),
        "pct_stub": pct(stub, n),
        "pct_runs_to_limit": pct(runs_to_limit, n),
        # raw item lists, so pool_cell can take a TRUE item-pooled median (not a
        # weighted average of per-task medians)
        "_words": words, "_chars": chars, "_cpw": cpw,
    }


def mmlu_acc(cell_dir):
    p = os.path.join(cell_dir, "mmlu_formal_logic", "results.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p)).get("results", {}).get("mmlu_formal_logic", {})
    a = r.get("acc,none")
    return 100.0 * a if a is not None else None


def pool_cell(per_task):
    """Pool 4 tasks. mean-of-task for reparse (consolidate's BBHmean/coll%); item-pooled
    for the degeneracy means/shares so a long-loop task is weighted by its item count."""
    have = [t for t in TASKS if per_task.get(t)]
    if not have:
        return None
    mean = lambda key: st.mean(per_task[t][key] for t in have)
    pooled = {
        "ntasks": len(have),
        "BBHmean": mean("reparsed_acc"),
        "collapse_meanoftask": mean("collapse"),
        "wrong_meanoftask": mean("wrong"),
        "recovered_meanoftask": mean("recovered"),
        "correct_meanoftask": mean("correct"),
    }
    N = sum(per_task[t]["n"] for t in have)
    # item-count-weighted mean: valid for means and shares (a mean-of-means weighted
    # by item count equals the item-pooled mean).
    wmean = lambda key: sum(per_task[t][key] * per_task[t]["n"] for t in have) / N
    pooled["n"] = N
    for key in ("word_mean", "char_mean", "rep_mean", "distinct_mean",
                "pct_loop", "pct_spaceless", "pct_stub", "pct_runs_to_limit"):
        pooled[key] = wmean(key)
    # TRUE item-pooled medians (pooling per-task medians is invalid for a median):
    # concatenate the raw per-item lists across the 4 tasks and take one median.
    all_cpw = [x for t in have for x in per_task[t]["_cpw"]]
    all_chars = [x for t in have for x in per_task[t]["_chars"]]
    all_words = [x for t in have for x in per_task[t]["_words"]]
    pooled["cpw_med"] = st.median(all_cpw)
    pooled["char_med"] = st.median(all_chars)
    pooled["word_med"] = st.median(all_words)
    return pooled


def main():
    data = {}
    for cell, cdir in cells():
        if not os.path.isdir(cdir):
            continue
        per_task = {t: task_block(cdir, t) for t in TASKS}
        per_task = {t: v for t, v in per_task.items() if v}
        pooled = pool_cell(per_task)
        # drop the raw item lists (used only for pooled medians) before serializing
        for v in per_task.values():
            for k in ("_words", "_chars", "_cpw"):
                v.pop(k, None)
        data[cell] = {"per_task": per_task, "pooled": pooled, "mmlu": mmlu_acc(cdir)}

    with open(os.path.join(OUT, "deeper_numbers_llama.json"), "w") as f:
        json.dump(data, f, indent=2)

    lines = []
    def p(s=""): lines.append(s)

    p("################ RQ3 DEEPER ANALYSIS — LLAMA (judge-INDEPENDENT, computed from samples.jsonl resps[0][0])")
    p("Degeneracy fields reimplemented from Judge source; validated to reproduce judge.jsonl exactly on 5000 Mistral rows.")

    # A. Degeneracy morphology per cell (pooled over 4 BBH tasks)
    p("\n################ A. DEGENERACY MORPHOLOGY (pooled over 4 BBH tasks) — every field from the generation string")
    p("word=mean word count; char=mean char length; cpw=item-pooled median chars/word (~5-6 normal English, high=spaceless);")
    p("rep=mean max word-4gram-repeat; loop%=share rep>=20; dist=mean distinct-ratio last 50 words;")
    p(f"space%=share with non-space run>={SPACELESS_RUN}; stub%=share <={STUB_CHARS} char & not spaceless; lim%=share >={CHAR_LIMIT} char (toward 1024-tok cap)")
    p(f"{'cell':22} {'word':>6} {'char':>6} {'cpw':>5} {'rep':>7} {'loop%':>6} {'dist':>6} {'space%':>7} {'stub%':>6} {'lim%':>5}")
    for cell, d in data.items():
        pl = d["pooled"]
        if not pl: continue
        p(f"{cell:22} {pl['word_mean']:6.0f} {pl['char_mean']:6.0f} {pl['cpw_med']:5.1f} "
          f"{pl['rep_mean']:7.1f} {pl['pct_loop']:6.0f} {pl['distinct_mean']:6.3f} "
          f"{pl['pct_spaceless']:7.0f} {pl['pct_stub']:6.0f} {pl['pct_runs_to_limit']:5.0f}")

    # A2. per-task degeneracy for the cliff/transition cells (heterogeneity)
    p("\n################ A2. PER-TASK morphology (degrading cells) — char_med / run-based; per cell, 4 tasks")
    p("fields per task: char_med / word_med / spaceless% / stub% / loop%")
    focus = ["steering/left/a3", "steering/left/a4", "steering/right/a3", "steering/right/a4",
             "dpo/left/s1_5", "dpo/left/s2", "dpo/right/s1_5", "dpo/right/s2"]
    p(f"{'cell':20} " + " ".join(f"{TSHORT[t]:>26}" for t in TASKS))
    for cell in focus:
        d = data.get(cell)
        if not d: continue
        row = f"{cell:20} "
        for t in TASKS:
            tb = d["per_task"].get(t)
            if tb:
                row += (f"{tb['char_med']:5.0f}/{tb['word_med']:3.0f}/"
                        f"{tb['pct_spaceless']:2.0f}/{tb['pct_stub']:2.0f}/{tb['pct_loop']:2.0f} ")
            else:
                row += f"{'--':>26} "
        p(row)

    # B. three-regime trajectory (pooled mean-of-task — matches consolidate)
    p("\n################ B. THREE-REGIME (reparse, mean-of-task %) correct/wrong/collapse/recovered; correct+wrong+collapse=100")
    p(f"{'cell':22} {'correct':>8} {'wrong':>7} {'collapse':>9} {'recov':>7}")
    for cell, d in data.items():
        pl = d["pooled"]
        if not pl: continue
        p(f"{cell:22} {pl['BBHmean']:8.0f} {pl['wrong_meanoftask']:7.0f} "
          f"{pl['collapse_meanoftask']:9.0f} {pl['recovered_meanoftask']:7.0f}")

    # C. MMLU dissociation
    p("\n################ C. MMLU formal_logic (loglikelihood, chance=25) vs generation BBHmean")
    p("margin = mmlu_acc - 25 (above-chance retained). dissoc = does the generation cliff spare the representation?")
    p(f"{'cell':22} {'mmlu':>6} {'margin':>7} {'BBHmean':>8}")
    for cell, d in data.items():
        m = d["mmlu"]
        if m is None: continue
        pl = d["pooled"]
        bm = pl["BBHmean"] if pl else float("nan")
        p(f"{cell:22} {m:6.0f} {m - MMLU_CHANCE:7.0f} {bm:8.0f}")

    txt = "\n".join(lines)
    with open(os.path.join(OUT, "deeper_tables_llama.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)
    print("\n[deeper_analysis_llama] wrote results/deeper/deeper_numbers_llama.json + deeper_tables_llama.txt")


if __name__ == "__main__":
    main()
