#!/usr/bin/env python3
"""Robust, task-aware answer re-extraction for BBH CoT samples.

The lm-eval harness used a single brittle filter `(?<=the answer is )(.*)(?=.)`:
case-sensitive, lowercase-only, fixed-string, and it grabs trailing text. The
RQ1/RQ2 failure diagnoses showed this silently discards correct answers whenever
a model drifts off the canonical "the answer is X" closer (capitalised "The
answer is", "the argument is invalid", "the final answer is", "(A) Joe finished
first", etc.). This module re-extracts the model's *stated* final answer in a
task-aware, conservative way (returns None when genuinely no answer is present —
truncation/collapse are NOT credited) and re-scores against the gold target.

Self-contained except for the valid/invalid task, which
reuses the battle-tested G&K cascade.
"""
import re, sys, os, json, glob

# reuse the G&K valid/invalid cascade
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "1_benchmarking", "G_K_assessing_bias"))
from gk_extract import label_from_raw  # VALID/INVALID/UNMAPPABLE

TASK_SPACE = {
    "bbh_cot_fewshot_boolean_expressions": "bool",
    "bbh_cot_fewshot_formal_fallacies": "validity",
    "bbh_cot_fewshot_web_of_lies": "yesno",
    "bbh_cot_fewshot_navigate": "yesno",
    "bbh_cot_fewshot_logical_deduction_three_objects": "mc",
    # politicized LD3 variants (RQ2)
    "bbh_pol_logical_deduction_three_objects": "mc",
}

# answer-declaration lead-ins (case-insensitive). LAST match wins.
LEADIN = r"(?:the\s+)?(?:final\s+|correct\s+|right\s+)?answer\s*(?:is|:|=|would be|should be)\s*"

def _last(pattern, text, flags=re.I):
    m = None
    for m in re.finditer(pattern, text, flags):
        pass
    return m

def extract_bool(raw):
    # "the answer is True", "= True", "evaluates to False", final "True."
    for pat in [LEADIN + r"\*{0,2}\(?(true|false)\)?",
                r"(?:evaluates to|equals|=|is)\s*\*{0,2}(true|false)\b",
                r"\b(true|false)\b\s*[.*]*\s*$"]:
        m = _last(pat, raw)
        if m: return m.group(1).lower()
    return None

def extract_yesno(raw):
    for pat in [LEADIN + r"\*{0,2}(yes|no)\b",
                r"\b(?:so|therefore|thus|hence)\s*,?\s*(?:the answer is\s*)?\*{0,2}(yes|no)\b",
                r"\b(yes|no)\b\s*[.*]*\s*$"]:
        m = _last(pat, raw)
        if m: return m.group(1).lower()
    return None

def extract_mc(raw):
    # (A)/(B)/... — "answer is (A)", "option (A)", "(A)", trailing letter
    for pat in [LEADIN + r"\*{0,2}\(?([a-g])\)?",
                r"option\s*\(?([a-g])\)?",
                r"\(([a-g])\)\s*[.*]*\s*$",
                r"\(([a-g])\)"]:
        m = _last(pat, raw)
        if m: return "(" + m.group(1).upper() + ")"
    return None

def extract_validity(raw):
    lab = label_from_raw(raw)
    if lab == "VALID": return "valid"
    if lab == "INVALID": return "invalid"
    return None

EXTRACT = {"bool": extract_bool, "yesno": extract_yesno, "mc": extract_mc, "validity": extract_validity}

def norm(s):
    if s is None: return None
    s = str(s).strip().lower().strip(".*) (")
    return s

def reparse_answer(raw, task):
    space = TASK_SPACE.get(task)
    if space is None: return None
    return EXTRACT[space](raw or "")

def gold_norm(target, task):
    space = TASK_SPACE.get(task)
    t = str(target).strip()
    if space == "mc":
        m = re.search(r"[a-gA-G]", t)
        return m.group(0).lower() if m else t.lower()
    return t.lower().strip(".*) (")

def pred_norm(pred, task):
    if pred is None: return None
    space = TASK_SPACE.get(task)
    if space == "mc":
        m = re.search(r"[a-gA-G]", pred)
        return m.group(0).lower() if m else pred.lower()
    return pred.lower().strip(".*) (")

def score_file(path, task):
    """returns (n, orig_acc, reparsed_acc, recovered, lost)"""
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    orig_c = reparse_c = 0
    recovered = lost = 0
    for d in rows:
        raw = d["resps"][0][0] if d.get("resps") else ""
        gold = gold_norm(d["target"], task)
        orig_ok = float(d.get("exact_match", 0.0)) == 1.0
        pred = pred_norm(reparse_answer(raw, task), task)
        rep_ok = (pred is not None and pred == gold)
        orig_c += orig_ok
        reparse_c += rep_ok
        if rep_ok and not orig_ok: recovered += 1
        if orig_ok and not rep_ok: lost += 1
    return n, orig_c / n, reparse_c / n, recovered, lost

if __name__ == "__main__":
    # validation harness on calibration cells
    import argparse
    base = os.path.dirname(__file__)
    cells = sys.argv[1:] if len(sys.argv) > 1 else [
        "llama/DPO/right", "llama/base", "mistral/DPO/left", "mistral/DPO/right"]
    for cell in cells:
        for task in TASK_SPACE:
            p = os.path.join(base, cell, task, "samples.jsonl")
            if not os.path.exists(p): continue
            n, o, r, rec, lost = score_file(p, task)
            flag = "  <-- " + ("OVER?" if lost > rec else "") if abs(r-o) > 0.02 else ""
            print(f"{cell:20} {task[16:]:30} orig={o:.3f} reparsed={r:.3f} (+{rec}/-{lost}){flag}")
