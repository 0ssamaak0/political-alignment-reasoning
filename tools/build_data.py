#!/usr/bin/env python3
"""
build_data.py — preprocess AIMS_project judge outputs into a static dataset for the
GitHub Pages dashboard.

It reads the raw (messy) judge files under RQ1/, RQ2/, RQ3/ and emits clean, compact
JSON under docs/data/. All the naming reconciliation lives here, so the UI stays
faithful to the thesis:

  - folder / cell strings  (steering/left_a2_5, mistral-dpo-right-s1_0, ...)  -> paper config labels
  - motivational_framing_bias                                                -> "Editorial framing bias"
  - strength tokens        (a0_5, s0_25)                                      -> numeric values for the slider

Run from anywhere:  python3 tools/build_data.py
Output:             docs/data/manifest.json
                    docs/data/examples/<rq>__<model>__<config>__<benchmark>.json
                    docs/data/rq2_aggregate/party_fixed.json
                    docs/data/rq3/<model>__<method>__<direction>.json
"""

import json
import os
import glob
import collections

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # AIMS_project/
OUT = os.path.join(ROOT, "docs", "data")
EX_DIR = os.path.join(OUT, "examples")
RQ2AGG_DIR = os.path.join(OUT, "rq2_aggregate")
RQ3_DIR = os.path.join(OUT, "rq3")
for d in (OUT, EX_DIR, RQ2AGG_DIR, RQ3_DIR):
    os.makedirs(d, exist_ok=True)

warnings = []
def warn(msg):
    warnings.append(msg)
    print("  [warn]", msg)


# --------------------------------------------------------------------------------------
# Faithful display vocabulary (paper terminology)
# --------------------------------------------------------------------------------------
MODELS = [
    {"id": "llama",   "label": "Llama-3-8B-Instruct"},
    {"id": "mistral", "label": "Mistral-7B-Instruct-v0.2"},
]

METHODS = [
    {"id": "base",     "label": "Base"},
    {"id": "roleplay", "label": "Roleplaying"},
    {"id": "steering", "label": "Activation steering"},
    {"id": "dpo",      "label": "DPO fine-tuning"},
]

# The seven primary categories. The raw key motivational_framing_bias is shown as the
# paper's name "Editorial framing bias".
PRIMARY_CATEGORY = [
    {"key": "faithful_task_performance",      "label": "Faithful task performance"},
    {"key": "post_hoc_reasoning",             "label": "Post-hoc reasoning"},
    {"key": "capability_error",               "label": "Capability error"},
    {"key": "instruction_following_failure",  "label": "Instruction-following failure"},
    {"key": "viewpoint_bias",                 "label": "Viewpoint bias"},
    {"key": "motivational_framing_bias",      "label": "Editorial framing bias"},
    {"key": "generation_collapse",            "label": "Generation collapse"},
]

OUTCOME = [
    {"key": "correct",    "label": "Correct"},
    {"key": "wrong",      "label": "Wrong"},
    {"key": "no_answer",  "label": "No answer"},
    {"key": "off_format", "label": "Off-format"},
]

REASONING_VALIDITY = [
    {"key": "valid",   "label": "Valid"},
    {"key": "invalid", "label": "Invalid"},
    {"key": "opaque",  "label": "Opaque"},
    {"key": "n/a",     "label": "N/A"},
]

FALLACY_LENS = [
    {"key": "none",                      "label": "None"},
    {"key": "equivocation",              "label": "Equivocation"},
    {"key": "false_dilemma",             "label": "False dilemma"},
    {"key": "token_bias_shortcut",       "label": "Token-bias shortcut"},
    {"key": "premise_truth_conflation",  "label": "Premise-truth conflation"},
    {"key": "illicit_premise_insertion", "label": "Illicit premise insertion"},
    {"key": "motivational_reasoning",    "label": "Motivational reasoning"},
]

CATEGORY_KEYS = [c["key"] for c in PRIMARY_CATEGORY]
VALIDITY_KEYS = [c["key"] for c in REASONING_VALIDITY]

# Config labels (paper). Steering deployed at alpha 2.5 (left) / 3.0 (right) in RQ1.
CONFIG_LABEL = {
    "base":          "Base",
    "roleplay-left": "Roleplay-L",
    "roleplay-right":"Roleplay-R",
    "steer-left":    "Steer-L (α2.5)",
    "steer-right":   "Steer-R (α3.0)",
    "dpo-left":      "DPO-L",
    "dpo-right":     "DPO-R",
}
CONFIG_META = {
    "base":          {"method": "base",     "direction": None},
    "roleplay-left": {"method": "roleplay", "direction": "left"},
    "roleplay-right":{"method": "roleplay", "direction": "right"},
    "steer-left":    {"method": "steering", "direction": "left"},
    "steer-right":   {"method": "steering", "direction": "right"},
    "dpo-left":      {"method": "dpo",      "direction": "left"},
    "dpo-right":     {"method": "dpo",      "direction": "right"},
}
# Canonical ordering for the UI
CONFIG_ORDER = ["base", "roleplay-left", "roleplay-right",
                "steer-left", "steer-right", "dpo-left", "dpo-right"]

BENCH_LABEL = {
    "boolean_expressions":             "Boolean expressions",
    "logical_deduction_three_objects": "Logical deduction (three objects)",
    "web_of_lies":                     "Web of lies",
    "navigate":                        "Navigate",
    "value_loaded":                    "Value-loaded benchmark",
    "party_fixed":                     "Party-fixed content-swap test",
}


def config_from_method_dir(method, direction):
    if method == "base":
        return "base"
    short = {"roleplay": "roleplay", "steering": "steer", "dpo": "dpo"}[method]
    return f"{short}-{direction}"


def parse_rq1_cell_path(rel):
    """'llama/DPO/right_2nd' -> ('llama', 'dpo-right'). Returns (model, config_id)."""
    parts = rel.split("/")
    model = parts[0]
    seg = parts[1].lower()
    if seg == "base":
        return model, "base"
    if seg == "roleplay":
        return model, config_from_method_dir("roleplay", parts[2])
    if seg == "dpo":
        direction = "left" if "left" in parts[2].lower() else "right"
        return model, config_from_method_dir("dpo", direction)
    if seg == "steering":
        direction = "left" if parts[2].lower().startswith("left") else "right"
        return model, config_from_method_dir("steering", direction)
    raise ValueError(f"unknown RQ1 cell path: {rel}")


def parse_rq2_cell_string(cell):
    """'mistral-DPO-left' / 'llama-steering-right' / 'llama-base' -> (model, config_id)."""
    c = cell.lower()
    model = "llama" if c.startswith("llama") else "mistral"
    if "base" in c:
        return model, "base"
    direction = "left" if "left" in c else "right"
    if "roleplay" in c:
        method = "roleplay"
    elif "steer" in c:
        method = "steering"
    elif "dpo" in c:
        method = "dpo"
    else:
        raise ValueError(f"unknown RQ2 cell string: {cell}")
    return model, config_from_method_dir(method, direction)


def parse_flipped_filename(fname):
    """metrics filename stem -> (model, config_id).
    e.g. 'mistral-pvsteer-ml-left-a2_5' -> ('mistral','steer-left'),
         'llama-politune-hf-right'      -> ('llama','dpo-right')."""
    stem = fname.lower()
    model = "llama" if stem.startswith("llama") else "mistral"
    if "base" in stem:
        return model, "base"
    direction = "left" if "left" in stem else "right"
    if "roleplay" in stem:
        method = "roleplay"
    elif "pvsteer" in stem or "steer" in stem:
        method = "steering"
    elif "politune" in stem or "dpo" in stem:
        method = "dpo"
    else:
        raise ValueError(f"unknown flipped filename: {fname}")
    return model, config_from_method_dir(method, direction)


# --------------------------------------------------------------------------------------
# Stats helper (computed straight from the records the dashboard will show)
# --------------------------------------------------------------------------------------
def pct(num, den):
    return round(100.0 * num / den, 1) if den else None


def fresh_acc():
    return {"n": 0, "correct": 0, "contaminated": 0, "collapsed": 0, "no_verdict": 0,
            "categories": {k: 0 for k in CATEGORY_KEYS},
            "validity": {k: 0 for k in VALIDITY_KEYS},
            "confidence": {str(i): 0 for i in range(1, 6)}}


def acc_add(acc, rec):
    acc["n"] += 1
    if rec["outcome"] == "correct":
        acc["correct"] += 1
    if rec["contaminated"]:
        acc["contaminated"] += 1
    if rec["collapsed"]:
        acc["collapsed"] += 1
    if rec["outcome"] in ("no_answer", "off_format"):
        acc["no_verdict"] += 1
    if rec["category"] in acc["categories"]:
        acc["categories"][rec["category"]] += 1
    v = rec.get("validity")
    if v in acc["validity"]:
        acc["validity"][v] += 1
    c = rec.get("confidence")
    if c is not None and str(c) in acc["confidence"]:
        acc["confidence"][str(c)] += 1


def acc_finalize(acc):
    n = acc["n"]
    return {
        "n": n,
        "judge_acc": pct(acc["correct"], n),
        "contam": pct(acc["contaminated"], n),
        "collapse": pct(acc["collapsed"], n),
        "no_verdict": pct(acc["no_verdict"], n),
        "categories": acc["categories"],
        "validity": acc["validity"],
        "confidence": acc["confidence"],
    }


# --------------------------------------------------------------------------------------
# RQ1 + RQ2 example browser
# --------------------------------------------------------------------------------------
def compact_rq1(r, task):
    return {
        "id": r.get("doc_id"),
        "task": task,
        "gold": r.get("gold"),
        "parsed": r.get("lm_eval_parsed"),
        "exact": int(r.get("lm_eval_exact_match") or 0),
        "outcome": r.get("outcome"),
        "contaminated": bool(r.get("contaminated")),
        "collapsed": bool(r.get("collapsed")),
        "validity": r.get("reasoning_validity"),
        "category": r.get("primary_category"),
        "fallacy": (r.get("fallacy_lens") or "none"),
        "confidence": r.get("confidence"),
        "lean": r.get("lean"),
        "ntok": r.get("n_tokens_generated"),
        "rep": r.get("max_4gram_repeat"),
        "distinct": r.get("distinct_ratio_last_50"),
        "prompt": r.get("text"),
        "response": r.get("raw_response"),
        "judge_reasoning": r.get("reasoning"),
        "justification": r.get("justification"),
    }


def compact_rq2(r):
    return {
        "id": r.get("gidx"),
        "item_lean": r.get("item_lean"),
        "valid": "valid" if r.get("gold_valid") else "invalid",
        "parsed_verdict": r.get("parsed_verdict"),
        "variation": r.get("variation"),
        "template_family": r.get("template_family"),
        "outcome": r.get("outcome"),
        "contaminated": bool(r.get("contaminated")),
        "collapsed": bool(r.get("collapsed")),
        "validity": r.get("reasoning_validity"),
        "category": r.get("primary_category"),
        "fallacy": (r.get("fallacy_lens") or "none"),
        "confidence": None,
        "engaged": bool(r.get("engaged")),
        "prompt": r.get("prompt"),
        "response": r.get("raw_response"),
        "judge_reasoning": r.get("judge_reasoning"),
        "justification": r.get("justification"),
    }


def build_examples():
    # examples[rq][model][config][bench] = {"file","n","stats"}
    examples = {"rq1": {}, "rq2": {}}
    # pooled[rq][model][config] = stats across benchmarks
    pooled = {"rq1": {}, "rq2": {}}
    pool_acc = {"rq1": {}, "rq2": {}}

    def write_cell(rq, model, config, bench, records):
        acc = fresh_acc()
        for rec in records:
            acc_add(acc, rec)
        stats = acc_finalize(acc)
        fname = f"{rq}__{model}__{config}__{bench}.json"
        with open(os.path.join(EX_DIR, fname), "w") as f:
            json.dump(records, f, separators=(",", ":"))
        examples[rq].setdefault(model, {}).setdefault(config, {})[bench] = {
            "file": f"examples/{fname}", "n": stats["n"], "stats": stats,
        }
        # accumulate pooled
        pa = pool_acc[rq].setdefault(model, {}).setdefault(config, fresh_acc())
        for rec in records:
            acc_add(pa, rec)

    # ---- RQ1: 4 BBH neutral tasks, 14 cells ----
    print("RQ1 (neutral BBH) ...")
    cell_paths = sorted({
        p[len("RQ1/"):].rsplit("/bbh_cot_fewshot_", 1)[0]
        for p in [os.path.relpath(x, ROOT) for x in
                  glob.glob(os.path.join(ROOT, "RQ1", "**", "judge.jsonl"), recursive=True)]
    })
    for rel in cell_paths:
        try:
            model, config = parse_rq1_cell_path(rel)
        except ValueError as e:
            warn(str(e)); continue
        cell_dir = os.path.join(ROOT, "RQ1", rel)
        for jpath in sorted(glob.glob(os.path.join(cell_dir, "bbh_cot_fewshot_*", "judge.jsonl"))):
            task = os.path.basename(os.path.dirname(jpath)).replace("bbh_cot_fewshot_", "")
            records = [compact_rq1(json.loads(l), task) for l in open(jpath) if l.strip()]
            write_cell("rq1", model, config, task, records)
        print(f"  {model:8} {config:14} <- {rel}")

    # ---- RQ2: value-loaded benchmark (judge_long.jsonl), 14 cells in one file ----
    print("RQ2 (value-loaded) ...")
    jl = os.path.join(ROOT, "RQ2", "G_K_assessing_bias", "patterns", "judge_long.jsonl")
    by_cell = collections.defaultdict(list)
    for l in open(jl):
        if not l.strip():
            continue
        r = json.loads(l)
        by_cell[r["cell"]].append(r)
    for cell, rows in sorted(by_cell.items()):
        try:
            model, config = parse_rq2_cell_string(cell)
        except ValueError as e:
            warn(str(e)); continue
        records = [compact_rq2(r) for r in rows]
        write_cell("rq2", model, config, "value_loaded", records)
        print(f"  {model:8} {config:14} <- {cell} ({len(records)})")

    # finalize pooled
    for rq in ("rq1", "rq2"):
        for model, cfgs in pool_acc[rq].items():
            for config, pa in cfgs.items():
                pooled[rq].setdefault(model, {})[config] = acc_finalize(pa)

    return examples, pooled


# --------------------------------------------------------------------------------------
# RQ2 party-fixed (flipped) aggregate-only
# --------------------------------------------------------------------------------------
def build_party_fixed():
    print("RQ2 (party-fixed, aggregate-only) ...")
    out = {}
    for fp in sorted(glob.glob(os.path.join(ROOT, "RQ2", "flipped", "metrics", "*.json"))):
        stem = os.path.splitext(os.path.basename(fp))[0]
        try:
            model, config = parse_flipped_filename(stem)
        except ValueError as e:
            warn(str(e)); continue
        d = json.load(open(fp))
        out.setdefault(model, {})[config] = {
            "config_label": CONFIG_LABEL[config],
            "cell_lean": d.get("cell_lean"),
            "overall": d.get("overall"),
            "by_arm_lean": d.get("by_arm_lean"),
            "signed_bias": d.get("signed_bias"),
            "net_political_belief_effect": d.get("net_political_belief_effect"),
        }
        print(f"  {model:8} {config:14} <- {stem}")
    with open(os.path.join(RQ2AGG_DIR, "party_fixed.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    return {"file": "rq2_aggregate/party_fixed.json", "models": sorted(out.keys())}


# --------------------------------------------------------------------------------------
# RQ3 strength explorer
# --------------------------------------------------------------------------------------
STEER_TOKEN = {"a0_5": 0.5, "a1": 1.0, "a2": 2.0, "a3": 3.0, "a4": 4.0, "a2_5": 2.5}
DPO_TOKEN = {"s0_25": 0.25, "s0_5": 0.5, "s1_0": 1.0, "s1_5": 1.5, "s2": 2.0}
DEPLOYED = {"steering": {"left": 2.5, "right": 3.0},
            "dpo": {"left": 1.0, "right": 1.0}}


def deeper_file(model):
    if model == "llama":
        return os.path.join(ROOT, "RQ3", "results", "deeper", "deeper_numbers_llama.json")
    return os.path.join(ROOT, "RQ3", "results", "deeper", "deeper_numbers.json")


def accuracy_points(model, method, direction):
    """strength(float) -> dict of BBH/judge metrics from deeper_numbers."""
    d = json.load(open(deeper_file(model)))
    pts = {}
    # base point (strength 0), direction-agnostic
    base = d.get("base", {}).get("pooled", {})
    if base:
        pts[0.0] = base
    tokmap = STEER_TOKEN if method == "steering" else DPO_TOKEN
    prefix = f"{method}/{direction}/"
    for key, cell in d.items():
        if not key.startswith(prefix):
            continue
        tok = key[len(prefix):]
        if tok not in tokmap:
            warn(f"RQ3 unknown strength token {tok} in {key}")
            continue
        pts[tokmap[tok]] = cell.get("pooled", {})
    return pts


def trait_points(model, method, direction):
    """strength(float) -> {trait, coherence, per_question?}."""
    pts = {}
    if method == "steering":
        fp = os.path.join(ROOT, "RQ3", "results", "trait_coherence", "steering",
                          f"sweep_{model}_{direction}.json")
        if not os.path.exists(fp):
            warn(f"missing steering sweep {fp}"); return pts
        d = json.load(open(fp))
        for s, v in d.get("per_coef", {}).items():
            pts[float(s)] = {"trait": v.get("trait_mean"), "coherence": v.get("coh_mean"),
                             "per_question": None}
    else:  # dpo
        main = os.path.join(ROOT, "RQ3", "results", "trait_coherence", "dpo",
                            f"sweep_{model}_{direction}.json")
        extra = os.path.join(ROOT, "RQ3", "results", "trait_coherence", "dpo",
                             f"sweep_{model}_{direction}_s0_25.json")
        files = [p for p in (main, extra) if os.path.exists(p)]
        for fp in files:
            d = json.load(open(fp))
            for s, v in d.get("per_scale", {}).items():
                pq = v.get("per_question")
                pq_out = None
                if pq:
                    pq_out = [{"question": q.get("question"), "response": q.get("response"),
                               "trait": q.get("trait_score"), "coherence": q.get("coh_score")}
                              for q in pq]
                pts[float(s)] = {"trait": v.get("trait_mean"), "coherence": v.get("coh_mean"),
                                 "per_question": pq_out}
        # base lean (strength 0) for the same base model comes from the steering sweep
        steer = os.path.join(ROOT, "RQ3", "results", "trait_coherence", "steering",
                             f"sweep_{model}_{direction}.json")
        if os.path.exists(steer):
            z = json.load(open(steer)).get("per_coef", {}).get("0.0")
            if z:
                pts[0.0] = {"trait": z.get("trait_mean"), "coherence": z.get("coh_mean"),
                            "per_question": None}
    return pts


def build_rq3():
    print("RQ3 (strength explorer) ...")
    index = {}
    for model in ("llama", "mistral"):
        for method in ("steering", "dpo"):
            for direction in ("left", "right"):
                accp = accuracy_points(model, method, direction)
                trap = trait_points(model, method, direction)
                strengths = sorted(set(accp) | set(trap))
                stops = []
                for s in strengths:
                    a = accp.get(s, {})
                    t = trap.get(s, {})
                    stops.append({
                        "strength": s,
                        "is_base": (s == 0.0),
                        "trait": t.get("trait"),
                        "coherence": t.get("coherence"),
                        "accuracy": a.get("BBHmean"),
                        "collapse": a.get("collapse_meanoftask"),
                        "judge_acc": a.get("judge_acc"),
                        "contam": a.get("contam"),
                        "no_verdict": a.get("no_verdict"),
                        "categories": a.get("cat"),
                        "per_question": t.get("per_question"),
                    })
                fname = f"{model}__{method}__{direction}.json"
                payload = {
                    "model": model, "method": method, "direction": direction,
                    "strength_field": "α" if method == "steering" else "s",
                    "deployed": DEPLOYED[method][direction],
                    "has_responses": any(st["per_question"] for st in stops),
                    "stops": stops,
                }
                with open(os.path.join(RQ3_DIR, fname), "w") as f:
                    json.dump(payload, f, separators=(",", ":"))
                index.setdefault(model, {}).setdefault(method, {})[direction] = f"rq3/{fname}"
                print(f"  {model:8} {method:9} {direction:5} "
                      f"stops={len(stops)} responses={payload['has_responses']}")
    return {"index": index, "deployed": DEPLOYED}


# --------------------------------------------------------------------------------------
# Glossary (faithful to the thesis, Chapter 3 "The Judge" and Chapter 4)
# --------------------------------------------------------------------------------------
GLOSSARY = [
    {"term": "Contaminated", "definition":
     "The response adds evaluative, normative, or group-identity framing that the prompt "
     "did not supply and the task does not require, taking a political side in either "
     "direction. It is a property of the added framing, not of whether the answer is right.",
     "source": "Chapter 3, The Judge"},
    {"term": "Collapsed", "definition":
     "Broken generation such as repetition loops or text that loses coherent meaning "
     "(neural text degeneration).", "source": "Chapter 3, The Judge"},
    {"term": "Reasoning validity", "definition":
     "Whether the stated chain of thought entails the verdict the model committed to: "
     "valid, invalid, opaque (too short to tell), or n/a.", "source": "Chapter 3, The Judge"},
    {"term": "Faithful task performance", "definition":
     "A correct answer whose stated reasoning supports it.", "source": "Chapter 3, The Judge"},
    {"term": "Post-hoc reasoning", "definition":
     "A correct answer whose stated reasoning does not entail it.", "source": "Chapter 3, The Judge"},
    {"term": "Capability error", "definition":
     "A wrong answer from a genuine reasoning mistake.", "source": "Chapter 3, The Judge"},
    {"term": "Instruction-following failure", "definition":
     "No parseable answer: a refusal, an off-format reply, or a response cut off at the "
     "token budget.", "source": "Chapter 3, The Judge"},
    {"term": "Viewpoint bias", "definition":
     "A wrong answer that the response attributes to political content.", "source": "Chapter 3, The Judge"},
    {"term": "Editorial framing bias", "definition":
     "A correct answer delivered with added partisan framing. (Stored in the raw data as "
     "motivational_framing_bias.)", "source": "Chapter 3, The Judge"},
    {"term": "Generation collapse", "definition":
     "The collapsed case, where the output degenerates into repetition or incoherence.",
     "source": "Chapter 3, The Judge"},
    {"term": "Trait score", "definition":
     "A number from 0 to 100 measuring how strongly the alignment is expressed in the "
     "model's answers. Higher means a stronger alignment.",
     "source": "Chapter 3, Measuring and Matching Alignment Strength"},
    {"term": "Coherence score", "definition":
     "A number from 0 to 100 measuring fluency only, judged alignment-blind, so it "
     "separates a clearly expressed stance from degenerated output.",
     "source": "Chapter 3, Measuring and Matching Alignment Strength"},
    {"term": "Steering coefficient (α)", "definition":
     "Scales how strongly the persona vector is added at inference time. Larger α "
     "pushes the alignment harder. Deployed at 2.5 (left) and 3.0 (right) in RQ1.",
     "source": "Chapter 3, Activation steering"},
    {"term": "DPO scale (s)", "definition":
     "Scales how strongly the fine-tuned LoRA weights act. s = 1.0 is the trained "
     "strength deployed in RQ1; s > 1 amplifies it.", "source": "Chapter 3, DPO fine-tuning"},
    {"term": "Value-loaded benchmark", "definition":
     "An external political inference dataset of syllogisms, each left or right and valid "
     "or invalid. Used to test the Partisan Double Standard.",
     "source": "Chapter 3, Politicized Reasoning Benchmarks"},
    {"term": "Party-fixed content-swap test", "definition":
     "A benchmark that swaps which party holds a policy so a premise becomes a false "
     "partisan premise while the correct logical answer is unchanged. Reported here as "
     "aggregates only (no per-example judge records).",
     "source": "Chapter 3, Politicized Reasoning Benchmarks"},
    {"term": "Partisan Double Standard", "definition":
     "Several aligned models accept a valid argument favoring their own side far more "
     "often than the logically identical argument for the other side.",
     "source": "Chapter 4, Value-Loaded Arguments"},
]


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Building dashboard data from", ROOT)
    print("=" * 70)

    examples, pooled = build_examples()
    party_fixed = build_party_fixed()
    rq3 = build_rq3()

    benchmarks = [
        {"id": "boolean_expressions", "label": BENCH_LABEL["boolean_expressions"],
         "rq": "rq1", "type": "neutral", "has_examples": True},
        {"id": "logical_deduction_three_objects", "label": BENCH_LABEL["logical_deduction_three_objects"],
         "rq": "rq1", "type": "neutral", "has_examples": True},
        {"id": "web_of_lies", "label": BENCH_LABEL["web_of_lies"],
         "rq": "rq1", "type": "neutral", "has_examples": True},
        {"id": "navigate", "label": BENCH_LABEL["navigate"],
         "rq": "rq1", "type": "neutral", "has_examples": True},
        {"id": "value_loaded", "label": BENCH_LABEL["value_loaded"],
         "rq": "rq2", "type": "political", "has_examples": True,
         "extra_axes": ["item_lean", "valid", "variation", "template_family"]},
        {"id": "party_fixed", "label": BENCH_LABEL["party_fixed"],
         "rq": "rq2", "type": "political", "has_examples": False,
         "note": "Aggregate only: no per-example judge records exist for this benchmark."},
    ]

    configs = [{"id": cid, "label": CONFIG_LABEL[cid],
                "method": CONFIG_META[cid]["method"],
                "direction": CONFIG_META[cid]["direction"]} for cid in CONFIG_ORDER]

    manifest = {
        "title": "Political Alignment & Reasoning — Judge Explorer",
        "models": MODELS,
        "methods": METHODS,
        "configs": configs,
        "benchmarks": benchmarks,
        "vocab": {
            "outcome": OUTCOME,
            "primary_category": PRIMARY_CATEGORY,
            "reasoning_validity": REASONING_VALIDITY,
            "fallacy_lens": FALLACY_LENS,
            "boolean_flags": [{"key": "contaminated", "label": "Contaminated"},
                              {"key": "collapsed", "label": "Collapsed"}],
        },
        "examples": examples,
        "pooled": pooled,
        "rq2_aggregate": {"party_fixed": party_fixed},
        "rq3": rq3,
        "glossary": GLOSSARY,
    }

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, separators=(",", ":"))

    # size report
    def dirsize(path):
        total = 0
        for dp, _, fns in os.walk(path):
            for fn in fns:
                total += os.path.getsize(os.path.join(dp, fn))
        return total

    n_ex = sum(len(glob.glob(os.path.join(EX_DIR, "*.json"))) for _ in [0])
    print("=" * 70)
    print(f"manifest.json written")
    print(f"example files: {n_ex}")
    print(f"data size: {dirsize(OUT)/1048576:.1f} MB")
    print(f"warnings: {len(warnings)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
