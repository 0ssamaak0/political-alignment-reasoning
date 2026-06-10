#!/usr/bin/env python3
"""Character-based collapse morphology (companion to deeper_analysis.py).

WHY: the judge.jsonl degeneracy fields are WORD-based — `n_tokens_generated` is a
whitespace word count and `max_4gram_repeat` counts word 4-grams. A *spaceless*
concatenation loop (e.g. "reverseordenreverseorden...", which dpo/right/s2 emits)
is one giant "word", so it registers as ~5 tokens and rep=1 and is misclassified
as a short stub. Character-length + a longest-non-space-run detector recover the
true morphology, so the loop-vs-stub split is measured on the actual string.

Per cell (pooled over the 4 BBH tasks): median word count, median char length,
median chars-per-word (a spaceless-loop tell: ~5-6 normal English, high = no
spaces), spaceless% (share with a >=60-char non-space run), and truestub% (share
that is genuinely short: <=120 chars AND not a spaceless loop). A coarse label is
assigned: LOOP (long, high char length / spaceless), STUB (short char length),
SERMON/coherent (medium length, normal spacing), or INTACT.

    python3 RQ3/char_morphology.py
Outputs: RQ3/results/deeper/char_morphology.txt
"""
import json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results", "mistral")
OUT = os.path.join(HERE, "results", "deeper", "char_morphology.txt")
TASKS = ["boolean_expressions", "logical_deduction_three_objects", "web_of_lies", "navigate"]
STEER = ["a0_5", "a1", "a2", "a3", "a4"]
DPO = ["s0_25", "s0_5", "s1_0", "s1_5", "s2"]


def cells():
    yield ("base", os.path.join(R, "base"))
    for lean in ("left", "right"):
        for s in STEER:
            yield (f"steer/{lean}/{s}", os.path.join(R, "steering", lean, s))
    for lean in ("left", "right"):
        for s in DPO:
            yield (f"dpo/{lean}/{s}", os.path.join(R, "dpo", lean, s))


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


def load(cell_dir):
    rows = []
    for t in TASKS:
        p = os.path.join(cell_dir, f"bbh_cot_fewshot_{t}", "judge.jsonl")
        if os.path.exists(p):
            rows += [json.loads(l) for l in open(p)]
    return rows


def main():
    lines = ["Character-based collapse morphology (pooled 4 BBH tasks). word_med=median n_tokens_generated (word count);",
             "char_med=median len(raw_response); cpw=median chars/word (~5-6 normal, high=spaceless loop);",
             "spaceless%=share with a >=60-char non-space run; truestub%=share <=120 chars AND not spaceless.",
             f"{'cell':14} {'n':>5} {'word_med':>8} {'char_med':>8} {'cpw':>5} {'spaceless%':>10} {'truestub%':>9}  label"]
    for name, d in cells():
        rows = load(d)
        if not rows:
            continue
        words = [r.get("n_tokens_generated") or 0 for r in rows]
        texts = [(r.get("raw_response") or "") for r in rows]
        chars = [len(t) for t in texts]
        cpw = [(len(t) / w if w else len(t)) for t, w in zip(texts, words)]
        runs = [longest_nonspace_run(t) for t in texts]
        n = len(rows)
        spaceless = 100 * sum(1 for x in runs if x >= 60) / n
        truestub = 100 * sum(1 for c, x in zip(chars, runs) if c <= 120 and x < 60) / n
        cm, wm = st.median(chars), st.median(words)
        # coarse label
        if cm < 130 and spaceless < 10:
            label = "STUB" if truestub > 30 else "short/off-format"
        elif spaceless > 15:
            label = "LOOP (spaceless)"
        elif cm > 900:
            label = "LOOP (spaced)"
        elif name == "base" or cm < 700 and spaceless < 5 and st.median(runs) < 30:
            label = "intact/coherent"
        else:
            label = "mixed"
        lines.append(f"{name:14} {n:5d} {wm:8.0f} {cm:8.0f} {st.median(cpw):5.1f} "
                     f"{spaceless:9.1f}% {truestub:8.1f}%  {label}")
    txt = "\n".join(lines)
    with open(OUT, "w") as f:
        f.write(txt + "\n")
    print(txt)
    print(f"\n[char_morphology] wrote {OUT}")


if __name__ == "__main__":
    main()
