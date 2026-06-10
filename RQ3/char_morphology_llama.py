#!/usr/bin/env python3
"""Character-based collapse morphology — LLAMA (companion to deeper_analysis_llama.py).

Same purpose as the Mistral char_morphology.py, but Llama has no judge.jsonl, so the
text and the word-count are taken from samples.jsonl (resps[0][0]) rather than the
judge's pre-computed fields. This keeps the analysis judge-independent and, more
importantly, avoids the word-count trap: a *spaceless* concatenation loop is one
giant "word", so word-count alone mislabels it a stub. char-length + a
longest-non-space-run detector recover the true morphology.

Per cell (pooled over 4 BBH tasks): median word count, median char length, median
chars-per-word (~5-6 normal English, high=spaceless loop), spaceless% (share with a
>=60-char non-space run), truestub% (share <=120 chars AND not spaceless). A coarse
label is assigned: LOOP (spaceless), LOOP (spaced), STUB / short-off-format,
intact/coherent, or mixed.

    python3 RQ3/char_morphology_llama.py
Outputs: RQ3/results/deeper/char_morphology_llama.txt
"""
import json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results", "llama")
OUT = os.path.join(HERE, "results", "deeper", "char_morphology_llama.txt")
TASKS = ["boolean_expressions", "logical_deduction_three_objects", "web_of_lies", "navigate"]
STEER = ["a0_5", "a1", "a2", "a3", "a4"]      # exclude a2_5 (outside RQ3 subset)
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
    """Return list of generation strings (resps[0][0]) for the 4 BBH tasks."""
    texts = []
    for t in TASKS:
        p = os.path.join(cell_dir, f"bbh_cot_fewshot_{t}", "samples.jsonl")
        if os.path.exists(p):
            for d in (json.loads(l) for l in open(p)):
                texts.append(d["resps"][0][0] if d.get("resps") else "")
    return texts


def main():
    lines = [
        "Character-based collapse morphology — LLAMA (pooled 4 BBH tasks; text from samples.jsonl resps[0][0]).",
        "word_med=median word count; char_med=median len(gen); cpw=median chars/word (~5-6 normal, high=spaceless loop);",
        "spaceless%=share with a >=60-char non-space run; truestub%=share <=120 chars AND not spaceless.",
        f"{'cell':14} {'n':>5} {'word_med':>8} {'char_med':>8} {'cpw':>5} {'spaceless%':>10} {'truestub%':>9}  label",
    ]
    for name, d in cells():
        texts = load(d)
        if not texts:
            continue
        words = [len(t.split()) for t in texts]
        chars = [len(t) for t in texts]
        cpw = [(len(t) / w if w else len(t)) for t, w in zip(texts, words)]
        runs = [longest_nonspace_run(t) for t in texts]
        n = len(texts)
        spaceless = 100 * sum(1 for x in runs if x >= 60) / n
        truestub = 100 * sum(1 for c, x in zip(chars, runs) if c <= 120 and x < 60) / n
        cm, wm = st.median(chars), st.median(words)
        # coarse label (same thresholds as the Mistral script)
        if cm < 130 and spaceless < 10:
            label = "STUB" if truestub > 30 else "short/off-format"
        elif spaceless > 15:
            label = "LOOP (spaceless)"
        elif cm > 900:
            label = "LOOP (spaced)"
        elif name == "base" or (cm < 700 and spaceless < 5 and st.median(runs) < 30):
            label = "intact/coherent"
        else:
            label = "mixed"
        lines.append(f"{name:14} {n:5d} {wm:8.0f} {cm:8.0f} {st.median(cpw):5.1f} "
                     f"{spaceless:9.1f}% {truestub:8.1f}%  {label}")
    txt = "\n".join(lines)
    with open(OUT, "w") as f:
        f.write(txt + "\n")
    print(txt)
    print(f"\n[char_morphology_llama] wrote {OUT}")


if __name__ == "__main__":
    main()
