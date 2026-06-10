#!/usr/bin/env python3
"""Reconcile deeper_numbers_llama.json (BBHmean, collapse%) against consolidate.py.

Imports consolidate's own task_metrics loop and recomputes BBHmean/collapse% the way
consolidate.py prints them, then asserts the deeper_analysis_llama numbers match to a
0.05 tolerance (pure rounding). Any mismatch is a bug in deeper_analysis_llama.
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import consolidate as C  # noqa: E402

FAM = "llama"
RESULTS = os.path.join(HERE, "results")
STEER = ["a0_5", "a1", "a2", "a3", "a4"]
DPO = ["s0_25", "s0_5", "s1_0", "s1_5", "s2"]


def cells():
    yield ("base", os.path.join(RESULTS, FAM, "base"))
    for lean in ("left", "right"):
        for s in STEER:
            yield (f"steering/{lean}/{s}", os.path.join(RESULTS, FAM, "steering", lean, s))
    for lean in ("left", "right"):
        for s in DPO:
            yield (f"dpo/{lean}/{s}", os.path.join(RESULTS, FAM, "dpo", lean, s))


def consolidate_pooled(cell_dir):
    raccs, colls = [], []
    for t in C.BBH:
        m = C.task_metrics(cell_dir, t)
        if m:
            raccs.append(m["racc"]); colls.append(m["collapse"])
    if not raccs:
        return None, None
    return sum(raccs) / len(raccs), sum(colls) / len(colls)


def main():
    data = json.load(open(os.path.join(RESULTS, "deeper", "deeper_numbers_llama.json")))
    fails = 0
    print(f"{'cell':22} {'BBHmean(mine/cons)':>22} {'collapse%(mine/cons)':>22}  status")
    for cell, cdir in cells():
        if not os.path.isdir(cdir):
            continue
        cb, cc = consolidate_pooled(cdir)
        if cb is None:
            continue
        pl = data.get(cell, {}).get("pooled")
        if not pl:
            print(f"{cell:22} MISSING in deeper_numbers_llama"); fails += 1; continue
        mb, mc = pl["BBHmean"], pl["collapse_meanoftask"]
        ok = abs(mb - cb) < 0.05 and abs(mc - cc) < 0.05
        fails += not ok
        print(f"{cell:22} {mb:9.3f}/{cb:<9.3f}     {mc:9.3f}/{cc:<9.3f}      {'OK' if ok else 'FAIL'}")
    print("\nRECONCILE:", "PASS — deeper_analysis_llama matches consolidate.py llama exactly"
          if fails == 0 else f"FAIL ({fails} cells)")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
