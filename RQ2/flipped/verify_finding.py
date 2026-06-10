"""Durable verification of the party-vs-content FINDING, consolidating the
independent-review recomputes into one re-runnable script. Closes the provenance
gap left when the workflow's ephemeral scratch scripts (mechanism_a.py,
_attack1_matched.py) did not persist.

Computes, from raw data only:
  (1) within-party content-flip MATCHED-PAIR nets (party fixed, content swapped on
      adjacent twins) per carrier + bases, with base-difference + sign-test p;
  (2) matched-ENGAGED paired-delta (immune to arm-asymmetric disengagement) — to
      state its magnitude honestly (unchanged, NOT larger; near-vacuous at high engagement);
  (3) Holm-Bonferroni over the 9 non-degenerate leaned cells on content perm_p;
  (4) judge `reasoning_validity`-keyed content/party signed bias (regex-INDEPENDENT
      replication: keys on the judge's read of the CoT, not extract_verdict's token);
  (5) aggregate sign-tracking among cells with |content_bias|>0.03.
"""
import json, os, sys
import numpy as np
from scipy import stats

ROOT = "/Users/0ssamaak0/Documents/polireason/RQ2/flipped"
BENCH = "/Users/0ssamaak0/Documents/polireason/1_benchmarking/RQ2_bench"
sys.path.insert(0, BENCH)
from extract import extract_verdict  # noqa
META = {r["row_idx"]: r for r in (json.loads(l) for l in open(f"{BENCH}/stimuli.jsonl"))}
DEC = json.load(open(f"{ROOT}/decompose_out.json"))["full"]

CARRIERS = ["mistral-pvsteer-ml-right-a3", "mistral-politune-hf-left",
            "mistral-politune-hf-right", "llama-politune-hf-left"]
IND = {"mistral-pvsteer-ml-right-a3": "R", "mistral-politune-hf-left": "L",
       "mistral-politune-hf-right": "R", "llama-politune-hf-left": "L"}
# full lean map (for the aggregate sign-tracking tally; carrier-only IND was a bug)
LEAN = {t: ("R" if t.split("-a")[0].endswith("right") or "right" in t else "L")
        for t in []}
def lean_of(tag):
    return "R" if "right" in tag else ("L" if "left" in tag else "-")
BASES = ["mistral-base", "llama-base"]
# 9 non-degenerate leaned cells (exclude bases + 3 degenerate llama cells)
LEANED9 = ["mistral-roleplay-left", "mistral-roleplay-right", "mistral-pvsteer-ml-left-a2_5",
           "mistral-pvsteer-ml-right-a3", "mistral-politune-hf-left", "mistral-politune-hf-right",
           "llama-roleplay-left", "llama-roleplay-right", "llama-politune-hf-left"]


def load(tag):
    R = {json.loads(l)["row_idx"]: json.loads(l) for l in open(f"{ROOT}/responses/{tag}.jsonl")}
    J = {}
    jp = f"{ROOT}/judges/{tag}.jsonl"
    if os.path.exists(jp):
        J = {json.loads(l)["row_idx"]: json.loads(l) for l in open(jp)}
    return R, J


def right_content(m):
    return (m["variant"] == "clean" and m["lean"] == "right") or (m["variant"] == "flipped" and m["lean"] == "left")


# ---------- (1)+(2) matched within-party content-flip ----------
def matched_pairs(tag):
    """Adjacent twins (2k clean, 2k+1 flipped) sharing lean+gold+skeleton. Returns
    per-arm net (favor_Rcontent - favor_Lcontent over discordant both-engaged pairs),
    pooled net, n_discordant, n_botheng, mean paired-delta."""
    R, _ = load(tag)
    rows = sorted(META.values(), key=lambda m: m["row_idx"])
    res = {"left": [0, 0, 0], "right": [0, 0, 0]}  # [net, disc, botheng]
    deltas = []
    for m in rows:
        if m["arm"] != "political" or m["variant"] != "clean":
            continue
        c, f = m["row_idx"], m["row_idx"] + 1
        mf = META.get(f)
        if not mf or mf["variant"] != "flipped" or mf["lean"] != m["lean"]:
            continue
        vc, vf = extract_verdict(R[c]["raw_response"]), extract_verdict(R[f]["raw_response"])
        if vc not in ("valid", "invalid") or vf not in ("valid", "invalid"):
            continue
        arm = m["lean"]
        # right-content member: left-arm -> flipped(f); right-arm -> clean(c)
        rc_say_valid = (vf == "valid") if arm == "left" else (vc == "valid")
        lc_say_valid = (vc == "valid") if arm == "left" else (vf == "valid")
        res[arm][2] += 1
        deltas.append(int(rc_say_valid) - int(lc_say_valid))
        if rc_say_valid != lc_say_valid:
            res[arm][1] += 1
            res[arm][0] += 1 if rc_say_valid else -1
    net = res["left"][0] + res["right"][0]
    disc = res["left"][1] + res["right"][1]
    bothe = res["left"][2] + res["right"][2]
    p = stats.binomtest(max(0, (disc + net) // 2), disc, 0.5).pvalue if disc else 1.0
    return {"L_net": res["left"][0], "R_net": res["right"][0], "net": net,
            "disc": disc, "botheng": bothe,
            "mean_delta": round(np.mean(deltas), 3) if deltas else None, "binom_p": round(p, 5)}


# ---------- (4) judge reasoning_validity-keyed content/party ----------
def judge_keyed(tag):
    R, J = load(tag)
    rows = []
    for ri, jr in J.items():
        m = META.get(ri)
        if not m or m["arm"] != "political":
            continue
        rv = jr.get("reasoning_validity")
        if rv not in ("valid", "invalid"):
            continue
        rows.append({"y": 1.0 if rv == "valid" else 0.0,
                     "gold": 1.0 if m["valid"] else -1.0,
                     "party": 1.0 if m["lean"] == "right" else -1.0,
                     "content": 1.0 if right_content(m) else -1.0})
    if len(rows) < 30:
        return None
    y = np.array([r["y"] for r in rows])
    if not (0 < y.mean() < 1):
        return None
    X = np.column_stack([np.ones_like(y)] + [np.array([r[k] for r in rows]) for k in ("gold", "party", "content")])
    # IRLS
    beta = np.zeros(X.shape[1]); Rr = 1e-3 * np.eye(X.shape[1])
    for _ in range(100):
        mu = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        W = np.clip(mu * (1 - mu), 1e-6, None)
        step = np.linalg.solve(X.T @ (W[:, None] * X) + Rr, X.T @ (y - mu) - 1e-3 * beta)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    mu = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
    W = np.clip(mu * (1 - mu), 1e-6, None)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (W[:, None] * X) + Rr)))
    def pz(i): return 2 * stats.norm.sf(abs(beta[i] / se[i]))
    return {"n": len(rows), "party": (round(beta[2], 3), round(pz(2), 4)),
            "content": (round(beta[3], 3), round(pz(3), 4))}


def main():
    print("=" * 96)
    print("(1)+(2) WITHIN-PARTY content-flip matched pairs (party FIXED, content swapped on adjacent twins)")
    print("=" * 96)
    print(f"{'cell':30} ind | {'L_net':>5} {'R_net':>5} {'net':>5} | disc botheng | {'mean_delta':>10} | binom_p")
    base_arm = {}
    for tag in BASES + CARRIERS:
        mp = matched_pairs(tag)
        ind = IND.get(tag, "-")
        if tag in BASES:
            base_arm[tag.split("-")[0]] = mp
        print(f"{tag:30} {ind:>2}  | {mp['L_net']:+5} {mp['R_net']:+5} {mp['net']:+5} | "
              f"{mp['disc']:>4} {mp['botheng']:>6} | {str(mp['mean_delta']):>10} | {mp['binom_p']}")
    print("\nbase-difference (carrier per-arm net minus same-family base per-arm net):")
    for tag in CARRIERS:
        mp = matched_pairs(tag); b = base_arm[tag.split("-")[0]]
        print(f"  {tag:30} ind={IND[tag]}  dL={mp['L_net']-b['L_net']:+d}  dR={mp['R_net']-b['R_net']:+d}  "
              f"(base {tag.split('-')[0]}: L={b['L_net']:+d}/R={b['R_net']:+d})")

    print("\n" + "=" * 96)
    print("(3) HOLM-BONFERRONI over 9 non-degenerate leaned cells on content perm_p")
    print("=" * 96)
    pv = sorted([(t, DEC[t]["content_perm_p"]) for t in LEANED9], key=lambda x: x[1])
    m = len(pv)
    print(f"{'rank':>4} {'cell':30} {'perm_p':>8} {'holm_thresh':>11} {'bonf_p':>8}  survives_FWER")
    for i, (t, p) in enumerate(pv):
        thr = 0.05 / (m - i)
        surv = all(pv[j][1] <= 0.05 / (m - j) for j in range(i + 1))
        print(f"{i+1:>4} {t:30} {p:8.4f} {thr:11.4f} {min(1,p*m):8.3f}  {'YES' if surv else 'no'}")

    print("\n" + "=" * 96)
    print("(4) JUDGE reasoning_validity-KEYED content/party (regex-INDEPENDENT replication)")
    print("=" * 96)
    print(f"{'cell':30} ind | n | {'content coef (p)':>18} | {'party coef (p)':>16}")
    for tag in CARRIERS + BASES:
        jk = judge_keyed(tag)
        ind = IND.get(tag, "-")
        if jk:
            c, cp = jk["content"]; pa, pp = jk["party"]
            print(f"{tag:30} {ind:>2}  | {jk['n']:>3} | {c:+.3f} (p={cp:.4f}) | {pa:+.3f} (p={pp:.4f})")

    print("\n" + "=" * 96)
    print("(5) Aggregate sign-tracking among cells with |content_bias|>0.03")
    print("=" * 96)
    big = [(t, DEC[t]["decomp"]["content_bias"], lean_of(t)) for t in LEANED9
           if abs(DEC[t]["decomp"]["content_bias"]) > 0.03]
    tracks = sum((cb < 0 and i == "L") or (cb > 0 and i == "R") for _, cb, i in big)
    p = stats.binomtest(tracks, len(big), 0.5).pvalue
    for t, cb, i in big:
        ok = (cb < 0 and i == "L") or (cb > 0 and i == "R")
        print(f"  {t:30} ind={i} content_bias={cb:+.3f} tracks={'YES' if ok else 'NO'}")
    print(f"  => {tracks}/{len(big)} track lean, two-sided binom p={p:.3f}")


if __name__ == "__main__":
    main()
