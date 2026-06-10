"""Party-vs-content decomposition of the partisan-validity bias on the
counter-attitudinal (clean/flipped) instrument — RQ2 "flipped" sub-study.

THE QUESTION (the spark):
  G&K-style signed bias swaps the PARTY and reads the skew as "partisan
  double-standard". But its clean party-swap confounds party-LABEL with
  content-CONGRUENCE: a Republican-arm item also carries right-congenial content.

  The flipped instrument holds the PARTY LABEL FIXED (same group, e.g. Democrats)
  and swaps only the POLICY CONTENT (congruent -> incongruent). So party and
  content are CROSSED and ORTHOGONAL here:
      party   = +1 right-arm (Rep label)         / -1 left-arm (Dem label)
      content = +1 right-coded policy in the item / -1 left-coded policy
                right-content := (clean & right-arm) OR (flipped & left-arm)
  gold validity is balanced within every party x content cell (design check below).

  Decomposition of the signed bias (FP-FN form, == results.md metric):
      party_bias   = signed_bias over party-arm     ~ (clean+flipped)/2
      content_bias = signed_bias over content-coding ~ (clean-flipped)/2
  A *tribal* (label) account predicts party_bias carries the effect and NO sign
  flip clean->flipped. A *belief-bias* (content) account predicts content_bias
  carries it, sign-tracking the induced lean, with party_bias ~ 0.

Outputs: console table + decompose_out.json (durable, for the workflow agents).
"""
import json, os, sys, math
from collections import defaultdict, Counter
import numpy as np
from scipy import stats

ROOT = "/Users/0ssamaak0/Documents/polireason/RQ2/flipped"
BENCH = "/Users/0ssamaak0/Documents/polireason/1_benchmarking/RQ2_bench"
sys.path.insert(0, BENCH)
from extract import extract_verdict  # noqa

STIM = [json.loads(l) for l in open(f"{BENCH}/stimuli.jsonl")]
META = {r["row_idx"]: r for r in STIM}
MR = {"motivational_reasoning", "premise_truth_conflation"}

CELLS = [
    ("mistral-base", "Base", "-"), ("mistral-roleplay-left", "RP-L", "L"),
    ("mistral-roleplay-right", "RP-R", "R"), ("mistral-pvsteer-ml-left-a2_5", "Steer-L", "L"),
    ("mistral-pvsteer-ml-right-a3", "Steer-R", "R"), ("mistral-politune-hf-left", "DPO-L", "L"),
    ("mistral-politune-hf-right", "DPO-R", "R"),
    ("llama-base", "Base", "-"), ("llama-roleplay-left", "RP-L", "L"),
    ("llama-roleplay-right", "RP-R", "R"), ("llama-pvsteer-ml-left-a2_5", "Steer-L", "L"),
    ("llama-pvsteer-ml-right-a3", "Steer-R", "R"), ("llama-politune-hf-left", "DPO-L", "L"),
    ("llama-politune-hf-right", "DPO-R", "R"),
]


def is_right_content(m):
    """right-coded policy attributed: clean&right-arm OR flipped&left-arm."""
    if m["variant"] == "clean":
        return m["lean"] == "right"
    return m["lean"] == "left"  # flipped


def signed_bias(rows):
    """((R_FP - R_FN) - (L_FP - L_FN)) / N_engaged over a 'group' key in r['_grp'].
    group +1 -> R side, -1 -> L side. rows already filtered to engaged."""
    if not rows:
        return None, 0
    def term(side):
        g = [r for r in rows if r["_grp"] == side]
        fp = sum((not r["gold"]) and r["_say_valid"] for r in g)
        fn = sum(r["gold"] and (not r["_say_valid"]) for r in g)
        return fp - fn
    return round((term(+1) - term(-1)) / len(rows), 4), len(rows)


def irls_logit(X, y, ridge=1e-3, iters=100):
    """Newton-IRLS logistic regression with tiny ridge for separation guard.
    Returns beta, se (sqrt diag of inv Fisher), and a converged flag."""
    n, p = X.shape
    beta = np.zeros(p)
    R = ridge * np.eye(p)
    last = None
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-6, None)
        XtWX = X.T @ (W[:, None] * X) + R
        grad = X.T @ (y - mu) - ridge * beta
        try:
            step = np.linalg.solve(XtWX, grad)
        except np.linalg.LinAlgError:
            return beta, np.full(p, np.nan), False
        beta = beta + step
        if last is not None and np.max(np.abs(step)) < 1e-8:
            break
        last = beta.copy()
    eta = np.clip(X @ beta, -30, 30)
    mu = 1 / (1 + np.exp(-eta))
    W = np.clip(mu * (1 - mu), 1e-6, None)
    cov = np.linalg.inv(X.T @ (W[:, None] * X) + R)
    se = np.sqrt(np.diag(cov))
    return beta, se, True


def perm_test_content(rows, observed, nperm=5000, seed=0):
    """Permutation p for |content_bias|: shuffle content labels within cell."""
    rng = np.random.default_rng(seed)
    grp = np.array([r["_content"] for r in rows])
    vals = []
    arr = [{"gold": r["gold"], "_say_valid": r["_say_valid"]} for r in rows]
    for _ in range(nperm):
        perm = rng.permutation(grp)
        for r, g in zip(arr, perm):
            r["_grp"] = g
        sb, _ = signed_bias(arr)
        vals.append(abs(sb) if sb is not None else 0.0)
    vals = np.array(vals)
    return float((np.sum(vals >= abs(observed)) + 1) / (nperm + 1))


def analyze(tag, ind, strict=False):
    rp, jp = f"{ROOT}/responses/{tag}.jsonl", f"{ROOT}/judges/{tag}.jsonl"
    if not os.path.exists(rp):
        return None
    R = [json.loads(l) for l in open(rp)]
    J = {}
    if os.path.exists(jp):
        for l in open(jp):
            jr = json.loads(l); J[jr.get("row_idx")] = jr
    items = []
    for r in R:
        m = META[r["row_idx"]]
        if m["arm"] != "political":
            continue
        if strict and (m["pattern"] == "T6" or m.get("soft_opposition")):
            continue
        v = extract_verdict(r["raw_response"])
        engaged = v in ("valid", "invalid")
        items.append({
            "row_idx": r["row_idx"], "gold": bool(m["valid"]),
            "lean": m["lean"], "variant": m["variant"],
            "rc": is_right_content(m), "verdict": v, "engaged": engaged,
            "_say_valid": v == "valid",
            "_grp_party": +1 if m["lean"] == "right" else -1,
            "_content": +1 if is_right_content(m) else -1,
            "judge": J.get(r["row_idx"]),
        })

    # ---- engagement by arm x variant x party (CHECK #1) ----
    eng_tab = {}
    for var in ("clean", "flipped"):
        for arm in ("left", "right"):
            sub = [it for it in items if it["variant"] == var and it["lean"] == arm]
            n = len(sub); e = sum(it["engaged"] for it in sub)
            eng_tab[f"{var}-{arm}"] = {"n": n, "engaged": e,
                                       "eng_rate": round(e / n, 3) if n else None}
    asym = {}
    for var in ("clean", "flipped"):
        l = eng_tab[f"{var}-left"]["eng_rate"]; rt = eng_tab[f"{var}-right"]["eng_rate"]
        asym[var] = round(abs((l or 0) - (rt or 0)), 3)

    eng = [it for it in items if it["engaged"]]

    def sb_over(rows, key):
        for r in rows:
            r["_grp"] = r["_grp_party"] if key == "party" else r["_content"]
        return signed_bias(rows)
    clean_rows = [it for it in eng if it["variant"] == "clean"]
    flip_rows = [it for it in eng if it["variant"] == "flipped"]
    sb_clean, n_cl = sb_over(clean_rows, "party")
    sb_flip, n_fl = sb_over(flip_rows, "party")
    sb_all, n_all = sb_over(eng, "party")
    party_bias, _ = sb_over(eng, "party")
    content_bias, _ = sb_over(eng, "content")

    glm = None
    if len(eng) >= 30:
        y = np.array([1.0 if it["_say_valid"] else 0.0 for it in eng])
        gold = np.array([1.0 if it["gold"] else -1.0 for it in eng])
        party = np.array([float(it["_grp_party"]) for it in eng])
        content = np.array([float(it["_content"]) for it in eng])
        X = np.column_stack([np.ones_like(y), gold, party, content])
        if 0 < y.mean() < 1:
            beta, se, conv = irls_logit(X, y)
            names = ["intercept", "gold", "party", "content"]
            glm = {}
            for i, nm in enumerate(names):
                z = beta[i] / se[i] if se[i] > 0 and np.isfinite(se[i]) else np.nan
                p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
                glm[nm] = {"coef": round(float(beta[i]), 4),
                           "se": round(float(se[i]), 4) if np.isfinite(se[i]) else None,
                           "z": round(float(z), 3) if np.isfinite(z) else None,
                           "p": float(p) if np.isfinite(p) else None,
                           "ci": [round(float(beta[i] - 1.96 * se[i]), 3),
                                  round(float(beta[i] + 1.96 * se[i]), 3)]
                                 if np.isfinite(se[i]) else None}
            glm["_converged"] = conv

    for r in eng:
        r["_grp"] = r["_content"]
    perm_p = perm_test_content(eng, content_bias) if len(eng) >= 30 and content_bias is not None else None

    def jrate(rows, fn):
        js = [it["judge"] for it in rows if it["judge"]]
        if not js:
            return None
        return round(sum(fn(j) for j in js) / len(js), 3)
    judge = {}
    for var in ("clean", "flipped"):
        sub = [it for it in items if it["variant"] == var]
        judge[var] = {
            "contam": jrate(sub, lambda j: bool(j.get("contaminated"))),
            "mr": jrate(sub, lambda j: j.get("fallacy_lens") in MR),
            "rsn_invalid": jrate([it for it in sub if it["judge"] and it["judge"].get("reasoning_validity") in ("valid", "invalid")],
                                 lambda j: j.get("reasoning_validity") == "invalid"),
        }
    conflict = [it for it in items if (it["gold"] and it["variant"] == "flipped") or
                ((not it["gold"]) and it["variant"] == "clean")]
    congru = [it for it in items if it not in conflict]
    judge["conflict_mr"] = jrate(conflict, lambda j: j.get("fallacy_lens") in MR)
    judge["congru_mr"] = jrate(congru, lambda j: j.get("fallacy_lens") in MR)
    judge["conflict_contam"] = jrate(conflict, lambda j: bool(j.get("contaminated")))
    judge["congru_contam"] = jrate(congru, lambda j: bool(j.get("contaminated")))

    return {
        "tag": tag, "ind": ind, "n_political": len(items), "n_engaged": len(eng),
        "engagement": eng_tab, "arm_asymmetry": asym,
        "signed_bias": {"clean": sb_clean, "flipped": sb_flip, "all": sb_all,
                        "n_clean": n_cl, "n_flipped": n_fl},
        "decomp": {"party_bias": party_bias, "content_bias": content_bias},
        "glm": glm, "content_perm_p": perm_p, "judge": judge,
    }


def main():
    out = {"full": {}, "strict": {}}
    for strict, key in [(False, "full"), (True, "strict")]:
        for tag, _short, ind in CELLS:
            res = analyze(tag, ind, strict=strict)
            if res:
                out[key][tag] = res
    with open(f"{ROOT}/decompose_out.json", "w") as f:
        json.dump(out, f, indent=1)

    print("=" * 110)
    print("PARTY vs CONTENT DECOMPOSITION  (engaged political items; signed bias FP-FN form)")
    print("  party_bias=(clean+flipped)/2 [label tribalism] | content_bias=(clean-flipped)/2 [belief bias]")
    print("  GLM: logit P(say valid) ~ gold + party + content ;  +content => right-belief-bias, sign should track lean")
    print("=" * 110)
    hdr = (f"{'cell':28} ind | {'sb_cln':>7} {'sb_flp':>7} {'sb_all':>7} | "
           f"{'PARTY':>6} {'CONTENT':>7} | {'glm_party(p)':>16} {'glm_content(p)':>18} | {'perm_p':>6} | "
           f"{'asym_cl/fl':>10} | {'eng%':>5}")
    print(hdr); print("-" * len(hdr))
    fam = None
    for tag, _short, ind in CELLS:
        r = out["full"].get(tag)
        if not r:
            continue
        f0 = tag.split("-")[0]
        if f0 != fam:
            fam = f0; print(f"== {fam} ==")
        d = r["decomp"]; sb = r["signed_bias"]; g = r["glm"] or {}
        gp = g.get("party", {}); gc = g.get("content", {})
        gp_s = f"{gp.get('coef'):+.2f}({gp.get('p'):.3f})" if gp.get("p") is not None else "      .       "
        gc_s = f"{gc.get('coef'):+.2f}({gc.get('p'):.3f})" if gc.get("p") is not None else "      .        "
        eng_pct = round(100 * r["n_engaged"] / r["n_political"], 1)
        print(f"{tag:28} {ind:>2}  | {sb['clean']:+7.3f} {sb['flipped']:+7.3f} {sb['all']:+7.3f} | "
              f"{d['party_bias']:+6.3f} {d['content_bias']:+7.3f} | {gp_s:>16} {gc_s:>18} | "
              f"{(r['content_perm_p'] or float('nan')):6.3f} | "
              f"{r['arm_asymmetry']['clean']:.2f}/{r['arm_asymmetry']['flipped']:.2f} | {eng_pct:5.1f}")
    print("\nLEAD-EXHIBIT lens (coherent cells = engaged>~90% AND arm_asym<0.15 both variants):")
    for tag, _short, ind in CELLS:
        r = out["full"].get(tag)
        if not r or ind == "-":
            continue
        coh = (r["n_engaged"] / r["n_political"] > 0.90 and
               r["arm_asymmetry"]["clean"] < 0.15 and r["arm_asymmetry"]["flipped"] < 0.15)
        if coh:
            cb = r["decomp"]["content_bias"]; tracks = (cb < 0 and ind == "L") or (cb > 0 and ind == "R")
            print(f"  {tag:28} induced={ind}  content_bias={cb:+.3f}  "
                  f"tracks_lean={'YES' if tracks else 'NO'}  party_bias={r['decomp']['party_bias']:+.3f}  "
                  f"perm_p={r['content_perm_p']:.3f}")

    print("\nJUDGE axis (patterns §3/§4 link): MR%/contam% clean vs flipped, and conflict vs congruent cells")
    print(f"{'cell':28} | {'MR cl/fl':>9} | {'contam cl/fl':>12} | {'MR conflict/congru':>18} | {'contam conf/cong':>16}")
    fam = None
    for tag, _short, ind in CELLS:
        r = out["full"].get(tag)
        if not r:
            continue
        f0 = tag.split("-")[0]
        if f0 != fam:
            fam = f0; print(f"== {fam} ==")
        j = r["judge"]
        def f2(a, b):
            a = "  . " if a is None else f"{100*a:3.0f}"; b = " . " if b is None else f"{100*b:3.0f}"
            return f"{a}/{b}"
        print(f"{tag:28} | {f2(j['clean']['mr'], j['flipped']['mr']):>9} | "
              f"{f2(j['clean']['contam'], j['flipped']['contam']):>12} | "
              f"{f2(j['conflict_mr'], j['congru_mr']):>18} | {f2(j['conflict_contam'], j['congru_contam']):>16}")
    print("\nwrote decompose_out.json")


if __name__ == "__main__":
    main()
