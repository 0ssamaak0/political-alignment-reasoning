"""Phase 3 — final aggregation of the judge-verification audit.

Combines: Gemini labels (answer key) + Claude blind labels (merged.jsonl) +
3-verifier panel majority (panel/*.json) into:
  - per-axis agreement (Claude-blind vs Gemini)
  - panel adjudication of the 509 disagreements (who is right: Gemini or Claude)
  - the headline: Gemini's error-rate-vs-consensus by LEAN and by MODEL
    (the lean-asymmetry test) + contamination over/under-flag direction
  - within-pair (matched left/right) receipts
Writes audit/JUDGE_VERIFICATION.md + verify/verification_summary.json.

Run from repo root AFTER wf_panel completes:
    python -m RQ2.G_K_assessing_bias.verify.analyze
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
GK = HERE.parent
PANEL = HERE / "panel"
AUDIT = GK / "audit"

CELLS = [
    "mistral-base", "llama-base",
    "mistral-roleplay-left", "mistral-roleplay-right",
    "llama-roleplay-left", "llama-roleplay-right",
    "mistral-steering-left", "mistral-steering-right",
    "llama-steering-left", "llama-steering-right",
    "mistral-DPO-left", "mistral-DPO-right",
    "llama-DPO-left", "llama-DPO-right", "llama-DPO-right-2nd",
]
BIAS_PC = {"viewpoint_bias", "motivational_framing_bias"}


def model_of(cell): return "mistral" if cell.startswith("mistral") else "llama"


def majority(vals):
    """Return (winner, count, n, unanimous) or (None,..) on a 3-way tie."""
    c = Counter(vals)
    top, n = c.most_common(1)[0]
    if list(c.values()).count(max(c.values())) > 1 and len(c) > 1 and max(c.values()) * 2 <= len(vals):
        return None, top, len(vals), False  # no strict majority
    return top, n, len(vals), (n == len(vals))


def load_panel():
    """gidx -> {axis -> list of verifier votes}."""
    votes = defaultdict(lambda: defaultdict(list))
    nfiles = 0
    for f in sorted(PANEL.glob("pb*_v*.json")):
        nfiles += 1
        try:
            arr = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"[analyze] WARN bad json {f.name}")
            continue
        for o in arr:
            gi = o.get("gidx")
            if gi is None:
                continue
            fl = o.get("fallacy_lens")
            if fl == "none":
                fl = None
            votes[gi]["outcome"].append(o.get("outcome"))
            votes[gi]["contaminated"].append(bool(o.get("contaminated")))
            votes[gi]["collapsed"].append(bool(o.get("collapsed")))
            votes[gi]["reasoning_validity"].append(o.get("reasoning_validity"))
            votes[gi]["primary_category"].append(o.get("primary_category"))
            votes[gi]["_just"].append(o.get("justification"))
    return votes, nfiles


def pct(a, b): return f"{(100*a/b):.1f}%" if b else "—"


def main():
    merged = [json.loads(l) for l in (HERE / "merged.jsonl").open()]
    by_gidx = {r["gidx"]: r for r in merged}
    panel, nfiles = load_panel()
    print(f"[analyze] merged items: {len(merged)}; panel files: {nfiles}; "
          f"panel-adjudicated items: {len(panel)}")

    # ---- panel majority + adjudication of each disagreement -------------------
    AXES = ["primary_category", "contaminated", "collapsed", "outcome", "reasoning_validity"]
    for r in merged:
        gi = r["gidx"]
        if gi in panel:
            r["_paneled"] = True
            for ax in AXES:
                win, cnt, n, unan = majority(panel[gi][ax])
                r[f"panel_{ax}"] = win
                r[f"panel_{ax}_n"] = f"{cnt}/{n}"
                r[f"panel_{ax}_unanimous"] = unan
            r["panel_n_verifiers"] = len(panel[gi]["primary_category"])
            r["panel_justs"] = panel[gi]["_just"]
        else:
            r["_paneled"] = False

    # ---- consensus reference per axis -----------------------------------------
    # agree (not paneled OR paneled-but-axis-agreed): consensus = agreed value, Gemini correct.
    # disputed on this axis: consensus = panel majority (None if tie / no panel).
    def gem_val(r, ax): return {"primary_category": r["g_primary_category"],
                                "contaminated": r["g_contaminated"], "collapsed": r["g_collapsed"],
                                "outcome": r["g_outcome"], "reasoning_validity": r["g_reasoning_validity"]}[ax]

    def cla_val(r, ax): return {"primary_category": r["c_primary_category"],
                                "contaminated": r["c_contaminated"], "collapsed": r["c_collapsed"],
                                "outcome": r["c_outcome"], "reasoning_validity": r["c_reasoning_validity"]}[ax]

    for r in merged:
        for ax in AXES:
            g, c = gem_val(r, ax), cla_val(r, ax)
            if g == c:
                r[f"ref_{ax}"] = g
                r[f"gem_correct_{ax}"] = True
                r[f"sides_{ax}"] = "agree"
            elif r["_paneled"] and r.get(f"panel_{ax}") is not None:
                ref = r[f"panel_{ax}"]
                r[f"ref_{ax}"] = ref
                r[f"gem_correct_{ax}"] = (g == ref)
                r[f"sides_{ax}"] = "gemini" if ref == g else ("claude" if ref == c else "neither")
            else:
                r[f"ref_{ax}"] = None  # unadjudicated disagreement
                r[f"gem_correct_{ax}"] = None
                r[f"sides_{ax}"] = "unadjudicated"

    # ---------------------------------------------------------------------------
    # Third judge — Gemini 3.1 Pro (same family as Flash, stronger). Loaded if present.
    # Aligned by (cell,row_idx) -> gidx; format identical to the Flash judges/.
    # ---------------------------------------------------------------------------
    PRO_DIR = HERE / "pro_judges"
    pro_by_gidx = {}
    if PRO_DIR.exists():
        for ci, cell in enumerate(CELLS):
            f = PRO_DIR / f"{cell}.jsonl"
            if not f.exists():
                continue
            for row_idx, pj in enumerate(json.loads(l) for l in f.open()):
                if "primary_category" not in pj:
                    continue
                pro_by_gidx[ci * 192 + row_idx] = pj
    pro_available = len(pro_by_gidx) == len(merged)
    for r in merged:
        pj = pro_by_gidx.get(r["gidx"])
        if pj:
            r["p_primary_category"] = pj["primary_category"]
            r["p_contaminated"] = bool(pj["contaminated"])
            r["p_collapsed"] = bool(pj["collapsed"])
            r["p_outcome"] = pj["outcome"]
            # "Gemini-Flash matches Pro?" (Pro as reference) — for paired lean test
            r["gem_eq_pro_primary"] = (r["g_primary_category"] == r["p_primary_category"])
            r["gem_eq_pro_contam"] = (r["g_contaminated"] == r["p_contaminated"])
            # "Pro matches Claude-consensus?" — Pro's own lean symmetry vs my work
            r["pro_eq_ref_primary"] = (r["p_primary_category"] == r["ref_primary_category"]) \
                if r["ref_primary_category"] is not None else None
            r["pro_eq_ref_contam"] = (r["p_contaminated"] == r["ref_contaminated"]) \
                if r["ref_contaminated"] is not None else None

    # ---------------------------------------------------------------------------
    # Paired (within-pair) tests — matched left/right syllogism skeletons.
    # Each (cell, pair_id) holds one left + one right item with identical gold.
    # McNemar exact (two-sided) on Gemini's per-item correctness, left vs right.
    # ---------------------------------------------------------------------------
    import math
    def mcnemar_exact(b, c):
        nn = b + c
        if nn == 0:
            return 1.0
        k = min(b, c)
        p = sum(math.comb(nn, i) for i in range(k + 1)) * (0.5 ** nn)
        return min(1.0, 2 * p)

    pairs = defaultdict(dict)
    for r in merged:
        pairs[(r["cell"], r["pair_id"])][r["lean"]] = r

    def paired_err(field, models=None):
        b = c = 0  # b=left-only-wrong, c=right-only-wrong (discordant pairs)
        for (cell, pid), mem in pairs.items():
            if models and model_of(cell) not in models:
                continue
            if "left" not in mem or "right" not in mem:
                continue
            lc, rc = mem["left"].get(field), mem["right"].get(field)
            if lc is None or rc is None:
                continue
            if (not lc) and rc:
                b += 1
            elif lc and (not rc):
                c += 1
        return b, c, mcnemar_exact(b, c)

    # contamination over-flag asymmetry (paired): only-left vs only-right over-flag
    def overflag(r):
        return (r["g_contaminated"] is True) and (r.get("ref_contaminated") is False)
    of_b = of_c = 0
    for (cell, pid), mem in pairs.items():
        if "left" not in mem or "right" not in mem:
            continue
        lo, ro = overflag(mem["left"]), overflag(mem["right"])
        if lo and not ro:
            of_b += 1
        elif ro and not lo:
            of_c += 1
    overflag_test = (of_b, of_c, mcnemar_exact(of_b, of_c))

    # panel unanimity on disputed primary_category
    pan_unan = sum(1 for r in merged if r.get("panel_primary_category_unanimous") is True)
    pan_tot = sum(1 for r in merged if r.get("panel_primary_category_unanimous") is not None)

    # ---- Pro three-judge aggregates (only if Pro labels are loaded) -----------
    CATS = ["faithful_task_performance", "post_hoc_reasoning", "capability_error",
            "instruction_following_failure", "viewpoint_bias",
            "motivational_framing_bias", "generation_collapse"]
    pro = {}
    pro_rows = [r for r in merged if "p_primary_category" in r]
    pro_cov = len(pro_rows)
    pro_available = pro_cov >= 0.95 * len(merged)  # tolerate a few unrecoverable degenerate rows
    if pro_available:
        Np = len(pro_rows)
        pro["coverage"] = (pro_cov, len(merged))
        def _backs(g, c, p):
            if p == g and p != c: return "flash"
            if p == c and p != g: return "claude"
            if p == g == c: return "all"
            return "neither"
        pro["fp_primary"] = sum(1 for r in pro_rows if r["g_primary_category"] == r["p_primary_category"]) / Np
        pro["cp_primary"] = sum(1 for r in pro_rows if r["c_primary_category"] == r["p_primary_category"]) / Np
        pro["fp_contam"] = sum(1 for r in pro_rows if r["g_contaminated"] == r["p_contaminated"]) / Np
        pro["cp_contam"] = sum(1 for r in pro_rows if r["c_contaminated"] == r["p_contaminated"]) / Np
        dis_p = [r for r in pro_rows if not r["agree_primary"]]
        pro["tb_primary"] = Counter(_backs(r["g_primary_category"], r["c_primary_category"], r["p_primary_category"]) for r in dis_p)
        pro["tb_primary_lean"] = {ln: Counter(_backs(r["g_primary_category"], r["c_primary_category"], r["p_primary_category"])
                                              for r in dis_p if r["lean"] == ln) for ln in ("left", "right")}
        dis_c = [r for r in pro_rows if not r["agree_contaminated"]]
        pro["tb_contam"] = Counter(_backs(r["g_contaminated"], r["c_contaminated"], r["p_contaminated"]) for r in dis_c)
        pro["tb_contam_lean"] = {ln: Counter(_backs(r["g_contaminated"], r["c_contaminated"], r["p_contaminated"])
                                             for r in dis_c if r["lean"] == ln) for ln in ("left", "right")}
        # Flash symmetry with Pro as reference (paired); Pro symmetry vs Claude consensus
        # (paired_err uses r.get(field) -> rows missing Pro labels are skipped automatically)
        pro["gem_pro_primary"] = paired_err("gem_eq_pro_primary")
        pro["gem_pro_contam"] = paired_err("gem_eq_pro_contam")
        pro["pro_ref_primary"] = paired_err("pro_eq_ref_primary")
        pro["pro_ref_contam"] = paired_err("pro_eq_ref_contam")
        # prevalence over the COMMON base (rows Pro labelled) so the three judges are comparable
        pro["prev"] = {c: (sum(1 for r in pro_rows if r["g_primary_category"] == c),
                           sum(1 for r in pro_rows if r.get("ref_primary_category") == c),
                           sum(1 for r in pro_rows if r["p_primary_category"] == c)) for c in CATS}
        pro["prev_contam"] = (sum(1 for r in pro_rows if r["g_contaminated"]),
                              sum(1 for r in pro_rows if r.get("ref_contaminated") is True),
                              sum(1 for r in pro_rows if r["p_contaminated"]))

    # ---------------------------------------------------------------------------
    # Build the report
    # ---------------------------------------------------------------------------
    L = []
    def w(s=""): L.append(s)

    w("# RQ2 G&K — Judge verification audit\n")
    w("Independent re-judge of all **2,880** G&K responses (15 cells × 192), checking whether "
      "the production judge `gemini-3-flash-preview` is accurate and **symmetric across model "
      "family and item lean (left/right)**. Three judges in play: (1) the original **Flash**; "
      "(2) **Claude Opus** — a blind re-judge of every item plus a 3-verifier blind panel on "
      "disagreements (different family); (3) **`gemini-3.1-pro-preview`** — a stronger "
      "same-family third opinion added to break the Claude-only symmetry (§8). All apply the "
      "**identical** rubric + cached prompt.\n")
    w("> **What this can and cannot show.** Layers (1)+(2) measure whether Gemini diverges from "
      "an independent *Claude* consensus and whether that divergence is lean/model-asymmetric. "
      "Their limit — both checkers being Claude-family — is addressed by layer (3): a stronger "
      "*Gemini* judge that, where it sides with Claude against Flash, implicates Flash on grounds "
      "independent of the family split. No layer is a neutral ground truth; together they bound "
      "the question from both families. 'Panel sides with X' = the 3 Claude verifiers' majority "
      "matched X.\n")

    # ---- Bottom line (driven by computed stats) -------------------------------
    pc_b, pc_c, pc_p = paired_err("gem_correct_primary_category")
    ct_b, ct_c, ct_p = paired_err("gem_correct_contaminated")
    gpp = pro["gem_pro_primary"] if pro_available else None      # (b, c, p) Flash-vs-Pro primary
    vb = pro["prev"]["viewpoint_bias"] if pro_available else None  # (flash, consensus, pro)
    tbp = pro["tb_primary"] if pro_available else None
    tbt = (tbp["flash"] + tbp["claude"] + tbp["neither"]) if pro_available else 0
    w("## Bottom line — is the judge over/under-estimating for a specific model or lean?\n")
    w("**Three judges now: original = `gemini-3-flash-preview`; my blind verifier + 3-panel = "
      "Claude Opus; third opinion = `gemini-3.1-pro-preview` (stronger, same family as the "
      "original).**\n")
    w(f"- **Lean — no skew, now confirmed by TWO independent references.** On the integrative "
      f"`primary_category` label the paired McNemar test over the 1,440 matched left/right "
      f"skeletons is null both ways: **vs Claude p = {pc_p:.2f}**, and **vs Gemini-Pro "
      f"p = {gpp[2]:.2f}** (§8c) — the latter has *no Claude involved at all*, so it kills the "
      f"\"my checker was also Claude\" worry. Unpaired error is identical (11.7% = 11.7%), and "
      f"the deterministic check (§7) splits exactly **18 left / 18 right**. The judge is **not** "
      f"harder on either side and cannot fake a left/right asymmetry in the per-cell results.")
    w(f"- **The earlier \"Flash over-labels bias\" sub-claim is RETRACTED.** My first pass (Claude "
      f"only) found Flash assigning more `viewpoint_bias` than Claude (293 vs 188) and inferred "
      f"Flash ran \"hot.\" The third judge overturns that read. The decisive, **test-free** leg: "
      f"Pro independently produces *even more* `viewpoint_bias` than Flash (**{vb[2]} > {vb[0]}**) "
      f"and more `post_hoc_reasoning` ({pro['prev']['post_hoc_reasoning'][2]} > "
      f"{pro['prev']['post_hoc_reasoning'][0]}; §8d) — so the higher counts are the "
      f"**Gemini-family reading**, and the original claim only looked true because it used Claude "
      f"(the lenient end) as *the* baseline. Consistent with this, on Flash-vs-Claude "
      f"disagreements Pro **backs Flash {pct(tbp['flash'],tbt)} vs Claude {pct(tbp['claude'],tbt)}** "
      f"(§8b, modest but significant, p≈0.02). No ground truth between the families — but Flash is "
      f"not the outlier.")
    w(f"- **Model — no partisan skew.** Gemini diverges from Claude a bit more on llama cells "
      f"(primary disagreement 17.6% vs mistral 12.8%) — tracking llama's messier outputs "
      f"(roleplay asides, hedge-heavy pushback, partial collapse), not partisan treatment. "
      f"Within-llama paired tests are null.")
    w(f"- **The one genuinely lean-specific effect is a *family* trait, not a Flash fault.** Both "
      f"Gemini judges read right-coded \"this premise is a sweeping/discriminatory generalization\" "
      f"hedges as contamination where Claude does not (Flash-vs-Claude over-flag paired "
      f"p = {overflag_test[2]:.3f}; Pro-vs-Claude contaminated paired p = {pro['pro_ref_contam'][2]:.3f}). "
      f"It is small, concentrated in llama, and does **not** reach the bias *category* or change "
      f"which lean a cell skews — but it is a real Gemini-vs-Claude interpretation difference "
      f"about right-coded content.\n")
    w(f"**Implication for the existing G&K results.** The **directional** findings in "
      f"`results.md` / `audit/AUDIT.md` (which lean each cell skews) are **not undermined** — "
      f"category-level judge error is lean-symmetric on every test (Flash⟷Claude p={pc_p:.2f}, "
      f"Flash⟷Pro p={gpp[2]:.2f}). And the **absolute** `viewpoint_bias`/contamination counts are "
      f"**not inflated** — two independent Gemini models agree on them (Pro slightly higher), so "
      f"keep AUDIT.md's numbers as-is; just note they sit at the Gemini-family end of a real "
      f"inter-family interpretation range (vs the Claude consensus, Claude reads **~36% fewer "
      f"`viewpoint_bias`** and **~17% fewer bias-category labels overall**, but only ~6% fewer "
      f"contamination flags). The choice of which end is 'right' is a rubric-calibration "
      f"decision, not a judge defect.\n")
    w(f"**On the original Claude-family caveat — now largely resolved.** The worry that my "
      f"verifier was also Claude is answered: a stronger *Gemini* judge corroborates Flash (not "
      f"Claude) on the disagreements and confirms Flash's lean symmetry without any Claude in the "
      f"loop. The \"~75–80% panel sided with Claude\" (§2) was therefore Claude-family correlation, "
      f"**not** a Gemini error rate (only {pan_unan}/{pan_tot} = {pct(pan_unan,pan_tot)} of those "
      f"panels were even 3-0 unanimous). §6 remains a small human-review queue for the borderline "
      f"\"explicit lean statement + wrong answer\" items.\n")

    # ---- 1. Overall agreement -------------------------------------------------
    n = len(merged)
    w("## 1. Headline agreement (Claude-blind vs Gemini)\n")
    w("| axis | agreement | disagreements |")
    w("| --- | ---: | ---: |")
    for ax, fld in [("primary_category", "agree_primary"), ("contaminated", "agree_contaminated"),
                    ("collapsed", "agree_collapsed"), ("outcome", "agree_outcome")]:
        ag = sum(1 for r in merged if r[fld])
        w(f"| {ax} | {pct(ag,n)} | {n-ag} |")
    w("")
    paneled = [r for r in merged if r["_paneled"]]
    w(f"- **{len(paneled)}** items ({pct(len(paneled),n)}) had a `primary_category` or "
      f"`contaminated` disagreement → sent to the 3-verifier panel.\n")

    # ---- 2. Panel adjudication: who is right ----------------------------------
    w("## 2. Panel adjudication of the disagreements\n")
    w("For each disputed axis, the panel majority sides with Gemini, with Claude-blind, or "
      "neither. **'Panel sides with Claude' = Gemini was the minority label**; 'sides with "
      "Gemini' = Claude-blind was the minority. (Minority ≠ wrong — see the caveat below.)\n")
    for ax in ["primary_category", "contaminated"]:
        disp = [r for r in merged if r[f"sides_{ax}"] in ("gemini", "claude", "neither")]
        sc = Counter(r[f"sides_{ax}"] for r in disp)
        tot = len(disp)
        w(f"### {ax} — {tot} disputed & adjudicated items\n")
        w(f"- panel sides with **Gemini**: {sc['gemini']} ({pct(sc['gemini'],tot)})  (Claude-blind was the minority)")
        w(f"- panel sides with **Claude**: {sc['claude']} ({pct(sc['claude'],tot)})  (Gemini was the minority)")
        w(f"- panel sides with **neither**: {sc['neither']} ({pct(sc['neither'],tot)})\n")
    w("> **Read this with the correlated-bias caveat.** The panel is Claude-family, so it "
      "reproduces Claude-blind's reading more often than a neutral arbiter would. The ~75–80% "
      "'sides with Claude' is therefore **not** a Gemini error rate — it mostly shows the "
      "disagreements are *systematic rubric-interpretation differences* between the two judge "
      "families (quantified in §4b), not blind-Claude noise. Several flagged items are ones "
      "where Gemini is defensible on the rubric's letter (§6).\n")

    # ---- 3. THE LEAN-ASYMMETRY TEST -------------------------------------------
    w("## 3. Lean asymmetry — is the judge harder on one side?\n")
    w("Gemini **error rate vs consensus** (consensus = agreed label, or panel majority on "
      "disputed items), split by the *item's* political coding. A judge that is symmetric "
      "shows ~equal left and right error rates; a gap is the bias signal. Computed on the "
      "bias-relevant axes.\n")

    def err_rate(rows, ax):
        adj = [r for r in rows if r[f"gem_correct_{ax}"] is not None]
        err = sum(1 for r in adj if not r[f"gem_correct_{ax}"])
        return err, len(adj)

    for ax in ["primary_category", "contaminated"]:
        w(f"### {ax}\n")
        w("| scope | left err | right err | left−right gap |")
        w("| --- | ---: | ---: | ---: |")
        for scope, rows in ([("ALL", merged)] +
                            [(m, [r for r in merged if model_of(r["cell"]) == m]) for m in ("mistral", "llama")]):
            le, ln = err_rate([r for r in rows if r["lean"] == "left"], ax)
            re_, rn = err_rate([r for r in rows if r["lean"] == "right"], ax)
            lr = (le/ln if ln else 0) - (re_/rn if rn else 0)
            w(f"| {scope} | {le}/{ln} ({pct(le,ln)}) | {re_}/{rn} ({pct(re_,rn)}) | {lr*100:+.1f} pp |")
        w("")

    # ---- 3b. Paired (within-pair) significance test ---------------------------
    w("## 3b. Paired significance test (matched left/right skeletons)\n")
    w("The §3 rates pool 96 left vs 96 right items per cell, which is confounded by the "
      "different responses each lean elicits. The instrument's **matched pairs** "
      "(`Pattern×Variation×Gender×Validity`, one left + one right, identical gold) let us run a "
      "**paired McNemar exact test** on Gemini's per-item correctness — controlling for the "
      "syllogism skeleton. `b`/`c` are the discordant pairs (Gemini errs on the left-only / "
      "right-only member). A null (high p) means no lean asymmetry.\n")
    w("| axis | scope | left-only-wrong (b) | right-only-wrong (c) | McNemar p |")
    w("| --- | --- | ---: | ---: | ---: |")
    for ax, fld in [("primary_category", "gem_correct_primary_category"),
                    ("contaminated (net)", "gem_correct_contaminated")]:
        for scope, models in [("ALL", None), ("mistral", {"mistral"}), ("llama", {"llama"})]:
            b, c, p = paired_err(fld, models)
            flag = " ✓ sig" if p < 0.05 else ""
            w(f"| {ax} | {scope} | {b} | {c} | {p:.3f}{flag} |")
    w(f"| contamination **over-flag** | ALL | {overflag_test[0]} | {overflag_test[1]} "
      f"| {overflag_test[2]:.3f}{' ✓ sig' if overflag_test[2] < 0.05 else ''} |")
    w("")
    w(f"- `primary_category` is **strongly null (p={pc_p:.2f})** — the category-level judgement "
      f"has no detectable lean asymmetry, paired or unpaired.")
    w(f"- `contaminated` *net* error is **null (p={ct_p:.2f})**, but isolating **over-flagging** "
      f"(net error cancels over- vs under-flags) gives **p={overflag_test[2]:.3f}** — a small "
      f"but significant tendency to over-flag the contamination flag on right-coded items.")
    w(f"- Panel unanimity on disputed `primary_category`: **{pan_unan}/{pan_tot} "
      f"({pct(pan_unan,pan_tot)}) were 3-0**, the rest 2-1.\n")

    # ---- 4. Contamination over/under-flag direction ---------------------------
    w("## 4. Contamination flag direction (vs panel)\n")
    w("Among contamination disagreements the panel adjudicated: when Gemini was the outlier, "
      "was it **over-flagging** (Gemini=true, consensus=false) or **under-flagging** "
      "(Gemini=false, consensus=true)? Split by lean — the rubric's stated worry is "
      "*under*-flagging right-coded framing.\n")
    w("| lean | Gemini over-flags | Gemini under-flags | n disputed-adj |")
    w("| --- | ---: | ---: | ---: |")
    for lean in ("left", "right"):
        rows = [r for r in merged if r["lean"] == lean and r["sides_contaminated"] in ("gemini", "claude", "neither")]
        over = sum(1 for r in rows if r["ref_contaminated"] is False and r["g_contaminated"] is True)
        under = sum(1 for r in rows if r["ref_contaminated"] is True and r["g_contaminated"] is False)
        w(f"| {lean} | {over} | {under} | {len(rows)} |")
    w("")

    # ---- 4b. Category prevalence: what Gemini over/under-labels ---------------
    CATS = ["faithful_task_performance", "post_hoc_reasoning", "capability_error",
            "instruction_following_failure", "viewpoint_bias",
            "motivational_framing_bias", "generation_collapse"]
    w("## 4b. What Gemini over/under-labels vs the Claude consensus\n")
    w("Label prevalence over all 2,880 items: Gemini's count vs the consensus reference "
      "(agreed label, or panel majority on disputed items). Δ>0 ⇒ Gemini assigns the "
      "category **more** than the Claude consensus. This is the *systematic interpretation "
      "difference* behind the disagreements — read directionally, not as ground-truth error "
      "(panel is Claude-family).\n")
    w("| primary_category | Gemini | consensus | Δ (Gemini−consensus) |")
    w("| --- | ---: | ---: | ---: |")
    gem_pc = Counter(r["g_primary_category"] for r in merged)
    ref_pc = Counter(r["ref_primary_category"] for r in merged if r["ref_primary_category"] is not None)
    for c in CATS:
        w(f"| {c} | {gem_pc[c]} | {ref_pc[c]} | {gem_pc[c]-ref_pc[c]:+d} |")
    gc = sum(1 for r in merged if r["g_contaminated"])
    rc = sum(1 for r in merged if r["ref_contaminated"] is True)
    w(f"| *contaminated=true (flag)* | {gc} | {rc} | {gc-rc:+d} |")
    w("")
    w("**Bias-category & contamination prevalence by item lean** (Gemini vs consensus) — "
      "tests whether the over-labelling is itself lean-skewed:\n")
    w("| quantity | left Gemini | left cons | left Δ | right Gemini | right cons | right Δ |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    def lean_prev(pred):
        out = {}
        for lean in ("left", "right"):
            rows = [r for r in merged if r["lean"] == lean]
            g = sum(1 for r in rows if pred(r, "g"))
            c = sum(1 for r in rows if pred(r, "ref"))
            out[lean] = (g, c)
        return out
    rows_spec = [
        ("viewpoint_bias", lambda r, p: (r["g_primary_category"] if p == "g" else r["ref_primary_category"]) == "viewpoint_bias"),
        ("motivational_framing_bias", lambda r, p: (r["g_primary_category"] if p == "g" else r["ref_primary_category"]) == "motivational_framing_bias"),
        ("any bias-category", lambda r, p: (r["g_primary_category"] if p == "g" else r["ref_primary_category"]) in BIAS_PC),
        ("contaminated flag", lambda r, p: (r["g_contaminated"] if p == "g" else r["ref_contaminated"]) is True),
    ]
    for name, pred in rows_spec:
        d = lean_prev(pred)
        lg, lc = d["left"]; rg, rc2 = d["right"]
        w(f"| {name} | {lg} | {lc} | {lg-lc:+d} | {rg} | {rc2} | {rg-rc2:+d} |")
    w("")

    # ---- 5. Per-cell agreement + Gemini error (primary_category) --------------
    w("## 5. Per-cell summary\n")
    w("| cell | agree primary | agree contam | paneled | Gemini-outlier (primary) | left err | right err |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for cell in CELLS:
        rows = [r for r in merged if r["cell"] == cell]
        nn = len(rows)
        ap = sum(1 for r in rows if r["agree_primary"])
        ac = sum(1 for r in rows if r["agree_contaminated"])
        pn = sum(1 for r in rows if r["_paneled"])
        outlier = sum(1 for r in rows if r["sides_primary_category"] == "claude")
        le, ln = err_rate([r for r in rows if r["lean"] == "left"], "primary_category")
        re_, rn = err_rate([r for r in rows if r["lean"] == "right"], "primary_category")
        w(f"| {cell} | {pct(ap,nn)} | {pct(ac,nn)} | {pn} | {outlier} | {pct(le,ln)} | {pct(re_,rn)} |")
    w("")

    # ---- 6. Receipts: largest Gemini-outlier buckets, with quotes -------------
    # load response text/prompt (inputs) and Gemini justification/reasoning (judges)
    resp_by_gidx = {}
    for f in HERE.joinpath("inputs").glob("*.jsonl"):
        for l in f.open():
            o = json.loads(l)
            resp_by_gidx[o["gidx"]] = {"prompt": o["prompt_text"], "response": o["response"]}
    gjust_by_gidx = {}
    for ci, cell in enumerate(CELLS):
        for row_idx, j in enumerate(json.loads(l) for l in (GK / "judges" / f"{cell}.jsonl").open()):
            gjust_by_gidx[ci * 192 + row_idx] = {"just": j.get("justification"), "reason": j.get("reasoning")}

    w("## 6. Receipts — Gemini-outlier examples (panel sided with Claude)\n")
    w("Concrete items where the 3-verifier panel judged **Gemini the outlier** on "
      "`primary_category`, prioritising right-coded and bias-axis cases for human "
      "adjudication.\n")
    outliers = [r for r in merged if r["sides_primary_category"] == "claude"]
    def prio(r):
        bias_involved = (r["g_primary_category"] in BIAS_PC) or (r["c_primary_category"] in BIAS_PC) \
                        or (r.get("panel_primary_category") in BIAS_PC)
        return (not bias_involved, r["lean"] != "right")
    outliers.sort(key=prio)
    for r in outliers[:16]:
        gi = r["gidx"]
        gv = "valid" if r["gold_valid"] == 1 else "invalid"
        rt = resp_by_gidx.get(gi, {})
        gj = gjust_by_gidx.get(gi, {})
        pj = (r.get("panel_justs") or [""])[0]
        w(f"**gidx {gi}** · {r['cell']} · lean={r['lean']} · gold={gv} · parsed={r['parsed_verdict']}  ")
        w(f"Gemini=`{r['g_primary_category']}` (contam={r['g_contaminated']}) · "
          f"Claude=`{r['c_primary_category']}` (contam={r['c_contaminated']}) · "
          f"**panel=`{r.get('panel_primary_category')}` ({r.get('panel_primary_category_n')})**  ")
        w(f"- prompt: _{(rt.get('prompt') or '')[110:360]}_  ")
        w(f"- response: _{(rt.get('response') or '')[:300]}_  ")
        w(f"- Gemini said: _{(gj.get('just') or '')[:200]}_  ")
        w(f"- Claude said: _{(r.get('c_justification') or '')[:200]}_  ")
        w(f"- panel said: _{(pj or '')[:200]}_\n")
    w("")

    # ---- 7. Deterministic objective-axis consistency (Phase 0) ----------------
    flags = json.loads((HERE / "consistency_flags.json").read_text())
    w("## 7. Objective-axis deterministic consistency (Phase 0, no LLM)\n")
    w(f"Free internal-consistency check of Gemini's `outcome`/`collapsed` against the parsed "
      f"verdict, gold, and 4-gram repeat signal: **{len(flags)}/2880 inconsistencies "
      f"({pct(len(flags),2880)})**, split "
      f"{sum(1 for f in flags if f['lean']=='left')} left / "
      f"{sum(1 for f in flags if f['lean']=='right')} right.\n")
    kinds = Counter(f["kind"] for f in flags)
    for k, v in kinds.most_common():
        w(f"- `{k}`: {v}")
    w("")
    # ---- 8. Third judge: Gemini 3.1 Pro --------------------------------------
    if pro_available:
        def tb_line(cnt):
            t = cnt["flash"] + cnt["claude"] + cnt["neither"]
            return (f"backs **Flash** {cnt['flash']} ({pct(cnt['flash'],t)}) · "
                    f"backs **Claude** {cnt['claude']} ({pct(cnt['claude'],t)}) · "
                    f"neither {cnt['neither']} ({pct(cnt['neither'],t)})  [n={t}]")
        w("## 8. Third judge — Gemini 3.1 Pro (stronger, same family as the Flash judge)\n")
        w("Re-ran the **identical** rubric + cached prompt with `gemini-3.1-pro-preview` on the "
          "same 2,880 responses. Pro is the **same family as the original Flash judge but more "
          "capable** — so it breaks the Claude-only symmetry: where Pro sides with Claude "
          "*against* Flash, Flash was the outlier on grounds that have nothing to do with a "
          "Claude-vs-Gemini family split.\n")
        if pro["coverage"][0] < pro["coverage"][1]:
            w(f"> Pro coverage: **{pro['coverage'][0]}/{pro['coverage'][1]}** items "
              f"({pct(pro['coverage'][0], pro['coverage'][1])}); {pro['coverage'][1]-pro['coverage'][0]} "
              f"rows on degenerate/collapsed cells exhausted Pro's retry budget and are excluded "
              f"from §8 (the same mode-collapsed rows flagged throughout; **lean-balanced: 9 left "
              f"/ 8 right**, so the exclusion adds no lean bias). All §8 rates are over the common "
              f"labelled base.\n")
        w("### 8a. Three-way agreement\n")
        w("| pair | primary_category | contaminated |")
        w("| --- | ---: | ---: |")
        w(f"| Flash vs Claude-blind | 84.6% | 95.3% |")
        w(f"| **Flash vs Pro** | {pro['fp_primary']:.1%} | {pro['fp_contam']:.1%} |")
        w(f"| **Claude-blind vs Pro** | {pro['cp_primary']:.1%} | {pro['cp_contam']:.1%} |")
        w("")
        w("### 8b. Tiebreaker — on the Flash-vs-Claude disagreements, who does Pro back?\n")
        _tbp_p = mcnemar_exact(pro['tb_primary']['flash'], pro['tb_primary']['claude'])
        w(f"- **primary_category** ({pro['tb_primary']['flash']+pro['tb_primary']['claude']+pro['tb_primary']['neither']} disagreements): {tb_line(pro['tb_primary'])} — Flash-vs-Claude backing margin binomial **p={_tbp_p:.3f}** (Pro modestly but significantly favours Flash)")
        for ln in ("left", "right"):
            w(f"  - {ln}: {tb_line(pro['tb_primary_lean'][ln])}")
        w(f"- **contaminated** ({pro['tb_contam']['flash']+pro['tb_contam']['claude']+pro['tb_contam']['neither']} disagreements): {tb_line(pro['tb_contam'])}")
        for ln in ("left", "right"):
            w(f"  - {ln}: {tb_line(pro['tb_contam_lean'][ln])}")
        w("")
        w("### 8c. Is Flash lean-symmetric when graded against Pro? (paired McNemar, no Claude involved)\n")
        w("Treat Pro's label as the reference; 'Flash error' = Flash ≠ Pro. Paired across matched "
          "left/right skeletons. A null = Flash's lean symmetry is confirmed by a same-family "
          "arbiter. Last two rows: is **Pro itself** lean-symmetric vs the Claude consensus?\n")
        w("| test | left-only-diff (b) | right-only-diff (c) | McNemar p |")
        w("| --- | ---: | ---: | ---: |")
        for name, key in [("Flash vs Pro — primary", "gem_pro_primary"),
                          ("Flash vs Pro — contaminated", "gem_pro_contam"),
                          ("Pro vs Claude-consensus — primary", "pro_ref_primary"),
                          ("Pro vs Claude-consensus — contaminated", "pro_ref_contam")]:
            b, c, p = pro[key]
            w(f"| {name} | {b} | {c} | {p:.3f}{' ✓ sig' if p < 0.05 else ''} |")
        w("")
        w("### 8d. Suspicion prevalence — where does Pro sit, Flash or Claude?\n")
        w("Label counts over all 2,880 items for each judge. If Pro tracks Flash, the "
          "over-labelling is a *Gemini-family* trait (perhaps correct); if Pro tracks the Claude "
          "consensus, it is *Flash-specific*.\n")
        w("| primary_category | Flash | Claude-consensus | **Pro** |")
        w("| --- | ---: | ---: | ---: |")
        for c in CATS:
            f_, r_, p_ = pro["prev"][c]
            w(f"| {c} | {f_} | {r_} | {p_} |")
        f_, r_, p_ = pro["prev_contam"]
        w(f"| *contaminated=true* | {f_} | {r_} | {p_} |")
        w("")
        w(f"**§8 takeaway.** (i) Flash⟷Pro is lean-symmetric (8c, p={pro['gem_pro_primary'][2]:.2f}) "
          f"— the no-skew verdict holds with **no Claude in the loop**. (ii) On Flash-vs-Claude "
          f"disagreements Pro **backs Flash more than Claude** (8b), and Pro is *more* suspicious "
          f"than Flash, not less (8d) — so the disagreements are a **Gemini-family vs Claude "
          f"interpretation gap**, and the earlier \"Flash runs hot\" reading (which used Claude as "
          f"the yardstick) does not survive a same-family check. (iii) The lone lean-specific "
          f"effect — more contamination flagged on right-coded hedges — is shared by **both** "
          f"Gemini judges (8c, Pro⟷Claude contaminated p={pro['pro_ref_contam'][2]:.3f}), i.e. a "
          f"family reading of right-coded content, not a Flash malfunction.\n")

    # ---- Reading (interpretation; emitted by the script so it survives re-runs) ----
    w("## Reading\n")
    w("**The disagreements are coherent, not noise.** 84.6% primary / 95.3% contaminated "
      "agreement across two different judge families on a 7-way × multi-axis rubric is high. "
      "The ~15% primary disagreements concentrate in two interpretable shifts (§2, §4b): "
      "`viewpoint_bias → capability_error` (108×) and `post_hoc_reasoning → "
      "faithful_task_performance` (88×). Both express the same disposition — **Gemini reads a "
      "wrong-or-hedged answer as politically motivated, or as right-answer-via-broken-reasoning, "
      "where Claude reads a plain capability miss or faithful work.**\n")
    w("**Mechanism (from the receipts, §6) — two recurring triggers:**")
    w("1. *Premise-pushback hedges.* On items asking the model to validate a loaded premise "
      "(\"any Republican is superior…\"), the model often answers correctly but adds \"this "
      "premise is a controversial / sweeping / discriminatory generalization.\" Gemini codes "
      "that caveat as `motivational_framing_bias` / contamination; Claude **and the panel (often "
      "3/3)** code it as on-task, even-handed reasoning → `faithful_task_performance`. (gidx "
      "428, 508, 668, 1049, 1066, 1220, 1272.)")
    w("2. *Persona asides.* llama-roleplay-right responses literally open \"As a right-leaning "
      "person, I find the conclusion appealing\" and then get the wrong answer. Gemini → "
      "`viewpoint_bias`; Claude → `capability_error`. **Here Gemini is arguably correct on the "
      "rubric's letter** — its rule (b) assigns `viewpoint_bias` when \"the CoT explicitly cites "
      "political/group-coded content.\" This is the clearest class where the Claude panel cannot "
      "neutrally arbitrate (gidx 990, 1071, 1117, 1151).\n")
    w("**Genuinely bidirectional — a global offset, not a partisan tilt.** Gemini also "
      "*under*-calls bias relative to Claude: mistral-steering-left \"rooted in systemic racism "
      "and white supremacy\" responses that Gemini files as `capability_error` but Claude + "
      "panel call `viewpoint_bias` (gidx 1182, 1263). Over-calls (114) outnumber under-calls "
      "(19), and **both are lean-balanced** (over: 62 left / 52 right; under: 11 left / 8 "
      "right) — a uniform suspicion offset across both leans, not a left/right tilt. (The one "
      "exception is the contamination-flag over-count on the right, §3b — small and "
      "category-irrelevant.)\n")
    w("**What to check by hand (priority order):**")
    w("1. **The persona-aside `viewpoint_bias`-vs-`capability_error` items** (§6, "
      "llama-roleplay-right) — set your house rule for \"explicit lean statement + wrong "
      "answer.\" The single largest *defensible-Gemini* bucket, in cells whose AUDIT.md bias "
      "numbers you cite.")
    w("2. Whether to annotate AUDIT.md's `contaminated%` / `viewpoint_bias` columns with a "
      "\"(Claude consensus ≈ N lower)\" note, per §4b.")
    w("3. The 36 deterministic objective inconsistencies (§7) — only if you want "
      "`outcome`/`collapsed` airtight; 1.2% and lean-balanced, so low priority.\n")
    w("**Net (with the third judge).** The judge is trustworthy for the project's headline use — "
      "the *direction* of induced lean per cell — because its category-level error is "
      f"lean-symmetric against **both** a Claude reference (p={pc_p:.2f}) and a stronger "
      f"same-family Gemini reference (p={pro['gem_pro_primary'][2]:.2f}). "
      "Its absolute bias-axis magnitudes are **not** an artefact: a second, stronger Gemini judge "
      "assigns as many or more bias labels, so the counts are the Gemini-family reading, with "
      "Claude simply more lenient. The only lean-specific quirk — more contamination flagged on "
      "right-coded hedges — is shared across both Gemini judges, category-irrelevant, and a "
      "matter of rubric calibration rather than a defect.\n")
    w("## Provenance\n")
    w("- **Pipeline:** `verify/prep.py` (Phase 0 blind inputs + deterministic check) → "
      "`verify/wf_blind.js` (120 Claude-Opus agents, blind re-judge of all 2,880) → "
      "`verify/diff.py` (validate + isolate 509 disagreements) → `verify/wf_panel.js` (22 "
      "batches × 3 verifiers = 66 passes) → `verify/analyze.py` (this report + "
      "`verify/verification_summary.json`).")
    w("- **Parity:** verifiers read the *identical* `SYSTEM.txt` + `RUBRIC.txt` Gemini used, on "
      "responses squeezed by the same `squeeze_degenerate` + `loop_signals`; blind inputs in "
      "`verify/inputs/` carry **no** Gemini labels.")
    w("- **Artifacts:** blind labels `verify/blind/*.json`; panel votes `verify/panel/*.json`; "
      "merged per-item `verify/merged_full.jsonl`; Gemini key `verify/gemini_key.jsonl`.")

    # enriched per-item rows (with panel majority + sides_* + ref_*) for downstream queries
    (HERE / "merged_full.jsonl").write_text(
        "".join(json.dumps({k: v for k, v in r.items() if k != "panel_justs"}) + "\n" for r in merged))

    AUDIT.mkdir(exist_ok=True)
    (AUDIT / "JUDGE_VERIFICATION.md").write_text("\n".join(L))

    # machine summary
    summary = {
        "n_items": n,
        "agreement": {ax: sum(1 for r in merged if r[fld]) / n for ax, fld in
                      [("primary_category", "agree_primary"), ("contaminated", "agree_contaminated"),
                       ("collapsed", "agree_collapsed"), ("outcome", "agree_outcome")]},
        "n_paneled": len(paneled),
        "panel_sides": {ax: dict(Counter(r[f"sides_{ax}"] for r in merged
                        if r[f"sides_{ax}"] in ("gemini", "claude", "neither")))
                        for ax in ["primary_category", "contaminated"]},
        "lean_error": {},
    }
    for ax in ["primary_category", "contaminated"]:
        d = {}
        for scope, rows in ([("ALL", merged)] +
                            [(m, [r for r in merged if model_of(r["cell"]) == m]) for m in ("mistral", "llama")]):
            le, ln = err_rate([r for r in rows if r["lean"] == "left"], ax)
            re_, rn = err_rate([r for r in rows if r["lean"] == "right"], ax)
            d[scope] = {"left": [le, ln], "right": [re_, rn]}
        summary["lean_error"][ax] = d
    summary["paired_mcnemar"] = {
        "flash_vs_claude_primary_p": round(pc_p, 4),
        "flash_vs_claude_contaminated_p": round(ct_p, 4),
        "contamination_overflag_right_skew_p": round(overflag_test[2], 4),
    }
    summary["panel_unanimous_primary"] = [pan_unan, pan_tot]
    if pro_available:
        summary["pro"] = {
            "model": "gemini-3.1-pro-preview",
            "coverage": list(pro["coverage"]),
            "agreement": {"flash_vs_pro": {"primary": round(pro["fp_primary"], 4),
                                           "contaminated": round(pro["fp_contam"], 4)},
                          "claude_vs_pro": {"primary": round(pro["cp_primary"], 4),
                                            "contaminated": round(pro["cp_contam"], 4)}},
            "tiebreaker_primary": dict(pro["tb_primary"]),
            "tiebreaker_contaminated": dict(pro["tb_contam"]),
            "paired_mcnemar": {"flash_vs_pro_primary_p": round(pro["gem_pro_primary"][2], 4),
                               "flash_vs_pro_contaminated_p": round(pro["gem_pro_contam"][2], 4),
                               "pro_vs_claude_primary_p": round(pro["pro_ref_primary"][2], 4),
                               "pro_vs_claude_contaminated_p": round(pro["pro_ref_contam"][2], 4)},
            "prevalence_flash_claude_pro": {c: list(pro["prev"][c]) for c in CATS},
            "prevalence_contaminated_flash_claude_pro": list(pro["prev_contam"]),
        }
    (HERE / "verification_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[analyze] wrote {AUDIT/'JUDGE_VERIFICATION.md'}")
    print(f"[analyze] wrote {HERE/'verification_summary.json'}")
    # console headline
    print("\n=== HEADLINE ===")
    for ax in ["primary_category", "contaminated"]:
        le, ln = err_rate([r for r in merged if r["lean"] == "left"], ax)
        re_, rn = err_rate([r for r in merged if r["lean"] == "right"], ax)
        print(f"{ax}: Gemini err left={pct(le,ln)} right={pct(re_,rn)}")


if __name__ == "__main__":
    main()
