"""Stdlib-only test for scale_sweep.run_scale_sweep (no torch / no pytest).

Run: python3 tests/test_scale_sweep.py   (exits non-zero on failure)
Imports only the DI core, so it needs neither a GPU nor the judge deps.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import scale_sweep  # noqa: E402

QS = ["q1", "q2", "q3"]


def make_fakes():
    state = {"applied": [], "gen_calls": 0}

    def chat_fn(tok, q):
        return f"PROMPT::{q}"

    def apply_scale(model, s):
        state["applied"].append(s)

    def generate_fn(model, tok, prompts, max_new_tokens):
        state["gen_calls"] += 1
        cur = state["applied"][-1]          # tag response with the active scale
        return [f"resp@{cur}::{p}" for p in prompts]

    return state, chat_fn, apply_scale, generate_fn


def test_basic_assembly():
    state, chat_fn, apply_scale, generate_fn = make_fakes()
    # judge is just a base offset; score = base + index
    def score_fn(judge, qa):
        return [judge["base"] + i for i in range(len(qa))]

    per = scale_sweep.run_scale_sweep(
        model=object(), tok=object(), questions=QS,
        trait_judge={"base": 50}, coh_judge={"base": 70},
        scales=[0.5, 1.0, 2.0], apply_scale=apply_scale,
        chat_fn=chat_fn, generate_fn=generate_fn, score_fn=score_fn,
    )

    # one apply per scale, in order; one generate per scale
    assert state["applied"] == [0.5, 1.0, 2.0], state["applied"]
    assert state["gen_calls"] == 3
    assert set(per.keys()) == {0.5, 1.0, 2.0}

    e = per[1.0]
    assert e["n_trait_parsed"] == 3 and e["n_coh_parsed"] == 3
    assert math.isclose(e["trait_mean"], (50 + 51 + 52) / 3)
    assert math.isclose(e["coh_mean"], (70 + 71 + 72) / 3)
    assert len(e["per_question"]) == 3
    assert e["per_question"][0]["question"] == "q1"
    # raw response is persisted and reflects the active scale
    assert e["per_question"][0]["response"] == "resp@1.0::PROMPT::q1"
    assert e["per_question"][2]["trait_score"] == 52
    assert e["per_question"][2]["coh_score"] == 72


def test_none_scores_dropped_from_mean_but_kept_in_raw():
    state, chat_fn, apply_scale, generate_fn = make_fakes()
    def score_fn(judge, qa):
        return [10.0, None, 20.0]            # middle item unparseable

    per = scale_sweep.run_scale_sweep(
        model=object(), tok=object(), questions=QS,
        trait_judge={"base": 0}, coh_judge={"base": 0},
        scales=[1.0], apply_scale=apply_scale,
        chat_fn=chat_fn, generate_fn=generate_fn, score_fn=score_fn,
    )
    e = per[1.0]
    assert e["n_trait_parsed"] == 2            # None excluded from count
    assert math.isclose(e["trait_mean"], 15.0)  # (10+20)/2
    assert e["per_question"][1]["trait_score"] is None  # raw None preserved
    assert len(e["per_question"]) == 3


def test_empty_scores_give_nan_not_crash():
    state, chat_fn, apply_scale, generate_fn = make_fakes()
    def score_fn(judge, qa):
        return [None, None, None]

    per = scale_sweep.run_scale_sweep(
        model=object(), tok=object(), questions=QS,
        trait_judge={"base": 0}, coh_judge={"base": 0},
        scales=[1.0], apply_scale=apply_scale,
        chat_fn=chat_fn, generate_fn=generate_fn, score_fn=score_fn,
    )
    assert math.isnan(per[1.0]["trait_mean"])
    assert per[1.0]["n_trait_parsed"] == 0


if __name__ == "__main__":
    test_basic_assembly()
    test_none_scores_dropped_from_mean_but_kept_in_raw()
    test_empty_scores_give_nan_not_crash()
    print("ALL PASS")
