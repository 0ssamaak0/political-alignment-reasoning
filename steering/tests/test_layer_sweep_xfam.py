"""Tests for the family-parameterized layer_sweep.py (no GPU / no network).

Covers:
- family_resources(family) → (repo, vector path template) for both families
- best_layer_excl_last_n excludes the highest-indexed N layers
- build_layer_sweep_audit picks 3 items (low / median / high) per layer
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
sys.path.insert(0, str(_SRC))

import build_layer_sweep_audit as audit_mod  # noqa: E402

# We can't import layer_sweep directly because it pulls in torch + transformers
# at module load. Instead, lift the two pure-Python helpers under test by
# reading the module's source via the stdlib AST machinery is overkill; just
# import the two functions through importlib with the heavy imports patched.


def _import_layer_sweep_minimal(monkeypatch):
    """Import layer_sweep with torch / transformers / gemini_judge stubbed."""
    import types
    fake_torch = types.ModuleType("torch")
    fake_torch.inference_mode = lambda: (lambda f: f)
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.Tensor = object
    fake_torch.dtype = type("dtype", (), {})
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoModelForCausalLM = type("AutoModelForCausalLM", (), {})
    fake_tf.AutoTokenizer = type("AutoTokenizer", (), {})
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    fake_steering = types.ModuleType("steering")
    fake_steering.load_vector = lambda *a, **k: None
    fake_steering.make_steerer = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "steering", fake_steering)
    fake_tq = types.ModuleType("trait_questions")
    fake_tq.load_eval_questions = lambda *a, **k: []
    fake_tq.load_eval_prompt = lambda *a, **k: ""
    monkeypatch.setitem(sys.modules, "trait_questions", fake_tq)
    fake_gj = types.ModuleType("gemini_judge")
    fake_gj.GeminiJudge = type("GeminiJudge", (), {})
    monkeypatch.setitem(sys.modules, "gemini_judge", fake_gj)
    # Now safe to import
    import importlib
    if "layer_sweep" in sys.modules:
        del sys.modules["layer_sweep"]
    return importlib.import_module("layer_sweep")


def test_family_resources_mistral(monkeypatch):
    mod = _import_layer_sweep_minimal(monkeypatch)
    repo, vec_template = mod.family_resources("mistral")
    assert "Mistral-7B-Instruct-v0.2" in repo
    assert "shared/vectors/mistral/" in vec_template
    assert "{direction}" in vec_template
    assert vec_template.format(direction="left").endswith("left_leaning_response_avg_diff.pt")


def test_family_resources_llama(monkeypatch):
    mod = _import_layer_sweep_minimal(monkeypatch)
    repo, vec_template = mod.family_resources("llama")
    assert "Meta-Llama-3-8B-Instruct" in repo
    assert "shared/vectors/llama/" in vec_template
    assert vec_template.format(direction="right").endswith("right_leaning_response_avg_diff.pt")


def test_family_resources_unknown(monkeypatch):
    mod = _import_layer_sweep_minimal(monkeypatch)
    with pytest.raises(KeyError):
        mod.family_resources("falcon")


def test_best_layer_excl_last_n_simple(monkeypatch):
    mod = _import_layer_sweep_minimal(monkeypatch)
    # L32 has the max (extraction artifact), but L17 is the real peak
    per_layer = {L: {"mean": float(L)} for L in range(1, 31)}
    per_layer[17] = {"mean": 89.0}
    per_layer[31] = {"mean": 35.0}
    per_layer[32] = {"mean": 100.0}    # artifact: max raw
    assert mod.best_layer_excl_last_n(per_layer, n_excluded=2) == 17  # excludes 31, 32
    assert mod.best_layer_excl_last_n(per_layer, n_excluded=0) == 32  # without exclusion, raw wins


def test_best_layer_excl_last_n_string_keys(monkeypatch):
    """JSON round-trips int keys as strings — make sure the helper handles that."""
    mod = _import_layer_sweep_minimal(monkeypatch)
    per_layer = {str(L): {"mean": float(L)} for L in (1, 17, 31, 32)}
    per_layer["32"]["mean"] = 100.0
    per_layer["17"]["mean"] = 89.0
    assert mod.best_layer_excl_last_n(per_layer, n_excluded=2) == 17


def test_pick_three_low_median_high():
    items = [{"score": s, "response": f"r{s}", "question": "q"} for s in (10, 30, 50, 70, 90)]
    picks = audit_mod._pick_three(items)
    labels = [lbl for lbl, _ in picks]
    scores = [it["score"] for _, it in picks]
    assert labels == ["low", "median", "high"]
    assert scores == [10, 50, 90]


def test_pick_three_skips_none_scores():
    items = [
        {"score": None, "response": "skipped", "question": "q"},
        {"score": 42,   "response": "kept",    "question": "q"},
    ]
    picks = audit_mod._pick_three(items)
    assert len(picks) == 3
    assert all(it["score"] == 42 for _, it in picks)


def test_pick_three_empty_returns_empty():
    assert audit_mod._pick_three([]) == []


def test_summarise_truncates_long_text():
    long = "x" * 5000
    out = audit_mod._summarise(long)
    assert "…[truncated]" in out
    assert len(out) < 700


def test_summarise_strips_newlines():
    out = audit_mod._summarise("line1\nline2\n  ")
    assert "\n" not in out
    assert out.endswith("line2")
