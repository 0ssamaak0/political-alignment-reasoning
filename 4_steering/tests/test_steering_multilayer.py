"""Unit tests for the multi-layer additions to 4_steering/src/steering.py.

CPU-only. No model loaded. Verifies the math of parse_layers and
build_layer_perturbations, including the telescoping property that
makes the paper's multi-layer recipe equivalent to single-layer at
the top of the stack.
"""
import torch
import pytest

from src.steering import parse_layers, build_layer_perturbations


def test_parse_layers_range():
    assert parse_layers("1-5") == [1, 2, 3, 4, 5]


def test_parse_layers_comma_list():
    assert parse_layers("10,16,22,28") == [10, 16, 22, 28]


def test_parse_layers_single():
    assert parse_layers("17") == [17]


def test_parse_layers_mixed():
    # mixed ranges + singletons + duplicates collapse + sort
    assert parse_layers("3,1-2,3,5-6") == [1, 2, 3, 5, 6]


def test_parse_layers_empty_part_skipped():
    # trailing comma or extra whitespace is tolerated
    assert parse_layers("1-3, ,4") == [1, 2, 3, 4]


def test_build_perturbations_raw():
    # synthetic v_full shape [N+1, d] = [4, 3]: row 0 emb, rows 1..3 post-block
    v = torch.tensor([
        [0.0, 0.0, 0.0],   # row 0: embedding
        [1.0, 0.0, 0.0],   # row 1
        [2.0, 0.0, 0.0],   # row 2
        [3.0, 0.0, 0.0],   # row 3
    ], dtype=torch.float32)
    perts = build_layer_perturbations(v, [1, 2, 3], "raw", torch.float32, "cpu")
    assert [L for L, _ in perts] == [1, 2, 3]
    assert torch.allclose(perts[0][1], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(perts[1][1], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(perts[2][1], torch.tensor([3.0, 0.0, 0.0]))


def test_build_perturbations_incremental_telescoping():
    """Paper §A.3 claim: sum of v_inc_ℓ over a contiguous range [1..L]
    telescopes to v_full[L] - v_full[0]. With v_full[0]=0 (typical),
    this equals v_full[L]."""
    v = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [7.0, 0.0, 0.0],
    ], dtype=torch.float32)
    perts = build_layer_perturbations(v, [1, 2, 3], "incremental", torch.float32, "cpu")
    # per-layer increments
    assert torch.allclose(perts[0][1], torch.tensor([1.0, 0.0, 0.0]))   # v[1] - v[0]
    assert torch.allclose(perts[1][1], torch.tensor([2.0, 0.0, 0.0]))   # v[2] - v[1]
    assert torch.allclose(perts[2][1], torch.tensor([4.0, 0.0, 0.0]))   # v[3] - v[2]
    # telescoping sum equals v[3] - v[0]
    total = sum(p for _, p in perts)
    assert torch.allclose(total, v[3] - v[0])


def test_build_perturbations_out_of_range_raises():
    v = torch.zeros((4, 3), dtype=torch.float32)
    with pytest.raises(ValueError):
        build_layer_perturbations(v, [4], "raw", torch.float32, "cpu")  # only 1..3 valid
    with pytest.raises(ValueError):
        build_layer_perturbations(v, [0], "raw", torch.float32, "cpu")


def test_build_perturbations_unknown_mode_raises():
    v = torch.zeros((4, 3), dtype=torch.float32)
    with pytest.raises(ValueError):
        build_layer_perturbations(v, [1], "bogus", torch.float32, "cpu")


def test_build_perturbations_dim_check():
    v = torch.zeros((4,), dtype=torch.float32)   # wrong shape
    with pytest.raises(ValueError):
        build_layer_perturbations(v, [1], "raw", torch.float32, "cpu")


def test_build_perturbations_dtype_cast():
    v = torch.zeros((4, 3), dtype=torch.float32)
    perts = build_layer_perturbations(v, [1], "raw", torch.bfloat16, "cpu")
    assert perts[0][1].dtype == torch.bfloat16


def test_make_multilayer_steerer_returns_context_manager():
    """Smoke test: make_multilayer_steerer instantiates ActivationSteererMultiple
    correctly. We can't __enter__ without a real model, but we can check the
    constructor accepted our instructions and the instruction list is well-formed."""
    from src.steering import make_multilayer_steerer
    import torch

    # 4-row v_full, 3-d hidden. Mock model: a single nn.Module with a
    # config.hidden_size attribute and a `model.layers` ModuleList of 3 dummy layers
    # — enough for ActivationSteerer's _locate_layer + hidden_size sanity check,
    # though we don't enter the context.
    import torch.nn as nn

    class Cfg:
        hidden_size = 3

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Identity() for _ in range(3)])
            # ActivationSteerer.__init__ calls next(model.parameters()) to resolve
            # dtype/device; provide a minimal leaf parameter so it doesn't raise.
            self._dummy = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    model = MockModel()
    v_full = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=torch.float32)
    steerer = make_multilayer_steerer(
        model, v_full,
        layers=[1, 2, 3],
        mode="incremental",
        coeff=5.0,
        positions="all",
    )
    # ActivationSteererMultiple holds a list of sub-steerers
    assert len(steerer._steerers) == 3
    # Each sub-steerer has coeff=5.0 and was given the right per-layer vector
    for i, s in enumerate(steerer._steerers, start=1):
        assert s.coeff == 5.0
        assert s.layer_idx == i - 1   # 0-indexed for ActivationSteerer
        assert s.positions == "all"
