"""Thin wrapper around ref_impl/activation_steer.py.

Exposes:
  - `make_steerer(model, v_full, layer_idx_1based, coeff, positions)` →
    ActivationSteerer context manager (1-indexed in, 0-indexed out).
  - `load_vector(path)` → torch.Tensor of shape [num_layers+1, d_model].

Use this module from 4_steering scripts; production inference goes
through `pipeline.adapters.steering_context()` (in 1_benchmarking).
"""
from __future__ import annotations
import sys
import torch
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_REF_IMPL = _REPO_ROOT / "4_steering" / "ref_impl"
if str(_REF_IMPL) not in sys.path:
    sys.path.insert(0, str(_REF_IMPL))
from activation_steer import ActivationSteerer, ActivationSteererMultiple  # noqa: E402


def load_vector(path: str | Path) -> torch.Tensor:
    p = Path(path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    v = torch.load(p, map_location="cpu", weights_only=True)
    if v.ndim != 2:
        raise ValueError(f"expected [num_layers+1, d_model], got {tuple(v.shape)}")
    return v


def make_steerer(
    model,
    v_full: torch.Tensor,
    layer_idx_1based: int,
    coeff: float,
    positions: str = "all",
) -> ActivationSteerer:
    """layer_idx_1based: 1..num_layers, matching the vector row convention.
    The underlying ActivationSteerer wants 0-indexed model.layers index."""
    if not (1 <= layer_idx_1based <= v_full.shape[0] - 1):
        raise IndexError(
            f"layer_idx_1based={layer_idx_1based} not in 1..{v_full.shape[0]-1}"
        )
    v_layer = v_full[layer_idx_1based]
    return ActivationSteerer(
        model,
        steering_vector=v_layer,
        coeff=float(coeff),
        layer_idx=layer_idx_1based - 1,
        positions=positions,
    )


def make_multilayer_steerer(
    model,
    v_full: torch.Tensor,
    layers: list[int],
    mode: str,
    coeff: float,
    positions: str = "all",
) -> ActivationSteererMultiple:
    """Hook every layer in `layers` with `coeff * pert(L)` where pert is built
    by build_layer_perturbations(mode={"raw","incremental"}).

    Returns an ActivationSteererMultiple context manager. Use as:

        with make_multilayer_steerer(model, v_full, [1,2,...,31],
                                     "incremental", coeff=5.0):
            out = model.generate(...)
    """
    # Resolve model dtype + device for vector casts.
    p = next(model.parameters())
    perts = build_layer_perturbations(v_full, layers, mode, p.dtype, p.device)
    instructions = [
        {
            "steering_vector": v,
            "coeff": float(coeff),
            "layer_idx": L - 1,       # convert to 0-indexed model.layers
            "positions": positions,
        }
        for L, v in perts
    ]
    return ActivationSteererMultiple(model, instructions)


def parse_layers(spec: str) -> list[int]:
    """Parse a layer spec like '17', '10,16,22,28', or '1-31'.

    Returns a sorted list of unique 1-indexed layer numbers.
    """
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            for i in range(int(a), int(b) + 1):
                out.add(i)
        else:
            out.add(int(part))
    return sorted(out)


def build_layer_perturbations(
    v_full: torch.Tensor,
    layers: list[int],
    mode: str,
    dtype: torch.dtype,
    device,
) -> list[tuple[int, torch.Tensor]]:
    """Return [(layer_idx_1based, perturbation_vector), ...].

    v_full shape: [num_layers + 1, d_model]. Row 0 = embedding output,
    rows 1..N = post-block residuals.

    raw:         pert(L) = v_full[L].
    incremental: pert(L) = v_full[L] - v_full[L-1]. Cumulative effect over
                 a contiguous range [1..L] telescopes to v_full[L] - v_full[0]
                 (≈ v_full[L] since row 0 is typically near zero).
    """
    if v_full.dim() != 2:
        raise ValueError(
            f"expected v_full shape [num_layers+1, d_model], got {tuple(v_full.shape)}"
        )
    num_layers_avail = v_full.shape[0] - 1
    out: list[tuple[int, torch.Tensor]] = []
    for L in layers:
        if L < 1 or L > num_layers_avail:
            raise ValueError(f"layer {L} out of range [1, {num_layers_avail}]")
        if mode == "raw":
            v = v_full[L]
        elif mode == "incremental":
            v = v_full[L] - v_full[L - 1]
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        out.append((L, v.to(device=device, dtype=dtype)))
    return out
