"""Forward-pass projection-out hook (paper §4 zero_ablate=True equivalent).

For each layer-15 residual-stream activation `h`, replaces it with
    h_new = h - alpha * (h · v / ||v||²) * v
where `v` is a cached `h_diff` direction. With alpha=1.0 this fully
projects out the v direction (the paper's `zero_ablate=True` mode).

Usage from run_all_hf.py:
    from custom_bench.projection_hook import (
        load_ablation_vector, install_at_layer, find_transformer_layer
    )

    direction = load_ablation_vector(env_var="ABLATION_VECTOR_PATH")
    handle = install_at_layer(model, layer_idx=15, direction=direction)
    try:
        ... run generation ...
    finally:
        handle.remove()

Hook handles both PEFT-wrapped and vanilla HF models by walking common
layer-access paths.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

import torch


def _find_layer_paths(model: Any, idx: int) -> list[tuple[str, Any]]:
    """Try common layer-access paths and return (path_str, module) for any
    that resolve. We try, in order:
      - model.model.layers[idx]                   (vanilla HF causal LM)
      - model.base_model.model.model.layers[idx]  (PEFT-wrapped HF causal LM)
      - model.base_model.model.layers[idx]        (alt PEFT nesting)
    """
    out: list[tuple[str, Any]] = []
    candidates = [
        ("model.model.layers", lambda m: m.model.layers[idx]),
        ("model.base_model.model.model.layers", lambda m: m.base_model.model.model.layers[idx]),
        ("model.base_model.model.layers", lambda m: m.base_model.model.layers[idx]),
    ]
    for path, fn in candidates:
        try:
            mod = fn(model)
        except (AttributeError, IndexError, TypeError):
            continue
        if mod is not None:
            out.append((path, mod))
    return out


def find_transformer_layer(model: Any, idx: int) -> Any:
    """Return the idx-th transformer layer module, regardless of PEFT
    wrapping. Raises RuntimeError if no path works."""
    found = _find_layer_paths(model, idx)
    if not found:
        raise RuntimeError(
            f"Could not locate transformer layer {idx} in model "
            f"of type {type(model).__name__}. Tried model.model.layers, "
            f"model.base_model.model.model.layers, "
            f"model.base_model.model.layers."
        )
    # Prefer the deepest path (PEFT-wrapped) over the shallow one if both work
    # — PEFT wraps the base model, so the deeper path points at the actual
    # layer that gets the LoRA-modified output.
    return found[-1][1]


def make_projection_out_hook(
    direction: torch.Tensor, alpha: float = 1.0
) -> Callable:
    """Return a forward hook that projects out (alpha × component along
    `direction`) from the layer's hidden-state output.

    Math: h_new[b, s] = h[b, s] - alpha * (h[b, s] · v / ||v||²) * v.

    With alpha=1.0 this is the paper's `zero_ablate=True` mode (fully
    remove the v direction). With alpha=0.0 it's a no-op (useful for
    sanity check). With alpha=2.0 it overshoots (negative coefficient).
    """
    if direction.ndim != 1:
        raise ValueError(f"direction must be 1-D, got shape {tuple(direction.shape)}")
    norm_sq_f32 = (direction.float() ** 2).sum().clamp_min(1e-8)

    def hook(module: Any, args: Any, output: Any) -> Any:
        # Layer modules return either a Tensor (some custom impls) or a
        # tuple (HF transformer block: hidden_states, optionally cached_kv,
        # attentions, etc.). We modify only the first element.
        if isinstance(output, tuple):
            hs = output[0]
            rest = output[1:]
        else:
            hs = output
            rest = None
        # hs: [batch, seq, hidden]
        if hs.ndim != 3:
            # Unexpected shape — pass through unchanged rather than crash.
            return output
        v = direction.to(hs.device, hs.dtype)
        norm_sq = norm_sq_f32.to(hs.device, hs.dtype)
        coeffs = (hs @ v) / norm_sq                  # [batch, seq]
        hs_new = hs - alpha * coeffs.unsqueeze(-1) * v   # [batch, seq, hidden]
        if rest is not None:
            return (hs_new,) + rest
        return hs_new

    return hook


def install_at_layer(
    model: Any, layer_idx: int, direction: torch.Tensor, alpha: float = 1.0
) -> Any:
    """Register the projection-out hook on the idx-th transformer layer.
    Returns the removal handle (caller must `handle.remove()` to detach)."""
    layer = find_transformer_layer(model, layer_idx)
    hook = make_projection_out_hook(direction, alpha=alpha)
    handle = layer.register_forward_hook(hook)
    return handle


def load_ablation_vector(
    env_var: str = "ABLATION_VECTOR_PATH",
) -> Optional[torch.Tensor]:
    """If env var is set and points to a file, load and return the 1-D
    vector. Otherwise return None (no ablation)."""
    path = os.environ.get(env_var, "").strip()
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"{env_var}={path} does not exist on disk")
    obj = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"{path} loaded as {type(obj).__name__}, expected torch.Tensor")
    if obj.ndim != 1:
        raise ValueError(f"{path} has shape {tuple(obj.shape)}, expected 1-D")
    return obj


def _self_test() -> None:
    """Smoke test the hook math without loading a real model."""
    d_model = 8
    direction = torch.randn(d_model)
    direction = direction / direction.norm()  # unit vector for easy verification

    # Create a fake "layer output" that is exactly the direction
    fake_h = direction.view(1, 1, -1).clone()  # [1, 1, 8]
    hook = make_projection_out_hook(direction, alpha=1.0)

    class FakeModule: pass
    out = hook(FakeModule(), None, fake_h)
    # After full projection-out, the result should be ~zero
    residual_norm = out.norm().item()
    assert residual_norm < 1e-5, f"projection-out failed: residual norm = {residual_norm}"

    # Test with random h, projection should leave orthogonal complement
    h = torch.randn(2, 3, d_model)
    out = hook(FakeModule(), None, h)
    # The output's component along direction should be ~0
    coeffs_after = (out @ direction)
    assert coeffs_after.abs().max().item() < 1e-4, \
        f"projection-out left component along direction: max={coeffs_after.abs().max()}"

    print("[projection_hook] _self_test passed")


if __name__ == "__main__":
    _self_test()
