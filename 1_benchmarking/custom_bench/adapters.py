"""In-tree adapter registry.

Maps adapter names to on-disk directories. Supports an `ADAPTERS_DIR`
env override so the same pipeline code can point at adapters mounted
at a different path.

Naming conventions:
- `<family>_politune_hf_<lean>`  -> HF-trained DPO adapters
                                    (politune_hf_train_native, `_fixed` recipe)
- `<family>_politune_<lean>`     -> original torchtune PoliTune adapters,
                                    after HF conversion (currently MISSING
                                    on this checkout; configs that need
                                    them are silently skipped by
                                    `_adapter_exists()`).
"""

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # /home/.../main

ADAPTERS_DIR = os.environ.get("ADAPTERS_DIR")

# `apply_lora_scale_override` now lives with the LoRA training code in
# politune_hf_train_native/lora_sweep.py. Re-export it here so existing
# `from custom_bench.adapters import apply_lora_scale_override` call sites
# (run_all_hf, classic_evals.run_lora_sweep, G_K_assessing_bias.run_eval)
# keep working unchanged. PROJECT_ROOT is the repo root, where
# politune_hf_train_native is importable as a (namespace) package.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from politune_hf_train_native.lora_sweep import apply_lora_scale_override  # noqa: E402,F401

_LORA_SUFFIX_RE = re.compile(r"-lora(\d+)_(\d+)$")
_LORA_ADAPTER_SUFFIX_RE = re.compile(r"_lora\d+_\d+$")


def parse_lora_scale(tag: str) -> tuple[str, float | None]:
    """Strip a -lora{x_x} suffix from `tag` and return (base_tag, scale|None).

    E.g. "mistral-politune-hf-left-lora1_5" -> ("mistral-politune-hf-left", 1.5)
    A tag without the suffix returns (tag, None).
    """
    m = _LORA_SUFFIX_RE.search(tag)
    if m is None:
        return tag, None
    base = tag[: m.start()]
    scale = float(m.group(1)) + float(m.group(2)) / (10 ** len(m.group(2)))
    return base, scale


def adapter_dir_for(name: str) -> Path:
    if ADAPTERS_DIR:
        # Strip _lora{x_x} suffix when ADAPTERS_DIR is set (containerised path).
        clean = _LORA_ADAPTER_SUFFIX_RE.sub("", name)
        return Path(ADAPTERS_DIR) / clean

    # Strip _lora{x_x} suffix before the standard resolution.
    name = _LORA_ADAPTER_SUFFIX_RE.sub("", name)

    if "_politune_hf_" in name:
        family, _, lean = name.partition("_politune_hf_")
        return (PROJECT_ROOT / "politune_hf_train_native"
                / "adapters_train" / f"{family}_{lean}_fixed")

    raise KeyError(f"No adapter resolution for {name!r}")


# ---------------------------------------------------------------------------
# Inference-time steering (4_steering subproject)
# ---------------------------------------------------------------------------

_STEERING_CFG_PATH = PROJECT_ROOT / "4_steering" / "configs" / "steering.yaml"


def _load_steering_configs() -> dict:
    """Load 4_steering/configs/steering.yaml. Returns {} if absent."""
    if not _STEERING_CFG_PATH.exists():
        return {}
    import yaml
    return yaml.safe_load(_STEERING_CFG_PATH.read_text()) or {}


# Loaded once at import time. If 4_steering/ is removed, this is {} and
# steering_context() becomes a permanent no-op.
STEERING_CONFIGS: dict = _load_steering_configs()


def is_steering_tag(name: str) -> bool:
    """True iff `name` is a configured `mistral-pvsteer-*` tag."""
    return name in STEERING_CONFIGS


def steering_context(model, tag: str):
    """Return a context manager that wraps model.generate() with the
    inference-time persona-vector hook for `tag`.

    For non-pvsteer tags returns nullcontext() — i.e. no-op. This means
    existing call sites can wrap unconditionally:

        with adapters.steering_context(model, tag):
            out = model.generate(...)

    For pvsteer tags loads the vector from disk, slices the chosen layer
    row, and instantiates ref_impl.activation_steer.ActivationSteerer.
    """
    if tag not in STEERING_CONFIGS:
        from contextlib import nullcontext
        return nullcontext()

    cfg = STEERING_CONFIGS[tag]

    import sys
    import torch
    ref_impl = PROJECT_ROOT / "4_steering" / "ref_impl"
    if str(ref_impl) not in sys.path:
        sys.path.insert(0, str(ref_impl))

    v_full = torch.load(
        PROJECT_ROOT / cfg["vector_path"],
        map_location="cpu",
        weights_only=True,
    )

    # Multi-layer dispatch — presence of `layers` key flips to ml path.
    if "layers" in cfg:
        # Lift the same builders 4_steering/src/steering.py uses, so we
        # don't add a Python-path tangle from 1_benchmarking to 4_steering.
        src_dir = str(PROJECT_ROOT / "4_steering" / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from steering import (  # noqa: E402
            parse_layers,
            make_multilayer_steerer,
        )
        return make_multilayer_steerer(
            model,
            v_full,
            layers=parse_layers(str(cfg["layers"])),
            mode=cfg.get("vector_mode", "incremental"),
            coeff=float(cfg["coeff"]),
            positions=cfg.get("positions", "all"),
        )

    # Single-layer path (existing).
    layer_idx_1based = int(cfg["layer_idx"])
    if layer_idx_1based <= 0:
        raise ValueError(
            f"steering tag {tag!r} has layer_idx={layer_idx_1based}; "
            f"Stage A must run first to fill this in"
        )

    from activation_steer import ActivationSteerer  # noqa: E402

    v_layer = v_full[layer_idx_1based]
    return ActivationSteerer(
        model,
        steering_vector=v_layer,
        coeff=float(cfg["coeff"]),
        layer_idx=layer_idx_1based - 1,
        positions=cfg.get("positions", "all"),
    )
