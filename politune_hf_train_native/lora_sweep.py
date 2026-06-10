"""LoRA-scale override utility for PoliTune-HF adapters.

Lives with the training code so that
``1_benchmarking/custom_bench/adapters.py`` can re-export
``apply_lora_scale_override`` (its import target). Used by the LoRA-scale
sweep (``classic_evals.run_lora_sweep``, ``G_K_assessing_bias.run_eval``):
tags like ``mistral-politune-hf-left-lora1_5`` request the adapter applied
at 1.5x its trained strength.

A PEFT LoRA layer's effective contribution is ``scaling = lora_alpha / r``.
We multiply that by the requested ``scale`` in place, so ``scale=1.0`` is a
no-op (trained strength) and ``scale=0.0`` disables the adapter.
"""
from __future__ import annotations


def apply_lora_scale_override(model, scale: float):
    """Multiply every LoRA layer's scaling by ``scale`` (in place). Returns model."""
    from peft.tuners.lora import LoraLayer

    n = 0
    for module in model.modules():
        if isinstance(module, LoraLayer):
            for adapter in list(module.scaling.keys()):
                base = module.lora_alpha[adapter] / module.r[adapter]
                module.scaling[adapter] = base * scale
                n += 1
    if n == 0:
        raise RuntimeError("apply_lora_scale_override: no LoRA layers found on model")
    return model
