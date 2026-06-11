"""Thin torchtune inference helpers for the comparison + MMLU scripts.

Mirrors the model setup in PoliTune/finetune/dpo_finetune.py but only the bits
needed for a forward pass / greedy generation.
"""
from __future__ import annotations
import os
from pathlib import Path

import torch


def _snap(repo_dir: str) -> Path:
    return next(Path(os.path.expanduser(f"~/.cache/huggingface/hub/{repo_dir}/snapshots")).iterdir())


def _safetensor_shards(snap: Path) -> list[str]:
    return sorted(
        f.name for f in snap.iterdir()
        if f.name.startswith("model-") and f.name.endswith(".safetensors")
    )


def _init_rope_fp32(model: torch.nn.Module, device: str) -> None:
    """Re-init RoPE buffers in fp32 on device.

    `to_empty` leaves buffers undefined; torchtune's default rope_cache is fp32
    and casting it to bf16 noticeably degrades the forward pass.
    """
    for m in model.modules():
        if hasattr(m, "rope_init"):
            m.rope_init()
            for name, buf in m.named_buffers(recurse=False):
                m.register_buffer(name, buf.to(device=device, dtype=torch.float32),
                                  persistent=False)


def load_mistral_tt(adapter_pt: str | os.PathLike | None, device: str = "cuda",
                    dtype=torch.bfloat16):
    """Load lora_mistral_7b with base weights from the HF cache and optionally an adapter."""
    from torchtune.training import FullModelHFCheckpointer, set_default_dtype
    from torchtune.models.mistral import lora_mistral_7b

    snap = _snap("models--mistralai--Mistral-7B-Instruct-v0.2")
    ckpt = FullModelHFCheckpointer(
        checkpoint_dir=str(snap),
        checkpoint_files=_safetensor_shards(snap),
        model_type="MISTRAL",
        output_dir="/tmp/politune_tt_scratch",
    )
    base_sd = ckpt.load_checkpoint()["model"]

    with set_default_dtype(dtype), torch.device("meta"):
        model = lora_mistral_7b(
            lora_attn_modules=["q_proj", "v_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=True,
            lora_rank=16,
            lora_alpha=32,
        )
    model = model.to_empty(device=device)
    _init_rope_fp32(model, device)
    model.load_state_dict(base_sd, strict=False)
    if adapter_pt is not None:
        model.load_state_dict(
            torch.load(adapter_pt, map_location="cpu", weights_only=True), strict=False
        )
    model.eval()
    return model


def load_llama3_tt(adapter_pt: str | os.PathLike | None, device: str = "cuda",
                   dtype=torch.bfloat16):
    """Load lora_llama3_8b with base weights from the HF cache and optionally an adapter."""
    from torchtune.training import FullModelHFCheckpointer, set_default_dtype
    from torchtune.models.llama3 import lora_llama3_8b

    snap = _snap("models--meta-llama--Meta-Llama-3-8B-Instruct")
    ckpt = FullModelHFCheckpointer(
        checkpoint_dir=str(snap),
        checkpoint_files=_safetensor_shards(snap),
        model_type="LLAMA3",
        output_dir="/tmp/politune_tt_scratch",
    )
    base_sd = ckpt.load_checkpoint()["model"]

    with set_default_dtype(dtype), torch.device("meta"):
        model = lora_llama3_8b(
            lora_attn_modules=["q_proj", "v_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=True,
            lora_rank=16,
            lora_alpha=32,
        )
    model = model.to_empty(device=device)
    _init_rope_fp32(model, device)
    model.load_state_dict(base_sd, strict=False)
    if adapter_pt is not None:
        model.load_state_dict(
            torch.load(adapter_pt, map_location="cpu", weights_only=True), strict=False
        )
    model.eval()
    return model


@torch.no_grad()
def tt_forward_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Full forward pass; return (B, T, V) logits tensor."""
    input_ids = input_ids.to(next(model.parameters()).device)
    return model(input_ids)


@torch.no_grad()
def tt_greedy_generate(model, input_ids: torch.Tensor, max_new_tokens: int = 32) -> torch.Tensor:
    """Greedy decoding without KV cache — simple and deterministic."""
    device = next(model.parameters()).device
    ids = input_ids.to(device).clone()
    for _ in range(max_new_tokens):
        logits = model(ids)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids
