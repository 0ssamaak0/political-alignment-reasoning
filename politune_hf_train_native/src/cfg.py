"""Config dataclass + YAML loader for the PoliTune HF/TRL DPO-LoRA trainer.

One YAML per cell (family x lean) under ``configs/``. Values mirror the
original torchtune PoliTune recipe (see polieval ``debate_*.yaml``); the
deviation is that this trainer applies each model's NATIVE chat template
instead of ``AlpacaInstructTemplate``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    # identity
    family: str                       # "mistral" | "llama"
    lean: str                         # "left" | "right"
    base_repo: str                    # HF model id
    dataset: str                      # HF dataset id (scale-lab/politune-{lean})
    output_subdir: str                # adapters_train/<output_subdir>/

    # LoRA (matches lora_rank 16 / lora_alpha 32, attn q/v + MLP + output)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "v_proj",                      # lora_attn_modules
            "gate_proj", "up_proj", "down_proj",     # apply_lora_to_mlp
            "lm_head",                               # apply_lora_to_output
        ]
    )

    # DPO loss (DPOLoss beta 0.1, sigmoid, no label smoothing)
    beta: float = 0.1
    loss_type: str = "sigmoid"
    label_smoothing: float = 0.0
    max_length: int = 1024            # max_seq_len: 1024
    max_prompt_length: int = 512

    # optimiser + schedule (AdamW lr 5e-4 wd 0.05; cosine warmup 100)
    learning_rate: float = 5.0e-4
    weight_decay: float = 0.05
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100

    # batching (batch_size 2 x grad_accum 16 = effective 32)
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16

    # duration: deployed cell = 60 opt-steps (left) / 80 (right).
    # Both < warmup(100), so LR never leaves the linear ramp -> the
    # cosine horizon is irrelevant and max_steps fully determines the run.
    max_steps: int = 60
    save_steps: int = 10
    logging_steps: int = 1
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        data = yaml.safe_load(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)
