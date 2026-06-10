"""PoliTune DPO-LoRA trainer (HuggingFace TRL + PEFT, native chat template).

Re-implementation of the original torchtune PoliTune DPO recipe. Mirrors the
hyperparameters (LoRA r16/a32, DPO beta 0.1 sigmoid, AdamW 5e-4, eff. batch
32) but tokenizes with each model's NATIVE chat template instead of
AlpacaInstructTemplate.

Usage:
    python -m src.train --config configs/mistral_left.yaml [--max_steps 5]

Outputs a PEFT adapter dir to adapters_train/<output_subdir>/ -- exactly where
1_benchmarking/custom_bench/adapters.py:adapter_dir_for() resolves the
`{family}_politune_hf_{lean}` cells.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer

from src.cfg import TrainConfig
from src.data import load_politune

HERE = Path(__file__).resolve().parent.parent          # politune_hf_train_native/
ADAPTERS_TRAIN = HERE / "adapters_train"


def build_tokenizer(cfg: TrainConfig) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(cfg.base_repo)
    # Mistral-7B-Instruct-v0.2 ships no pad token; DPO's collator needs one.
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    return tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a cell YAML")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="override max_steps (e.g. 5 for a smoke test)")
    ap.add_argument("--output_root", default=str(ADAPTERS_TRAIN),
                    help="where adapters_train/<subdir> is written")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    set_seed(cfg.seed)

    out_dir = Path(args.output_root) / cfg.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] {cfg.family}/{cfg.lean}  base={cfg.base_repo}  "
          f"dataset={cfg.dataset}  max_steps={cfg.max_steps}  out={out_dir}")

    tokenizer = build_tokenizer(cfg)

    # --- prove native templating before spending GPU time (advisor) ---
    sample_prompt = "Express your opinion on universal healthcare."
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": sample_prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    print("[template] one templated prompt (verify [INST]/header wrapping):")
    print(repr(templated))

    train_ds = load_politune(cfg.dataset)
    print(f"[data] {cfg.dataset}: {len(train_ds)} examples; "
          f"eff. batch {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_repo, torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dpo_config = DPOConfig(
        output_dir=str(out_dir),
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        label_smoothing=cfg.label_smoothing,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_steps=cfg.max_steps,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        seed=cfg.seed,
        report_to=[],
        save_total_limit=None,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                 # PEFT: reference = adapter-disabled policy
        args=dpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # Final adapter = the deployed cell (max_steps).
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[done] adapter saved to {out_dir}")


if __name__ == "__main__":
    main()
