"""Dataset loading for PoliTune DPO.

``scale-lab/politune-{left,right}`` ship plain-string rows
``{prompt, chosen, rejected}``. We convert them to TRL's *conversational*
format (lists of role/content messages). TRL then applies the model's
native ``chat_template`` automatically (Mistral ``[INST]...[/INST]``,
Llama-3 header format) -- this is the whole point of the rebuild vs the
original AlpacaInstructTemplate path.
"""
from __future__ import annotations

import time

from datasets import Dataset, load_dataset


def _load_with_retry(dataset_id: str, split: str, tries: int = 5) -> Dataset:
    """load_dataset with backoff -- GCP->HF Hub connections reset intermittently."""
    last = None
    for i in range(tries):
        try:
            return load_dataset(dataset_id, split=split)
        except Exception as e:  # noqa: BLE001 - retry any transient hub/network error
            last = e
            wait = 2 ** i
            print(f"[data] load_dataset attempt {i + 1}/{tries} failed ({e}); "
                  f"retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"load_dataset({dataset_id}) failed after {tries} tries") from last


def _to_conversational(row: dict) -> dict:
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def load_politune(dataset_id: str, split: str = "train") -> Dataset:
    """Load a politune split and map to conversational prompt/chosen/rejected."""
    ds = _load_with_retry(dataset_id, split)
    expected = {"prompt", "chosen", "rejected"}
    missing = expected - set(ds.column_names)
    if missing:
        raise ValueError(f"{dataset_id} missing columns {missing}; has {ds.column_names}")
    return ds.map(_to_conversational, remove_columns=ds.column_names)
