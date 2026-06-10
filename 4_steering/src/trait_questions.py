"""Load eval-set trait questions for the layer-effectiveness sweep.

`shared/trait_data/{direction}_leaning.json` has the schema:
    {"instruction": [5 pairs of system prompts],
     "questions":   [40 trait-eliciting user questions],
     "eval_prompt": "<judge rubric, 0-100>"}

Stage A uses questions[20:40] — the latter half — to avoid contamination
with the first 20 used by extract_persona_responses.py to build the vector.

Returns a Python list of question strings.
"""
from __future__ import annotations
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TRAIT_DIR = _REPO_ROOT / "3_persona_vectors" / "shared" / "trait_data"


def load_eval_questions(direction: str, n: int = 20) -> list[str]:
    if direction not in ("left", "right"):
        raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")
    path = _TRAIT_DIR / f"{direction}_leaning.json"
    data = json.loads(path.read_text())
    qs = data["questions"]
    if len(qs) < 40:
        raise ValueError(f"{path}: expected at least 40 questions, got {len(qs)}")
    return qs[20:20 + n]


def load_eval_prompt(direction: str) -> str:
    path = _TRAIT_DIR / f"{direction}_leaning.json"
    return json.loads(path.read_text())["eval_prompt"]


def load_extraction_system_prompts(direction: str) -> list[dict]:
    """The 5 (pos, neg) contrastive system-prompt pairs. Not used by Stage A
    (Stage A just adds the vector; no system prompt) but exposed for sanity
    inspection."""
    path = _TRAIT_DIR / f"{direction}_leaning.json"
    return json.loads(path.read_text())["instruction"]
