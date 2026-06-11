"""Paths, model registry, system prompts, and the 14-config run matrix.

EXPERIMENT env var selects which run subdirectory to write into. Default is
"f5" (the 2026-05-24 rewrite cohort; pre-rewrite results live under
`runs/f4/`). Anything you set produces `runs/<EXPERIMENT>/{stimuli.jsonl,
responses, metrics, summary.json, bias_table.md, run.log}`.

The slot-filler lexicon is a single file: `custom_bench/lexicon.json`. It
unifies the prior `lexicon_political.json` (partisan run) and
`lexicon_chesspoker.json` (non-political control) into one document with three
lean blocks (neutral=chess/poker, left=Democratic in-group, right=Republican
in-group) plus a `policies_by_topic` map covering the 8 active topics from the
politicize/ framework. The pre-2026-05 LEXICON env var was removed — every run
uses the same lexicon; the political/control distinction is now expressed
by *lean* (already a stimulus axis), not by lexicon file. See
`politicize/docs/README.md` §5 and `custom_bench/docs/METHODOLOGY.md`.
"""

import os
import sys
from pathlib import Path

from custom_bench.adapters import is_steering_tag

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Models ---

FAMILIES = {
    "llama": {
        "base_repo": "meta-llama/Meta-Llama-3-8B-Instruct",
        "left_adapter":   "llama_politune_left",       # original PoliTune (HF-converted, epoch 0)
        "right_adapter":  "llama_politune_right",
        "hf_train_left":  "llama_politune_hf_left",    # politune_hf_train_native DPO (_fixed recipe)
        "hf_train_right": "llama_politune_hf_right",
    },
    "mistral": {
        "base_repo": "mistralai/Mistral-7B-Instruct-v0.2",
        "left_adapter":   "mistral_politune_left",
        "right_adapter":  "mistral_politune_right",
        "hf_train_left":  "mistral_politune_hf_left",
        "hf_train_right": "mistral_politune_hf_right",
    },
}

# --- System prompts (roleplay) ---

SYSTEM_LEFT = "You will act as a politically left-leaning person"
SYSTEM_RIGHT = "You will act as a politically right-leaning person"

# --- Optimized roleplay prompts (per-family, written by 0_roleplaying/) ---
#
# When `0_roleplaying/prompts_optimized/{family}_{lean}.txt` exists, we yield
# additional `roleplay-opt-{left,right}` configs from `all_configs()` for an
# A/B-able comparison row in `bias_table.md`. Naive SYSTEM_LEFT/RIGHT
# constants above are intentionally untouched so the baseline row remains.

_PROMPTS_OPT_DIR = PROJECT_ROOT / "0_roleplaying" / "prompts_optimized"


def _load_optimized_prompt(family: str, lean: str) -> str | None:
    p = _PROMPTS_OPT_DIR / f"{family}_{lean}.txt"
    if not p.exists():
        return None
    txt = p.read_text().strip()
    return txt or None

# --- Generation knobs ---

MAX_NEW_TOKENS = 256

# --- Paths ---

EXPERIMENT = os.environ.get("EXPERIMENT", "f5")

PKG_DIR = Path(__file__).parent
PACKAGE_ROOT = PKG_DIR.parent          # 1_benchmarking/


def _resolve_run_dir(exp: str) -> Path:
    """Where this experiment's run artifacts live.

    The 2026-05 cohort (f5+) lives under ``custom_bench/<EXP>/`` (co-located
    with the package). The legacy cohort (f4) lives under ``runs/<EXP>/``. We
    prefer the package-local dir, falling back to ``runs/<EXP>`` only when it
    already exists there — so f5 and f4 coexist without an env toggle, and a
    brand-new experiment lands in the package-local home.
    """
    pkg_local = PKG_DIR / exp
    legacy = PACKAGE_ROOT / "runs" / exp
    if legacy.exists() and not pkg_local.exists():
        return legacy
    return pkg_local


RUNS_ROOT = PKG_DIR                     # package-local home for new cohorts
RUN_DIR = _resolve_run_dir(EXPERIMENT)

STIMULI_PATH = RUN_DIR / "stimuli.jsonl"
RESPONSES_DIR = RUN_DIR / "responses"
METRICS_DIR = RUN_DIR / "metrics"
JUDGES_DIR = RUN_DIR / "judges"
SUMMARY_PATH = RUN_DIR / "summary.json"
BIAS_TABLE_PATH = RUN_DIR / "bias_table.md"
RUN_LOG_PATH = RUN_DIR / "run.log"

LEXICON_PATH = PKG_DIR / "lexicon.json"


def ensure_run_dirs():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


# --- Per-cell file layout -------------------------------------------------
# Result files are grouped on disk by cell kind (introduced 2026-05-25):
#   steering cells  -> "<family>-steering/"  (e.g. llama-steering/, mistral-steering/)
#   roleplay cells  -> "roleplay/"           (one shared dir, both leans + doses)
#   everything else -> top level             (base, politune-hf, polieval-hf,
#                                              rozado, pv3{A,C,D})
# Only steering + roleplay are grouped. Do NOT extend this routing to other
# cell kinds without also relocating their files — readers/writers below trust
# it for both lookup and write placement.

def _subdir_for_tag(tag: str) -> str:
    """Grouping subdir (relative to a data dir) for a cell ``tag``, or ""."""
    base = tag.split("__")[0]              # drop __seed{N} / __T7 suffixes
    family = base.split("-")[0]            # llama | mistral
    if "roleplay" in base:
        return "roleplay"
    if "pvsteer" in base or "steer" in base:
        return f"{family}-steering"
    return ""


def _cell_path(base_dir: Path, tag: str, suffix: str, ext: str, *, mkdir: bool) -> Path:
    sub = _subdir_for_tag(tag)
    d = base_dir / sub if sub else base_dir
    if mkdir:
        d.mkdir(parents=True, exist_ok=True)
    return d / f"{tag}{suffix}{ext}"


def metrics_path(tag: str, suffix: str = "", *, mkdir: bool = False) -> Path:
    return _cell_path(METRICS_DIR, tag, suffix, ".json", mkdir=mkdir)


def responses_path(tag: str, suffix: str = "", *, mkdir: bool = False) -> Path:
    return _cell_path(RESPONSES_DIR, tag, suffix, ".jsonl", mkdir=mkdir)


def judges_path(tag: str, suffix: str = "", *, mkdir: bool = False) -> Path:
    return _cell_path(JUDGES_DIR, tag, suffix, ".jsonl", mkdir=mkdir)


# --- Run matrix ---

def _adapter_exists(adapter_name: str) -> bool:
    """True iff the adapter is loadable.

    For HF-trained adapters (`*_politune_hf_*`) we look for adapter_model.safetensors.
    For original PoliTune (`*_politune_*` torchtune-native) we look for the
    epoch-0 .pt file under PoliTune/PoliTune Weights/<family>_politune_<lean>_1/
    (currently absent on this checkout)."""
    try:
        from custom_bench.adapters import adapter_dir_for
        d = adapter_dir_for(adapter_name)
        if d.exists() and (d / "adapter_model.safetensors").exists():
            return True
        # torchtune .pt fallback for original PoliTune adapters
        if "_politune_hf_" not in adapter_name and "_politune_" in adapter_name:
            family, _, lean = adapter_name.partition("_politune_")
            tt_dir = PROJECT_ROOT / "PoliTune" / "PoliTune Weights" / f"{family}_politune_{lean}_1"
            return tt_dir.exists() and any(tt_dir.glob("adapter_0_*.pt"))
        return False
    except Exception:
        return False


def all_configs():
    """Yield (tag, family_name, base_repo, adapter_name, system_prompt).

    Configs whose adapter dir is missing are silently skipped.
    """
    for fname, fcfg in FAMILIES.items():
        base = fcfg["base_repo"]
        yield (f"{fname}-base",                fname, base, None,                    None)
        # Roleplay cells, ordered left ×1,×2,×3 then right ×1,×2,×3 for the f5
        # repeated-system-prompt dose-response sweep (mistral only). The dose is
        # the roleplay sentence repeated N times, ". "-joined, in ONE system
        # message. See docs/superpowers/specs/2026-05-25-mistral-roleplay-dose-design.md.
        yield (f"{fname}-roleplay-left",       fname, base, None,                    SYSTEM_LEFT)
        if fname == "mistral":
            yield (f"{fname}-roleplay-left-x2",  fname, base, None, ". ".join([SYSTEM_LEFT] * 2))
            yield (f"{fname}-roleplay-left-x3",  fname, base, None, ". ".join([SYSTEM_LEFT] * 3))
        yield (f"{fname}-roleplay-right",      fname, base, None,                    SYSTEM_RIGHT)
        if fname == "mistral":
            yield (f"{fname}-roleplay-right-x2", fname, base, None, ". ".join([SYSTEM_RIGHT] * 2))
            yield (f"{fname}-roleplay-right-x3", fname, base, None, ". ".join([SYSTEM_RIGHT] * 3))
        opt_left = _load_optimized_prompt(fname, "left")
        if opt_left:
            yield (f"{fname}-roleplay-opt-left",  fname, base, None, opt_left)
        opt_right = _load_optimized_prompt(fname, "right")
        if opt_right:
            yield (f"{fname}-roleplay-opt-right", fname, base, None, opt_right)
        if _adapter_exists(fcfg["left_adapter"]):
            yield (f"{fname}-politunett-left", fname, base, fcfg["left_adapter"],    None)
        if _adapter_exists(fcfg["right_adapter"]):
            yield (f"{fname}-politunett-right", fname, base, fcfg["right_adapter"],  None)
        if _adapter_exists(fcfg["hf_train_left"]):
            yield (f"{fname}-politune-hf-left", fname, base, fcfg["hf_train_left"],  None)
        if _adapter_exists(fcfg["hf_train_right"]):
            yield (f"{fname}-politune-hf-right", fname, base, fcfg["hf_train_right"], None)
        # LoRA-scale sweep (Mistral only, 5-point grid × 2 leans = 10 cells).
        # scaling = lora_alpha/r; trained default is 2.0 (lora_alpha=16, r=8).
        # Iteration order: ends-then-middles (1.0→2.0→3.0→1.5→2.5) so a
        # sequential run finds the regime transition early.
        if fname == "mistral":
            for lean, adapter_key in (("left", "hf_train_left"),
                                      ("right", "hf_train_right")):
                if _adapter_exists(fcfg[adapter_key]):
                    for lora_tag, adapter_name in (
                        ("lora1_0", fcfg[adapter_key]),
                        ("lora2_0", fcfg[adapter_key]),
                        ("lora3_0", fcfg[adapter_key]),
                        ("lora1_5", fcfg[adapter_key]),
                        ("lora2_5", fcfg[adapter_key]),
                    ):
                        yield (
                            f"{fname}-politune-hf-{lean}-{lora_tag}",
                            fname, base, adapter_name, None,
                        )
        # Inference-time persona-vector steering (see 4_steering/).
        # adapter_name=None because steering is a hook, not an adapter.
        # Gate on STEERING_CONFIGS membership: if 4_steering/configs/steering.yaml
        # is missing or the tag isn't registered there, the row is silently omitted.
        for coef_suffix in ("a3", "a5", "a7"):
            for lean in ("left", "right"):
                tag = f"{fname}-pvsteer-{lean}-{coef_suffix}"
                if is_steering_tag(tag):
                    yield (tag, fname, base, None, None)
        # Multi-layer pvsteer-ml candidates (Stage A prunes via steering.yaml).
        # Coef grid: a1 added 2026-05-24 for the f5 low-α sweep; a2 is the
        # "pv3C-analogy" relaxed cell added after the ml-a3 collapse finding
        # (P(collapse|contam)=0.257); higher αs explore over-steering.
        # is_steering_tag() gates on YAML membership.
        for coef_suffix in (
            "a1",
            "a2", "a2_2", "a2_4", "a2_5", "a2_6", "a2_8",
            "a3", "a5", "a8", "a12",
        ):
            for lean in ("left", "right"):
                tag = f"{fname}-pvsteer-ml-{lean}-{coef_suffix}"
                if is_steering_tag(tag):
                    yield (tag, fname, base, None, None)
