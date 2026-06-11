"""Per-cell file layout for the G&K partisan-inference probe.

Result/response/judge files are grouped on disk by cell kind (introduced
2026-05-25), mirroring the custom_bench layout:
    steering cells  -> "<family>-steering/"  (e.g. mistral-steering/)
    roleplay cells  -> "roleplay/"
    everything else -> top level             (base, summary CSVs, ...)

Only steering + roleplay are grouped. The summary CSVs written by
``compute_bias`` / ``aggregate_judges`` stay at the dir root by construction
(their names route to "").
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESPONSES_DIR = HERE / "responses"
JUDGES_DIR = HERE / "judges"


def _subdir_for(name: str) -> str:
    """Grouping subdir (relative to a data dir) for a cell ``name``, or ""."""
    base = name.split("__")[0]
    family = base.split("-")[0]
    if "roleplay" in base:
        return "roleplay"
    if "pvsteer" in base or "steer" in base:
        return f"{family}-steering"
    return ""


def _cell_path(base_dir: Path, name: str, ext: str, *, mkdir: bool) -> Path:
    sub = _subdir_for(name)
    d = base_dir / sub if sub else base_dir
    if mkdir:
        d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}{ext}"


def result_csv(name: str, *, mkdir: bool = False) -> Path:
    return _cell_path(RESULTS_DIR, name, ".csv", mkdir=mkdir)


def response_jsonl(name: str, *, mkdir: bool = False) -> Path:
    return _cell_path(RESPONSES_DIR, name, ".jsonl", mkdir=mkdir)


def judge_jsonl(name: str, *, mkdir: bool = False) -> Path:
    return _cell_path(JUDGES_DIR, name, ".jsonl", mkdir=mkdir)
