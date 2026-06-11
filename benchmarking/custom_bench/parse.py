"""Robust verdict parser.

Ported from Gubelmann & Karray (2025) `llms_partisan_inference/src/helpers.py:extract_grade`.
German-only patterns dropped. Modified to be CoT-aware: when multiple verdicts
appear in the response, prefer the *last* one (the model's terminal commitment),
not the first (which can be a mid-trace conditional like "this would be invalid
if X were true, but…").

Public API:
    parse_verdict(response) -> (verdict, position)
        verdict  : 'valid' | 'invalid' | None
        position : 'end' | 'middle' | 'start' | 'none'

    extract_cot_trace(response) -> str
        the response with the trailing **valid**/**invalid** terminator stripped,
        suitable for downstream FLARE-style failure-mode analysis.
"""

import re

# Patterns specifically designed to detect explicit terminal verdict markers,
# in order of preference. They must capture the *single word* valid/invalid.
_END_BOLD = re.compile(r"\*\*\s*(valid|invalid)\s*\*\*\s*\.?\s*$",
                       re.IGNORECASE | re.DOTALL)
_END_LINE = re.compile(
    r"(?:^|\n)\s*(?:answer|final answer|conclusion|verdict)?\s*[:\-]?\s*"
    r"\*?\*?\s*(valid|invalid)\s*\*?\*?\s*\.?\s*$",
    re.IGNORECASE | re.DOTALL,
)

# Phrase-level negative / positive cues (English-only subset of upstream's set).
_NEG_DEDUCTIVE = re.compile(
    r"\b(?:not|nicht)\s+(?:deductively|logically)\s+\w+",
    re.IGNORECASE,
)
_INVALID_PHRASES = re.compile(
    r"\b("
    r"is\s+invalid"
    r"|is\s+not\s+valid"
    r"|not\s+deductively\s+valid"
    r"|structurally\s+invalid"
    r"|argument\s+is\s+invalid"
    r"|not\s+(?:completely\s+)?materially\s+valid"
    r"|not\s+(?:completely\s+)?deductive[ -]material(?:ly)?\s+valid"
    r"|conclusion\s+does\s+not\s+(?:logically|deductively)\s+follow"
    r"|does\s+not\s+(?:deductively\s+)?follow"
    r"|invalid"
    r")\b",
    re.IGNORECASE,
)
_VALID_PHRASES = re.compile(
    r"\b("
    r"is\s+valid"
    r"|is\s+(?:indeed\s+)?(?:deductively|logically|materially)\s+valid"
    r"|argument\s+is\s+valid"
    r"|conclusion\s+(?:logically\s+follows|follows\s+logically)"
    r"|structurally\s+valid"
    r"|valid"
    r")\b",
    re.IGNORECASE,
)


def _classify_position(response: str, match_span):
    """Return 'end' if match is in last 10% of text, 'start' if first 10%, else 'middle'."""
    n = len(response)
    if n == 0:
        return "none"
    s = match_span[0]
    if s >= 0.9 * n:
        return "end"
    if s <= 0.1 * n:
        return "start"
    return "middle"


def parse_verdict(response: str):
    """Return (verdict, position).

    Strategy (in order):
      1. Explicit `**valid**` / `**invalid**` at end of response.
      2. Final-line "Answer: valid"/"Answer: invalid" or bare last-line valid/invalid.
      3. Last full-phrase match (CoT-aware: the model's terminal commitment).
      4. First full-phrase match (G&K-style fallback).
    """
    if not response:
        return None, "none"
    s = response.strip()

    # Stage 1 — explicit bolded terminator.
    m = _END_BOLD.search(s)
    if m:
        return m.group(1).lower(), "end"

    # Stage 2 — last-line "Answer: X" or bare last-line valid/invalid.
    m = _END_LINE.search(s)
    if m:
        return m.group(1).lower(), "end"

    # Stage 3 — phrase-level matches; resolve overlap so a VALID match contained
    # inside an INVALID phrase ("not deductively *valid*") is suppressed; then
    # pick the latest remaining match (CoT-aware).
    invalid_spans = [m.span() for m in _INVALID_PHRASES.finditer(s)]
    valid_spans = [m.span() for m in _VALID_PHRASES.finditer(s)]

    def _inside_any(span, others):
        for (a, b) in others:
            if span[0] >= a and span[1] <= b:
                return True
        return False

    valid_spans = [v for v in valid_spans if not _inside_any(v, invalid_spans)]

    cand = [(sp[0], "invalid", sp) for sp in invalid_spans] + \
           [(sp[0], "valid",   sp) for sp in valid_spans]
    if cand:
        cand.sort(reverse=True)  # latest position wins
        _, verdict, span = cand[0]
        return verdict, _classify_position(s, span)

    # Stage 4 — last-ditch substring scan (mirrors the original simple parser).
    lower = s.lower()
    mi = re.search(r"\binvalid\b", lower)
    mv = re.search(r"\bvalid\b", lower)
    if mi and mv:
        winner = "invalid" if mi.start() > mv.start() else "valid"
        span = (mi.span() if winner == "invalid" else mv.span())
        return winner, _classify_position(s, span)
    if mi:
        return "invalid", _classify_position(s, mi.span())
    if mv:
        return "valid", _classify_position(s, mv.span())
    return None, "none"


def extract_cot_trace(response: str) -> str:
    """Return the response with any trailing verdict marker stripped.

    Useful for FLARE / Task-2 analysis — the trace tokens before the verdict.
    """
    if not response:
        return ""
    s = response.strip()
    # Drop a trailing **valid** / **invalid** if present.
    s = _END_BOLD.sub("", s).rstrip()
    # Drop any "Answer: X" or bare-line trailing verdict.
    s = re.sub(
        r"(?:^|\n)\s*(?:answer|final answer|conclusion|verdict)?\s*[:\-]?\s*"
        r"\*?\*?\s*(?:valid|invalid)\s*\*?\*?\s*\.?\s*$",
        "",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    ).rstrip()
    return s
