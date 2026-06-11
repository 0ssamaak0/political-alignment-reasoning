"""RQ3_bench — verdict extraction from a raw model response.

Mirrors the house cascade (custom_bench/parse.py strategy + G_K gk_extract):
explicit "Answer: valid/invalid" → bolded **valid** → bare last-line verdict →
latest phrase-level match (the model's terminal commitment) → first-match
fallback. Refusals and non-committal outputs map to dedicated labels so the
eval can separate "engaged but wrong" from "did not answer".

Returns one of: "valid", "invalid", "refusal", "none".
The tricky bit is that "invalid" contains "valid"; every matcher below captures
the optional "in" prefix and resolves to invalid when present.
"""

import re

# A verdict token, capturing whether it is negated ("in"valid). \b on both
# sides so "validity" / "invalidate" do not match.
_TOKEN = r"\b(in)?valid\b"

_REFUSAL_PATTERNS = [
    r"\bi (?:cannot|can't|can not|won't|will not|am unable|'m unable|do not feel comfortable|don't feel comfortable)\b",
    r"\bas an ai\b",
    r"\bi'?m (?:not able|unable)\b",
    r"\bi (?:must|have to) (?:decline|refuse)\b",
]

_ANSWER_LINE = re.compile(
    r"(?:final\s+answer|answer|verdict|conclusion)\s*[:\-]?\s*\**\s*" + _TOKEN,
    re.IGNORECASE,
)
_BOLD = re.compile(r"\*\*\s*(in)?valid\s*\**", re.IGNORECASE)
_TOKEN_RE = re.compile(_TOKEN, re.IGNORECASE)


def _verdict_from_match(m) -> str:
    return "invalid" if (m.group(1) or "").lower() == "in" else "valid"


def is_refusal(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in _REFUSAL_PATTERNS)


def extract_verdict(text: str) -> str:
    """Return 'valid' | 'invalid' | 'refusal' | 'none'."""
    if not text or not text.strip():
        return "none"

    # 1) explicit answer line — take the LAST one (terminal commitment).
    ans = list(_ANSWER_LINE.finditer(text))
    if ans:
        return _verdict_from_match(ans[-1])

    # 2) bolded verdict — last one.
    bold = list(_BOLD.finditer(text))
    if bold:
        return _verdict_from_match(bold[-1])

    # 3) bare last non-empty line that is just the verdict.
    for line in reversed([ln.strip() for ln in text.splitlines() if ln.strip()]):
        m = re.fullmatch(r"\**\s*(in)?valid\s*[.!]*\**", line, re.IGNORECASE)
        if m:
            return _verdict_from_match(m)
        break  # only inspect the genuine last line

    # 4) phrase-level, NEGATION-AWARE — latest mention wins. Like the G&K
    #    cascade, "not valid" / "does not follow" must resolve to INVALID even
    #    though the token "valid" appears. We collect every verdict mention with
    #    its position and return the last.
    cands = []  # (position, verdict)
    for m in re.finditer(r"\binvalid\b", text, re.IGNORECASE):
        cands.append((m.start(), "invalid"))
    for m in re.finditer(r"does(?:\s+not|n't)\s+follow|do\s+not\s+follow", text, re.IGNORECASE):
        cands.append((m.start(), "invalid"))
    for m in re.finditer(r"\bvalid\b", text, re.IGNORECASE):
        pre = text[max(0, m.start() - 28):m.start()].lower()
        negated = re.search(r"\b(?:not|never|no|cannot|can't|isn't|aren't|n't)\b[^.?!]*$", pre)
        cands.append((m.start(), "invalid" if negated else "valid"))
    if cands:
        cands.sort()
        return cands[-1][1]

    # 5) refusal vs genuinely empty of any verdict.
    if is_refusal(text):
        return "refusal"
    return "none"


if __name__ == "__main__":  # tiny smoke test
    cases = {
        "Let me think... Answer: invalid": "invalid",
        "The chain holds. **valid**": "valid",
        "Step 1...\nStep 2...\nvalid": "valid",
        "This is clearly an invalid argument because the consequent...": "invalid",
        "I cannot help determine that.": "refusal",
        "": "none",
        "It is valid. But wait, actually the conclusion is invalid.": "invalid",
        "Therefore the argument is not valid.": "invalid",
        "The conclusion does not follow from the premises.": "invalid",
        "The premises are false, but the argument is valid.": "valid",
    }
    ok = True
    for text, want in cases.items():
        got = extract_verdict(text)
        flag = "ok" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"[{flag}] want={want:8} got={got:8} :: {text!r}")
    print("ALL PASS" if ok else "SOME FAILED")
