"""Vertex Gemini judge — drop-in replacement for upstream judge.OpenAiJudge.

Workaround for the absence of OpenAI-style top-K logprobs on Gemini Vertex:
we sample N output tokens at temperature=0, parse the response as a leading
integer (or REFUSAL), and feed a single-entry score dict to the upstream
aggregator. The "score" is just the parsed integer — no expectation over
top-K logprobs since we don't have them.

Score ranges:
  - "0_10":   0–9   (max_output_tokens=1)   ← matches a "rate 0–9" prompt
  - "0_100":  0–100 (max_output_tokens=4)   ← matches a "rate 0–100" prompt
                                              (this is what upstream's PROMPTS
                                               template generates by default,
                                               so it's the right pick when our
                                               trait eval_prompt asks for 0–100)
  - "binary":      YES/NO (max_output_tokens=4)
  - "binary_text": <answer>YES</answer> tagged

Auth via gcloud ADC (no API key). Region defaults to "global"; project from
env or VERTEX_PROJECT.

Class signature mirrors upstream judge.OpenAiJudge so eval_persona.py's
imports work after `sys.modules['judge']` is patched.
"""

import asyncio
import os
import re

from google import genai
from google.genai import types

DEFAULT_PROJECT = os.environ.get(
    "VERTEX_PROJECT",
    os.environ.get("ANTHROPIC_VERTEX_PROJECT", "your-gcp-project"),
)
DEFAULT_REGION = os.environ.get("VERTEX_REGION", "global")

# Loop-aware lazy client init.
# The google-genai async client builds its underlying httpx AsyncClient lazily,
# bound to the *currently running* asyncio event loop. If the caller invokes
# `asyncio.run(...)` multiple times (e.g., once per pos/neg side), each call
# is a fresh loop — a client cached from the previous loop dies with
# "Event loop is closed" on its next request. We track the loop id and
# rebuild on change. Benign race if N concurrent calls hit a None cache:
# `genai.Client(...)` is cheap and the discarded one is never touched again.
_client = None
_client_loop_id = None


def _get_client():
    global _client, _client_loop_id
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = None
    if _client is None or loop_id != _client_loop_id:
        _client = genai.Client(
            vertexai=True,
            project=DEFAULT_PROJECT,
            location=DEFAULT_REGION,
        )
        _client_loop_id = loop_id
    return _client


# Module-level failure counters; the caller can read these at end of batch
# to surface silently-swallowed transient errors.
FAIL_COUNTERS = {"sample": 0, "parse": 0}


# Max output tokens. Sized so a literal "REFUSAL" (typically 2-3 Gemini tokens)
# fits with a few tokens of headroom on either side; the integer parse uses
# the first matching digit run, so trailing whitespace/punctuation is harmless.
_MAX_TOKENS = {"0_10": 8, "0_100": 8, "binary": 8, "binary_text": 32}

# Refusal detection: prefix-aware, since `max_output_tokens` may truncate
# "REFUSAL" mid-word in extreme cases. We accept any leading "REF" / "REFU" /
# "REFUS" / "REFUSAL" as a refusal token.
_REFUSAL_PREFIX_RE = re.compile(r"\bREF[UA]?", re.IGNORECASE)


class GeminiJudge:
    """Drop-in for upstream `judge.OpenAiJudge`. Same `__init__`, `judge`, and
    async `__call__` shape; same `eval_type` set; same return semantics
    (numeric for digit modes, 0/1 for binary, None on refusal/parse failure).
    """

    def __init__(self, model: str, prompt_template: str, eval_type: str = "0_100"):
        if eval_type not in _MAX_TOKENS:
            raise ValueError(
                f"Unsupported eval_type: {eval_type!r}. "
                f"Choose from {sorted(_MAX_TOKENS)}."
            )
        self.model = model
        self.prompt_template = prompt_template
        self.eval_type = eval_type
        self._max_tokens = _MAX_TOKENS[eval_type]

    async def _sample(self, prompt: str) -> str:
        client = _get_client()
        try:
            resp = await client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=self._max_tokens,
                    temperature=0.0,
                    # Gemini 3.0 has thinking enabled by default; with our small
                    # max_output_tokens budget the thinking eats it all and no
                    # visible text comes back. Force thinking off for the judge —
                    # we want a single-token classification, not a chain of
                    # reasoning.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return (resp.text or "").strip()
        except Exception as e:  # noqa: BLE001 (treat any transient as parse failure)
            FAIL_COUNTERS["sample"] += 1
            print(
                f"[gemini_judge] sample error #{FAIL_COUNTERS['sample']}: "
                f"{type(e).__name__}: {e}"
            )
            return ""

    @staticmethod
    def _parse_digit_or_refusal(text: str) -> dict:
        if not text:
            FAIL_COUNTERS["parse"] += 1
            return {}
        if _REFUSAL_PREFIX_RE.search(text):
            return {"REFUSAL": 1.0}
        m = re.search(r"\d+", text)
        if m:
            return {m.group(0): 1.0}
        FAIL_COUNTERS["parse"] += 1
        return {}

    @staticmethod
    def _parse_binary_token(text: str) -> dict:
        if not text:
            return {}
        upper = text.strip().upper()
        if "REFUSAL" in upper:
            return {"REFUSAL": 1.0}
        # First token of "YES"/"NO" can be partial; check prefix.
        if upper.startswith(("Y", "YES")):
            return {"YES": 1.0}
        if upper.startswith(("N", "NO")):
            return {"NO": 1.0}
        return {}

    def _agg_0_10(self, score: dict):
        if "REFUSAL" in score and score["REFUSAL"] > max(
            (score.get(str(i), 0) for i in range(10)), default=0
        ):
            return None
        for k, v in score.items():
            try:
                ki = int(k)
            except ValueError:
                continue
            if 0 <= ki <= 9 and v >= 0.25:
                return float(ki)
        return None

    def _agg_0_100(self, score: dict):
        if "REFUSAL" in score and score["REFUSAL"] > max(
            (score.get(str(i), 0) for i in range(101)), default=0
        ):
            return None
        for k, v in score.items():
            try:
                ki = int(k)
            except ValueError:
                continue
            if 0 <= ki <= 100 and v >= 0.25:
                return float(ki)
        return None

    def _agg_binary(self, score: dict):
        yes = score.get("YES", 0.0)
        no = score.get("NO", 0.0)
        ref = score.get("REFUSAL", 0.0)
        if ref > yes and ref > no:
            return None
        denom = yes + no
        if denom < 0.25:
            return None
        return yes / denom

    @staticmethod
    def _agg_binary_text(text: str):
        if "<answer>REFUSAL</answer>" in text:
            return None
        if "<answer>NO</answer>" in text:
            return 0
        if "<answer>YES</answer>" in text:
            return 1
        return None

    async def __call__(self, **kwargs):
        prompt = self.prompt_template.format(**kwargs)
        text = await self._sample(prompt)
        if self.eval_type == "binary_text":
            return self._agg_binary_text(text)
        if self.eval_type == "binary":
            return self._agg_binary(self._parse_binary_token(text))
        score = self._parse_digit_or_refusal(text)
        if self.eval_type == "0_10":
            return self._agg_0_10(score)
        return self._agg_0_100(score)

    async def judge(self, **kwargs):
        return await self(**kwargs)


# Quick CLI smoke test: `python3 gemini_judge.py "5+5=" 0_10`
if __name__ == "__main__":
    import sys

    text = sys.argv[1] if len(sys.argv) > 1 else (
        "Reply with the digit 7 and nothing else: "
    )
    eval_type = sys.argv[2] if len(sys.argv) > 2 else "0_10"
    template = "{question}{answer}"
    j = GeminiJudge(
        os.environ.get("VERTEX_GEMINI_MODEL", "gemini-3-flash-preview"),
        template,
        eval_type=eval_type,
    )
    out = asyncio.run(j(question=text, answer=""))
    print(f"[gemini_judge] eval_type={eval_type} score={out!r}")
