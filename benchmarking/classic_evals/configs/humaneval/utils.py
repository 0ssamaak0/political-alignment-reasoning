"""Custom HumanEval scoring for instruct models (chat-template + full-function
code-block extraction). Avoids the two failure modes of the stock lm-eval
HumanEval tasks on Mistral-/Llama-Instruct:
  - `humaneval_instruct` + chat template -> model ends its turn early -> empties
  - completion-style (`humaneval`, or instruct w/o chat) -> off-by-one indent
Here the model is asked to emit the WHOLE function in a ```python block; we
extract that block verbatim (consistent indentation) and run the unit tests.
"""
import re

import evaluate as hf_evaluate

compute_ = hf_evaluate.load("code_eval")  # requires HF_ALLOW_CODE_EVAL=1


def pass_at_k(references, predictions, k=None):
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(references=references, predictions=predictions, k=k)
    return res[0]


def build_predictions_block(resps, docs):
    """Extract the model's code as the full prediction. Robust to a dangling
    opening fence (the `until="\\n```"` stop strips the *closing* fence, so the
    generation is ```python\\n<code> with no closing fence). We strip a leading
    ```python / ``` fence wherever it appears, then truncate at any closing
    fence. Falls back to prompt+resp if the result omits the target function
    (so a body-only reply still gets its signature)."""
    out = []
    for resp, doc in zip(resps, docs):
        preds = []
        for r in resp:
            code = r
            m = re.search(r"```(?:python)?[ \t]*\n", code)  # leading fence
            if m:
                code = code[m.end():]
            idx = code.find("```")  # closing fence, if any survived
            if idx != -1:
                code = code[:idx]
            if ("def " + doc["entry_point"]) not in code:
                code = doc["prompt"] + code
            preds.append(code)
        out.append(preds)
    return out
