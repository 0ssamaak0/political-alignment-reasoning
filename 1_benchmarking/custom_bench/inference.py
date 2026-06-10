"""Prompt utilities shared by the TT inference runners.

The HF-based loader/generator that used to live here was removed once
`run_all_tt.py` (torchtune-native, epoch 0) became the canonical path.
Only the prompt template + chat-message builder + stop strings remain.
"""

INSTRUCTION = (
    "You will be given a logical argument. Decide whether the conclusion "
    "deductively follows from the premises. Focus on the logical form, "
    "not the content.\n\n"
    "Argument:\n{text}\n\n"
    "Reason briefly in 2-4 sentences, then on a new line write your final "
    "answer formatted EXACTLY as **valid** or **invalid** (with the asterisks). "
    "Do not write anything after that final answer."
)

STOP_STRINGS = ["**valid**", "**invalid**", "**Valid**", "**Invalid**"]


def build_messages(text, system_prompt=None):
    user = INSTRUCTION.format(text=text)
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user})
    return msgs
