"""Prompt construction and code extraction, shared by every generator (JAX, vLLM, API)."""

import re

INSTRUCTION = "Reply with the complete Python function in a single ```python code block."


def build_messages(item: dict) -> list[dict]:
    """Chat messages for one item. Apply the model's chat template outside this module
    (Qwen3: enable_thinking=False)."""
    if item["source"] == "humaneval":
        content = f"Complete the following Python function.\n```python\n{item['text']}```\n{INSTRUCTION}"
    else:
        content = (f"{item['text']}\nYour code should pass this test:\n"
                   f"```python\n{item['example_test']}\n```\n{INSTRUCTION}")
    return [{"role": "user", "content": content}]


_DEF = re.compile(r"^\s*(?:async\s+)?def\s+\w+", re.MULTILINE)
_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|$)", re.DOTALL)


def extract_code(completion: str, item: dict | None = None) -> str:
    """Last fenced block that defines a function (models often append a usage/test block),
    else the last block, else the raw text. Strips Qwen <think> blocks."""
    text = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL)
    blocks = _BLOCK.findall(text)
    with_def = [b for b in blocks if _DEF.search(b)]
    code = (with_def or blocks or [text])[-1]
    if item is not None and item["source"] == "humaneval":
        if f"def {item['entry_point']}" not in code:  # model returned only the body
            code = item["text"] + code
    return code.strip() + "\n"
