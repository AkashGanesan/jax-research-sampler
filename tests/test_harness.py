import math

import pytest

from harness.prompts import build_messages, extract_code
from harness.stats import pass_at_k, summarize
from harness.verifier import run_program, verify_many

ITEM = {"id": "t/1", "source": "mbpp", "text": "Add two numbers.", "example_test": "assert add(1, 2) == 3",
        "setup": "", "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
        "plus_test": "for a, b in [(0, 0), (5, 7)]:\n    assert add(a, b) == a + b"}


def test_pass_at_k_matches_closed_form():
    # 1 - C(n-c, k) / C(n, k)
    for n, c, k in [(10, 3, 1), (10, 3, 5), (20, 0, 4), (8, 8, 8), (16, 5, 8)]:
        expected = 1 - math.comb(n - c, k) / math.comb(n, k)
        assert pass_at_k(n, c, k) == pytest.approx(expected)
    with pytest.raises(ValueError):
        pass_at_k(4, 1, 8)


def test_summarize_reports_k_up_to_half_n():
    rows = [{"id": f"i{i}", "basic": "pass" if s < i % 9 else "fail"} for i in range(30) for s in range(8)]
    out = summarize(rows, "basic")
    assert out["samples_per_item"] == 8 and "pass@4" in out and "pass@8" not in out
    lo, hi = out["pass@1"]["ci95"]
    assert lo <= out["pass@1"]["mean"] <= hi


def test_extract_code_prefers_last_python_block_and_strips_think():
    c = "<think>```python\nbad\n```</think>Sure:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    assert extract_code(c) == "def add(a, b):\n    return a + b\n"
    assert extract_code("```python\ndef f(): pass") == "def f(): pass\n"  # unterminated block


def test_extract_code_skips_trailing_usage_block():
    c = "```python\ndef add(a, b):\n    return a + b\n```\nExample:\n```python\nassert add(1, 2) == 3\n```"
    assert extract_code(c).startswith("def add")


def test_humaneval_body_only_gets_prompt_prepended():
    item = {"source": "humaneval", "entry_point": "f", "text": "def f(x):\n    \"\"\"doc\"\"\"\n"}
    assert extract_code("```python\n    return x\n```", item).startswith("def f(x):")
    assert "Complete the following" in build_messages(item)[0]["content"]


def test_verifier_statuses():
    ok = "def add(a, b):\n    return a + b\n"
    wrong = "def add(a, b):\n    return a - b\n"
    (r_ok, r_wrong), stats = verify_many([(ITEM, ok), (ITEM, wrong)])
    assert r_ok == {"basic": "pass", "plus": "pass"}
    assert r_wrong["basic"] == "fail" and r_wrong["plus"] == "fail"
    assert stats["programs"] == 4
    assert run_program("while True: pass", timeout_s=2)["status"] == "timeout"
    assert run_program("raise ValueError('x')")["status"] == "error"
    assert run_program("x = bytearray(8 << 30)", mem_mb=1024)["status"] == "error"  # mem limit


def test_sandbox_blocks_destructive_calls(tmp_path):
    victim = tmp_path / "keep.txt"
    victim.write_text("x")
    r = run_program(f"import os\nos.remove({str(victim)!r})")
    assert r["status"] == "error" and victim.exists()
