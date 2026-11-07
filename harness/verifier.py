"""Run model code against unit tests in sandboxed subprocesses, in parallel.

Threads + subprocess (no fork of the parent's state), so this is safe to call from a process
that already imported jax: the parent only waits, and each test is a fresh `python -I` process.
"""

import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


RUNNER = str(Path(__file__).with_name("_sandbox_runner.py"))
PYTHON = sys.executable  # jax env python: has numpy, which MBPP+ tests need


def run_program(program: str, timeout_s: int = 10, mem_mb: int = 4096) -> dict:
    """Returns {"status": pass|fail|error|timeout, "detail": str}."""
    with tempfile.TemporaryDirectory(prefix="verify_") as d:
        path = os.path.join(d, "program.py")
        with open(path, "w") as f:
            f.write(program)
        env = {"PATH": "/usr/bin:/bin", "HOME": d, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        try:
            r = subprocess.run([PYTHON, "-I", "-B", RUNNER, path, str(timeout_s), str(mem_mb)],
                               cwd=d, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                               timeout=timeout_s, start_new_session=True)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "detail": f">{timeout_s}s"}
    if r.returncode == 0:
        return {"status": "pass", "detail": ""}
    err = r.stderr.decode(errors="replace").strip().splitlines()
    last = err[-1][:300] if err else f"exit {r.returncode}"
    return {"status": "fail" if last.startswith("AssertionError") else "error", "detail": last}


def programs_for(item: dict, code: str) -> dict[str, str]:
    """Verifier name -> program text. 'basic' = MBPP tests, 'plus' = MBPP+/HumanEval+ tests."""
    progs = {}
    if item["tests"]:
        progs["basic"] = "\n".join([code, item["setup"], *item["tests"]]) + "\n"
    if item.get("plus_test"):
        progs["plus"] = code + "\n" + item["plus_test"] + "\n"
    return progs


def verify_many(jobs: list[tuple[dict, str]], workers: int | None = None,
                timeouts: dict | None = None) -> tuple[list[dict], dict]:
    """jobs: [(item, code)]. Returns per-job {verifier: status, verifier_detail: ...} and stats."""
    timeouts = timeouts or {"basic": 10, "plus": 60}
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    tasks = [(i, name, prog) for i, (item, code) in enumerate(jobs)
             for name, prog in programs_for(item, code).items()]
    results = [{} for _ in jobs]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(run_program, prog, timeouts[name]): (i, name) for i, name, prog in tasks}
        for fut, (i, name) in futs.items():
            r = fut.result()
            results[i][name] = r["status"]
            if r["detail"]:
                results[i][f"{name}_detail"] = r["detail"]
    dt = time.perf_counter() - t0
    stats = {"programs": len(tasks), "seconds": round(dt, 2), "workers": workers,
             "programs_per_s": round(len(tasks) / dt, 1) if dt else None}
    return results, stats
