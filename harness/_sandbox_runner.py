"""Executed as `python -I -B _sandbox_runner.py program.py <timeout_s> <mem_mb>` in a temp dir.

stdlib only (runs under -I, so the harness package is not importable). Sets resource limits,
pre-imports common test deps, disables destructive functions (in the style of the human-eval
reliability guard), then runs the program. Exit code 0 means every assertion passed.

This is NOT a security boundary (there's no network or filesystem isolation). It only protects
against accidents from model-written code. Use a container/VM for untrusted agents (week 7).
"""

import builtins
import faulthandler
import os
import resource
import shutil
import subprocess
import sys

prog, timeout_s, mem_mb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

resource.setrlimit(resource.RLIMIT_AS, (mem_mb << 20, mem_mb << 20))
resource.setrlimit(resource.RLIMIT_CPU, (timeout_s + 1, timeout_s + 1))
resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
faulthandler.disable()

# Pre-import what MBPP+/HumanEval+ test code uses, before the guard removes os functions.
import bisect  # noqa: F401
import collections  # noqa: F401
import functools  # noqa: F401
import heapq  # noqa: F401
import itertools  # noqa: F401
import math  # noqa: F401
import re  # noqa: F401
import string  # noqa: F401
import typing  # noqa: F401

try:
    import numpy  # noqa: F401
except ImportError:
    pass

src = open(prog).read()

for name in ["kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid",
             "fork", "forkpty", "killpg", "rename", "renames", "truncate", "replace", "unlink",
             "fchmod", "fchown", "chmod", "chown", "chroot", "lchown", "chdir"]:
    if hasattr(os, name):
        setattr(os, name, None)
shutil.rmtree = shutil.move = shutil.chown = None
subprocess.Popen = None
builtins.exit = builtins.quit = None
for mod in ["ipdb", "joblib", "resource", "psutil", "tkinter"]:
    sys.modules[mod] = None

sys.setrecursionlimit(10000)
exec(compile(src, "program.py", "exec"), {"__name__": "__main__"})
