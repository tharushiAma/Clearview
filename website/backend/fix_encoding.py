#!/usr/bin/env python3
"""Fix emoji characters in backend_server.py that cause UnicodeEncodeError on Windows."""

import re

INPUT  = "backend_server.py"
OUTPUT = "backend_server.py"

with open(INPUT, encoding="utf-8") as f:
    content = f.read()

# Replace every emoji / special unicode with safe ASCII equivalents
emoji_map = [
    ("\u2705", "[OK]"),
    ("\u26a0\ufe0f ", "[WARN] "),
    ("\u26a0\ufe0f", "[WARN]"),
    ("\u274c", "[ERR]"),
    ("\U0001f680", "[START]"),
    ("\U0001f3af", "[MODEL]"),
    ("\U0001f504", "[LOAD]"),
    ("\u267b\ufe0f ", "[CACHE] "),
    ("\u267b\ufe0f", "[CACHE]"),
    ("\U0001f4ca", "[DATA]"),
    ("\u23f1\ufe0f", "[TIME]"),
    ("\u23f1", "[TIME]"),
    ("\U0001f50d", "[SEARCH]"),
    ("\U0001f4cd", "[LOC]"),
]
for old, new in emoji_map:
    content = content.replace(old, new)

# Insert UTF-8 reconfigure right after "import sys"
# This ensures even legacy emoji that slip through can still render
lines = content.split("\n")
for i, line in enumerate(lines):
    if line.strip() == "import sys":
        lines.insert(i + 1, "import io")
        lines.insert(i + 2, "if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')")
        lines.insert(i + 3, "if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8', errors='replace')")
        break

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Done: emoji replaced, UTF-8 stdout/stderr configured.")
