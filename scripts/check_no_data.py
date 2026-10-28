"""Fail if data-like files or restricted identifiers are tracked in git.

python scripts/check_no_data.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

BLOCKED_SUFFIXES = {
    ".csv",
    ".tsv",
    ".feather",
    ".parquet",
    ".arrow",
    ".sas7bdat",
    ".sav",
    ".dta",
    ".xlsx",
    ".xls",
    ".rds",
    ".rdata",
    ".rda",
    ".pkl",
    ".pickle",
    ".h5",
    ".hdf5",
    ".db",
    ".sqlite",
    ".zip",
    ".gz",
    ".pdf",
    ".docx",
    ".pptx",
}
# Generic patterns only. Project-specific ones (field names of the licensed source
# tables, analysis-server paths) live in an untracked local file, one regex per
# line, so the guard itself never discloses them.
BLOCKED_PATTERNS = [re.compile(r"/home/[\w.-]+/", re.I)]
LOCAL_PATTERNS = Path(".data-guard-patterns")
SELF = Path(__file__).resolve()


def load_patterns() -> list[re.Pattern[str]]:
    pats = list(BLOCKED_PATTERNS)
    if LOCAL_PATTERNS.is_file():
        for line in LOCAL_PATTERNS.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                pats.append(re.compile(line.strip(), re.I))
    return pats


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True)
    return [Path(p) for p in out.stdout.decode().split("\0") if p]


def main() -> int:
    problems = []
    patterns = load_patterns()
    for path in tracked_files():
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            problems.append(f"{path}: data-like file type")
            continue
        if path.resolve() == SELF or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pat in patterns:
            if pat.search(text):
                problems.append(f"{path}: matches {pat.pattern!r}")
    for p in problems:
        print("✗", p)
    print("data guard:", "FAILED" if problems else "ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
