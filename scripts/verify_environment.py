"""Fail fast when DashApp is run from an unsupported or incomplete environment."""

from __future__ import annotations

import argparse
import importlib.util
import sys

MINIMUM_PYTHON = (3, 10)
MAXIMUM_PYTHON_EXCLUSIVE = (3, 13)
REQUIRED_MODULES = (
    "alembic",
    "numpy",
    "pandas",
    "pyarrow",
    "pytest",
    "ruff",
    "sqlalchemy",
    "streamlit",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Check the interpreter version without importing installed dependencies.",
    )
    args = parser.parse_args()
    problems: list[str] = []
    version = sys.version_info[:2]
    if not MINIMUM_PYTHON <= version < MAXIMUM_PYTHON_EXCLUSIVE:
        problems.append(
            "DashApp supports Python 3.10 through 3.12; "
            f"this environment uses Python {sys.version.split()[0]}."
        )
    if not args.python_only:
        for module in REQUIRED_MODULES:
            if importlib.util.find_spec(module) is None:
                problems.append(f"{module} is not installed")
    if problems:
        details = "\n- ".join(problems)
        raise SystemExit(
            "Environment verification failed:\n"
            f"- {details}\n\n"
            "Create a fresh Python 3.12 virtual environment, install requirements-dev.txt, "
            "and run checks through .venv/bin/python. See README.md."
        )
    print(f"Environment verified with Python {sys.version.split()[0]}.")


if __name__ == "__main__":
    main()
