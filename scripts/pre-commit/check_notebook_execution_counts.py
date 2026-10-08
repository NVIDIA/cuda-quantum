#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2023 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Check that staged Jupyter notebooks have sequential code-cell execution counts."""

import json
import sys
from pathlib import Path


def check_notebook(path: Path) -> list[str]:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: cannot read notebook JSON: {exc}"]

    code_cells = [
        cell for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]

    # A completely `unexecuted` notebook has no execution ordering to enforce.
    if all(cell.get("execution_count") is None for cell in code_cells):
        return []

    errors = []
    for index, cell in enumerate(code_cells, start=1):
        actual = cell.get("execution_count")
        if actual != index:
            errors.append(
                f"{path}: code cell {index} has execution_count={actual!r}; "
                f"expected {index}")
    return errors


def main() -> int:
    notebooks = [Path(name) for name in sys.argv[1:] if name.endswith(".ipynb")]
    errors = [error for path in notebooks for error in check_notebook(path)]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(
            "Re-run the notebook from top to bottom and stage the updated file.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
