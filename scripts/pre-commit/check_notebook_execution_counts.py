#!/usr/bin/env python3
"""Check that staged Jupyter notebooks have sequential code-cell execution counts."""

import json
import sys
from pathlib import Path


def check_notebook(path: Path) -> list[str]:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: cannot read notebook JSON: {exc}"]

    errors = []
    expected = 1
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        actual = cell.get("execution_count")
        if actual != expected:
            errors.append(
                f"{path}: code cell {index} has execution_count={actual!r}; "
                f"expected {expected}"
            )
        expected += 1
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
