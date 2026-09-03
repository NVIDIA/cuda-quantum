# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

import sys
from pathlib import Path


def _resolve_source() -> Path:
    for cand in (Path("source"), Path("preview/logical/docs/source")):
        if (cand / "conf.py").exists():
            return cand
    raise SystemExit(
        "could not locate docs source; run from preview/logical/docs or the repo root"
    )


def _require_extra(command: str, extra: str,
                   exc: ModuleNotFoundError) -> SystemExit:
    return SystemExit(
        f"{command} needs the '{extra}' extra ({exc.name} is missing); "
        f"install with: uv sync --extra {extra}  (or pip install '.[{extra}]')")


def main(argv: list[str] | None = None) -> int:
    try:
        from sphinx.cmd.build import main as sphinx_main
    except ModuleNotFoundError as exc:
        raise _require_extra("build-docs", "build-deps", exc)

    extra = list(sys.argv[1:] if argv is None else argv)
    source = _resolve_source()
    build = source.parent / "_build" / "html"
    return sphinx_main(["-W", "-b", "html", str(source), str(build), *extra])


def test_main(argv: list[str] | None = None) -> int:
    try:
        import pytest
        import sybil  # noqa: F401
    except ModuleNotFoundError as exc:
        raise _require_extra("test-docs", "test-deps", exc)

    extra = list(sys.argv[1:] if argv is None else argv)
    source = _resolve_source()
    return pytest.main(["-p", "no:cacheprovider", str(source), *extra])
