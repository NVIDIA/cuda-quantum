from __future__ import annotations

import sys
from pathlib import Path

from sphinx.cmd.build import main as sphinx_main


def _resolve_source() -> Path:
    for cand in (Path("source"), Path("preview/logical/docs/source")):
        if (cand / "conf.py").exists():
            return cand
    raise SystemExit(
        "could not locate docs source; run from preview/logical/docs or the repo root"
    )


def main(argv: list[str] | None = None) -> int:
    extra = list(sys.argv[1:] if argv is None else argv)
    source = _resolve_source()
    build = source.parent / "_build" / "html"
    return sphinx_main(["-W", "-b", "html", str(source), str(build), *extra])
