# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Execute inline Python snippets from the MyST docs against cudaq.logical."""

from __future__ import annotations

from sybil import Sybil
from sybil.parsers.myst import PythonCodeBlockParser, SkipParser


def _setup(namespace):
    from pathlib import Path
    import runpy

    import cudaq.logical.compiler.build as build_mod

    # Sybil executes snippets with exec(), so @ql.program providers keep
    # __module__ as None and serialized builds would record source_modules=[None].
    if not getattr(build_mod.Build.__init__, "_qlx_docs_patched", False):

        def _init(self, /, **kwargs):
            source_modules = kwargs.get("source_modules", ())
            kwargs["source_modules"] = tuple(
                dict.fromkeys(
                    item if isinstance(item, str) and item else "__main__"
                    for item in source_modules))
            return build_mod.Build._qlx_original_init(self, **kwargs)

        build_mod.Build._qlx_original_init = build_mod.Build.__init__
        build_mod.Build.__init__ = _init
        build_mod.Build.__init__._qlx_docs_patched = True

    root = Path.cwd()
    while (root != root.parent and
           not (root / "preview/logical/examples").is_dir()):
        root = root.parent
    if not (root / "preview/logical/examples").is_dir():
        raise RuntimeError(
            "could not locate preview/logical/examples; run doc tests from "
            "the repository root or preview/logical/docs")

    def load_ql_example(relpath, *names):
        mod = runpy.run_path(str(root / relpath))
        if len(names) == 1:
            return mod[names[0]]
        return tuple(mod[name] for name in names)

    namespace["load_ql_example"] = load_ql_example


pytest_collect_file = Sybil(
    parsers=[PythonCodeBlockParser(), SkipParser()],
    pattern="*.md",
    setup=_setup,
).pytest()
