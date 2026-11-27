# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Console entry points for package-private native CUDA-Q Logical tools."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import NoReturn


def _exec_native(tool: str) -> NoReturn:
    executable = Path(__file__).resolve().parent / "_bin" / tool
    environment = os.environ.copy()
    from cudaq import core
    core_lib = Path(core.__file__).parent / "lib"
    if core_lib.is_dir():
        # exec starts a new loader: Python's already-loaded providers are lost.
        # Core can be in a different user/virtual-environment installation prefix.
        variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        paths = [str(core_lib)]
        if environment.get(variable):
            paths.append(environment[variable])
        environment[variable] = os.pathsep.join(paths)
    os.execve(str(executable), [sys.argv[0], *sys.argv[1:]], environment)
    raise AssertionError("os.execve returned unexpectedly")


def qlx_opt() -> NoReturn:
    """Replace the wrapper process with the installed ``qlx-opt`` binary."""

    _exec_native("qlx-opt")


def qlx_translate() -> NoReturn:
    """Replace the wrapper process with the installed ``qlx-translate`` binary."""

    _exec_native("qlx-translate")
