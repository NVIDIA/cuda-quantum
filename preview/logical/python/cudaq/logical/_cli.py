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
    os.execv(str(executable), [sys.argv[0], *sys.argv[1:]])
    raise AssertionError("os.execv returned unexpectedly")


def qlx_opt() -> NoReturn:
    """Replace the wrapper process with the installed ``qlx-opt`` binary."""

    _exec_native("qlx-opt")


def qlx_translate() -> NoReturn:
    """Replace the wrapper process with the installed ``qlx-translate`` binary."""

    _exec_native("qlx-translate")
