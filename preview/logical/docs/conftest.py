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

pytest_collect_file = Sybil(
    parsers=[PythonCodeBlockParser(), SkipParser()],
    pattern="*.md",
).pytest()
