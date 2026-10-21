# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
Regression tests for `cudaq.kernel.utils.emitWarning`. The helper previously
built the formatted diagnostic into a local `msg` and then returned without
ever printing it, so every warning routed through it was silently dropped
(the only visible effect was a stray bold-escape with no text). It also
labelled the diagnostic "error:" even though it is a warning.
"""

import pytest

from cudaq.kernel.utils import emitWarning


def test_emit_warning_prints_message(capsys):
    emitWarning("a distinctive warning body")
    captured = capsys.readouterr()
    # The warning text must actually reach the user.
    assert "a distinctive warning body" in captured.out


def test_emit_warning_uses_warning_label(capsys):
    emitWarning("some advisory")
    captured = capsys.readouterr()
    # It is a warning, not an error.
    assert "warning:" in captured.out
    assert "error:" not in captured.out
