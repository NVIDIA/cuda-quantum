# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_ROOT = ROOT / "examples"
KERNEL_EXAMPLES = tuple(sorted(EXAMPLE_ROOT.glob("0[0-5]_*.py")))
STANDALONE_EXAMPLES = tuple(
    sorted((EXAMPLE_ROOT / "standalone").glob("[0-9][0-9]_*.py")))


@pytest.fixture(autouse=True)
def reset_cudaq_target_after_test():
    yield
    import cudaq

    cudaq.reset_target()


def test_examples_execute(capsys, monkeypatch):
    import cudaq

    assert len(KERNEL_EXAMPLES) == 6
    assert len(STANDALONE_EXAMPLES) == 6
    monkeypatch.syspath_prepend(str(EXAMPLE_ROOT))
    output = {}
    for example in (*KERNEL_EXAMPLES, *STANDALONE_EXAMPLES):
        monkeypatch.setattr(sys, "argv", [str(example)])
        runpy.run_path(str(example), run_name="__main__")
        output[example.name] = capsys.readouterr().out
