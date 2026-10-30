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
KERNEL_EXAMPLES = (
    *sorted(EXAMPLE_ROOT.glob("[0-9][0-9]_*.py")),
    EXAMPLE_ROOT / "steane_bell_e2e_workflow.py",
)
STANDALONE_EXAMPLES = tuple(
    sorted((EXAMPLE_ROOT / "standalone").glob("[0-9][0-9]_*.py")))


@pytest.fixture(autouse=True)
def reset_cudaq_target_after_test():
    yield
    import cudaq

    cudaq.reset_target()


def test_examples_execute(capsys, monkeypatch, tmp_path):
    import cudaq

    assert len(KERNEL_EXAMPLES) == 7
    assert len(STANDALONE_EXAMPLES) == 10
    monkeypatch.syspath_prepend(str(EXAMPLE_ROOT))
    # 05_memory_edem.py exports its artifact relative to the working directory.
    monkeypatch.chdir(tmp_path)
    output = {}
    for example in (*KERNEL_EXAMPLES, *STANDALONE_EXAMPLES):
        # Without a GPU, example 07 runs on `density-matrix-cpu`, which is slow.
        if (example.name == "07_t_factory_evaluation.py" and
                cudaq.num_available_gpus() == 0):
            continue
        cudaq.reset_target()
        monkeypatch.setattr(sys, "argv", [str(example)])
        runpy.run_path(str(example), run_name="__main__")
        output[example.name] = capsys.readouterr().out
