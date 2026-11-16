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

GADGET_EXAMPLES = tuple(
    sorted((EXAMPLE_ROOT / "gadgets").glob("[0-9][0-9]_*.py")))


@pytest.fixture(autouse=True)
def reset_cudaq_target_after_test():
    yield
    import cudaq

    cudaq.reset_target()


ALL_EXAMPLES = (*KERNEL_EXAMPLES, *STANDALONE_EXAMPLES, *GADGET_EXAMPLES)

# Bump these when adding or removing an example so it gets a test (and a
# timing entry in `pytest --durations`).
EXPECTED_EXAMPLE_COUNTS = {
    "kernel": 7,
    "standalone": 8,
    "gadgets": 5,
}


def test_expected_example_count():
    assert {
        "kernel": len(KERNEL_EXAMPLES),
        "standalone": len(STANDALONE_EXAMPLES),
        "gadgets": len(GADGET_EXAMPLES),
    } == EXPECTED_EXAMPLE_COUNTS, (
        "Example count changed; update EXPECTED_EXAMPLE_COUNTS.")


@pytest.mark.parametrize(
    "example",
    ALL_EXAMPLES,
    ids=[str(e.relative_to(EXAMPLE_ROOT)) for e in ALL_EXAMPLES],
)
def test_example_executes(example, monkeypatch, tmp_path):
    import cudaq

    # Without a GPU, example 08 runs on `density-matrix-cpu`, which is slow.
    if (example.name == "08_t_factory_evaluation.py" and
            cudaq.num_available_gpus() == 0):
        pytest.skip("requires a GPU; the CPU density-matrix fallback is slow")
    monkeypatch.syspath_prepend(str(EXAMPLE_ROOT))
    # Memory examples export their artifacts relative to the working directory.
    monkeypatch.chdir(tmp_path)
    cudaq.reset_target()
    monkeypatch.setattr(sys, "argv", [str(example)])
    runpy.run_path(str(example), run_name="__main__")
