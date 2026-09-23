# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Register the phase-folding benchmark targets for this test tree."""

import pathlib

import pytest

_BENCH_TARGET_DIR = pathlib.Path(__file__).parent / "phase_folding"
_BENCH_TEST_DIRS = {"phase_folding", "generated"}


@pytest.fixture(autouse=True)
def _phase_folding_bench_targets(request):
    # Deliberately no module-level `import cudaq` in this file: tests such as
    # utils/target_env_var_check.py set CUDAQ_DEFAULT_SIMULATOR at import time
    # and rely on the runtime not having initialized yet. Importing here, at
    # setup time, keeps that ordering intact.
    if request.path.parent.name not in _BENCH_TEST_DIRS:
        return

    import cudaq

    for config in sorted(_BENCH_TARGET_DIR.glob("phase-folding-bench*.yml")):
        # Returns False once already registered, so repeat calls are harmless.
        cudaq._register_target_config(str(config))
