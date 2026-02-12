# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import cudaq
from pathlib import Path

_NON_TEST_MODULES = frozenset(
    {"conftest.py", "__init__.py", "kernels.py", "test_ptsbe.py"})


def pytest_configure(config):
    here = Path(__file__).resolve().parent
    for path in sorted(here.glob("*.py")):
        if path.name not in _NON_TEST_MODULES:
            config.addinivalue_line("python_files", path.name)


@pytest.fixture(autouse=True)
def density_matrix_target():
    cudaq.set_target("density-matrix-cpu")
    cudaq.set_random_seed(42)
    yield
    cudaq.reset_target()


@pytest.fixture
def depol_noise():
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.1), num_controls=1)
    return noise
