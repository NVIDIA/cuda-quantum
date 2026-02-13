# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import cudaq


@pytest.fixture(autouse=True)
def ptsbe_target():
    cudaq.set_target("density-matrix-cpu")
    cudaq.set_random_seed(42)
    yield
    cudaq.reset_target()


@pytest.fixture
def depol_noise():
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.1), num_controls=1)
    return noise
