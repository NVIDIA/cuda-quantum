# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import warnings

import pytest

import cudaq


def test_set_noise_emits_deprecation_warning():
    cudaq.reset_target()
    cudaq.set_target("density-matrix-cpu")
    noise = cudaq.NoiseModel()
    try:
        with pytest.warns(
                DeprecationWarning,
                match=
                "set_noise is deprecated; please use launch arguments or launch options."
        ):
            cudaq.set_noise(noise)
    finally:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cudaq.unset_noise()
        cudaq.reset_target()


def test_unset_noise_emits_deprecation_warning():
    cudaq.reset_target()
    cudaq.set_target("density-matrix-cpu")
    noise = cudaq.NoiseModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cudaq.set_noise(noise)
    try:
        with pytest.warns(
                DeprecationWarning,
                match=
                "unset_noise is deprecated; please use launch arguments or launch options."
        ):
            cudaq.unset_noise()
    finally:
        cudaq.reset_target()


def test_sample_noise_model_kwarg_does_not_emit_set_noise_deprecation():
    """Recommended launch path must not warn about the deprecated global API."""
    cudaq.reset_target()
    cudaq.set_target("density-matrix-cpu")
    try:
        kernel = cudaq.make_kernel()
        q = kernel.qalloc()
        kernel.x(q)
        kernel.mz(q)

        noise = cudaq.NoiseModel()
        noise.add_all_qubit_channel("x", cudaq.BitFlipChannel(1.0))

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            result = cudaq.sample(kernel, shots_count=100, noise_model=noise)

        deprecation = [
            w for w in recorded
            if issubclass(w.category, DeprecationWarning) and
            "set_noise is deprecated" in str(w.message)
        ]
        assert not deprecation
        assert result["0"] == 100
    finally:
        cudaq.reset_target()
