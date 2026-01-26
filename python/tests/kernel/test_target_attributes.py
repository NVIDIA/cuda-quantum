# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import cudaq

skipIfNoGPU = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target('nvidia')),
    reason="nvidia backend not available")


@pytest.fixture
def do_something():
    cudaq.set_target("nvidia", option="fp32")
    yield "Running the tests."
    cudaq.__clearKernelRegistries()
    cudaq.reset_target()


@skipIfNoGPU
def test_target_valid_option_attribute(do_something):
    """Tests the target valid option attribute."""

    target = cudaq.get_target()
    assert target.platform == "default"


@skipIfNoGPU
def test_target_invalid_options_attribute():
    """Tests the target invalid options attribute."""

    with pytest.raises(RuntimeError) as e:
        cudaq.set_target("nvidia", options="fp32")
    assert "The keyword `options` argument is not supported in cudaq.set_target()" in str(
        e.value)
