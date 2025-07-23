# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import cudaq


def test_target_valid_option_attribute():
    """Tests the target valid option attribute."""

    cudaq.set_target("nvidia", option="mqpu")
    target = cudaq.get_target()
    assert target.platform == "mqpu"


def test_target_invalid_options_attribute():
    """Tests the target invalid options attribute."""

    with pytest.raises(RuntimeError) as e:
        cudaq.set_target("nvidia", options="mqpu")
    assert "The keyword `options` argument is not supported in cudaq.set_target()" in str(
        e.value)
