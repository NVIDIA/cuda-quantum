# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import json
import os
import pytest

skipIfPasqalNotInstalled = pytest.mark.skipif(
    not cudaq.has_target("qilimanjaro"),
    reason='Could not find `qilimanjaro` in installation'
)


@pytest.fixture(scope="session", autouse=True)
def do_something():
    # NOTE: Credentials can be set with environment variables
    cudaq.set_target("qilimanjaro")
    yield "Running the tests."
    cudaq.reset_target()

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])