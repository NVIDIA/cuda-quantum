# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
import os
import pytest

# The `pasqal` target only exists when CUDA-Q was configured with the Pasqal
# backend enabled (CUDAQ_ENABLE_PASQAL_BACKEND). Guard on what the build
# actually produced -- the session fixture below calls set_target during
# setup, which raises on a build that correctly does not provide it.
pytestmark = pytest.mark.skipif(
    not cudaq.has_target("pasqal"),
    reason="Could not find `pasqal` in installation")


@pytest.fixture(scope="session", autouse=True)
def set_up_target():
    # NOTE: Credentials can be set with environment variables.
    # This test covers the direct `pasqal` backend only.
    # QRMI-routed execution is validated separately because it requires a
    # supported QRMI build and a compatible cluster resource manager.
    cudaq.set_target("pasqal")
    yield "Running the tests."
    cudaq.reset_target()


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
