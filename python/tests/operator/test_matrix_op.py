# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq, numpy as np, pytest
from cudaq.ops import * # FIXME: module name

def test_properties():
    number(1).dump()
    pass

# Run with: pytest -rP
if __name__ == "__main__":
    pytest.main(["-rP"])