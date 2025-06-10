# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s

import os
import pytest

def test_cudaq_alias_as_quda():
    """Tests aliasing cudaq as quda"""
    import cudaq as quda

    @quda.kernel
    def simple():
        q = quda.qubit()
    
    results = quda.sample(simple)
    results.dump()
    assert '1000' in result

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
