# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq
import numpy as np
import sys

cudaq.set_target("quake_fake")


@cudaq.extern_kernel
def wait(duration: float, q: cudaq.qubit) -> None:
    ...


@cudaq.kernel
def ramsey_single(wait_duration: float) -> bool:
    qubit = cudaq.qubit()
    rx(np.pi / 2, qubit)
    wait(wait_duration, qubit)
    rx(np.pi / 2, qubit)
    return mz(qubit)


try:
    results = cudaq.run(ramsey_single, 1.0, shots_count=100)
    assert len(results) == 100, f"expected 100 results, got {len(results)}"
    for shot in results:
        assert shot in (True, False), f"unexpected result {shot}"
except Exception as e:
    print(e)
    sys.exit(1)
