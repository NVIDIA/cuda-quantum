# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from contextlib import contextmanager
import cudaq
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime


@contextmanager
def PreallocatedQubitsContext(n_qubits: int,
                              n_shots: int = 1000,
                              context_name: str = "sample"):
    """
    Context manager for running simulations with a fixed number of qubits
    pre-allocated ahead of time.
    
    Example:
    ```python
        with PreallocatedQubitsContext(2, 1000, "sample") as ctx:
            kernel()
        results = ctx.result
    ```
    """
    context = cudaq_runtime.ExecutionContext(context_name, n_shots)
    cudaq.testing.toggleDynamicQubitManagement()
    qubits = cudaq.testing.allocateQubits(n_qubits)
    with context:
        try:
            yield context
        finally:
            cudaq.testing.toggleDynamicQubitManagement()
            cudaq.testing.deallocateQubits(qubits)
