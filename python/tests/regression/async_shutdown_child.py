# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Helper program for `test_async_shutdown.py`.

Launches one asynchronous GHZ job per entry in `QUBIT_COUNTS` and retrieves
only the first result, leaving the rest outstanding when the interpreter shuts
down.

Run as:
    async_shutdown_child.py <run|sample|observe>
"""

import sys

import cudaq

# GHZ widths for the queued jobs.
QUBIT_COUNTS = (2, 4, 8, 16)

SHOTS = 20


@cudaq.kernel
def ghz_measured(num_qubits: int) -> bool:
    """GHZ state returning a measurement, for `run_async`."""
    q = cudaq.qvector(num_qubits)
    h(q[0])
    for i in range(num_qubits - 1):
        x.ctrl(q[i], q[i + 1])
    return mz(q[0])


@cudaq.kernel
def ghz(num_qubits: int):
    """GHZ state, for `sample_async` and `observe_async`.
    """
    q = cudaq.qvector(num_qubits)
    h(q[0])
    for i in range(num_qubits - 1):
        x.ctrl(q[i], q[i + 1])


def launch(api):
    """Enqueue one job per qubit count and return the handles."""
    if api == "run":
        return [
            cudaq.run_async(ghz_measured, n, shots_count=SHOTS)
            for n in QUBIT_COUNTS
        ]
    if api == "sample":
        return [
            cudaq.sample_async(ghz, n, shots_count=SHOTS) for n in QUBIT_COUNTS
        ]
    if api == "observe":
        hamiltonian = cudaq.spin.z(0)
        return [cudaq.observe_async(ghz, hamiltonian, n) for n in QUBIT_COUNTS]
    raise ValueError(f"unknown api: {api}")


def main():
    handles = launch(sys.argv[1])

    # Retrieve only the first result. The execution queue is serial FIFO, so
    # the remaining jobs are necessarily still queued or running at this point.
    handles[0].get()


main()
