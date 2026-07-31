# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Helper program for `test_async_shutdown.py`.

For `run`/`sample`/`observe`, launches one asynchronous GHZ job per entry in
`QUBIT_COUNTS` and retrieves only the first result, leaving the rest
outstanding when the interpreter shuts down.

`thread` submits from a non-daemon thread that outlives `main`, and
`target_swap` leaves jobs outstanding on a platform that is no longer the
current target.

Run as:
    async_shutdown_child.py <run|sample|observe|thread|target_swap>
"""

import sys
import threading
import time

import cudaq

# GHZ widths for the queued jobs.
QUBIT_COUNTS = (2, 4, 8, 16)

# Wider jobs for `target_swap`, which needs them outstanding on a GPU platform.
TARGET_SWAP_QUBIT_COUNTS = (4, 24, 26, 28)

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


def submit_after_main_returns():
    """Submit from a non-daemon thread that outlives `main`.
    """

    def worker():
        time.sleep(2)
        cudaq.sample_async(ghz, QUBIT_COUNTS[1], shots_count=SHOTS).get()
        # Marker for the test to check.
        print("SUCCESS", flush=True)

    threading.Thread(target=worker).start()


def leave_jobs_on_a_swapped_out_platform():
    """Leave jobs outstanding on a platform that is no longer current.
    """
    cudaq.set_target("nvidia", option="mqpu")
    handles = [
        cudaq.sample_async(ghz, n, shots_count=SHOTS)
        for n in TARGET_SWAP_QUBIT_COUNTS
    ]
    handles[0].get()
    # Swap the target then shutdown, leaving the remaining jobs on the previous platform.
    cudaq.set_target("qpp-cpu")


def main():
    test_case = sys.argv[1]
    if test_case == "thread":
        return submit_after_main_returns()
    if test_case == "target_swap":
        return leave_jobs_on_a_swapped_out_platform()

    handles = launch(test_case)

    # Retrieve only the first result. The execution queue is serial FIFO, so
    # the remaining jobs are necessarily still queued or running at this point.
    handles[0].get()


main()
