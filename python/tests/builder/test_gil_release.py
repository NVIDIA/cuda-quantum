# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #

import signal
import cudaq


def _assert_sigalrm_fires_during(compile_and_run):
    """Run compile_and_run and verify SIGALRM fires (GIL released)."""
    timed_out = False

    def handler(_signum, _frame):
        nonlocal timed_out
        timed_out = True
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, 0.05)
    try:
        compile_and_run()
        signal.setitimer(signal.ITIMER_REAL, 0)
    except TimeoutError:
        pass
    finally:
        signal.signal(signal.SIGALRM, old)
        cudaq.set_target('qpp-cpu')

    assert timed_out, (
        "SIGALRM did not fire during compilation (GIL not released?)")


def test_gil_release_estimate_resources():
    cudaq.set_target('circuit-opt-bench', device='path(50)')
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(50)
    for i in range(49):
        kernel.cx(q[i], q[(i + 2) % 50])
    _assert_sigalrm_fires_during(lambda: cudaq.estimate_resources(kernel))
