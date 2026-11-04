# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Captured lists in repeated kernel calls must not exhaust the host stack."""

import subprocess
import sys

import cudaq
import numpy as np
import pytest


@pytest.mark.skipif(sys.platform != 'linux',
                    reason='requires Linux stack limits')
@pytest.mark.parametrize('case', ['flat', 'nested', 'controlled'])
def test_captured_list_stack(case):
    # A regression causes SIGSEGV, so isolate it from the pytest process and
    # enforce a stack limit even when the test runner has an unlimited stack.
    result = subprocess.run([sys.executable, __file__, case],
                            capture_output=True,
                            text=True,
                            timeout=120)
    assert result.returncode == 0, (f'{case} failed: {result.returncode}\n'
                                    f'{result.stdout}\n{result.stderr}')


def _run(case):
    import resource

    _, hard_limit = resource.getrlimit(resource.RLIMIT_STACK)
    stack_limit = 8 * 1024 * 1024
    if hard_limit != resource.RLIM_INFINITY:
        stack_limit = min(stack_limit, hard_limit)
    resource.setrlimit(resource.RLIMIT_STACK, (stack_limit, hard_limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cudaq.set_target('qpp-cpu')
    table = [i % 4 for i in range(60)]
    tables = [table.copy() for _ in range(4)]

    if case == 'nested':

        @cudaq.kernel
        def step(q: cudaq.qview):
            for j in range(4):
                for i in range(60):
                    if tables[j][i] == 7:
                        x(q[0])
            x(q[0])
    else:

        @cudaq.kernel
        def step(q: cudaq.qview):
            for i in range(60):
                if table[i] == 7:
                    x(q[0])
            x(q[0])

    if case == 'controlled':

        @cudaq.kernel
        def loop(n: int):
            q = cudaq.qvector(2)
            x(q[0])
            for _ in range(n):
                cudaq.control(step, q[0], q[1:])

        expected = '11'
    else:

        @cudaq.kernel
        def loop(n: int):
            q = cudaq.qvector(1)
            for _ in range(n):
                step(q)

        expected = '1'

    def stack_kib():
        with open('/proc/self/status') as status:
            return next(
                int(line.split()[1])
                for line in status
                if line.startswith('VmStk:'))

    cudaq.get_state(loop, 10)  # Warm up JIT compilation.
    before = stack_kib()
    cudaq.get_state(loop, 2000)
    assert stack_kib() - before < 256
    state = cudaq.get_state(loop, 30001)
    assert np.isclose(abs(state[int(expected, 2)]), 1.0)
    assert stack_kib() - before < 256
    counts = cudaq.sample(loop, 30001, shots_count=2)
    assert counts[expected] == 2


if __name__ == '__main__':
    _run(sys.argv[1])
