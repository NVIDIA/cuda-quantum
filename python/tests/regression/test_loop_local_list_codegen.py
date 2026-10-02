# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Code generation for a list that is assigned inside of a loop.

After the loop is unrolled, each iteration leaves a scope with its own stack
allocation for the list. Those allocations must be preallocated and the stack
save and restore operations around them must not be left behind. Otherwise,
neither of these targets can be generated:

- `oqc` emulation, which generates QIR for the base profile. Its verifier
  rejects any call that is not a QIR call, such as `llvm.stacksave`.
- OpenQASM 2, whose emitter does not know how to name a stack allocation.
"""

import cudaq


@cudaq.kernel
def loop_local_angles():
    q = cudaq.qvector(2)
    for i in range(2):
        angles = [0.5, 0.25]
        rx(angles[i], q[i])
    mz(q)


@cudaq.kernel
def top_loop_write_only_list():
    q = cudaq.qvector(2)
    for i in range(2):
        res = [False, False]
        h(q[i])
        res[i] = mz(q[i])


def test_loop_local_list_qir_base_profile():
    # The QIR is verified against the base profile when the kernel is JIT
    # compiled for the emulated target.
    cudaq.set_target("oqc", emulate=True)
    try:
        result = cudaq.sample(loop_local_angles, shots_count=10)
        assert sum(result.values()) == 10
    finally:
        cudaq.reset_target()


def test_loop_local_write_only_list_openqasm2():
    qasm = cudaq.translate(top_loop_write_only_list, format="openqasm2")
    assert "OPENQASM 2.0;" in qasm
    assert "qreg var0[2];" in qasm
    assert "h var0[0];" in qasm
    assert "h var0[1];" in qasm
    assert qasm.count("measure ") == 2
