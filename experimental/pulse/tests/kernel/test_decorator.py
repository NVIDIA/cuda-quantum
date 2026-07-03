# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudaq_pulse as pulse


def test_kernel_basic_decoration():

    @pulse.kernel
    def my_kernel(q0):
        pass

    assert hasattr(my_kernel, "__cudaq_pulse_emitter__")


def test_kernel_caching():

    @pulse.kernel
    def my_kernel(q0):
        pass

    q = pulse.qudit_ref()
    my_kernel(q)
    emitter1 = my_kernel.__cudaq_pulse_emitter__
    my_kernel(q)
    emitter2 = my_kernel.__cudaq_pulse_emitter__
    assert emitter1 is not None
    assert emitter1 is emitter2


def test_qudit_ref_creation():
    q = pulse.qudit_ref()
    assert q is not None
    from cudaq_pulse.kernel.decorator import QuditRef
    assert isinstance(q, QuditRef)
    assert hasattr(q, "_vid")


def test_qvec_ref_creation():
    qv = pulse.qvec_ref(4)
    assert len(qv) == 4
    from cudaq_pulse.kernel.decorator import QuditRef
    q0 = qv[0]
    assert isinstance(q0, QuditRef)
    assert hasattr(q0, "_vid")


def test_qvec_ref_indexing():
    qv = pulse.qvec_ref(3)
    for i in range(3):
        assert qv[i] is not None


def test_kernel_invocation():

    @pulse.kernel
    def echo(q0):
        pass

    q = pulse.qudit_ref()
    result = echo(q)
    assert result is not None


def test_kernel_internal_qudit_alloc():
    """qudit_ref() inside the kernel emits a pulse.qudit_alloc op."""
    import cudaq_pulse

    @pulse.kernel
    def internal():
        q = cudaq_pulse.qudit_ref()
        d0, t0 = cudaq_pulse.get_drive_line(q)

    program = internal()
    assert program is not None
    op_kinds = [op.kind for op in program.ops]
    assert "pulse.qudit_alloc" in op_kinds
    assert "pulse.get_drive_line" in op_kinds


def test_kernel_internal_bare_qudit_ref():
    """qudit_ref() as a bare name inside the kernel also works."""

    @pulse.kernel
    def internal():
        q = qudit_ref()
        d0, t0 = get_drive_line(q)

    program = internal()
    op_kinds = [op.kind for op in program.ops]
    assert "pulse.qudit_alloc" in op_kinds
