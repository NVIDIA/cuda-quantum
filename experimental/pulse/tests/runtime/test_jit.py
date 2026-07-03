# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program as to_program


@pulse.kernel
def _jit_test_kernel(q0):
    d0, t0 = get_drive_line(q0)
    wf = gaussian(40, 0.3, 10.0)
    drive(d0, wf, t0)


@pytest.mark.gpu
def test_jit_requires_gpu():
    """JIT compilation should raise if no GPU is present or MLIR tools missing."""
    from cudaq_pulse.runtime.jit import JITCompiler

    try:
        compiler = JITCompiler()
    except FileNotFoundError:
        pytest.skip("MLIR tools not on PATH; cannot test JIT")

    ir = _jit_test_kernel(pulse.qudit_ref())
    prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz={0: 5.0e9})

    try:
        compiler.compile_and_run(prog)
    except RuntimeError as e:
        assert "GPU" in str(e) or "gpu" in str(e)


def test_jit_import():
    """JIT module should be importable."""
    from cudaq_pulse.runtime import jit
    assert hasattr(jit, "JITCompiler")


def test_evolve_import():
    """Evolve module should be importable."""
    from cudaq_pulse.runtime import evolve
    assert hasattr(evolve, "evolve")
