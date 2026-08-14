# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np

import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program as to_program
from cudaq_pulse.runtime.jit import _check_gpu_available
from cudaq_pulse.targets.base import Qubit, Target


@pulse.kernel
def _jit_test_kernel(q0):
    d0, t0 = get_drive_line(q0)
    wf = gaussian(40, 0.3, 10.0)
    drive(d0, wf, t0)


@pytest.mark.gpu
@pytest.mark.skipif(not _check_gpu_available(), reason="No GPU/cuDensityMat")
def test_jit_executes_target_aware_program():
    """The direct JIT path executes scheduled, target-aware pulse MLIR."""
    ir = _jit_test_kernel(pulse.qudit_ref())
    target = Target(
        name="jit-test",
        qubits={
            0:
                Qubit(index=0,
                      frequency_hz=5.0e9,
                      anharmonicity_hz=-200.0e6,
                      t1_us=0.0,
                      t2_star_us=0.0)
        },
    )
    result = pulse.evolve(ir,
                          target=target,
                          t_start=0.0,
                          t_end=20.0,
                          num_steps=200)
    state = result.final_state
    assert state.shape == (2,)
    assert np.vdot(state, state).real == pytest.approx(1.0, abs=1.0e-8)


def test_jit_import():
    """JIT module should be importable."""
    from cudaq_pulse.runtime import jit

    assert hasattr(jit, "JITCompiler")


def test_evolve_import():
    """Evolve module should be importable."""
    from cudaq_pulse.runtime import evolve

    assert hasattr(evolve, "evolve")


def test_evolve_builds_target_aware_mlir(monkeypatch):
    from cudaq_pulse.runtime import evolve as evolve_module

    target = Target(
        name="test",
        qubits={
            0:
                Qubit(index=0,
                      frequency_hz=5.0e9,
                      anharmonicity_hz=-200.0e6,
                      t1_us=50.0,
                      t2_star_us=30.0)
        },
    )
    captured = {}

    class Result:

        @staticmethod
        def to_numpy():
            return np.eye(2, dtype=np.complex128)

    def fake_run(mlir, *, entry, n_qubits):
        captured["mlir"] = mlir
        captured["n_qubits"] = n_qubits
        return [Result()]

    monkeypatch.setattr(evolve_module, "compile_and_run_pulse", fake_run)
    ir = _jit_test_kernel(pulse.qudit_ref())
    result = pulse.evolve(ir,
                          target=target,
                          t_start=0.0,
                          t_end=20.0,
                          num_steps=100)
    assert result.final_state.shape == (2, 2)
    assert result.times.shape == (101,)
    assert captured["n_qubits"] == 1
    assert "pulse.t1_times" in captured["mlir"]
    assert 'qop.integrator = "rk4"' in captured["mlir"]


@pytest.mark.parametrize("integrator",
                         ["rk1", "rk2", "rk4", "magnus", "crank_nicolson"])
def test_evolve_emits_selected_integrator(monkeypatch, integrator):
    """Every supported integrator name lowers to its dialect attribute."""
    from cudaq_pulse.runtime import evolve as evolve_module

    target = Target(
        name="test",
        qubits={
            0:
                Qubit(index=0,
                      frequency_hz=5.0e9,
                      anharmonicity_hz=-200.0e6,
                      t1_us=50.0,
                      t2_star_us=30.0)
        },
    )
    captured = {}

    class Result:

        @staticmethod
        def to_numpy():
            return np.eye(2, dtype=np.complex128)

    def fake_run(mlir, *, entry, n_qubits):
        captured["mlir"] = mlir
        return [Result()]

    monkeypatch.setattr(evolve_module, "compile_and_run_pulse", fake_run)
    ir = _jit_test_kernel(pulse.qudit_ref())
    pulse.evolve(ir,
                 target=target,
                 t_start=0.0,
                 t_end=20.0,
                 num_steps=100,
                 integrator=integrator)
    assert f'qop.integrator = "{integrator}"' in captured["mlir"]


def test_evolve_rejects_unimplemented_options():
    target = Target(name="test", qubits={})
    with pytest.raises(ValueError, match="Unknown integrator"):
        pulse.evolve(object(),
                     target=target,
                     t_start=0.0,
                     t_end=1.0,
                     num_steps=1,
                     integrator="magnus_cf4")
    with pytest.raises(NotImplementedError, match="observable"):
        pulse.evolve(
            object(),
            target=target,
            t_start=0.0,
            t_end=1.0,
            num_steps=1,
            observables={"z": object()},
        )
