# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""High-level evolve() entry point for pulse-level time evolution."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..passes.verify import verify as _verify_pass
from ..passes.canonicalize import run_canonicalize as _run_canonicalize
from ..passes.virtual_z import run_virtual_z as _run_virtual_z
from ..passes.fusion import run_fusion as _run_fusion
from ..passes.loop_passes import run_licm as _run_licm
from ..passes.scheduling import schedule_alap as _schedule_alap
from ..passes.to_pulse_mlir import program_to_pulse_mlir as _program_to_pulse_mlir
from .jit import compile_and_run_pulse

_VALID_INTEGRATORS = frozenset({
    "magnus_cf4",
    "lanczos",
    "rk1",
    "rk2",
    "rk4",
    "sse",
})


@dataclass()
class EvolveResult:
    """Result of a pulse-level time evolution."""

    final_state: np.ndarray
    times: np.ndarray
    expectation_values: Optional[Dict[str, np.ndarray]] = None


def evolve(
    program: Any,
    *,
    target: Any,
    t_start: float,
    t_end: float,
    num_steps: int,
    integrator: str = "magnus_cf4",
    observables: Optional[Dict[str, Any]] = None,
) -> EvolveResult:
    """Run a pulse program through the full compilation pipeline and evolve.

    The default path uses the MLIR lowering stack:
      1. verify (linearity, monotone time, drive exclusivity)
      2. canonicalize + virtual-z + fusion + LICM
      3. schedule (ALAP)
      4. program_to_pulse_mlir() (emit pulse dialect MLIR)
      5. pulse-to-qop -> qop-to-cudm -> cudm-to-llvm (MLIR passes)
      6. JIT compile & execute on GPU

    Set CUDAQ_PULSE_LEGACY_PYTHON_PATH=1 to use the legacy Python-only
    lowering path (pulse_to_operator -> emit_cudm_mlir) instead.

    Parameters
    ----------
    program:
        A ``PythonIRBuilder`` (from calling a ``@cudaq_pulse.kernel``) or
        a ``Program`` (from ``to_program()``).
    target:
        A ``Target`` providing the Hamiltonian, decoherence, and connectivity.
    t_start, t_end:
        Time window in nanoseconds.
    num_steps:
        Number of integration time steps.
    integrator:
        Integration strategy (default ``"magnus_cf4"``).
    observables:
        Optional dict mapping names to operator expressions.

    Returns
    -------
    EvolveResult
    """
    if target is None:
        raise ValueError(
            "target is required. Pass a Target to specify the system "
            "Hamiltonian and decoherence model.")
    if integrator not in _VALID_INTEGRATORS:
        raise ValueError(
            f"Unknown integrator {integrator!r}. "
            f"Choose from: {', '.join(sorted(_VALID_INTEGRATORS))}")
    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) must be > t_start ({t_start})")
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")

    use_legacy = os.environ.get("CUDAQ_PULSE_LEGACY_PYTHON_PATH", "") == "1"

    ir_program = _extract_program(program)

    if use_legacy:
        warnings.warn(
            "Using legacy Python-only lowering path. This is deprecated and "
            "will be removed in a future release. Unset "
            "CUDAQ_PULSE_LEGACY_PYTHON_PATH to use the MLIR pipeline.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _evolve_legacy(ir_program,
                              target=target,
                              t_start=t_start,
                              t_end=t_end,
                              num_steps=num_steps,
                              integrator=integrator,
                              observables=observables)

    return _evolve_mlir(ir_program,
                        target=target,
                        t_start=t_start,
                        t_end=t_end,
                        num_steps=num_steps,
                        integrator=integrator,
                        observables=observables)


def _evolve_mlir(
    ir_program: Any,
    *,
    target: Any,
    t_start: float,
    t_end: float,
    num_steps: int,
    integrator: str,
    observables: Optional[Dict[str, Any]],
) -> EvolveResult:
    """MLIR lowering path: pulse -> qop -> cudm -> LLVM -> GPU."""
    from ..passes.ir_types import Program

    if isinstance(ir_program, Program):
        _run_verify_suite(ir_program)
        ir_program = _run_canonicalize(ir_program)
        ir_program = _run_virtual_z(ir_program)
        ir_program = _run_fusion(ir_program)
        ir_program = _run_licm(ir_program)
        _events, _metrics = _schedule_alap(ir_program)

        n_qubits = len(ir_program.qubit_freq_hz) or 1
        pulse_mlir = _program_to_pulse_mlir(ir_program)
    else:
        raise TypeError(f"Expected a Program, got {type(ir_program).__name__}")

    results = compile_and_run_pulse(pulse_mlir, entry="main", n_qubits=n_qubits)

    if not results:
        raise RuntimeError("JIT execution returned no results.")

    times = np.linspace(t_start, t_end, num_steps + 1)
    final_state = results[0].to_numpy()

    return EvolveResult(final_state=final_state, times=times)


def _evolve_legacy(
    ir_program: Any,
    *,
    target: Any,
    t_start: float,
    t_end: float,
    num_steps: int,
    integrator: str,
    observables: Optional[Dict[str, Any]],
) -> EvolveResult:
    """Legacy Python-only lowering path (reference implementation)."""
    from ..passes.ir_types import Program
    from ..passes.pulse_to_operator import run_pulse_to_operator
    from ..passes.to_cudm_mlir import emit_cudm_mlir
    from .jit import compile_and_run

    if isinstance(ir_program, Program):
        _run_verify_suite(ir_program)
        _events, _metrics = _schedule_alap(ir_program)
        operator_ir = run_pulse_to_operator(ir_program, target=target)
        cudm_mlir = emit_cudm_mlir(operator_ir, t_start, t_end, num_steps,
                                   integrator)
    else:
        raise TypeError(f"Expected a Program, got {type(ir_program).__name__}")

    n_qubits = operator_ir.n_qubits
    results = compile_and_run(cudm_mlir, args=[], n_qubits=n_qubits)

    if not results:
        raise RuntimeError("JIT execution returned no results.")

    times = np.linspace(t_start, t_end, num_steps + 1)
    final_state = results[0].to_numpy()

    return EvolveResult(final_state=final_state, times=times)


def _extract_program(program: Any) -> Any:
    """Extract the IR program, dispatching on type."""
    from ..kernel.ir_builder import PythonIRBuilder
    from ..passes.ir_types import Program

    if isinstance(program, PythonIRBuilder):
        return program
    if isinstance(program, Program):
        return program

    emitter = getattr(program, "__cudaq_pulse_emitter__", None)
    if emitter is not None:
        return emitter

    raise TypeError(
        f"Expected a PythonIRBuilder, Program, or compiled @cudaq_pulse.kernel, "
        f"got {type(program).__name__}. Call the kernel to build its IR first.")


def _run_verify_suite(program: Any) -> None:
    """Run the verification pass suite. Raises on failure."""
    from ..passes.ir_types import Program
    if isinstance(program, Program):
        issues = _verify_pass(program)
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            msg = "\n".join(f"  {e}" for e in errors)
            raise RuntimeError(
                f"Verification failed with {len(errors)} error(s):\n{msg}")
