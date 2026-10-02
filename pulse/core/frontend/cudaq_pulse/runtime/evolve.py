# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""High-level evolve() entry point for pulse-level time evolution."""

from __future__ import annotations

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
    "rk1",
    "rk2",
    "rk4",
    "magnus",
    "crank_nicolson",
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
    integrator: str = "rk4",
    clock_ghz: float = 2.0,
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
        Time-integration strategy for the cuDensityMat runtime. One of the
        explicit Runge-Kutta schemes ``"rk1"``, ``"rk2"``, ``"rk4"``, the
        ``"magnus"`` Taylor-series midpoint expansion, or the
        ``"crank_nicolson"`` predictor-corrector method.
    clock_ghz:
        Pulse virtual-clock rate in GHz when converting a traced kernel IR.
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
    if getattr(target, "architecture", None) != "transmon":
        raise NotImplementedError(
            "GPU evolution currently supports two-level transmon targets "
            "only; neutral-atom and multilevel target lowering is not yet "
            "implemented")
    if integrator not in _VALID_INTEGRATORS:
        raise ValueError(
            f"Unknown integrator {integrator!r}. "
            f"Choose from: {', '.join(sorted(_VALID_INTEGRATORS))}")
    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) must be > t_start ({t_start})")
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if clock_ghz <= 0:
        raise ValueError(f"clock_ghz must be positive, got {clock_ghz}")
    if observables:
        raise NotImplementedError(
            "observable evaluation is not implemented in the research "
            "preview; evolve the state and evaluate observables explicitly")

    ir_program = _extract_program(program, target=target, clock_ghz=clock_ghz)

    return _evolve_mlir(
        ir_program,
        target=target,
        t_start=t_start,
        t_end=t_end,
        num_steps=num_steps,
        integrator=integrator,
        observables=observables,
    )


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

        n_qubits = max(ir_program.qubit_freq_hz, default=0) + 1
        pulse_mlir = _program_to_pulse_mlir(
            ir_program,
            target=target,
            t_start=t_start,
            t_end=t_end,
            num_steps=num_steps,
            integrator=integrator,
        )
    else:
        raise TypeError(f"Expected a Program, got {type(ir_program).__name__}")

    results = compile_and_run_pulse(pulse_mlir, entry="main", n_qubits=n_qubits)

    if not results:
        raise RuntimeError("JIT execution returned no results.")

    times = np.linspace(t_start, t_end, num_steps + 1)
    final_state = results[0].to_numpy()

    return EvolveResult(final_state=final_state, times=times)


def _extract_program(program: Any, *, target: Any, clock_ghz: float) -> Any:
    """Extract the IR program, dispatching on type."""
    from ..kernel.ir_builder import PythonIRBuilder
    from ..lower import _to_program
    from ..passes.ir_types import Program

    if isinstance(program, PythonIRBuilder):
        return _to_program(program,
                           clock_ghz=clock_ghz,
                           qubit_freq_hz=target.frequencies)
    if isinstance(program, Program):
        return program

    emitter = getattr(program, "__cudaq_pulse_emitter__", None)
    if emitter is not None:
        if not isinstance(emitter, PythonIRBuilder):
            raise TypeError("compiled kernel emitter is not a PythonIRBuilder")
        return _to_program(emitter,
                           clock_ghz=clock_ghz,
                           qubit_freq_hz=target.frequencies)

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
