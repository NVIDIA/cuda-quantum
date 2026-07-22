# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cudaq-pulse compiler passes (experimental pass-authoring API).

Provides verification, scheduling, canonicalization, and optimization passes
that operate on the lightweight ``Program``/``Op`` IR, plus the
``ProgramBuilder`` used to construct programs directly.

Each pass is a plain ``Program -> Program`` (or ``Program -> (events, metrics)``
for schedulers) function, so you can compose the built-ins or write your own
transform and apply it, then emit MLIR with ``program_to_pulse_mlir``. For the
common "compile a kernel end-to-end" path, use ``cudaq_pulse.compile()``.

This surface is experimental and may change without notice.
"""

from .ir_types import Op, OpKind, Program, Value, ValueType, _mk, clone_program, duration_of, is_loop_or_barrier
from .scheduling import ScheduledEvent, ScheduleMetrics, MachineModel
from ._builder import ProgramBuilder

from .verify import verify
from .verify import (
    check_linearity,
    check_monotone_time,
    check_drive_exclusivity,
    check_loop_structure,
    check_waveform_validity,
)
from .scheduling import (
    schedule_asap,
    schedule_alap,
    schedule_rcp,
)
from .canonicalize import run_canonicalize
from .virtual_z import run_virtual_z
from .fusion import run_fusion
from .loop_passes import run_licm, run_loop_strength_reduction
from .pulse_to_operator import run_pulse_to_operator, OperatorProgram
from .to_cudm_mlir import emit_cudm_mlir
from .to_pulse_mlir import program_to_pulse_mlir

__all__ = [
    "Op",
    "OpKind",
    "Program",
    "Value",
    "ValueType",
    "_mk",
    "clone_program",
    "duration_of",
    "is_loop_or_barrier",
    "ScheduledEvent",
    "ScheduleMetrics",
    "MachineModel",
    "ProgramBuilder",
    "OperatorProgram",
    "verify",
    "check_linearity",
    "check_monotone_time",
    "check_drive_exclusivity",
    "check_loop_structure",
    "check_waveform_validity",
    "schedule_asap",
    "schedule_alap",
    "schedule_rcp",
    "run_canonicalize",
    "run_virtual_z",
    "run_fusion",
    "run_licm",
    "run_loop_strength_reduction",
    "run_pulse_to_operator",
    "program_to_pulse_mlir",
    "emit_cudm_mlir",
]
