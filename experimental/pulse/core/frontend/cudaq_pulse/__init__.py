# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cudaq-pulse: pulse-level quantum programming on MLIR.

Usage::

    import cudaq_pulse as pulse

    @pulse.kernel
    def rabi(qubit):
        drive_line, tone = get_drive_line(qubit)
        drive(drive_line, gaussian(64, 0.5, 16.0), tone)

    compiled_kernel = pulse.compile(rabi, [pulse.qudit_ref()],
                                    qubit_freq_hz={0: 5.0e9})

Importing this module injects the kernel DSL vocabulary (``drive``,
``gaussian``, ``get_drive_line``, etc.) into the caller's namespace so
they can be used as bare names inside ``@pulse.kernel`` functions.
"""

from __future__ import annotations

import sys as _sys

__version__ = "0.1.0"

# ── Infrastructure (accessed via pulse.*) ────────────────────────────────

from .kernel import kernel as kernel
from .kernel.decorator import qudit_ref as qudit_ref
from .kernel.decorator import qvec_ref as qvec_ref
from .kernel.decorator import QuditRef as QuditRef
from .kernel.decorator import QvecRef as QvecRef
from .compile import compile as compile
from .compile import CompiledKernel as CompiledKernel
from .compile import CompileMetrics as CompileMetrics
from .kernel.ir_builder import Parameter as Parameter

# ── Kernel DSL ops (injected as bare names into importer's namespace) ────

from .ops import get_drive_line as get_drive_line
from .ops import get_readout_line as get_readout_line
from .ops import drive as drive
from .ops import readout as readout
from .ops import wait as wait
from .ops import sync as sync
from .ops import shift_phase as shift_phase
from .ops import set_phase as set_phase
from .ops import shift_frequency as shift_frequency
from .ops import set_frequency as set_frequency
from .ops import gaussian as gaussian
from .ops import square as square
from .ops import drag as drag
from .ops import cosine as cosine
from .ops import tanh_ramp as tanh_ramp
from .ops import gaussian_square as gaussian_square
from .ops import custom as custom
from .ops import custom_samples as custom_samples
from .ops import wf_add as wf_add
from .ops import wf_sub as wf_sub
from .ops import wf_mul as wf_mul
from .ops import wf_scale as wf_scale
from .ops import wf_neg as wf_neg

# DSL names that get injected into the importing module's globals.
_DSL_EXPORTS: list[str] = [
    "get_drive_line",
    "get_readout_line",
    "drive",
    "readout",
    "wait",
    "sync",
    "shift_phase",
    "set_phase",
    "shift_frequency",
    "set_frequency",
    "gaussian",
    "square",
    "drag",
    "cosine",
    "tanh_ramp",
    "gaussian_square",
    "custom",
    "custom_samples",
    "wf_add",
    "wf_sub",
    "wf_mul",
    "wf_scale",
    "wf_neg",
]

__all__: list[str] = [
    # infrastructure
    "kernel",
    "compile",
    "CompiledKernel",
    "CompileMetrics",
    "Parameter",
    "qudit_ref",
    "qvec_ref",
    "QuditRef",
    "QvecRef",
    # kernel DSL ops
    *_DSL_EXPORTS,
]


def _inject_dsl() -> None:
    """Inject kernel DSL ops into the importing module's namespace."""
    caller = _sys._getframe(1)
    if caller is not None:
        caller_globals = caller.f_globals
        our_globals = globals()
        for name in _DSL_EXPORTS:
            caller_globals.setdefault(name, our_globals[name])


_inject_dsl()
