# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bridge between Python Program IR and C++ PulseModuleBuilder.

Provides ``emit_pulse_module_packed`` which delegates to ``packed_emit.py``
for zero-copy Program → MLIR module construction via the packed-buffer path.
"""
from __future__ import annotations

from typing import Any

from .passes.ir_types import Program


def emit_pulse_module_packed(prog: Program) -> Any:
    """Build an in-memory PulseModule via the packed-buffer zero-copy path.

    Encodes the entire ``Program`` as a flat ``numpy.ndarray[int64]``
    and sends it to C++ in a single FFI call — zero per-op overhead.

    Returns a ``PulseModule`` (from ``_cudaq_pulse_native``).
    Raises ImportError if native bindings are not available.
    """
    from .packed_emit import emit_pulse_module_packed as _impl
    return _impl(prog)
