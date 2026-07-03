# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bytecode-based kernel capture.

Uses ``fn.__code__`` directly instead of ``inspect.getsource`` + ``ast.parse``.
All version-specific bytecode handling is in ``_bytecode_normalize.py``.
"""

from __future__ import annotations

from typing import Any, Callable

from .ir_builder import CompilationError, IRValue, PythonIRBuilder
from .decorator import QuditRef, QvecRef
from ._bytecode_normalize import normalize
from ._structure import recover_structure
from ._emitter import PulseIREmitter


def compile_kernel_bytecode(fn: Callable) -> Callable[..., PythonIRBuilder]:
    """Compile a pulse kernel function via bytecode analysis.

    Returns a callable that, given concrete arguments, produces a
    populated ``PythonIRBuilder``.
    """
    code = fn.__code__
    params = list(code.co_varnames[:code.co_argcount])

    instrs = normalize(code)
    regions = recover_structure(instrs)

    fn_globals = getattr(fn, "__globals__", {})
    fn_closures = _extract_closures(fn)

    def _emit(*args: Any, **kwargs: Any) -> PythonIRBuilder:
        if kwargs:
            raise CompilationError(
                "keyword arguments not supported in pulse kernels")
        if len(args) != len(params):
            raise CompilationError(
                f"{fn.__name__}: expected {len(params)} args, got {len(args)}")

        builder = PythonIRBuilder(name=fn.__name__)
        emitter = PulseIREmitter(builder,
                                 fn_globals=fn_globals,
                                 fn_closures=fn_closures)

        for name, val in zip(params, args):
            if isinstance(val, QuditRef):
                (ir_val,) = builder.emit("pulse.qudit_arg", (), ("qref",))
                emitter.locals[name] = ir_val
            elif isinstance(val, QvecRef):
                ir_vals = []
                for i in range(len(val)):
                    (v,) = builder.emit("pulse.qudit_arg", (), ("qref",),
                                        {"index": i})
                    ir_vals.append(v)
                emitter.locals[name] = ir_vals
            else:
                emitter.locals[name] = val

        emitter.emit_regions(regions)
        return builder

    return _emit


def _trace_kernel_with_builder(fn: Callable, builder, args) -> None:
    """Re-trace *fn* using *builder* (any object with an ``emit()`` method).

    This allows ``compile()`` to inject an ``MLIRIRBuilder`` that writes
    directly to an in-memory MLIR module instead of a ``PythonIRBuilder``.
    """
    code = fn.__code__
    params = list(code.co_varnames[:code.co_argcount])

    instrs = normalize(code)
    regions = recover_structure(instrs)

    fn_globals = getattr(fn, "__globals__", {})
    fn_closures = _extract_closures(fn)

    if len(args) != len(params):
        raise CompilationError(
            f"{fn.__name__}: expected {len(params)} args, got {len(args)}")

    emitter = PulseIREmitter(builder,
                             fn_globals=fn_globals,
                             fn_closures=fn_closures)

    for name, val in zip(params, args):
        if isinstance(val, QuditRef):
            (ir_val,) = builder.emit("pulse.qudit_alloc", (), ("qref",))
            emitter.locals[name] = ir_val
        elif isinstance(val, QvecRef):
            ir_vals = []
            for i in range(len(val)):
                (v,) = builder.emit("pulse.qudit_alloc", (), ("qref",),
                                    {"index": i})
                ir_vals.append(v)
            emitter.locals[name] = ir_vals
        else:
            emitter.locals[name] = val

    emitter.emit_regions(regions)


def _extract_closures(fn: Callable) -> dict[str, Any]:
    """Extract closure variable values from the function."""
    closures: dict[str, Any] = {}
    code = fn.__code__
    if fn.__closure__ and code.co_freevars:
        for name, cell in zip(code.co_freevars, fn.__closure__):
            try:
                closures[name] = cell.cell_contents
            except ValueError:
                pass
    return closures
