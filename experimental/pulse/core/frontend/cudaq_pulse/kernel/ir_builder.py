# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared IR types and builder for the pulse kernel compiler.

This module contains the core data structures used by the bytecode
compiler and downstream passes: Op, IRValue, Parameter, PythonIRBuilder,
OP_TABLE, LINEAR_TYPES, and CompilationError.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any

Op = namedtuple("Op", ["kind", "operands", "results", "attrs"])


class CompilationError(Exception):
    pass


class Parameter:
    """Sentinel for a symbolic kernel parameter (compile-once, evaluate-many).

    Instances track an index and type so the packed IR builder can emit
    a PARAM opcode instead of a concrete value.
    """
    __slots__ = ("name", "index", "dtype")

    def __init__(self, name: str, index: int, dtype: str = "f64"):
        self.name = name
        self.index = index
        self.dtype = dtype

    def __repr__(self) -> str:
        return f"Parameter({self.name!r}, idx={self.index}, {self.dtype})"

    # Arithmetic on Parameters is disallowed -- force direct parameter usage
    def _unsupported(self, op_name: str) -> None:
        raise CompilationError(
            f"Cannot perform '{op_name}' on Parameter({self.name!r}). "
            f"Parameters must be passed directly to pulse ops.")

    def __add__(self, other: Any) -> Any:
        self._unsupported("+")

    def __radd__(self, other: Any) -> Any:
        self._unsupported("+")

    def __sub__(self, other: Any) -> Any:
        self._unsupported("-")

    def __rsub__(self, other: Any) -> Any:
        self._unsupported("-")

    def __mul__(self, other: Any) -> Any:
        self._unsupported("*")

    def __rmul__(self, other: Any) -> Any:
        self._unsupported("*")

    def __truediv__(self, other: Any) -> Any:
        self._unsupported("/")

    def __neg__(self) -> Any:
        self._unsupported("neg")

    def __float__(self) -> float:
        self._unsupported("float()")
        return 0.0  # unreachable

    def __int__(self) -> int:
        self._unsupported("int()")
        return 0  # unreachable


class IRValue:
    __slots__ = ("vid", "vtype", "name")

    def __init__(self, vid: int, vtype: str, name: str = ""):
        self.vid = vid
        self.vtype = vtype
        self.name = name

    def __repr__(self) -> str:
        return f"{self.name or f'%v{self.vid}'}:{self.vtype}"


LINEAR_TYPES = frozenset({"drive_line", "readout_line", "tone"})

# (n_value_args, attr_names, result_types)
#   n_value_args: leading args that are IR values; -1 = variadic all-values
#   result_types: None = mirror operand types
OP_TABLE: dict[str, tuple[int, tuple[str, ...], tuple[str, ...] | None]] = {
    "get_drive_line": (1, (), ("drive_line", "tone")),
    "get_readout_line": (1, (), ("readout_line", "tone")),
    "gaussian": (0, ("duration", "amplitude", "sigma"), ("waveform",)),
    "square": (0, ("duration", "amplitude"), ("waveform",)),
    "drag": (0, ("duration", "amplitude", "sigma", "beta"), ("waveform",)),
    "cosine": (0, ("duration", "amplitude", "frequency"), ("waveform",)),
    "tanh_ramp": (0, ("duration", "amplitude", "sigma"), ("waveform",)),
    "gaussian_square":
        (0, ("duration", "amplitude", "sigma", "width"), ("waveform",)),
    "custom": (0, ("duration", "name"), ("waveform",)),
    "custom_samples": (0, ("samples",), ("waveform",)),
    "drive": (3, (), ("drive_line", "tone")),
    "readout": (3, (), ("readout_line", "tone", "measurement")),
    "wait": (1, ("duration",), None),
    "sync": (-1, (), None),
    "shift_phase": (1, ("phase_rad",), ("tone",)),
    "set_phase": (1, ("phase_rad",), ("tone",)),
    "shift_frequency": (1, ("freq_hz",), ("tone",)),
    "set_frequency": (1, ("freq_hz",), ("tone",)),
    "wf_add": (2, (), ("waveform",)),
    "wf_sub": (2, (), ("waveform",)),
    "wf_mul": (2, (), ("waveform",)),
    "wf_scale": (1, ("scale",), ("waveform",)),
    "wf_neg": (1, (), ("waveform",)),
    "qudit_ref": (0, (), ("qref",)),
    "qvec_ref": (0, ("size",), ("qref",)),
}


class PythonIRBuilder:
    """Lightweight in-memory IR builder (swap for real MLIR bindings)."""

    def __init__(self, name: str = "main"):
        self.name = name
        self.ops: list[Op] = []
        self._next_id = 0

    def _mk(self, vtype: str, name: str = "") -> IRValue:
        v = IRValue(self._next_id, vtype, name)
        self._next_id += 1
        return v

    def emit(
        self,
        kind: str,
        operands: tuple[IRValue, ...] = (),
        result_types: tuple[str, ...] = (),
        attrs: dict[str, Any] | None = None,
    ) -> tuple[IRValue, ...]:
        results = tuple(self._mk(rt) for rt in result_types)
        self.ops.append(Op(kind, operands, results, attrs or {}))
        return results

    def pretty(self) -> str:
        lines = [f"func.func @{self.name}() {{"]
        for op in self.ops:
            res = ", ".join(repr(r) for r in op.results)
            ops_s = ", ".join(repr(o) for o in op.operands)
            att = ", ".join(f"{k}={v!r}" for k, v in op.attrs.items())
            parts = [s for s in (ops_s, att) if s]
            lines.append(f"  {res} = {op.kind}({', '.join(parts)})")
        lines.append("}")
        return "\n".join(lines)
