# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Version-isolated bytecode normalizer.

This is the ONLY file that contains Python-version-specific bytecode
knowledge. All downstream modules operate on CanonicalInstr sequences
and never see raw dis.Instruction objects.

Adding support for a new CPython version: add an ``_OPNAME_MAP_3XX``
dict and a branch in ``_select_map``.
"""

from __future__ import annotations

import dis
import operator
import sys
from dataclasses import dataclass
from types import CodeType
from typing import Any

_PY_MAJOR, _PY_MINOR = sys.version_info[:2]

# ── Canonical instruction ────────────────────────────────────────────


@dataclass(frozen=True)
class CanonicalInstr:
    op: str
    arg: Any
    offset: int
    lineno: int


# ── Opcodes to silently skip (CPython internals, no semantic meaning) ─

_SKIP_OPS = frozenset({
    "RESUME",
    "CACHE",
    "NOP",
    "PUSH_NULL",
    "COPY",
    "PRECALL",
    "END_FOR",
    "TO_BOOL",
    "COPY_FREE_VARS",
    "EXTENDED_ARG",
})

# ── Binary op sub-codes (3.11+ encode op kind in BINARY_OP arg) ─────

_NB_OP_TO_STR: dict[int, str] = {
    0: "+",
    1: "&",
    2: "//",
    3: "<<",
    5: "*",
    6: "%",
    7: "|",
    8: "**",
    9: ">>",
    10: "-",
    11: "/",
    12: "^",
    13: "+=",
    14: "&=",
    15: "//=",
    16: "<<=",
    17: "@=",
    18: "*=",
    19: "%=",
    20: "|=",
    21: "**=",
    22: ">>=",
    23: "-=",
    24: "/=",
    25: "^=",
}

# ── Per-version opname canonicalization maps ─────────────────────────

_OPNAME_MAP_39: dict[str, str] = {
    "CALL_FUNCTION": "CALL",
    "CALL_METHOD": "CALL",
    "LOAD_METHOD": "LOAD_ATTR",
    "BINARY_ADD": "BINARY_OP",
    "BINARY_SUBTRACT": "BINARY_OP",
    "BINARY_MULTIPLY": "BINARY_OP",
    "BINARY_TRUE_DIVIDE": "BINARY_OP",
    "BINARY_FLOOR_DIVIDE": "BINARY_OP",
    "BINARY_MODULO": "BINARY_OP",
    "BINARY_POWER": "BINARY_OP",
    "UNARY_NEGATIVE": "UNARY_NEGATIVE",
    "POP_JUMP_IF_FALSE": "JUMP_IF_FALSE",
    "POP_JUMP_IF_TRUE": "JUMP_IF_TRUE",
    "JUMP_ABSOLUTE": "JUMP",
    "JUMP_FORWARD": "JUMP",
    "RETURN_VALUE": "RETURN",
    "IMPORT_NAME": "IMPORT_NAME",
}

_BINARY_NAME_TO_STR_39: dict[str, str] = {
    "BINARY_ADD": "+",
    "BINARY_SUBTRACT": "-",
    "BINARY_MULTIPLY": "*",
    "BINARY_TRUE_DIVIDE": "/",
    "BINARY_FLOOR_DIVIDE": "//",
    "BINARY_MODULO": "%",
    "BINARY_POWER": "**",
}

_OPNAME_MAP_312: dict[str, str] = {
    "CALL": "CALL",
    "BINARY_OP": "BINARY_OP",
    "POP_JUMP_IF_FALSE": "JUMP_IF_FALSE",
    "POP_JUMP_IF_TRUE": "JUMP_IF_TRUE",
    "POP_JUMP_FORWARD_IF_FALSE": "JUMP_IF_FALSE",
    "POP_JUMP_FORWARD_IF_TRUE": "JUMP_IF_TRUE",
    "JUMP_FORWARD": "JUMP",
    "JUMP_BACKWARD": "JUMP",
    "RETURN_VALUE": "RETURN",
    "RETURN_CONST": "RETURN",
    "IMPORT_NAME": "IMPORT_NAME",
}


def _select_map(major: int, minor: int) -> dict[str, str]:
    if major != 3:
        raise NotImplementedError(f"Python {major}.{minor} is not supported")
    if minor in (9, 10):
        return _OPNAME_MAP_39
    if minor in (11, 12, 13, 14):
        return _OPNAME_MAP_312
    raise NotImplementedError(
        f"Python {major}.{minor} is not yet supported by the bytecode bridge. "
        f"Supported: 3.9, 3.10, 3.12 (primary). Add an opname map to "
        f"_bytecode_normalize.py to add support.")


# ── Main normalize function ──────────────────────────────────────────


def normalize(code: CodeType) -> list[CanonicalInstr]:
    """Convert a code object to a version-agnostic canonical instruction stream."""
    raw = list(dis.get_instructions(code))
    opname_map = _select_map(_PY_MAJOR, _PY_MINOR)
    is_39 = _PY_MINOR in (9, 10)

    result: list[CanonicalInstr] = []
    prev_line = 0

    for instr in raw:
        if instr.opname in _SKIP_OPS:
            continue

        sl = getattr(instr, "line_number", None) or getattr(
            instr, "starts_line", None)
        lineno = sl if sl is not None else prev_line
        if sl is not None:
            prev_line = sl

        canonical_op = opname_map.get(instr.opname, instr.opname)
        arg = instr.argval

        if canonical_op == "BINARY_OP":
            if is_39:
                arg = _BINARY_NAME_TO_STR_39.get(instr.opname, instr.opname)
            elif isinstance(instr.argval, int):
                arg = _NB_OP_TO_STR.get(instr.argval, str(instr.argval))
            # else argval is already a string like "+" on some versions

        if canonical_op == "RETURN" and instr.opname == "RETURN_CONST":
            arg = instr.argval

        if canonical_op == "JUMP":
            arg = instr.argval  # target offset (already resolved by dis)

        result.append(
            CanonicalInstr(
                op=canonical_op,
                arg=arg,
                offset=instr.offset,
                lineno=lineno,
            ))

    return result
