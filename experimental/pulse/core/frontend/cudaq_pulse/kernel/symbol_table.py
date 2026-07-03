# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Optional

LINEAR_TYPES = frozenset({"drive_line", "readout_line", "tone"})
VALUE_TYPES = frozenset({"waveform", "duration", "measurement", "iq_data"})


class SymbolTable:
    """Manages name -> mlir.Value mappings with lexical scoping.

    Linear-typed values (drive_line, readout_line, tone) are automatically
    rebound when an op produces a new SSA value of the same type: the
    *source operand's* name is located and updated in-place.

    Value-typed results (waveform, duration, measurement, iq_data) must be
    captured via explicit ``=`` assignment in the source program.
    """

    def __init__(self) -> None:
        self._scopes: List[Dict[str, Any]] = [{}]

    def push_scope(self) -> None:
        self._scopes.append({})

    def pop_scope(self) -> Dict[str, Any]:
        if len(self._scopes) <= 1:
            raise RuntimeError("Cannot pop the global scope")
        return self._scopes.pop()

    def bind(self, name: str, value: Any) -> None:
        self._scopes[-1][name] = value

    def lookup(self, name: str) -> Optional[Any]:
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return None

    def rebind_linear(self, operand_value: Any, new_value: Any) -> None:
        """Find the name currently bound to *operand_value* and rebind it
        to *new_value*.  Used for ops that consume and re-produce a
        linear-typed SSA value (e.g. ``drive`` consumes a drive_line and
        yields an updated one)."""
        for scope in reversed(self._scopes):
            for name, val in scope.items():
                if val is operand_value:
                    scope[name] = new_value
                    return
        raise KeyError("operand value not found in any scope")
