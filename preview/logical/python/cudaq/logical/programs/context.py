# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

from ..errors import NoActiveTrace

_active_trace: ContextVar[Any | None] = ContextVar("qlx_trace", default=None)


def current_trace():
    return _active_trace.get()


def push_trace(trace) -> Token:
    # Compiler-internal dependency materialization may temporarily trace a
    # callee while its caller is being materialized. ContextVar tokens restore
    # the caller exactly; user-visible nested builders still cannot share SSA
    # values because every proxy carries its owner token.
    return _active_trace.set(trace)


def pop_trace(token: Token) -> None:
    _active_trace.reset(token)


def require_trace(operation: str):
    trace = current_trace()
    if trace is None:
        raise NoActiveTrace(
            f"qlx.{operation}() requires an active CUDA-Q Logical trace")
    return trace
