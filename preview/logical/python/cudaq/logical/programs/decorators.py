# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from inspect import currentframe
from typing import Any, Callable, get_type_hints

from ..programs.definition import ProgramDefinition


def _definition_decorator(
    fn: Callable[..., Any] | None,
    *,
    kind: str,
    machine: Any = None,
    selection: Any = None,
    objective_kind: str = "auto",
    estimate_only: bool = False,
    name: str | None = None,
    localns=None,
):

    def wrap(provider: Callable[..., Any]) -> ProgramDefinition:
        hints = get_type_hints(
            provider,
            globalns=provider.__globals__,
            localns={} if localns is None else localns,
        )
        return ProgramDefinition(
            provider,
            machine=machine,
            selection=selection,
            kind=kind,
            objective_kind=objective_kind,
            estimate_only=estimate_only,
            name=name,
            type_hints=hints,
        )

    return wrap(fn) if fn is not None else wrap


def program(
    fn: Callable[..., Any] | None = None,
    *,
    machine: Any = None,
    selection: Any = None,
    name: str | None = None,
    estimate_only: bool = False,
):
    frame = currentframe()
    localns = dict(frame.f_back.f_locals) if frame and frame.f_back else {}
    return _definition_decorator(
        fn,
        kind="program",
        machine=machine,
        selection=selection,
        estimate_only=estimate_only,
        name=name,
        localns=localns,
    )


def objective(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    kind: str = "auto",
):
    if kind not in {"auto", "action", "instrument"}:
        raise ValueError("objective kind must be auto, action, or instrument")
    frame = currentframe()
    localns = dict(frame.f_back.f_locals) if frame and frame.f_back else {}
    return _definition_decorator(
        fn,
        kind="objective",
        objective_kind=kind,
        name=name,
        localns=localns,
    )
