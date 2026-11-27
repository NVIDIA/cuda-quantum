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

from cudaq.logical.programs.definition import ProgramDefinition


def _definition_decorator(
    fn: Callable[..., Any] | None,
    *,
    kind: str,
    machine: Any = None,
    selection: Any = None,
    action_kind: str = "auto",
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
            action_kind=action_kind,
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


def action(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    kind: str = "auto",
):
    """Declare the ideal logical circuit implemented by a gadget.

    An action can include preparation and measurement. ``kind="unitary"``
    requires a body of unitary actions; ``kind="instrument"``
    permits classical outcomes and changes to the quantum boundary. The
    default infers the kind. ``kind="action"`` is the legacy spelling of
    ``"unitary"``.
    """

    if kind not in {"auto", "unitary", "action", "instrument"}:
        raise ValueError("action kind must be auto, unitary, or instrument")
    frame = currentframe()
    localns = dict(frame.f_back.f_locals) if frame and frame.f_back else {}
    return _definition_decorator(
        fn,
        kind="action",
        action_kind=kind,
        name=name,
        localns=localns,
    )


# An alias, rather than a wrapper, preserves construction-time local type
# annotations by keeping the same caller frame as the canonical decorator.
objective = action
