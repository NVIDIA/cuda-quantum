# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Public Stim-family emission facade."""

from __future__ import annotations

from dataclasses import dataclass

from .stim_common import (
    CompiledInterfaceManifest,
    _symbol,
)
from .stim_logical import _Emitter


@dataclass(frozen=True, slots=True)
class StimEmission:
    """Stim text plus the typed interface of that exact circuit projection."""

    text: str
    interface: CompiledInterfaceManifest | None
    argument_data: tuple[tuple[int, ...], ...] = ()


def emit_stim_artifact(
    module,
    *,
    root_symbol,
):
    """Project one folded CUDA-Q Logical graph to Stim text and typed boundary metadata."""

    try:
        verified = module.operation.verify()
    except Exception as exc:
        raise ValueError(
            "Stim emission requires a module that passes native verification"
        ) from exc
    if not verified:
        raise ValueError(
            "Stim emission requires a module that passes native verification")
    symbols = {
        _symbol(view.operation): view.operation
        for view in module.body.operations
        if _symbol(view.operation) is not None
    }
    root = symbols.get(root_symbol)
    if root is None or root.name != "fabric.gadget":
        raise ValueError("Stim emission requires a legalized P2 entry gadget")
    instance = _Emitter(module)
    text = instance.emit(root_symbol)
    return StimEmission(
        text,
        getattr(instance, "interface_manifest", None),
        getattr(instance, "input_port_data", ()),
    )


def emit_stim(
    module,
    *,
    root_symbol,
    return_boundary: bool = False,
):
    """Emit Stim-family text from a folded CUDA-Q Logical Fabric call graph.

    With ``return_boundary=True`` the result is ``(text, input_port_data)``,
    where ``input_port_data`` lists the data-qubit indices of each patch-typed
    input port in interface order — the emitter's actual carrier layout, which
    analysis backends need to place per-port input states.
    """
    emission = emit_stim_artifact(
        module,
        root_symbol=root_symbol,
    )
    if return_boundary:
        return emission.text, emission.argument_data
    return emission.text


__all__ = [
    "StimEmission",
    "emit_stim",
    "emit_stim_artifact",
]
