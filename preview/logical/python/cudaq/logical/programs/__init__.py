# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Programs, objectives, selection intents, and their typed references."""

from .._core.lazy import public_dir as _public_dir, resolve as _resolve

_EXPORTS = {
    "Definition": ".definition:Definition",
    "DefinitionHandle": ".definition:DefinitionHandle",
    "ProgramDefinition": ".definition:ProgramDefinition",
    "ObjectiveOperandRef": ".binding:ObjectiveOperandRef",
    "ObjectiveOperands": ".binding:ObjectiveOperands",
    "LogicalPortRef": ".binding:LogicalPortRef",
    "LogicalPorts": ".binding:LogicalPorts",
    "SelectionIntent": ".selection:SelectionIntent",
    "require": ".selection:require",
    "condition_results": ".selection:condition_results",
    "abort_on": ".selection:abort_on",
    "program": ".decorators:program",
    "objective": ".decorators:objective",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return _resolve(globals(), _EXPORTS, name)


def __dir__():
    return _public_dir(globals(), _EXPORTS)
