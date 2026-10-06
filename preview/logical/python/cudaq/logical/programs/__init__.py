# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Programs, objectives, selection intents, and their typed references."""

from cudaq.logical._core.lazy import public_dir as _public_dir, resolve as _resolve

_EXPORTS = {
    "Definition": "cudaq.logical.programs.definition:Definition",
    "DefinitionHandle": "cudaq.logical.programs.definition:DefinitionHandle",
    "ProgramDefinition": "cudaq.logical.programs.definition:ProgramDefinition",
    "ObjectiveOperandRef": "cudaq.logical.programs.binding:ObjectiveOperandRef",
    "ObjectiveOperands": "cudaq.logical.programs.binding:ObjectiveOperands",
    "LogicalPortRef": "cudaq.logical.programs.binding:LogicalPortRef",
    "LogicalPorts": "cudaq.logical.programs.binding:LogicalPorts",
    "SelectionIntent": "cudaq.logical.programs.selection:SelectionIntent",
    "require": "cudaq.logical.programs.selection:require",
    "condition_results": "cudaq.logical.programs.selection:condition_results",
    "abort_on": "cudaq.logical.programs.selection:abort_on",
    "program": "cudaq.logical.programs.decorators:program",
    "objective": "cudaq.logical.programs.decorators:objective",
    "UnplacedBuilder": "cudaq.logical.programs.builder:UnplacedBuilder",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return _resolve(globals(), _EXPORTS, name)


def __dir__():
    return _public_dir(globals(), _EXPORTS)
