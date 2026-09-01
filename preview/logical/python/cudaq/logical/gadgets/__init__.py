# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Gadget definitions, typed records, verification, and reusable factories."""

from .._core.lazy import public_dir as _public_dir, resolve as _resolve

_INTERFACE_NAMES = (
    "BlockEndpoint",
    "BlockFlow",
    "EndpointCollection",
    "GadgetInterface",
    "patch",
)
_RECORD_NAMES = (
    "GadgetRecords",
    "RecordFamily",
    "RecordParity",
    "RecordRef",
    "RecordVectorParity",
    "StructuredRecord",
)
_SEMANTIC_NAMES = (
    "CommitPoint",
    "CommitPointKind",
    "OutcomeMap",
    "OutcomeRole",
    "OutcomeSyndromeTerm",
    "ParameterMap",
    "RetryExhaustion",
    "RetryPolicy",
    "before_output",
    "before_resource_output",
)
_SPECIFICATION_NAMES = ("GadgetSpec", "Port")
_DEFINITION_NAMES = ("GadgetDefinition", "gadget")
_FACTORY_NAMES = (
    "css_memory_round",
    "logical_measure",
    "logical_pauli",
    "measure_x",
    "measure_z",
    "prepare_plus",
    "prepare_zero",
    "stabilizer_preparation",
)

_EXPORTS = {
    **{
        name: f".interface:{name}" for name in _INTERFACE_NAMES
    },
    **{
        name: f".records:{name}" for name in _RECORD_NAMES
    },
    **{
        name: f".semantics:{name}" for name in _SEMANTIC_NAMES
    },
    **{
        name: f".specification:{name}" for name in _SPECIFICATION_NAMES
    },
    **{
        name: f".definition:{name}" for name in _DEFINITION_NAMES
    },
    **{
        name: f".factories:{name}" for name in _FACTORY_NAMES
    },
    "GadgetBuilder": ".builder:GadgetBuilder",
    "clifford_action": ".analysis:clifford_action",
}
_PRIVATE_EXPORTS = {
    "InputSyndromeRef":
        ".records:InputSyndromeRef",
    "ProfileParity":
        ".records:ProfileParity",
    "_PredicateProvenance":
        ".semantics:_PredicateProvenance",
    "_resolve_gadget_code_profile":
        ".records:_resolve_gadget_code_profile",
    "_stabilizer_preparation_circuit":
        ".factories:_stabilizer_preparation_circuit",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        return _resolve(globals(), _EXPORTS, name)
    except AttributeError:
        return _resolve(globals(), _PRIVATE_EXPORTS, name)


def __dir__():
    return _public_dir(globals(), _EXPORTS)
