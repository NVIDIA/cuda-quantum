# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Gadget definitions, contracts, and reusable factories."""

from cudaq.logical._core.lazy import public_dir as _public_dir, resolve as _resolve

_INTERFACE_NAMES = (
    "BlockEndpoint",
    "BlockFlow",
    "EndpointCollection",
    "GadgetInterface",
    "patch",
)
_RECORD_NAMES = (
    "GadgetRecords",
    "InputSyndromeRef",
    "ProfileBinding",
    "ProfileParity",
    "ProfileVectorExpr",
    "RecordFamily",
    "RecordParity",
    "RecordRef",
    "RecordVectorParity",
    "StructuredRecord",
    "SyndromeBundleRef",
)
_SEMANTIC_NAMES = (
    "CommitPoint",
    "CommitPointKind",
    "Detector",
    "LogicalObservable",
    "ProfileSemanticError",
    "OutcomeRole",
    "OutcomeMap",
    "OutcomeSyndromeTerm",
    "OutputSyndromeAssignment",
    "ParameterMap",
    "RetryExhaustion",
    "RetryPolicy",
    "SuccessPredicate",
    "before_output",
    "before_resource_output",
)
_SPECIFICATION_NAMES = ("GadgetSpec", "Port")
_CONTRACT_NAMES = ("GadgetContract", "GadgetContractBuilder")
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

_VERIFICATION_NAMES = (
    "ContractVerification",
    "GadgetVerificationReceipt",
    "LogicalActionVerification",
    "VerificationFailure",
    "verify_gadget",
)

_EXPORTS = {
    **{
        name: f"cudaq.logical.gadgets.interface:{name}" for name in _INTERFACE_NAMES
    },
    **{
        name: f"cudaq.logical.gadgets.records:{name}" for name in _RECORD_NAMES
    },
    **{
        name: f"cudaq.logical.gadgets.semantics:{name}" for name in _SEMANTIC_NAMES
    },
    **{
        name: f"cudaq.logical.gadgets.specification:{name}" for name in _SPECIFICATION_NAMES
    },
    **{
        name: f"cudaq.logical.gadgets.contracts:{name}" for name in _CONTRACT_NAMES
    },
    **{
        name: f"cudaq.logical.gadgets.definition:{name}" for name in _DEFINITION_NAMES
    },
    **{
        name: f"cudaq.logical.gadgets.factories:{name}" for name in _FACTORY_NAMES
    },
    **{
        name: f"cudaq.logical.gadgets.verification:{name}" for name in _VERIFICATION_NAMES
    },
    "GadgetBuilder":
        "cudaq.logical.gadgets.builder:GadgetBuilder",
    "GadgetContractLowerer":
        "cudaq.logical.gadgets.contract_lowerer:GadgetContractLowerer",
    "clifford_action":
        "cudaq.logical.gadgets.analysis:clifford_action",
    "accept_all":
        "cudaq.logical.gadgets.success:accept_all",
    "all_zero":
        "cudaq.logical.gadgets.success:all_zero",
    "all_false":
        "cudaq.logical.gadgets.success:all_false",
}
_PRIVATE_EXPORTS = {
    "_PredicateProvenance":
        "cudaq.logical.gadgets.semantics:_PredicateProvenance",
    "_gadget_boundary_profiles":
        "cudaq.logical.gadgets.specification:_gadget_boundary_profiles",
    "_resolve_gadget_code_profile":
        "cudaq.logical.gadgets.records:_resolve_gadget_code_profile",
    "_scalar_profile_parities":
        "cudaq.logical.gadgets.semantics:_scalar_profile_parities",
    "_stabilizer_preparation_circuit":
        "cudaq.logical.gadgets.factories:_stabilizer_preparation_circuit",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        return _resolve(globals(), _EXPORTS, name)
    except AttributeError:
        return _resolve(globals(), _PRIVATE_EXPORTS, name)


def __dir__():
    return _public_dir(globals(), _EXPORTS)
