# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Integrated QEC families, synthesis, and lowering recipes."""

from cudaq.logical._core.lazy import public_dir as _public_dir, resolve as _resolve

# Generic definitions now belong to codes/gadgets/protocols. These
# Lazy aliases preserve the established CUDA-Q Logical QEC surface, but they
# are omitted
# from ``__all__`` so discovery presents this package's actual responsibilities.
_API_NAMES = (
    "Block",
    "CarrierRoleMap",
    "CSSBlock",
    "CSSCode",
    "Code",
    "CodeProfile",
    "Concatenated",
    "BinaryPolynomial",
    "BBPermutationMap",
    "BBSyndromeMoment",
    "BBSyndromeSchedule",
    "BivariateBicycleCode",
    "CyclicProduct",
    "Distance",
    "DistanceScope",
    "binary_polynomial",
    "Encoding",
    "EncodingEpoch",
    "EncodingEpochSchema",
    "EncodingHierarchy",
    "EncodingProjection",
    "PatchTransform",
    "QECActionSelection",
    "QECBlockBinding",
    "QECBlockOwner",
    "QECBlockRequest",
    "QECSelectionWitness",
    "FixedPort",
    "GF2Matrix",
    "GaugeMeasurementMap",
    "MeasurementPhase",
    "MetaChecks",
    "OutcomeRole",
    "BlockEndpoint",
    "BlockFlow",
    "EndpointCollection",
    "GadgetDefinition",
    "GadgetInterface",
    "GadgetProfile",
    "GadgetSpec",
    "OutcomeMap",
    "OutcomeSyndromeTerm",
    "ParameterMap",
    "Port",
    "ProfileBinding",
    "ProfileGraph",
    "ProfileParity",
    "ProfileVectorExpr",
    "CommitPoint",
    "CommitPointKind",
    "RetryExhaustion",
    "RetryPolicy",
    "ParameterizedCode",
    "ProtocolDefinition",
    "RecordParity",
    "RecordRef",
    "RecordVectorParity",
    "RecordLogicalMap",
    "RecordFamily",
    "SelectionIntent",
    "Schedule",
    "StabilizerCode",
    "SubsystemCode",
    "SyndromeBundleRef",
    "SubsystemFragmentObjective",
    "SuccessPredicate",
    "abort_on",
    "before_output",
    "before_resource_output",
    "code",
    "condition_results",
    "qec_block",
    "require",
    "gadget",
    "gauge",
    "expose",
    "fix",
    "protocol",
    "GeneratedQECArtifact",
    "QECCompiler",
    "QECNetworkCompiler",
    "QECNetworkContext",
    "QECLowering",
    "ActionSiteHandle",
    "QECCompilerContext",
    "qec_lowering",
    "subsystem_fragment",
)
_COMPAT_EXPORTS = {name: f"cudaq.logical.qec.api:{name}" for name in _API_NAMES}
_EXPORTS = {
    **{
        name: f"cudaq.logical.qec.lowering:{name}" for name in (
            "ActionSiteHandle",
            "GeneratedQECArtifact",
            "QECCompiler",
            "QECCompilerContext",
            "QECLowering",
            "QECNetworkCompiler",
            "QECNetworkContext",
            "qec_lowering",
        )
    },
    "SubsystemFragmentObjective":
        "cudaq.logical.qec.objectives:SubsystemFragmentObjective",
    "subsystem_fragment": "cudaq.logical.qec.objectives:subsystem_fragment",
    **{
        name: f"cudaq.logical.qec.{name}" for name in (
            "floquet",
            "lattice_surgery",
            "magic",
            "neutral_atoms",
            "pinnacle",
            "product_rotation",
            "qldpc",
            "reichardt",
            "rounds",
            "steane",
            "surface",
            "synthesis",
            "transport",
        )
    },
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        return _resolve(globals(), _EXPORTS, name)
    except AttributeError:
        return _resolve(globals(), _COMPAT_EXPORTS, name)


def __dir__():
    return _public_dir(globals(), _EXPORTS)
