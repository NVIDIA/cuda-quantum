# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Code definitions, encodings, profiles, blocks, and reusable families."""

from .._core.lazy import public_dir as _public_dir, resolve as _resolve

_SELECTION_NAMES = (
    "QECActionSelection",
    "QECBlockBinding",
    "QECBlockOwner",
    "QECBlockRequest",
    "QECSelectionWitness",
    "qec_block",
)
_STRUCTURE_NAMES = ("Block", "CarrierRoleMap", "CSSBlock", "PatchTransform")
_DISTANCE_NAMES = ("Distance", "DistanceScope", "Schedule")
_PROFILE_NAMES = (
    "CodeProfile",
    "EncodingEpoch",
    "EncodingEpochSchema",
    "GaugeMeasurementMap",
    "MeasurementPhase",
    "MetaChecks",
    "RecordLogicalMap",
)
_ENCODING_NAMES = (
    "Concatenated",
    "Encoding",
    "EncodingHierarchy",
    "EncodingProjection",
    "FixedPort",
    "expose",
    "fix",
    "gauge",
)
_DEFINITION_NAMES = (
    "CSSCode",
    "Code",
    "ParameterizedCode",
    "StabilizerCode",
    "SubsystemCode",
    "code",
)
_CATALOG_NAMES = (
    "BareQubit",
    "Repetition",
    "ReedMuller15",
    "RM15",
    "Steane",
    "Surface",
    "rotated_surface",
)

_EXPORTS = {
    **{
        name: f".selection:{name}" for name in _SELECTION_NAMES
    },
    **{
        name: f".structure:{name}" for name in _STRUCTURE_NAMES
    },
    **{
        name: f".distance:{name}" for name in _DISTANCE_NAMES
    },
    **{
        name: f".profiles:{name}" for name in _PROFILE_NAMES
    },
    **{
        name: f".encodings:{name}" for name in _ENCODING_NAMES
    },
    **{
        name: f".definition:{name}" for name in _DEFINITION_NAMES
    },
    **{
        name: f".catalog:{name}" for name in _CATALOG_NAMES
    },
}
_PRIVATE_EXPORTS = {
    "_materialized_code_identity": ".definition:_materialized_code_identity",
    "_materialized_code_metadata": ".definition:_materialized_code_metadata",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        return _resolve(globals(), _EXPORTS, name)
    except AttributeError:
        return _resolve(globals(), _PRIVATE_EXPORTS, name)


def __dir__():
    return _public_dir(globals(), _EXPORTS)
