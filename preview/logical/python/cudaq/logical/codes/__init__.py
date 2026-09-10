# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Code definitions, encodings, profiles, blocks, and reusable families."""

from cudaq.logical._core.lazy import public_dir as _public_dir, resolve as _resolve

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
_BB_NAMES = (
    "BinaryPolynomial",
    "BBPermutationMap",
    "BBSyndromeMoment",
    "BBSyndromeSchedule",
    "BivariateBicycleCode",
    "CyclicProduct",
    "binary_polynomial",
)
_CATALOG_NAMES = (
    "BB3x3",
    "BareQubit",
    "Repetition",
    "ReedMuller15",
    "RM15",
    "Steane",
    "Surface",
    "Tesseract",
    "Toric",
    "TriangularColor",
    "bivariate_bicycle",
    "PINNACLE_GB_INSTANCES",
    "PUBLISHED_GB_SEEDS",
    "PinnacleGBInstance",
    "pinnacle_gb",
    "pinnacle_gb_instance",
    "rotated_surface",
    "toric",
    "triangular_color",
    "zxxz_surface",
)

_EXPORTS = {
    **{
        name: f"cudaq.logical.codes.selection:{name}" for name in _SELECTION_NAMES
    },
    **{
        name: f"cudaq.logical.codes.structure:{name}" for name in _STRUCTURE_NAMES
    },
    **{
        name: f"cudaq.logical.codes.distance:{name}" for name in _DISTANCE_NAMES
    },
    **{
        name: f"cudaq.logical.codes.profiles:{name}" for name in _PROFILE_NAMES
    },
    **{
        name: f"cudaq.logical.codes.encodings:{name}" for name in _ENCODING_NAMES
    },
    **{
        name: f"cudaq.logical.codes.definition:{name}" for name in _DEFINITION_NAMES
    },
    **{
        name: f"cudaq.logical.codes.bb:{name}" for name in _BB_NAMES
    },
    **{
        name: f"cudaq.logical.codes.catalog:{name}" for name in _CATALOG_NAMES
    },
}
_PRIVATE_EXPORTS = {
    "_quotient_basis":
        "cudaq.logical.codes.catalog:_quotient_basis",
    "_torus_index":
        "cudaq.logical.codes.catalog:_torus_index",
    "_materialized_code_identity":
        "cudaq.logical.codes.definition:_materialized_code_identity",
    "_materialized_code_metadata":
        "cudaq.logical.codes.definition:_materialized_code_metadata",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        return _resolve(globals(), _EXPORTS, name)
    except AttributeError:
        return _resolve(globals(), _PRIVATE_EXPORTS, name)


def __dir__():
    return _public_dir(globals(), _EXPORTS)
