# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from enum import Enum
from typing import Optional


class Stage(str, Enum):
    """Semantic commitment stage of one verified QLX root."""

    P0 = "p0"
    P1 = "p1"
    P2 = "p2"
    P3 = "p3"

    def __str__(self) -> str:
        return self.value


class Facet(str, Enum):
    """Orthogonal verified facts attached to a semantic-stage root.

    Facets are products or capabilities, not additional points in the P0-P3
    lowering order.  Several may coexist for one immutable semantic root.
    """

    QEC_SPEC = "qec_spec"
    QEC_REALIZATION = "qec_realization"
    PROTOCOL_NETWORK = "protocol_network"
    PATCH_GRAPH = "patch_graph"
    PATCH_MAPPING = "patch_mapping"
    NATIVE_LEGALIZATION = "native_legalization"
    CARRIER_MAPPING = "carrier_mapping"
    PHYSICAL_ROUTING = "physical_routing"
    ZONED_MOVEMENT = "zoned_movement"
    PHYSICAL_SCHEDULE = "physical_schedule"

    def __str__(self) -> str:
        return self.value


P0 = Stage.P0
P1 = Stage.P1
P2 = Stage.P2
P3 = Stage.P3

QEC_SPEC = Facet.QEC_SPEC
QEC_REALIZATION = Facet.QEC_REALIZATION
PROTOCOL_NETWORK = Facet.PROTOCOL_NETWORK
PATCH_GRAPH = Facet.PATCH_GRAPH
PATCH_MAPPING = Facet.PATCH_MAPPING
NATIVE_LEGALIZATION = Facet.NATIVE_LEGALIZATION
CARRIER_MAPPING = Facet.CARRIER_MAPPING
PHYSICAL_ROUTING = Facet.PHYSICAL_ROUTING
ZONED_MOVEMENT = Facet.ZONED_MOVEMENT
PHYSICAL_SCHEDULE = Facet.PHYSICAL_SCHEDULE

_legacy_products = {
    "p2s": (Stage.P2, (Facet.QEC_SPEC,)),
    "p2a": (Stage.P2, ()),
    "p2n": (Stage.P2, (Facet.QEC_SPEC, Facet.PROTOCOL_NETWORK)),
}


def stage_and_facets(
    value: str | Stage,) -> tuple[Optional[Stage], tuple[Facet, ...]]:
    """Normalize an implementation product spelling to stage plus facets."""

    if isinstance(value, Stage):
        return value, ()
    text = str(value)
    if text == "common":
        return None, ()
    if text in _legacy_products:
        return _legacy_products[text]
    return Stage(text), ()


def normalize_facets(values) -> tuple[Facet, ...]:
    result: list[Facet] = []
    for value in values or ():
        facet = value if isinstance(value, Facet) else Facet(str(value))
        if facet not in result:
            result.append(facet)
    return tuple(result)


def facets_for_kind(kind) -> tuple[Facet, ...]:
    """Return the semantic facets established by one root definition kind."""

    name = kind if isinstance(kind, str) else getattr(kind, "__name__",
                                                      str(kind))
    mapping = {
        "Code": (Facet.QEC_SPEC,),
        "CodeProfile": (Facet.QEC_SPEC,),
        "Encoding": (Facet.QEC_SPEC,),
        "EncodingEpoch": (Facet.QEC_SPEC,),
        "EncodingEpochSchema": (Facet.QEC_SPEC,),
        "EncodingHierarchy": (Facet.QEC_SPEC,),
        "EncodingProjection": (Facet.QEC_SPEC,),
        "PatchTransform": (Facet.QEC_SPEC,),
        "gadget": (Facet.QEC_SPEC, Facet.QEC_REALIZATION),
        "protocol": (Facet.PROTOCOL_NETWORK,),
        "physical_schedule": (Facet.PHYSICAL_SCHEDULE,),
    }
    return mapping.get(name, ())


__all__ = [
    "Stage",
    "Facet",
    "P0",
    "P1",
    "P2",
    "P3",
    "QEC_SPEC",
    "QEC_REALIZATION",
    "PROTOCOL_NETWORK",
    "PATCH_GRAPH",
    "PATCH_MAPPING",
    "NATIVE_LEGALIZATION",
    "CARRIER_MAPPING",
    "PHYSICAL_ROUTING",
    "ZONED_MOVEMENT",
    "PHYSICAL_SCHEDULE",
    "stage_and_facets",
    "normalize_facets",
    "facets_for_kind",
]
