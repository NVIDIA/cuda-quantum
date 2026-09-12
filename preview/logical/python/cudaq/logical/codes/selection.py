# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from inspect import signature
import json
import math
import re
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from ..errors import InvalidCodeAlgebra
from cudaq.logical._core.immutable import ImmutableValue
from cudaq.logical.algebra.clifford import CliffordAction
from cudaq.logical.algebra.gf2 import (
    GF2Matrix,
    _normalize_binary_value,
    _normalize_binary_values,
    _row_bits,
)
from cudaq.logical.architecture.logical import (
    LogicalValueGroup,
    LogicalValueRef,
)


def _deep_freeze(value, *, what: str = "structured value"):
    """Detach and recursively freeze the supported structured value surface.

    The public metadata/evidence surface deliberately accepts JSON-like
    containers plus QLX's frozen provenance value.  Copying an arbitrary
    object is not sufficient: a custom ``__deepcopy__`` implementation can
    return ``self`` (or otherwise retain mutable aliases), which would make a
    validated profile mutable through an external reference.
    """

    from ..analysis.evidence import Provenance

    if isinstance(value, Provenance):
        if not isinstance(value.kind, str) or not isinstance(
                value.reference, str):
            raise TypeError(f"{what} Provenance fields must be strings")
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            frozen_key = _deep_freeze(key, what=f"{what} key")
            try:
                hash(frozen_key)
            except TypeError as exc:
                raise TypeError(f"{what} keys must be immutable") from exc
            frozen[frozen_key] = _deep_freeze(item, what=f"{what}[{key!r}]")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(
            _deep_freeze(item, what=f"{what}[{index}]")
            for index, item in enumerate(value))
    if isinstance(value, (set, frozenset)):
        return frozenset(
            _deep_freeze(item, what=f"{what} member") for item in value)
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    raise TypeError(f"{what} contains unsupported mutable or custom value "
                    f"{type(value).__name__!r}")


def _canonical_label_index(labels, width: int, *, what: str):
    """Validate the shared sorted-rank label/index contract.

    Matrix row and column positions are semantic indices. A labeled view is
    safe only when label position and sorted-label rank agree.
    """

    if isinstance(labels, (str, bytes)):
        raise TypeError(f"{what} must be an iterable of labels, not one string")
    normalized = tuple(labels)
    if len(normalized) != width:
        raise ValueError(f"{what} width must equal {width}")
    try:
        unique = set(normalized)
    except TypeError as exc:
        raise TypeError(f"{what} must contain hashable labels") from exc
    if len(unique) != width:
        raise ValueError(f"{what} must contain unique labels")
    try:
        canonical = tuple(sorted(normalized))
    except TypeError as exc:
        raise TypeError(
            f"{what} must contain mutually sortable labels") from exc
    if normalized != canonical:
        raise ValueError(
            f"{what} must use canonical sorted order so label rank equals index"
        )
    return normalized, MappingProxyType({
        label: index for index, label in enumerate(normalized)
    })


@dataclass(frozen=True, slots=True)
class QECBlockRequest:
    """P2 request to realize several placed logical owners in one QEC block."""

    values: tuple[LogicalValueRef, ...]
    encoding: "Encoding"

    def __post_init__(self) -> None:
        from .encodings import Encoding

        values = tuple(self.values)
        if not values or any(
                not isinstance(value, LogicalValueRef) for value in values):
            raise TypeError(
                "QECBlockRequest values must be logical value references")
        if len(set(values)) != len(values):
            raise ValueError("QECBlockRequest cannot repeat one logical owner")
        if not isinstance(self.encoding, Encoding):
            raise TypeError(
                "QECBlockRequest encoding must be a concrete Encoding")
        if len(values) > self.encoding.code.k:
            raise ValueError(
                f"encoding {self.encoding.name!r} exposes "
                f"{self.encoding.code.k} logical ports, not {len(values)}")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class QECBlockOwner:
    """One P1 owner mapped to a logical port of a selected P2 block."""

    placement: str
    logical_index: int
    source_allocation: int | None = None
    source_group: str | None = None
    source_path: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.placement, str) or not self.placement:
            raise TypeError("QECBlockOwner placement must be a nonempty string")
        if (not isinstance(self.logical_index, int) or
                isinstance(self.logical_index, bool) or self.logical_index < 0):
            raise TypeError("QECBlockOwner logical_index must be nonnegative")
        if self.source_allocation is not None and (
                not isinstance(self.source_allocation, int) or
                isinstance(self.source_allocation, bool) or
                self.source_allocation < 0):
            raise TypeError(
                "QECBlockOwner source_allocation must be nonnegative or None")
        if self.source_group is not None and (not isinstance(
                self.source_group, str) or not self.source_group):
            raise TypeError(
                "QECBlockOwner source_group must be a nonempty string or None")
        source_path = tuple(self.source_path)
        if any(not isinstance(value, int) or isinstance(value, bool) or
               value < 0 for value in source_path):
            raise TypeError(
                "QECBlockOwner source_path must contain nonnegative integers")
        object.__setattr__(self, "source_path", source_path)


@dataclass(frozen=True, slots=True)
class QECBlockBinding:
    """P2 witness for code/encoding choice and logical-port assignment."""

    block: str
    space: str
    code: str
    encoding: str
    logical_capacity: int
    owners: tuple[QECBlockOwner, ...]

    def __post_init__(self) -> None:
        for field_name in ("block", "space", "code", "encoding"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise TypeError(
                    f"QECBlockBinding {field_name} must be a nonempty string")
        if (not isinstance(self.logical_capacity, int) or
                isinstance(self.logical_capacity, bool) or
                self.logical_capacity <= 0):
            raise TypeError(
                "QECBlockBinding logical_capacity must be a positive integer")
        owners = tuple(self.owners)
        if not owners or any(
                not isinstance(value, QECBlockOwner) for value in owners):
            raise TypeError(
                "QECBlockBinding owners must be QECBlockOwner values")
        if len({value.placement for value in owners}) != len(owners):
            raise ValueError("QECBlockBinding owner placements must be unique")
        if len({value.logical_index for value in owners}) != len(owners):
            raise ValueError("QECBlockBinding logical ports must be unique")
        if any(value.logical_index >= self.logical_capacity
               for value in owners):
            raise ValueError(
                "QECBlockBinding logical ports must fit its capacity")
        object.__setattr__(self, "owners", owners)


@dataclass(frozen=True, slots=True)
class QECActionSelection:
    """P2 witness for one deterministic action/gadget selection."""

    site: str
    kind: str
    objective: str
    placements: tuple[str, ...]
    feasible_candidates: tuple[str, ...]
    selected: str
    provider: str = "fixed"
    version: str = "linked"
    manifest_sha256: str | None = None
    tie_break: str = "fixed-before-generated-then-symbol-order"
    channel: str | None = None
    channel_capability: str | None = None
    endpoints: tuple[str, ...] = ()
    direction: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
                "site",
                "kind",
                "objective",
                "selected",
                "provider",
                "version",
                "tie_break",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise TypeError(
                    f"QECActionSelection {field_name} must be a nonempty string"
                )
        placements = tuple(self.placements)
        feasible = tuple(self.feasible_candidates)
        if any(not isinstance(value, str) or not value for value in placements):
            raise TypeError(
                "QECActionSelection placements must be nonempty strings")
        if (not feasible or any(
                not isinstance(value, str) or not value for value in feasible)):
            raise TypeError(
                "QECActionSelection feasible_candidates must be nonempty strings"
            )
        if self.selected not in feasible:
            raise ValueError(
                "QECActionSelection selected candidate must be feasible")
        object.__setattr__(self, "placements", placements)
        object.__setattr__(self, "feasible_candidates", feasible)
        if self.provider != "fixed" and self.manifest_sha256 is None:
            raise ValueError(
                "versioned QECActionSelection requires a manifest digest")
        if self.manifest_sha256 is not None:
            prefix = "sha256:"
            payload = (self.manifest_sha256[len(prefix):]
                       if isinstance(self.manifest_sha256, str) and
                       self.manifest_sha256.startswith(prefix) else "")
            if len(payload) != 64 or any(
                    value not in "0123456789abcdef" for value in payload):
                raise ValueError(
                    "QECActionSelection manifest_sha256 must be canonical")
        endpoints = tuple(self.endpoints)
        object.__setattr__(self, "endpoints", endpoints)
        present = (
            self.channel is not None,
            self.channel_capability is not None,
            bool(endpoints),
            self.direction is not None,
        )
        if any(present) != all(present):
            raise ValueError(
                "QECActionSelection communication fields must be all absent "
                "or all present")
        if all(present) and (not isinstance(self.channel, str) or
                             not self.channel or
                             not isinstance(self.channel_capability, str) or
                             not self.channel_capability or
                             any(not isinstance(endpoint, str) or not endpoint
                                 for endpoint in endpoints) or
                             not isinstance(self.direction, str) or
                             not self.direction):
            raise ValueError(
                "QECActionSelection communication fields must be nonempty")


@dataclass(frozen=True, slots=True)
class QECSelectionWitness:
    """Replayable P1-to-P2 encoding and block-selection witness."""

    input_p1: str
    blocks: tuple[QECBlockBinding, ...]
    actions: tuple[QECActionSelection, ...] = ()
    code: str | None = None
    encoding: str | None = None
    objective: str = "policy_then_device_then_candidate"
    tie_break: str = "declaration_order"
    network_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input_p1, str) or not self.input_p1:
            raise TypeError("QECSelectionWitness input_p1 must be nonempty")
        if (not isinstance(self.objective, str) or not self.objective or
                not isinstance(self.tie_break, str) or not self.tie_break):
            raise TypeError(
                "QECSelectionWitness objective and tie_break must be nonempty")
        if (self.code is None) != (self.encoding is None):
            raise ValueError(
                "QECSelectionWitness code and encoding must occur together")
        for field_name in ("code", "encoding"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise TypeError(
                    f"QECSelectionWitness {field_name} must be nonempty or None"
                )
        blocks = tuple(self.blocks)
        actions = tuple(self.actions)
        if any(not isinstance(value, QECBlockBinding) for value in blocks):
            raise TypeError(
                "QECSelectionWitness blocks must be QECBlockBinding values")
        if any(not isinstance(value, QECActionSelection) for value in actions):
            raise TypeError(
                "QECSelectionWitness actions must be QECActionSelection values")
        if len({value.block for value in blocks}) != len(blocks):
            raise ValueError(
                "QECSelectionWitness block identities must be unique")
        owners = tuple(
            owner.placement for block in blocks for owner in block.owners)
        if len(set(owners)) != len(owners):
            raise ValueError("QECSelectionWitness owners must be unique")
        if len({value.site for value in actions}) != len(actions):
            raise ValueError("QECSelectionWitness action sites must be unique")
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "actions", actions)
        if self.network_manifest_sha256 is None:
            return
        prefix = "sha256:"
        payload = (self.network_manifest_sha256[len(prefix):]
                   if isinstance(self.network_manifest_sha256, str) and
                   self.network_manifest_sha256.startswith(prefix) else "")
        if len(payload) != 64 or any(
                value not in "0123456789abcdef" for value in payload):
            raise ValueError(
                "QECSelectionWitness network manifest must be canonical")
        network_actions = tuple(
            action for action in self.actions
            if action.manifest_sha256 == self.network_manifest_sha256)
        if not network_actions:
            raise ValueError(
                "network QEC selection must contain an action that commits "
                "its exact manifest")


def qec_block(values, *, code=None, encoding=None) -> QECBlockRequest:
    """Request one encoded block during P1-to-P2 QEC realization.

    This is intentionally a P2 policy value, not a P0-to-P1 placement
    constraint.  P1 decides logical residency; P2 chooses the concrete code,
    encoding, block identity, and logical-port map.
    """

    from .definition import Code
    from .encodings import Encoding

    if isinstance(values, LogicalValueRef):
        values = (values,)
    elif isinstance(values, LogicalValueGroup):
        values = tuple(values)
    else:
        values = tuple(values)
    if not values or any(
            not isinstance(value, LogicalValueRef) for value in values):
        raise TypeError(
            "cudaq.logical.qec_block expects one or more logical value "
            "references")
    if len(set(values)) != len(values):
        raise ValueError(
            "cudaq.logical.qec_block cannot repeat one logical owner")
    if (code is None) == (encoding is None):
        raise TypeError(
            "cudaq.logical.qec_block requires exactly one of code= or encoding="
        )
    if code is not None:
        if not isinstance(code, Code):
            raise TypeError(
                "cudaq.logical.qec_block code= requires a concrete Code")
        encoding = code.default_encoding
    if not isinstance(encoding, Encoding):
        raise TypeError(
            "cudaq.logical.qec_block encoding= requires a concrete Encoding")
    if len(values) > encoding.code.k:
        raise ValueError(
            f"encoding {encoding.name!r} exposes {encoding.code.k} logical ports, "
            f"not {len(values)}")
    return QECBlockRequest(values, encoding)
