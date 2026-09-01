# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from inspect import Signature
from math import isfinite
from types import MappingProxyType, NoneType
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Mapping,
    TypeVar,
    get_args,
    get_origin,
)

from ..programs.binding import (
    LogicalPortRef,
    ObjectiveOperandRef,
)
from .._core.immutable import ImmutableValue

EncodingT = TypeVar("EncodingT")

from .interface import EndpointCollection, _freeze_gadget_metadata, patch
from .records import GadgetRecords, RecordRef
from .semantics import OutcomeMap
from .specification import (
    GadgetSpec,
    Port,
    _build_gadget_interface,
    _normalize_gadget_logical_ports,
    _verify_explicit_spec,
)


class GadgetDefinition(ImmutableValue):
    __slots__ = (
        "provider",
        "implements",
        "name",
        "signature",
        "logical_ports",
        "transform",
        "type_hints",
        "metadata",
        "profile",
        "stage",
        "facets",
        "interface",
        "spec",
        "_inferred_outcome_map",
        "device",
        "__dict__",
    )

    def __init__(
        self,
        provider: Callable[..., Any],
        *,
        implements=None,
        spec: GadgetSpec | None = None,
        name: str | None = None,
        logical_ports: Mapping[ObjectiveOperandRef | str, LogicalPortRef | str]
        | None = None,
        transform=None,
        device=None,
        metadata: Mapping[str, Any] | None = None,
        type_hints: Mapping[str, Any] | None = None,
    ) -> None:
        if spec is not None:
            if not isinstance(spec, GadgetSpec):
                raise TypeError(
                    "@cudaq.logical.gadget spec= must be a cudaq.logical.GadgetSpec"
                )
            if implements is None:
                implements = spec.implements
            elif implements is not spec.implements and implements != spec.implements:
                from ..errors import ObjectiveMismatch

                raise ObjectiveMismatch(
                    "@cudaq.logical.gadget implements= and spec.implements disagree; "
                    "declare the objective once on the explicit spec")
        from .._core.definition_signature import resolve_definition_signature

        resolved_signature, resolved_hints = resolve_definition_signature(
            provider,
            definition_kind="gadget",
            patch_type=patch,
            localns={},
            allow_gadget_scalars=spec is not None,
            type_hints=type_hints,
        )
        self.provider = provider
        self.implements = implements
        self.spec = spec
        self._inferred_outcome_map = None
        if device is not None:
            from ..devices.definition import Device

            if not isinstance(device, Device):
                raise TypeError(
                    "@cudaq.logical.gadget device= requires a cudaq.logical.Device"
                )
            if implements is not None:
                raise TypeError(
                    "@cudaq.logical.gadget device= is available only on an "
                    "objective-free top-level entry gadget")
        self.device = device
        self.name = name or provider.__name__
        self.signature: Signature = resolved_signature
        self.type_hints = MappingProxyType(dict(resolved_hints))
        self.logical_ports = MappingProxyType({})
        if transform is not None:
            from ..codes import PatchTransform

            if not isinstance(transform, PatchTransform):
                raise TypeError(
                    "gadget transform= must be a cudaq.logical.PatchTransform")
        self.transform = transform
        self.metadata = _freeze_gadget_metadata(
            dict(metadata or {}), what="GadgetDefinition.metadata")
        self.profile = "p2a"
        from ..stages import (
            P2,
            QEC_REALIZATION,
            QEC_SPEC,
        )

        self.stage = P2
        self.facets = (QEC_SPEC, QEC_REALIZATION)
        self.interface = _build_gadget_interface(self)
        if self.implements is None and self.interface.inputs.blocks:
            from ..errors import MissingObjective

            raise MissingObjective(
                "an objective-free @cudaq.logical.gadget must be a zero-quantum-input "
                "entry point; gadgets with patch inputs require implements= "
                "or an explicit spec=")
        self.logical_ports = _normalize_gadget_logical_ports(
            self.implements, self.interface, logical_ports)
        if spec is not None:
            _verify_explicit_spec(self)
        self.__name__ = provider.__name__
        self.__qualname__ = provider.__qualname__
        self.__doc__ = provider.__doc__
        self.__module__ = provider.__module__
        self.__annotations__ = dict(getattr(provider, "__annotations__", {}))
        self._seal()

    def _authoritative_spec_metadata(self):
        """Return metadata serialized on the canonical gadget specification."""

        values = dict(self.metadata)
        if self.spec is not None:
            values.update(self.spec.metadata)
        return MappingProxyType(values)

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)

    def _attach_inferred_outcome_map(self, outcome_map: OutcomeMap) -> None:
        previous = self._inferred_outcome_map
        if previous is not None and previous != outcome_map:
            from ..errors import ObjectiveMismatch

            raise ObjectiveMismatch(
                f"gadget {self.name!r} derived a context-dependent OutcomeMap")
        if previous is None:
            object.__setattr__(self, "_inferred_outcome_map", outcome_map)

    @property
    def outcome_map(self) -> OutcomeMap | None:
        """Return the one typed explicit or compiler-derived outcome map."""

        if self.spec is not None:
            return self.spec.outcome_map
        if self._inferred_outcome_map is None and self.implements is not None:
            self.materialize()
        return self._inferred_outcome_map

    def _attach_direct_snapshot(self, build) -> None:
        if hasattr(self, "_qlx_direct_snapshot"):
            raise RuntimeError(
                "gadget definition already has a direct snapshot")
        object.__setattr__(self, "_qlx_direct_snapshot", build)

    def record(self, name: str) -> RecordRef:
        return RecordRef(self, name)

    @property
    def records(self) -> GadgetRecords:
        return GadgetRecords(self)

    @property
    def inputs(self) -> EndpointCollection:
        return self.interface.inputs

    @property
    def outputs(self) -> EndpointCollection:
        return self.interface.outputs

    def __call__(self, *args, **kwargs):
        from ..programs.context import current_trace

        trace = current_trace()
        if trace is None:
            raise RuntimeError(
                f"{self.name} is a CUDA-Q Logical gadget definition; materialize it or call "
                "it inside a compatible protocol/gadget trace")
        if self.implements is None:
            raise TypeError(
                f"objective-free gadget {self.name!r} is a root entry point "
                "and cannot be selected or called as a nested realization")
        return trace.call(self, args, kwargs)


def gadget(
    fn=None,
    *,
    implements=None,
    spec: GadgetSpec | None = None,
    name: str | None = None,
    logical_ports: Mapping[ObjectiveOperandRef | str, LogicalPortRef | str] |
    None = None,
    transform=None,
    device=None,
    metadata: Mapping[str, Any] | None = None,
):
    from inspect import currentframe

    frame = currentframe()
    localns = dict(frame.f_back.f_locals) if frame and frame.f_back else {}

    def decorate(provider):
        from .._core.definition_signature import resolve_definition_signature

        _, hints = resolve_definition_signature(
            provider,
            definition_kind="gadget",
            patch_type=patch,
            localns=localns,
            allow_gadget_scalars=spec is not None,
        )
        return GadgetDefinition(
            provider,
            implements=implements,
            spec=spec,
            name=name,
            logical_ports=logical_ports,
            transform=transform,
            device=device,
            metadata=metadata,
            type_hints=hints,
        )

    return decorate(fn) if fn is not None else decorate
