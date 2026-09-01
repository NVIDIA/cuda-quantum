# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from inspect import Signature
from typing import Any, Callable, Mapping, get_args, get_origin

from .._core.immutable import (
    ImmutableValue,
    freeze_mapping,
)
from ..types.semantic import resource


def _flatten_boundary_annotations(annotation):
    if get_origin(annotation) is tuple:
        flattened = []
        for member in get_args(annotation):
            flattened.extend(_flatten_boundary_annotations(member))
        return tuple(flattened)
    return (annotation,)


class ProtocolDefinition(ImmutableValue):
    """Lazy folded P2 network of typed gadget/protocol calls."""

    __slots__ = (
        "provider",
        "implements",
        "name",
        "signature",
        "metadata",
        "type_hints",
        "profile",
        "stage",
        "facets",
        "_factory_region",
        "__dict__",
    )

    def __init__(
        self,
        provider: Callable[..., Any],
        *,
        implements=None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        type_hints: Mapping[str, Any] | None = None,
        _factory_region=None,
    ) -> None:
        from .._core.definition_signature import resolve_definition_signature
        from ..gadgets import patch

        resolved_signature, resolved_hints = resolve_definition_signature(
            provider,
            definition_kind="protocol",
            patch_type=patch,
            localns={},
            type_hints=type_hints,
        )
        self.provider = provider
        self.implements = implements
        self.name = name or provider.__name__
        self.signature: Signature = resolved_signature
        self.type_hints = freeze_mapping(resolved_hints)
        self._verify_production_boundary()
        self.metadata = freeze_mapping(metadata)
        self.profile = "p2n"
        from ..stages import (
            P2,
            PROTOCOL_NETWORK,
        )

        self.stage = P2
        self.facets = (PROTOCOL_NETWORK,)
        self._factory_region = _factory_region
        self.__name__ = provider.__name__
        self.__qualname__ = provider.__qualname__
        self.__doc__ = provider.__doc__
        self.__module__ = provider.__module__
        self.__annotations__ = dict(getattr(provider, "__annotations__", {}))
        self._seal()

    def _verify_production_boundary(self) -> None:
        from ..std import ResourceFlowRef

        objective = self.implements
        if (not isinstance(objective, ResourceFlowRef) or
                objective.kind != "produce"):
            return
        annotation = self.type_hints.get("return",
                                         self.signature.return_annotation)
        boundaries = _flatten_boundary_annotations(annotation)
        if (len(boundaries) != 1 or get_origin(boundaries[0]) is not resource or
                get_args(boundaries[0]) != (objective.resource,)):
            raise TypeError(
                "a protocol implementing cudaq.logical.std.produce(kind) must "
                "return exactly cudaq.logical.types.resource[kind]")

    def _resource_input_kinds(self):
        kinds = []
        for name, parameter in self.signature.parameters.items():
            annotation = self.type_hints.get(name, parameter.annotation)
            if get_origin(annotation) is resource:
                kinds.append(get_args(annotation)[0])
        return tuple(kinds)

    def _resource_output_kinds(self):
        return tuple(
            get_args(value)[0]
            for value in self._boundary_output_annotations()
            if get_origin(value) is resource)

    def _boundary_output_annotations(self):
        annotation = self.type_hints.get("return",
                                         self.signature.return_annotation)
        return _flatten_boundary_annotations(annotation)

    def _bind_factory(self, region):
        """Return a device-authoring clone with a derived allocation region."""

        from ..architecture.logical import Space

        if not isinstance(region, Space):
            raise TypeError("protocol factory binding requires a logical Space")
        if self._factory_region is not None:
            if self._factory_region is region:
                return self
            raise ValueError(
                f"protocol {self.name!r} is already bound to another factory")
        return ProtocolDefinition(
            self.provider,
            implements=self.implements,
            name=self.name,
            metadata=self.metadata,
            type_hints=self.type_hints,
            _factory_region=region,
        )

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)

    def _attach_direct_snapshot(self, build) -> None:
        if hasattr(self, "_qlx_direct_snapshot"):
            raise RuntimeError(
                "protocol definition already has a direct snapshot")
        object.__setattr__(self, "_qlx_direct_snapshot", build)

    def __call__(self, *args, **kwargs):
        from ..programs.context import current_trace

        trace = current_trace()
        if trace is None:
            raise RuntimeError(
                f"{self.name} is a CUDA-Q Logical protocol definition; materialize it or "
                "call it inside a compatible protocol trace")
        return trace.call(self, args, kwargs)


def protocol(
    fn=None,
    *,
    implements=None,
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
):
    from inspect import currentframe

    frame = currentframe()
    localns = dict(frame.f_back.f_locals) if frame and frame.f_back else {}

    def decorate(provider):
        from .._core.definition_signature import resolve_definition_signature
        from ..gadgets import patch

        _, hints = resolve_definition_signature(
            provider,
            definition_kind="protocol",
            patch_type=patch,
            localns=localns,
        )
        return ProtocolDefinition(
            provider,
            implements=implements,
            name=name,
            metadata=metadata,
            type_hints=hints,
        )

    return decorate(fn) if fn is not None else decorate
