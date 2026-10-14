# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass, field
from inspect import Parameter, Signature, signature
from types import MappingProxyType
from typing import Any, Callable, Generic, Mapping, TypeVar, get_type_hints

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class DefinitionHandle(Generic[T]):
    """A typed reference to one materialized MLIR symbol."""

    symbol: str
    kind: type[T] | str
    profile: str

    @property
    def stage(self):
        from cudaq.logical.stages import stage_and_facets

        return stage_and_facets(self.profile)[0]

    @property
    def facets(self):
        from cudaq.logical.stages import (
            facets_for_kind,
            stage_and_facets,
        )

        _, legacy = stage_and_facets(self.profile)
        return tuple(dict.fromkeys((*legacy, *facets_for_kind(self.kind))))


@dataclass(frozen=True, slots=True)
class Definition(Generic[T]):
    """Detached immutable semantic definition.

    Ordinary Python object references connect detached definitions.  After
    materialization, :class:`DefinitionHandle` symbol references are
    authoritative; there is deliberately no universal artifact identifier.
    """

    name: str | None = None
    dependencies: tuple["Definition[Any]", ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "metadata",
                           MappingProxyType(dict(self.metadata)))

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)


class ProgramDefinition:
    """Lazy executable-program or ideal-objective provider."""

    __slots__ = (
        "provider",
        "name",
        "machine",
        "selection",
        "kind",
        "profile",
        "stage",
        "signature",
        "type_hints",
        "metadata",
        "specialization",
        "base",
        "objective_kind",
        "estimate_only",
        "_cudaq_kernel",
        "_kernel_declaration",
        "_sealed",
        "__dict__",
    )

    _SEMANTIC_FIELDS = frozenset({
        "provider",
        "name",
        "machine",
        "selection",
        "kind",
        "profile",
        "stage",
        "signature",
        "type_hints",
        "metadata",
        "specialization",
        "base",
        "objective_kind",
        "estimate_only",
        "_cudaq_kernel",
        "_kernel_declaration",
        "_sealed",
    })

    def __setattr__(self, name, value) -> None:
        if (getattr(self, "_sealed", False) and name in self._SEMANTIC_FIELDS):
            raise AttributeError("ProgramDefinition semantics are immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        provider: Callable[..., Any],
        *,
        machine: Any = None,
        selection: Any = None,
        kind: str = "program",
        objective_kind: str = "auto",
        estimate_only: bool = False,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        type_hints: Mapping[str, Any] | None = None,
        specialization: Mapping[str, Any] | None = None,
        base: "ProgramDefinition | None" = None,
        cudaq_kernel: Any = None,
    ) -> None:
        self._sealed = False
        self.provider = provider
        self.name = name or provider.__name__
        self.machine = machine
        if selection is not None:
            from cudaq.logical.programs.selection import SelectionIntent

            if not isinstance(selection, SelectionIntent):
                raise TypeError(
                    "selection= expects cudaq.logical.require(...), "
                    "cudaq.logical.condition_results(...), or "
                    "cudaq.logical.abort_on(...)")
        self.selection = selection
        self.kind = kind
        if kind not in {"program", "objective"}:
            raise ValueError("definition kind must be program or objective")
        if objective_kind not in {"auto", "action", "instrument"}:
            raise ValueError(
                "objective_kind must be auto, action, or instrument")
        if kind == "program" and objective_kind != "auto":
            raise ValueError("program definitions cannot set objective_kind")
        if cudaq_kernel is not None:
            from cudaq.kernel.kernel_decorator import isa_kernel_decorator

            if not isa_kernel_decorator(cudaq_kernel):
                raise TypeError("cudaq_kernel must be an @cudaq.kernel")
            if kind != "objective" or objective_kind == "instrument":
                raise ValueError(
                    "cudaq_kernel is supported only by action objectives")
        self.objective_kind = objective_kind
        self._cudaq_kernel = cudaq_kernel
        self._kernel_declaration = None
        if type(estimate_only) is not bool:
            raise TypeError("estimate_only must be a bool")
        if kind != "program" and estimate_only:
            raise ValueError("only program definitions may be estimate-only")
        if machine is not None and estimate_only:
            raise ValueError("estimate-only programs must begin at unplaced P0")
        self.estimate_only = estimate_only
        self.profile = "p1" if machine is not None else "p0"
        from cudaq.logical.stages import (
            P0,
            P1,
        )

        self.stage = P1 if machine is not None else P0
        self.signature: Signature = signature(provider)
        self.type_hints = MappingProxyType(
            dict(
                get_type_hints(provider) if type_hints is None else type_hints))
        self.metadata = MappingProxyType(dict(metadata or {}))
        self.specialization = MappingProxyType(dict(specialization or {}))
        self.base = base

        # Retain normal Python introspection without turning this into an
        # eagerly executing function.
        self.__name__ = provider.__name__
        self.__qualname__ = provider.__qualname__
        self.__doc__ = provider.__doc__
        self.__module__ = provider.__module__
        self.__annotations__ = dict(getattr(provider, "__annotations__", {}))
        self._sealed = True

    @property
    def ref(self) -> "ProgramDefinition":
        return self

    @property
    def cudaq_kernel(self):
        """Original CUDA-Q kernel retained for lazy objective lowering."""

        return self._cudaq_kernel

    @property
    def kernel_declaration(self):
        """Body-less CUDA-Q declaration for an ownership-preserving objective."""

        if self.kind != "objective" or self.objective_kind == "instrument":
            return None
        if self._kernel_declaration is None:
            from .kernel_objective import kernel_declaration_from_objective

            object.__setattr__(self, "_kernel_declaration",
                               kernel_declaration_from_objective(self))
        return self._kernel_declaration

    @property
    def operands(self):
        """Typed objective-operand references for semantic port bindings."""

        if self.kind != "objective":
            raise AttributeError(
                "only @cudaq.logical.objective definitions expose operands")
        from cudaq.logical.programs.binding import ObjectiveOperands

        return ObjectiveOperands(self, self.signature.parameters)

    @staticmethod
    def _specialization_value(value):
        from cudaq.logical.algebra.angle import Angle

        if value is None or isinstance(value, (bool, int, float, str, Angle)):
            return value
        if isinstance(value, tuple):
            return tuple(
                ProgramDefinition._specialization_value(item) for item in value)
        raise TypeError(
            "specialization values must be immutable scalar/tuple literals")

    @staticmethod
    def _specialization_fragment(value) -> str:
        raw = str(value).lower()
        fragment = "".join(
            char if char.isalnum() else "_" for char in raw).strip("_")
        return fragment or "value"

    def specialize(self, **bindings) -> "ProgramDefinition":
        """Bind named construction-time parameters and remove them from ABI."""

        if not bindings:
            return self
        unknown = set(bindings) - set(self.signature.parameters)
        if unknown:
            raise TypeError(
                f"unknown specialization parameter(s): {sorted(unknown)!r}")
        normalized = {
            name: self._specialization_value(value)
            for name, value in bindings.items()
        }
        for name in normalized:
            kind = self.signature.parameters[name].kind
            if kind in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}:
                raise TypeError("variadic parameters cannot be specialized")

        original = self.provider
        original_signature = self.signature
        remaining = tuple(
            parameter
            for name, parameter in original_signature.parameters.items()
            if name not in normalized)
        specialized_signature = original_signature.replace(parameters=remaining)

        def provider(*args, **kwargs):
            dynamic = specialized_signature.bind(*args, **kwargs)
            dynamic.apply_defaults()
            values = {**dynamic.arguments, **normalized}
            positional = []
            keywords = {}
            for name, parameter in original_signature.parameters.items():
                if name in values:
                    value = values[name]
                elif parameter.default is not Parameter.empty:
                    value = parameter.default
                else:
                    raise TypeError(
                        f"specialized definition is missing parameter {name!r}")
                if parameter.kind in {
                        Parameter.POSITIONAL_ONLY,
                        Parameter.POSITIONAL_OR_KEYWORD,
                }:
                    positional.append(value)
                elif parameter.kind is Parameter.KEYWORD_ONLY:
                    keywords[name] = value
            return original(*positional, **keywords)

        provider.__name__ = original.__name__
        provider.__qualname__ = original.__qualname__
        provider.__doc__ = original.__doc__
        provider.__module__ = original.__module__
        provider.__signature__ = specialized_signature
        hints = {
            name: value
            for name, value in self.type_hints.items()
            if name == "return" or name not in normalized
        }
        provider.__annotations__ = dict(hints)
        combined = {**self.specialization, **normalized}
        suffix = "__".join(f"{name}_{self._specialization_fragment(value)}"
                           for name, value in combined.items())
        return ProgramDefinition(
            provider,
            machine=self.machine,
            selection=self.selection,
            kind=self.kind,
            objective_kind=self.objective_kind,
            estimate_only=self.estimate_only,
            name=f"{(self.base or self).name}__{suffix}",
            metadata=self.metadata,
            type_hints=hints,
            specialization=combined,
            base=self.base or self,
            cudaq_kernel=self.cudaq_kernel,
        )

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)

    def __call__(self, *args, **kwargs):
        from cudaq.logical.programs.context import current_trace

        trace = current_trace()
        if trace is None:
            raise RuntimeError(
                f"{self.name} is a QLX definition, not an ordinary Python "
                "function; pass it to cudaq.logical.compile() or call it inside an "
                "active compatible QLX trace")
        return trace.call(self, args, kwargs)
