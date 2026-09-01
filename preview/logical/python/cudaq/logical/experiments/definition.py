# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from ..programs.definition import DefinitionHandle
from ..stages import (
    Stage,
    normalize_facets,
)


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _semantic_value(value):
    """Convert one semantic binding to deterministic bundle data."""

    from ..algebra.angle import Angle

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Angle):
        numerator, denominator = value.pi_fraction
        return {
            "kind": "rational_pi_angle",
            "numerator": numerator,
            "denominator": denominator,
        }
    if isinstance(value, Enum):
        return _semantic_value(value.value)
    if isinstance(value, DefinitionHandle):
        return {
            "symbol": value.symbol,
            "kind": (value.kind if isinstance(value.kind, str) else
                     f"{value.kind.__module__}.{value.kind.__qualname__}"),
            "profile": value.profile,
        }
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_value(item)
            for key, item in sorted(value.items(),
                                    key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_semantic_value(item) for item in value]
    semantic_manifest = getattr(value, "_qlx_semantic_manifest", None)
    if callable(semantic_manifest):
        return _semantic_value(semantic_manifest())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _semantic_value(getattr(value, field.name))
            for field in fields(value)
        }
    manifest = getattr(value, "manifest", None)
    if callable(manifest):
        return _semantic_value(manifest())
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return {
            "name": name,
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
        }
    raise TypeError("experiment binding is not replayable: "
                    f"{type(value).__module__}.{type(value).__qualname__}")


@dataclass(frozen=True, slots=True)
class Experiment:
    """One immutable CUDA-Q Logical compilation point.

    Before compilation ``root`` is a normal CUDA-Q Logical definition or build. A
    successful build exposes the same class with ``root`` rebound to its typed
    :class:`DefinitionHandle`, plus the verified stage/facets, exact pass
    recipe, and linked symbol closure.
    """

    root: Any
    device: Any = None
    device_provenance: str | None = None
    placement: Any = ()
    policy: Any = None
    parameters: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}))
    objective: Any = None
    profile: str | None = None
    stage: Stage | None = None
    facets: tuple[Any, ...] = ()
    pass_recipe: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] = ()
    closure: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.device_provenance not in (None, "entry_gadget"):
            raise ValueError(
                "experiment device_provenance must be None or 'entry_gadget'")
        if self.device_provenance is not None and self.device is None:
            raise ValueError(
                "experiment device_provenance requires a bound device")
        if self.placement is None:
            object.__setattr__(self, "placement", ())
        elif isinstance(self.placement, (tuple, list)):
            object.__setattr__(self, "placement", tuple(self.placement))
        object.__setattr__(self, "parameters", _mapping(self.parameters))
        object.__setattr__(self, "pass_recipe", tuple(self.pass_recipe or ()))
        object.__setattr__(self, "closure", tuple(self.closure or ()))
        if self.stage is not None and not isinstance(self.stage, Stage):
            object.__setattr__(self, "stage", Stage(str(self.stage)))
        object.__setattr__(self, "facets", normalize_facets(self.facets))

    @property
    def compiled(self) -> bool:
        return isinstance(self.root,
                          DefinitionHandle) and self.stage is not None

    def bind(
        self,
        *,
        root: DefinitionHandle,
        profile: str,
        stage,
        facets,
        pipeline,
        closure,
        device=None,
        placement=None,
        objective=None,
    ) -> "Experiment":
        recipe = ()
        if pipeline is not None:
            recipe = tuple((
                item.name,
                tuple((str(key), _semantic_value(value))
                      for key, value in item.options),
            )
                           for item in pipeline.passes)
        return Experiment(
            root=root,
            device=self.device if device is None else device,
            device_provenance=self.device_provenance,
            placement=self.placement if placement is None else placement,
            policy=self.policy,
            parameters=self.parameters,
            objective=self.objective if objective is None else objective,
            profile=str(profile),
            stage=stage,
            facets=tuple(facets),
            pass_recipe=recipe,
            closure=tuple(dict.fromkeys(str(symbol) for symbol in closure)),
        )

    def bindings(self) -> dict[str, Any]:
        """Return canonical JSON/MLIR-ready semantic bindings."""

        values = {
            "device": self.device,
            "device_provenance": self.device_provenance,
            "placement": self.placement,
            "policy": self.policy,
            "parameters": self.parameters,
            "objective": self.objective,
        }
        return {
            key: _semantic_value(value)
            for key, value in values.items()
            if value is not None and value != () and value != {}
        }

    def to_bundle(self) -> dict[str, Any]:
        if not self.compiled:
            raise ValueError("only a compiled Experiment can be serialized")
        return {
            "root": _semantic_value(self.root),
            "profile": self.profile,
            "stage": _semantic_value(self.stage),
            "facets": _semantic_value(self.facets),
            "pass_recipe": _semantic_value(self.pass_recipe),
            "closure": list(self.closure),
            "bindings": self.bindings(),
        }

    @classmethod
    def from_bundle(cls, value: Mapping[str, Any]) -> "Experiment":
        root = value["root"]
        bindings = dict(value.get("bindings", {}))
        for name, binding in tuple(bindings.items()):
            if (isinstance(binding, Mapping) and
                    set(binding) == {"symbol", "kind", "profile"}):
                bindings[name] = DefinitionHandle(
                    symbol=binding["symbol"],
                    kind=binding["kind"],
                    profile=binding["profile"],
                )
        return cls(
            root=DefinitionHandle(
                symbol=root["symbol"],
                kind=root["kind"],
                profile=root["profile"],
            ),
            profile=value["profile"],
            stage=value.get("stage"),
            facets=tuple(value.get("facets", ())),
            pass_recipe=tuple(
                (item[0], tuple((entry[0], entry[1])
                                for entry in item[1]))
                for item in value.get("pass_recipe", ())),
            closure=tuple(value.get("closure", ())),
            **bindings,
        )


__all__ = ["Experiment"]
