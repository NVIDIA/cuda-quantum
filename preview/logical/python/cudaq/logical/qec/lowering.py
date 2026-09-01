# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from hashlib import sha256
from inspect import signature
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from ..std import LogicalActionRef, LogicalInstrumentRef
from .._core.immutable import ImmutableValue
from ..programs.definition import ProgramDefinition
from ..gadgets import GadgetDefinition
from ..architecture.logical import CapabilityKey
from ..protocols.definition import ProtocolDefinition
from ..codes import (
    Code,
    Encoding,
    _materialized_code_identity,
)


@dataclass(frozen=True, slots=True)
class ActionSiteHandle:
    """Typed immutable view of one selected P1 action site."""

    symbol: str
    kind: str
    objective_family: str
    objective: str | None
    placements: tuple[str, ...]
    parameters: Mapping[str, Any]
    input_arity: int
    result_arity: int
    resource_kind: str | None = None
    resource_stream: str | None = None
    resource_stream_owner: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "placements", tuple(self.placements))
        object.__setattr__(self, "parameters",
                           MappingProxyType(dict(self.parameters)))
        for field_name in ("resource_kind", "resource_stream",
                           "resource_stream_owner"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise TypeError(
                    f"action-site {field_name.replace('_', ' ')} must be a "
                    "nonempty string or None")
        resource_present = (
            self.resource_kind is not None,
            self.resource_stream is not None,
            self.resource_stream_owner is not None,
        )
        if any(resource_present) and not all(resource_present):
            raise ValueError(
                "action-site resource provenance requires kind, stream, and "
                "stream owner together")


@dataclass(frozen=True, slots=True)
class QECCompilerContext:
    """Private typed context passed to a versioned QEC compiler provider."""

    device: Any
    lowering: "QECLowering"
    code: Code
    encoding: Encoding
    policy: Mapping[str, Any]
    dependencies: tuple[Any, ...]
    placements: tuple[Any, ...] = ()
    qec_selection: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", MappingProxyType(dict(self.policy)))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "placements", tuple(self.placements))


@dataclass(frozen=True, slots=True)
class GeneratedQECArtifact:
    """One generated definition plus inspectable specialization evidence."""

    definition: Any
    specialization: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "specialization",
            MappingProxyType(dict(self.specialization)),
        )


class QECCompiler:
    """Typed site implementation carried by one :class:`QECLowering` manifest."""

    __slots__ = ()


class _FunctionSiteCompiler(QECCompiler):
    """Compatibility adapter for the ordinary ``(site, context)`` surface."""

    __slots__ = ("_provider",)

    def __init__(self, provider: Callable[..., Any]) -> None:
        if not callable(provider):
            raise TypeError("QECLowering provider must be callable")
        if len(signature(provider).parameters) != 2:
            raise TypeError("QEC compiler provider must accept (site, context)")
        self._provider = provider

    @property
    def name(self) -> str:
        return self._provider.__name__

    @property
    def provider(self) -> Callable[..., Any]:
        return self._provider

    def compile_site(self, site, context):
        return self._provider(site, context)


def _manifest_type_name(value) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def _manifest_value(value, active=None):
    """Return a lossless JSON value for referenced QEC definition payloads."""

    active = set() if active is None else active
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(
                "QEC lowering referenced definitions require finite floats")
        return {"kind": "float", "value": repr(value)}
    if isinstance(value, type):
        return {
            "kind": "type",
            "name": f"{value.__module__}.{value.__qualname__}",
        }
    if isinstance(value, Enum):
        return {
            "kind": _manifest_type_name(value),
            "value": _manifest_value(value.value, active),
        }
    if isinstance(value, Code):
        return {
            "kind":
                "fabric.code",
            "payload":
                _manifest_value(_materialized_code_identity(value), active),
        }
    if isinstance(value, Encoding):
        return _manifest_encoding_identity(value, active)

    identity = id(value)
    if identity in active:
        raise ValueError(
            "QEC lowering referenced definition identity must be acyclic")
    if isinstance(value, Mapping):
        active.add(identity)
        try:
            entries = [(
                _manifest_value(key, active),
                _manifest_value(item, active),
            ) for key, item in value.items()]
        finally:
            active.remove(identity)
        entries.sort(key=lambda item: json.dumps(
            item[0],
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ))
        return {"kind": "mapping", "entries": entries}
    if isinstance(value, (tuple, list)):
        active.add(identity)
        try:
            items = [_manifest_value(item, active) for item in value]
        finally:
            active.remove(identity)
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": items,
        }
    if isinstance(value, (set, frozenset)):
        active.add(identity)
        try:
            items = [_manifest_value(item, active) for item in value]
        finally:
            active.remove(identity)
        items.sort(key=lambda item: json.dumps(
            item,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ))
        return {"kind": "set", "items": items}
    if is_dataclass(value) and not isinstance(value, type):
        active.add(identity)
        try:
            members = {
                field.name: _manifest_value(getattr(value, field.name), active)
                for field in fields(value)
            }
        finally:
            active.remove(identity)
        return {"kind": _manifest_type_name(value), "fields": members}

    slots = tuple(
        slot for kind in type(value).__mro__
        for slot in getattr(kind, "__slots__", ()) if isinstance(slot, str) and
        slot not in {"__dict__", "__weakref__", "_immutable_sealed"} and
        not slot.startswith("__"))
    if slots:
        active.add(identity)
        try:
            members = {
                slot: _manifest_value(getattr(value, slot), active)
                for slot in dict.fromkeys(slots)
                if hasattr(value, slot)
            }
        finally:
            active.remove(identity)
        return {"kind": _manifest_type_name(value), "fields": members}
    raise TypeError(
        "QEC lowering referenced definitions require canonical typed values")


def _manifest_encoding_identity(value: Encoding, active):
    identity = id(value)
    if identity in active:
        raise ValueError(
            "QEC lowering encoding identity must be an acyclic definition")
    active.add(identity)
    try:
        payload = {
            "code":
                _manifest_value(value.code, active),
            "profile":
                _manifest_value(value.profile, active),
            "block":
                value.block,
            "logical_ports":
                _manifest_value(value.logical_ports, active),
            "logical_port_indices":
                _manifest_value(value.logical_port_indices, active),
            "layout":
                _manifest_value(value.layout, active),
            "hierarchy":
                _manifest_value(value.hierarchy, active),
            "epoch_schema":
                _manifest_value(value.epoch_schema, active),
            "metadata":
                _manifest_value(value.metadata, active),
        }
    finally:
        active.remove(identity)
    return {"kind": "fabric.encoding", "payload": payload}


def _manifest_reference(value):
    """Return one lossless, type-qualified authored manifest reference."""

    if isinstance(value, Encoding):
        return {
            "kind": "encoding",
            "name": value.name,
            "code": value.code.name,
            "definition": _manifest_encoding_identity(value, set()),
        }
    if isinstance(value, Code):
        return {
            "kind": "code",
            "name": value.name,
            "definition": _manifest_value(value),
        }
    if isinstance(value, str):
        return {"kind": "symbol", "name": value}
    if isinstance(value, (LogicalActionRef, LogicalInstrumentRef)):
        return {"kind": _manifest_type_name(value), "name": value.name}
    if isinstance(value,
                  (ProgramDefinition, GadgetDefinition, ProtocolDefinition)):
        return {"kind": _manifest_type_name(value), "name": value.name}
    if isinstance(value, CapabilityKey):
        return {"kind": _manifest_type_name(value), "name": value.key}
    raise TypeError(
        "QEC lowering manifest references require a typed CUDA-Q Logical program, "
        "gadget, protocol, "
        "capability, logical objective, or symbol string")


def _manifest_objective(value):
    """Return one exact typed objective identity or fail before hashing."""

    if value is None:
        return None
    if isinstance(value, (LogicalActionRef, LogicalInstrumentRef)):
        return _manifest_reference(value)
    if isinstance(value, ProgramDefinition) and value.kind == "objective":
        return _manifest_reference(value)
    raise TypeError(
        "QECLowering objective must be a typed logical action, instrument, "
        "or @cudaq.logical.objective definition")


def _manifest_mapping(value, *, field: str) -> Mapping[str, str]:
    """Normalize one string-valued manifest dictionary without collisions."""

    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError(f"QECLowering {field} must be a mapping")
    normalized = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(
                f"QECLowering {field} keys must be nonempty strings")
        if isinstance(item, str):
            text = item
        elif isinstance(item, bool):
            text = str(item)
        elif isinstance(item, int):
            text = str(item)
        elif isinstance(item, float) and math.isfinite(item):
            text = str(item)
        else:
            raise TypeError(
                f"QECLowering {field} values must be finite scalar values")
        normalized[key] = text
    return MappingProxyType(normalized)


def _qec_lowering_manifest_sha256(definition) -> str:
    """Commit the stable semantic identity of one QEC-lowering manifest."""

    payload = {
        "name":
            definition.name,
        "objective_family":
            definition.objective_family,
        "objective":
            _manifest_objective(definition.objective),
        "codes": [_manifest_reference(value) for value in definition.codes],
        "requirements": [
            _manifest_reference(value) for value in definition.requires
        ],
        "dependencies": [
            _manifest_reference(value) for value in definition.dependencies
        ],
        "compiler_plugin":
            definition.plugin,
        "compiler_symbol":
            definition.compiler.name,
        "compiler_version":
            definition.version,
        "policy_schema":
            dict(definition.policy_schema),
        "metadata":
            dict(definition.metadata),
        "input_stage":
            definition.input_stage.value,
        "output_stage":
            definition.output_stage.value,
        "provides_facets": [
            value.value for value in definition.provides_facets
        ],
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


class QECLowering(ImmutableValue):
    """Immutable module-linked P1-to-P2 QEC compiler manifest."""

    __slots__ = (
        "compiler",
        "objective_family",
        "objective",
        "codes",
        "requires",
        "dependencies",
        "plugin",
        "version",
        "name",
        "policy_schema",
        "metadata",
        "profile",
        "input_stage",
        "output_stage",
        "provides_facets",
        "manifest_sha256",
    )

    def __init__(
        self,
        provider: Callable[..., Any] | None = None,
        *,
        compiler: QECCompiler | None = None,
        objective_family: str,
        objective=None,
        codes: Iterable[Code | Encoding] = (),
        requires: Iterable[Any] = (),
        dependencies: Iterable[Any] = (),
        plugin: str,
        version: str,
        name: str | None = None,
        policy_schema: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if provider is not None and compiler is not None:
            raise TypeError(
                "QECLowering accepts provider= or compiler=, not both")
        if compiler is None:
            compiler = _FunctionSiteCompiler(provider)
        elif not isinstance(compiler, QECCompiler):
            raise TypeError("QECLowering compiler must be a typed QECCompiler")
        if not isinstance(objective_family, str) or not objective_family:
            raise ValueError("QECLowering objective_family must be nonempty")
        if (not isinstance(plugin, str) or not plugin or
                not isinstance(version, str) or not version):
            raise ValueError("QECLowering requires versioned plugin provenance")
        normalized_codes = tuple(codes)
        if any(not isinstance(code, (Code, Encoding))
               for code in normalized_codes):
            raise TypeError(
                "QECLowering codes must contain Code or Encoding values")
        compiler_name = getattr(compiler, "name", None)
        if not isinstance(compiler_name, str) or not compiler_name:
            raise TypeError("QECCompiler must expose a nonempty string name")
        compile_site = getattr(compiler, "compile_site", None)
        if not callable(compile_site):
            raise TypeError(
                "QECCompiler must implement compile_site(site, context)")
        if len(signature(compile_site).parameters) != 2:
            raise TypeError(
                "QECCompiler.compile_site must accept (site, context)")
        _manifest_objective(objective)
        if name is not None and (not isinstance(name, str) or not name):
            raise TypeError("QECLowering name must be a nonempty string")
        self.compiler = compiler
        self.objective_family = objective_family
        self.objective = objective
        self.codes = normalized_codes
        self.requires = tuple(requires)
        self.dependencies = tuple(dependencies)
        self.plugin = plugin
        self.version = version
        self.name = name or compiler_name
        self.policy_schema = _manifest_mapping(policy_schema,
                                               field="policy_schema")
        self.metadata = _manifest_mapping(metadata, field="metadata")
        self.profile = "common"
        from ..stages import (
            P1,
            P2,
            QEC_REALIZATION,
        )

        self.input_stage = P1
        self.output_stage = P2
        self.provides_facets = (QEC_REALIZATION,)
        self.manifest_sha256 = _qec_lowering_manifest_sha256(self)
        self._seal()

    @property
    def provider(self):
        """Legacy view of an ordinary site provider."""

        provider = getattr(self.compiler, "provider", None)
        if provider is None:
            raise AttributeError(f"QECLowering {self.name!r} has no provider")
        return provider

    def compile_site(self, site, context):
        compile_site = getattr(self.compiler, "compile_site", None)
        if not callable(compile_site):
            raise TypeError(
                f"QECLowering {self.name!r} is not a site-scoped compiler")
        return compile_site(site, context)

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)


def _inferred_objective_family(objective) -> str:
    """Infer the selector family of one exact typed logical objective."""

    if isinstance(objective, ProgramDefinition):
        if (objective.kind == "objective" and
                objective.objective_kind in {"action", "instrument"}):
            return objective.objective_kind
        raise TypeError("qec_lowering requires objective_family= for an "
                        "@cudaq.logical.objective(kind='auto') definition")
    if not isinstance(
            objective,
        (LogicalActionRef, LogicalInstrumentRef),
    ):
        raise TypeError(
            "qec_lowering requires objective_family= when objective= is not "
            "a typed logical objective")
    name = objective.name
    explicit_family = getattr(objective, "family", None)
    if explicit_family:
        return str(explicit_family)
    normalized = name.removeprefix("qlx_standard_")
    if normalized == "mpp" or normalized.startswith("mpp_"):
        return "pauli_product_measurement"
    if normalized == "pauli_rotation" or normalized.startswith(
            "pauli_rotation_"):
        return "pauli_product_rotation"
    if normalized.startswith("measure_"):
        return "logical_measurement"
    if normalized.startswith("prepare_"):
        return "logical_preparation"
    if isinstance(objective, LogicalInstrumentRef):
        return "instrument"
    if isinstance(objective, LogicalActionRef):
        return "action"
    raise AssertionError(f"unhandled typed QEC objective {objective!r}")


def qec_lowering(
        fn=None,
        *,
        objective_family=None,
        objective=None,
        codes=(),
        requires=(),
        dependencies=(),
        plugin=None,
        version=None,
        name=None,
        policy_schema=None,
):

    def decorate(provider):
        family = (_inferred_objective_family(objective)
                  if objective_family is None else objective_family)
        provider_plugin = plugin
        if provider_plugin is None:
            provider_plugin = getattr(provider, "__module__", None) or "linked"
        provider_version = "linked" if version is None else version
        return QECLowering(
            provider,
            objective_family=family,
            objective=objective,
            codes=codes,
            requires=requires,
            dependencies=dependencies,
            plugin=provider_plugin,
            version=provider_version,
            name=name,
            policy_schema=policy_schema,
        )

    return decorate(fn) if fn is not None else decorate
