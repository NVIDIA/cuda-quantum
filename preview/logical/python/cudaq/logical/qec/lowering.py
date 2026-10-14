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
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from ..std import LogicalActionRef, LogicalInstrumentRef, ResourceKind
from cudaq.logical._core.immutable import ImmutableValue
from cudaq.logical.programs.definition import ProgramDefinition
from cudaq.logical.gadgets import GadgetDefinition
from cudaq.logical.architecture.logical import CapabilityKey
from cudaq.logical.protocols.definition import ProtocolDefinition
from cudaq.logical.codes import (
    Code,
    Encoding,
    QECSelectionWitness,
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
    channel: str | None = None
    channel_capability: CapabilityKey | None = None
    endpoints: tuple[str, ...] = ()
    direction: str | None = None
    resource_kind: str | None = None
    resource_stream: str | None = None
    resource_stream_owner: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "placements", tuple(self.placements))
        object.__setattr__(self, "parameters",
                           MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "endpoints", tuple(self.endpoints))
        present = (
            self.channel is not None,
            self.channel_capability is not None,
            bool(self.endpoints),
            self.direction is not None,
        )
        if any(present) and not all(present):
            raise ValueError(
                "action-site communication obligation requires channel, "
                "channel_capability, endpoints, and direction together")
        if self.channel_capability is not None and not isinstance(
                self.channel_capability, CapabilityKey):
            raise TypeError(
                "action-site communication capability must be a CapabilityKey")
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
    physical: Any = None
    channel: Any = None
    qec_selection: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", MappingProxyType(dict(self.policy)))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "placements", tuple(self.placements))

    def qec_region_for(self, placement: str):
        """Return the selected device QEC region for one P1 placement."""

        if not isinstance(placement, str) or not placement:
            raise TypeError("qec_region_for requires a P1 placement name")
        selected = next(
            (item for item in self.placements
             if getattr(item, "placement", None) == placement),
            None,
        )
        if selected is None:
            raise ValueError(
                f"placement {placement!r} is not part of this QEC action site")
        space = getattr(selected, "space", None)
        regions = tuple(binding.qec_region
                        for binding in self.device.logical_to_qec
                        if binding.logical_region.name == space)
        if len(regions) != 1:
            raise ValueError(
                f"placement {placement!r} has no unique selected QEC region")
        return regions[0]

    def qec_block_for(self, placement: str) -> str:
        """Return the compiler-selected encoded block for one placement."""

        if not isinstance(placement, str) or not placement:
            raise TypeError("qec_block_for requires a P1 placement name")
        selected = next(
            (item for item in self.placements
             if getattr(item, "placement", None) == placement),
            None,
        )
        block = None if selected is None else getattr(selected, "block", None)
        if not isinstance(block, str) or not block:
            raise ValueError(
                f"placement {placement!r} has no selected encoded block")
        return block


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
    """Typed implementation carried by one :class:`QECLowering` manifest.

    Concrete compiler type determines its compilation scope.  Ordinary
    callable providers are normalized to an internal site compiler, while a
    network compiler such as ``LatticeSurgeryCompiler`` implements the whole
    selected P1 network contract.  No public string-valued scope selector is
    involved.
    """

    __slots__ = ()


class QECNetworkCompiler(QECCompiler):
    """Typed compiler for a closed set of placed QEC action sites.

    Core QLX owns source traversal, region boundaries, stage transitions, and
    :class:`Build` construction.  A network compiler owns compatibility,
    complete spatial/temporal planning, and typed protocol emission for the
    regions selected through its :class:`QECLowering` manifests.

    The concrete request, plan, and emission-context types live in
    :mod:`cudaq.logical.qec.lattice_surgery` so providers can implement this
    contract without
    importing compiler internals or MLIR bindings.
    """

    __slots__ = ()

    @property
    def key(self) -> str:
        """Canonical ``plugin:name@version`` planning-provider identity."""

        raise NotImplementedError


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
    if isinstance(value, ResourceKind):
        return {
            "kind": _manifest_type_name(value),
            "name": value.name,
            "payload_roles": list(value.payload_roles),
        }
    if isinstance(value,
                  (ProgramDefinition, GadgetDefinition, ProtocolDefinition)):
        return {"kind": _manifest_type_name(value), "name": value.name}
    if isinstance(value, CapabilityKey):
        return {"kind": _manifest_type_name(value), "name": value.key}
    raise TypeError(
        "QEC lowering manifest references require a typed QLX program, "
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
        "consumes": [
            _manifest_reference(value) for value in definition.consumes
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
        "consumes",
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
        consumes: Iterable[ResourceKind] = (),
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
        normalized_consumes = tuple(consumes)
        if any(not isinstance(kind, ResourceKind)
               for kind in normalized_consumes):
            raise TypeError(
                "QECLowering consumes must contain ResourceKind values")
        if len({kind.name for kind in normalized_consumes
               }) != len(normalized_consumes):
            raise ValueError(
                "QECLowering consumes must not repeat a resource kind")
        compiler_name = getattr(compiler, "name", None)
        if not isinstance(compiler_name, str) or not compiler_name:
            raise TypeError("QECCompiler must expose a nonempty string name")
        if isinstance(compiler, QECNetworkCompiler):
            expected_key = f"{plugin}:{compiler_name}@{version}"
            if compiler.key != expected_key:
                raise ValueError("QECNetworkCompiler key must be canonical "
                                 f"{expected_key!r}")
        _manifest_objective(objective)
        if name is not None and (not isinstance(name, str) or not name):
            raise TypeError("QECLowering name must be a nonempty string")
        self.compiler = compiler
        self.objective_family = objective_family
        self.objective = objective
        self.codes = normalized_codes
        self.requires = tuple(requires)
        self.dependencies = tuple(dependencies)
        self.consumes = normalized_consumes
        self.plugin = plugin
        self.version = version
        self.name = name or compiler_name
        self.policy_schema = _manifest_mapping(policy_schema,
                                               field="policy_schema")
        self.metadata = _manifest_mapping(metadata, field="metadata")
        self.profile = "common"
        from cudaq.logical.stages import (
            P1,
            P2,
            PROTOCOL_NETWORK,
            QEC_REALIZATION,
        )

        self.input_stage = P1
        self.output_stage = P2
        self.provides_facets = (QEC_REALIZATION, PROTOCOL_NETWORK)
        self.manifest_sha256 = _qec_lowering_manifest_sha256(self)
        self._seal()

    @property
    def provider(self):
        """Legacy view of an ordinary site provider.

        Network compilers intentionally have no callable site provider.  New
        compiler code should inspect ``compiler`` and call ``compile_site``
        only for a site compiler.
        """

        provider = getattr(self.compiler, "provider", None)
        if provider is None:
            raise AttributeError(
                f"QECLowering {self.name!r} is implemented by a network compiler"
            )
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


@dataclass(frozen=True, slots=True)
class QECNetworkContext:
    """Compiler-owned P1 provenance supplied to a network materializer.

    This context is transient transaction input, not another serialized plan.
    The plan binds the exact provider; this value supplies the authenticated P1
    artifact and selected lowering witness that the generated P2 root must
    retain in its ordinary QLX provenance chain. Canonical whole-network
    lowering also commits ``selection.network_manifest_sha256``; the explicit
    operation-only compatibility route leaves that marker unset.
    """

    source: Any
    lowering: QECLowering
    selection: QECSelectionWitness
    selection_digest: str
    device: Any = None
    policy: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from ..compiler.build import (
            Build,
            _qec_selection_sha256,
            _verify_qec_network_source,
        )

        if not isinstance(self.source, Build) or self.source.profile != "p1":
            raise TypeError("QEC network context source must be a P1 Build")
        if not isinstance(self.lowering, QECLowering):
            raise TypeError("QEC network context requires a QECLowering")
        if not isinstance(self.selection, QECSelectionWitness):
            raise TypeError(
                "QEC network context selection must be a QECSelectionWitness")
        object.__setattr__(self, "policy", MappingProxyType(dict(self.policy)))
        if self.selection.input_p1 != self.source.root.symbol:
            raise ValueError(
                "QEC network selection must reference the exact P1 root")
        if not self.selection.actions or any(
                action.selected != self.lowering.name or action.provider != self
                .lowering.plugin or action.version != self.lowering.version or
                action.manifest_sha256 != self.lowering.manifest_sha256
                for action in self.selection.actions):
            raise ValueError(
                "QEC network selection must commit the exact lowering manifest")
        if self.selection.network_manifest_sha256 not in {
                None,
                self.lowering.manifest_sha256,
        }:
            raise ValueError(
                "QEC network witness identifies a different manifest")
        prefix = "sha256:"
        payload = (self.selection_digest[len(prefix):]
                   if isinstance(self.selection_digest, str) and
                   self.selection_digest.startswith(prefix) else "")
        if len(payload) != 64 or any(
                value not in "0123456789abcdef" for value in payload):
            raise ValueError(
                "QEC network selection digest must be canonical sha256 evidence"
            )
        if self.selection_digest != _qec_selection_sha256(self.selection):
            raise ValueError(
                "QEC network selection digest differs from its witness")
        _verify_qec_network_source(
            self.source,
            self.lowering,
            self.selection,
        )


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
        consumes=(),
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
            consumes=consumes,
            plugin=provider_plugin,
            version=provider_version,
            name=name,
            policy_schema=policy_schema,
        )

    return decorate(fn) if fn is not None else decorate
