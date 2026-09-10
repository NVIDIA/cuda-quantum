# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import math
import os
import time

import cudaq.mlir.ir as mlir_ir

from ..errors import (
    PhysicalProjectionCapacityError,
    PlacementInfeasible,
)
from cudaq.logical.programs.definition import DefinitionHandle
from cudaq.logical.devices.definition import Device
from cudaq.logical.stages import P3
from cudaq.logical.architecture.physical_definition import (
    QuantumProcess,
    PhysicalAction,
)
from cudaq.logical.architecture.capabilities import (
    NATIVE_PAULI_PRODUCT_ROTATION,
    PhysicalCapability,
)
from ..architecture.physical_instruments import MX, MZ
from .build import Build, EvidenceRecord, _attr_text, _symbol_path
from .context import CompilationContext
from .legalize import NATIVE_DECOMPOSITIONS, SEMANTIC_ACTIONS

_COMMUNICATION_CALL_ATTRIBUTES = (
    "channel",
    "channel_capability",
    "endpoints",
    "action_site",
    "generated_by",
)


class PhysicalProjector:
    """Mixin contract for a device-contributed P2-to-P3 projector.

    Projectors are separate from P2 materializers even when one provider
    implements both capabilities on the same immutable object.  The distinct
    method names prevent planning compatibility from being confused with
    physical-projection compatibility.
    """

    __slots__ = ()

    @property
    def key(self) -> str:
        raise NotImplementedError

    @property
    def projection_pipeline(self):
        raise NotImplementedError

    def accepts_projection(self, source: Build, device: Device) -> bool:
        raise NotImplementedError

    def projection_architecture_digest(self, device: Device) -> str:
        raise NotImplementedError

    def accepts_projection_pipeline(self, pipeline) -> bool:
        return pipeline == self.projection_pipeline

    def projection_architecture(self, projection, device: Device):
        """Return the physical-machine view for canonical network emission."""

        raise NotImplementedError

    def emit_projection(
        self,
        projection,
        device: Device,
        *,
        builder,
        pipeline,
        experiment=None,
    ) -> "PhysicalProjectionEmission":
        """Emit into a core-prepared builder for a canonical network plan."""

        raise NotImplementedError

    def project_p3(
        self,
        source: Build,
        device: Device,
        *,
        pipeline,
        experiment=None,
    ) -> Build:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PhysicalProjectionEmission:
    """Unfinished provider emission finalized into a P3 Build by core QLX."""

    context: object
    module: object
    root: DefinitionHandle
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(
                self.root,
                DefinitionHandle) or self.root.kind != "physical_graph":
            raise TypeError(
                "physical projection emission requires a physical-graph handle")
        evidence = tuple(self.evidence)
        if any(not isinstance(value, EvidenceRecord) for value in evidence):
            raise TypeError(
                "physical projection emission evidence must be EvidenceRecord values"
            )
        object.__setattr__(self, "evidence", evidence)


class PhysicalProjectionBuilder:
    """Core-prepared physical graph for one canonical network projection.

    QLX imports the exact P2 closure, retains its source/device identities, and
    commits the canonical request/plan before provider emission begins.  The
    provider authors only physical events and provider-private sidecars through
    :attr:`events`, seals the graph, and returns an unfinished emission.
    """

    __slots__ = (
        "_device",
        "_events",
        "_pipeline",
        "_projection",
        "_root",
        "_source_handle",
        "_device_handle",
    )

    def __init__(
        self,
        name: str,
        *,
        architecture,
        projection,
        device: Device,
        pipeline,
    ) -> None:
        from ..qec import lattice_surgery
        from cudaq.logical.compiler.authoring import PhysicalBuilder

        if not isinstance(projection, lattice_surgery.QECNetworkProjection):
            raise TypeError(
                "PhysicalProjectionBuilder requires QECNetworkProjection")
        if not isinstance(device, Device):
            raise TypeError("PhysicalProjectionBuilder requires a Device")
        if not (projection.device is device or
                projection.device._has_same_static_stack(device)):
            raise ValueError(
                "physical projection device differs from the network plan")
        events = PhysicalBuilder(name, architecture=architecture)
        source = projection.source
        source_handle = events.transaction._import_direct_snapshot(
            object(), source)
        imported_root = events.transaction.find_symbol(source_handle.symbol,
                                                       "fabric.protocol")
        device_handle = None
        if imported_root is not None and "metadata" in imported_root.attributes:
            metadata = imported_root.attributes["metadata"]
            if "device" in metadata:
                device_symbol = _text(metadata["device"])
                if events.transaction.find_symbol(device_symbol,
                                                  "qlx.device") is not None:
                    device_handle = DefinitionHandle(device_symbol, "device",
                                                     "device")
        if device_handle is None:
            device_handle = events.transaction.materialize(device)
        with events.context:
            attributes = events._backend.operation.attributes
            attributes["source_protocol"] = mlir_ir.FlatSymbolRefAttr.get(
                source_handle.symbol, context=events.context)
            attributes["qlx.qec_network_request_sha256"] = (
                mlir_ir.StringAttr.get(projection.request.digest,
                                       context=events.context))
            attributes["qlx.qec_network_plan_sha256"] = mlir_ir.StringAttr.get(
                projection.plan.digest, context=events.context)
            attributes["qlx.qec_network_artifact_sha256"] = (
                mlir_ir.StringAttr.get(projection.plan.artifact.sha256,
                                       context=events.context))
            attributes["qlx.qec_network_projector"] = mlir_ir.StringAttr.get(
                projection.plan.required_projector_key, context=events.context)
            attributes["qlx.qec_network_projector_pipeline_sha256"] = (
                mlir_ir.StringAttr.get(
                    projection.plan.required_projector_pipeline_sha256,
                    context=events.context,
                ))
        self._device = device
        self._events = events
        self._pipeline = pipeline
        self._projection = projection
        self._root = None
        self._source_handle = source_handle
        self._device_handle = device_handle

    @property
    def events(self):
        """Advanced physical-event builder for provider-specific emission."""

        return self._events

    @property
    def source_handle(self) -> DefinitionHandle:
        return self._source_handle

    @property
    def device_handle(self) -> DefinitionHandle:
        return self._device_handle

    def seal(self, *outputs) -> DefinitionHandle:
        if self._root is not None:
            raise RuntimeError("physical projection graph is already sealed")
        self._root = self._events.seal(
            *outputs,
            source=self._projection.source,
            source_imported=True,
        )
        return self._root

    def finish(self, *, evidence=()) -> PhysicalProjectionEmission:
        if self._root is None:
            raise RuntimeError(
                "physical projection graph must be sealed before finish()")
        return PhysicalProjectionEmission(
            context=self._events.context,
            module=self._events.module,
            root=self._root,
            evidence=tuple(evidence),
        )


def _text(attribute) -> str:
    value = getattr(attribute, "value", None)
    return str(value if value is not None else attribute).strip('"').lstrip("@")


def _symbol(operation) -> str | None:
    try:
        return _text(operation.attributes["sym_name"])
    except KeyError:
        return None


def _code_name(type_) -> str | None:
    text = str(type_)
    prefix = "!fabric.patch<@"
    if not text.startswith(prefix):
        return None
    return text[len(prefix):-1].split(",", 1)[0].strip().lstrip("@")


def _patch_references(type_):
    text = str(type_)
    prefix = "!fabric.patch<"
    if not text.startswith(prefix):
        return ()
    return tuple(
        value.strip().lstrip("@") for value in text[len(prefix):-1].split(","))


def _fabric_resource_kind(type_):
    text = str(type_)
    prefix = "!fabric.resource<@"
    return text[len(prefix):-1] if text.startswith(prefix) else None


def _fabric_event_resource_kind(type_):
    text = str(type_)
    prefix = '!event.handle<!fabric.resource<@'
    if not text.startswith(prefix):
        return None
    return text[len(prefix):].split(">", 1)[0]


def _partition(attribute) -> str:
    text = str(attribute)
    return text[text.find("<") + 1:text.rfind(">")]


def _capability_key(attribute) -> str:
    text = str(attribute)
    prefix = "#lvm.capability<"
    if text.startswith(prefix) and text.endswith(">"):
        return text[len(prefix):-1].strip('"')
    return _text(attribute)


def _pairs(attribute, control_width, target_width):
    """Parse Fabric's canonical bounded matching."""

    text = _text(attribute)
    if text == "index":
        return tuple(
            (index, index) for index in range(min(control_width, target_width)))
    if ":" in text:
        return tuple(
            tuple(int(value.strip())
                  for value in entry.split(":", 1))
            for entry in text.split(","))
    return tuple(ast.literal_eval(text))


def _pair_text(pairs) -> str:
    return ",".join(f"{control}:{target}" for control, target in pairs)


def _i64(context, value):
    return mlir_ir.IntegerAttr.get(
        mlir_ir.IntegerType.get_signless(64, context=context), int(value))


def _f64(context, value):
    with mlir_ir.Location.unknown(context):
        return mlir_ir.FloatAttr.get(mlir_ir.F64Type.get(context=context),
                                     float(value))


def _empty_array_attr(context):
    """Create an empty array across CUDA-Q's supported MLIR bindings."""

    # CUDA-Q's packaged LLVM 22 binding requires a concrete list. Newer local
    # LLVM bindings accept any Sequence, which let tuple call sites escape CI.
    return mlir_ir.ArrayAttr.get([], context=context)


def _string_dict(context, values):
    if not values:
        return None
    return mlir_ir.DictAttr.get(
        {
            str(key): mlir_ir.StringAttr.get(str(value), context=context)
            for key, value in values.items()
        },
        context=context,
    )


def _f64_dict(context, values):
    return mlir_ir.DictAttr.get(
        {
            str(key): _f64(context, value) for key, value in values.items()
        },
        context=context,
    )


@dataclass(slots=True)
class _Carrier:
    value: object
    resource: str
    resource_class: str
    region: str | None = None
    qec_region: str | None = None
    physical_binding: str | None = None

    def with_value(self, value, *, resource=None):
        return _Carrier(
            value,
            self.resource if resource is None else resource,
            self.resource_class,
            self.region,
            self.qec_region,
            self.physical_binding,
        )


@dataclass(frozen=True, slots=True)
class _RegionBinding:
    logical_region: str
    qec_region: object
    physical_binding: object


@dataclass(slots=True)
class _Patch:
    code: str
    encoding: str | None
    partitions: dict[str, list[_Carrier]]
    region: str | None
    patch_id: str
    slot: int | None = None
    patch_topology: str | None = None
    qec_region: str | None = None
    physical_binding: str | None = None
    live: bool = True

    def all(self):
        for partition in self.partitions.values():
            yield from partition

    def selection(self, partition):
        if partition == "all":
            return tuple((name, index, carrier)
                         for name, carriers in self.partitions.items()
                         for index, carrier in enumerate(carriers))
        try:
            return tuple(
                (partition, index, carrier)
                for index, carrier in enumerate(self.partitions[partition]))
        except KeyError as exc:
            raise PlacementInfeasible(
                f"patch code @{self.code} has no partition {partition!r}"
            ) from exc

    def clone(self):
        return _Patch(
            self.code,
            self.encoding,
            {
                name: list(values) for name, values in self.partitions.items()
            },
            self.region,
            self.patch_id,
            self.slot,
            self.patch_topology,
            self.qec_region,
            self.physical_binding,
            self.live,
        )


@dataclass(slots=True)
class _PatchBundle:
    """A folded hierarchy view over carriers still owned by one parent patch.

    Children borrow disjoint carrier slices from the parent.  ``remainder``
    retains parent-level syndrome carriers that do not belong to an inner
    child, so packing is a lossless ownership operation rather than a fresh
    allocation.
    """

    parent_code: str
    children: list[_Patch]
    remainder: dict[str, list[_Carrier]]
    region: str | None
    hierarchy: str
    slot_group: str
    live: bool = True


@dataclass(slots=True)
class _AllocationBinding:
    """One invocation-local patch lifetime bound to concrete carriers."""

    allocation: str
    resource_class: str
    resources: tuple[str, ...]
    indices: tuple[int, ...]
    acquire: str
    release: str | None = None
    after: tuple[str, ...] = ()
    slot: int | None = None
    slot_binding: str | None = None
    scope: str | None = None
    qec_region: str | None = None
    physical_binding: str | None = None


class _P2ToP3:
    """Project one selected Fabric call graph into a physical event graph.

    The projector deliberately expands only finite, selected P2 structure.  It
    never guesses an MPP decomposition, adaptive policy, hierarchy projection,
    or routing decision.  Those constructs must first be resolved by their
    owning compiler pass, which keeps P3 honest as a physical projection rather
    than a second logical compiler.
    """

    def __init__(self, source: Build, device: Device) -> None:
        if source.profile not in {"p2a", "p2n"}:
            raise ValueError(
                "physical projection requires a selected P2A/P2N Build")
        if not isinstance(device, Device):
            raise TypeError(
                "physical projection requires one concrete qlx device")
        if device.physical is None:
            raise ValueError(
                "physical projection requires a device with a PhysicalMachine")
        device.require_physical_completeness(context="physical projection")
        profile = os.getenv("QLX_PROFILE_P2_TO_P3") is not None

        def checkpoint(label, started):
            now = time.perf_counter()
            if profile:
                print(
                    f"p2-to-p3 prepare-{label} {now - started:.6f}s",
                    flush=True,
                )
            return now

        started = time.perf_counter()
        self.source = source
        self.device = device
        self.transaction = CompilationContext.replay(source)
        started = checkpoint("clone", started)
        self.context = self.transaction.context
        self.location = self.transaction.location
        self.module = self.transaction.module
        retained_top_level = len(tuple(self.module.body.operations))
        # ``CompilationContext.replay`` has already indexed every retained
        # symbol while opening the private clone.  Rewalking the paper-scale
        # P2 closure here used to rebuild the same 220k-entry table in Python
        # before projection.  Share the transaction-local index so symbols
        # materialized below also become visible without another scan.
        self.symbols = self.transaction._symbol_operations
        started = checkpoint("symbol-index", started)
        self.codes = self._read_codes()
        self.architecture = self._ensure_architecture()
        started = checkpoint("architecture", started)
        resource_requests, specialized_requests = (
            self._canonicalize_resource_requests())
        started = checkpoint("resource-requests", started)
        self._validate_scheduled_macro_homes(resource_requests)
        started = checkpoint("macro-homes", started)
        self._materialize_factory_models(resource_requests)
        started = checkpoint("factory-models", started)
        self._materialize_component_models()
        started = checkpoint("component-models", started)
        # The source Build already authenticates the retained P2 closure. The
        # private projection clone may add device-stack definitions and bind
        # resource-request placeholders to that device. Verify exactly those
        # increments now so the native projector can later verify only its new
        # P3 graph and sidecars instead of rescanning the unchanged P2 bodies.
        added_top_level = tuple(
            self.module.body.operations)[retained_top_level:]
        for view in added_top_level:
            if not view.operation.verify():
                raise ValueError(
                    "physical projection added an invalid device definition")
        for operation in specialized_requests:
            if not operation.verify():
                raise ValueError(
                    "physical projection produced an invalid device-bound "
                    "resource request")
        started = checkpoint("increment-verification", started)
        candidate = self.symbols.get(self.source.root.symbol)
        self.root_profile = (candidate if candidate is not None and
                             candidate.name == "fabric.gadget_profile" else
                             None)
        self.root = self._resolve_root()
        checkpoint("root", started)
        self.graph_symbol = self.transaction.unique_symbol(
            f"{_symbol(self.root)}_physical")
        self.record_projection_symbol = self.transaction.unique_symbol(
            f"{self.graph_symbol}_record_projection")
        self._event = 0
        self._sidecar = 0
        self._resource_offsets = {
            resource.name: 0 for resource in device.physical.resource_classes
        }
        self._allocated_indices = {
            resource.name: set() for resource in device.physical.resource_classes
        }
        self._released_index_events = {
            resource.name: {} for resource in device.physical.resource_classes
        }
        self._ever_allocated_indices = {
            resource.name: set() for resource in device.physical.resource_classes
        }
        self._patch_allocated_indices = {
            resource.name: set() for resource in device.physical.resource_classes
        }
        self._scratch = {}
        self._patch_instance = 0
        self._slot_allocations = {
            binding.name: set()
            for binding in device.qec_to_physical
            if binding.patch_topology is not None
        }
        self._ever_slot_allocations = {
            binding.name: set()
            for binding in device.qec_to_physical
            if binding.patch_topology is not None
        }
        patch_symbols = {
            _symbol(operation)
            for operation in self.transaction.walk()
            if operation.name == "phys.patch_topology"
        }
        self._patch_topology_symbols = {
            binding.name:
                next(symbol
                     for symbol in patch_symbols
                     if symbol == f"{binding.name}_patch_topology" or
                     symbol.startswith(f"{binding.name}_patch_topology_"))
            for binding in device.qec_to_physical
            if binding.patch_topology is not None
        }
        self._patch_nodes = []
        self._patch_interactions = []
        self._patch_assignments = []
        self._mapping_initial = []
        self._allocation_bindings = []
        self._allocation_by_patch = {}
        self._route_steps = []
        self._route = 0
        self._route_event_capture = None
        self._communication_stack = []
        self._encode_zero_stack = []
        self._exclusive_path = []
        self._exclusive_control = 0
        self._communication_bridges = 0
        self._legalizations = []
        self._legalized_actions: dict[str, tuple[tuple[str, tuple[int, ...]],
                                                 ...]] = {}
        self._resource_zones: dict[str, str] = {}
        self._zone_roles = {
            str(topology.parameters["role"]): topology.name
            for topology in device.physical.topologies
            if topology.kind == "zone" and "role" in topology.parameters
        }
        self._zone_routes = {
            (
                str(topology.parameters["source"]),
                str(topology.parameters["destination"]),
            ):
                topology
            for topology in device.physical.topologies
            if topology.kind == "shuttle" and "source" in topology.parameters
            and "destination" in topology.parameters
        }
        self._zoned_movements = 0
        self.patch_graph_symbol = self.transaction.unique_symbol(
            f"{_symbol(self.root)}_patch_graph")
        self._record_values = {}
        self._record_ids_by_value = {}
        self._record_projection = {}
        self._record_projection_repeats = {}
        self._repeat_context = []
        self._provenance_sidecars = []
        self._resource_indices = {}
        self._communication_reachability = {}
        self._call_instance = 0
        self._hierarchy_projection = 0
        self.graph = None
        self.block = None
        self.ip = None

    def _read_codes(self):
        codes = {}
        for name, operation in self.symbols.items():
            if operation.name != "fabric.code":
                continue
            raw = {
                str(named.name): int(named.attr)
                for named in operation.attributes["partitions"]
            }
            order = [key for key in ("data", "sx", "sz") if key in raw]
            order.extend(key for key in raw if key not in order)
            codes[name] = tuple((key, raw[key]) for key in order)
        return codes

    def _ensure_architecture(self):
        operation = self.transaction.find_symbol(self.device.name, "qlx.device")
        if operation is None:
            self.transaction.materialize(self.device)
            operation = self.transaction.find_symbol(self.device.name,
                                                     "qlx.device")
        if operation is None:
            raise ValueError(f"failed to link device @{self.device.name}")
        if "physical" not in operation.attributes:
            raise ValueError(
                f"device @{self.device.name} has no physical machine")
        physical = _text(operation.attributes["physical"])
        if self.transaction.find_symbol(physical, "phys.machine") is None:
            raise ValueError(
                f"device references missing physical machine @{physical}")
        self.operating_point = (_text(operation.attributes["operating_point"])
                                if "operating_point" in operation.attributes
                                else None)
        return physical

    def _materialize_factory_models(self, resource_requests):
        """Derive typed compact P3 factory models from device refinements."""

        profile = os.getenv("QLX_PROFILE_P2_TO_P3") is not None
        started = time.perf_counter()

        def checkpoint(label):
            nonlocal started
            now = time.perf_counter()
            if profile:
                print(
                    f"p2-to-p3 factory-models-{label} "
                    f"{now - started:.6f}s",
                    flush=True,
                )
            started = now

        requested = {
            _symbol_path(operation.attributes["stream"])
            for operation in resource_requests
        }
        checkpoint("requests")
        backed_requested = []
        for stream in self.device.logical.streams:
            if (stream.produced_by is None or stream.region is None or
                ((self.device.logical.name, stream.name) not in requested and
                 (stream.name,) not in requested)):
                continue
            resolved = self._binding_for_region(stream.region.name)
            if resolved.physical_binding.factory_model is not None:
                backed_requested.append((stream, resolved))
        checkpoint("bindings")
        if not backed_requested:
            self._factory_models = {}
            return

        if self.device.operating_point is None or self.operating_point is None:
            raise ValueError(
                "physical factory model requires a selected operating point")
        timing = self.device.operating_point.timing
        try:
            cycle_ns = float(timing["surface_cycle_ns"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("physical factory model requires finite positive "
                             "surface_cycle_ns") from error
        if not math.isfinite(cycle_ns) or cycle_ns <= 0.0:
            raise ValueError("physical factory model requires finite positive "
                             "surface_cycle_ns")

        models = {}
        for stream, resolved in backed_requested:
            path = (self.device.logical.name, stream.name)
            retained = self._region_symbol(
                (self.device.logical.name, stream.name), "lvm.stream")
            checkpoint("stream")
            if retained is None:
                raise ValueError(
                    f"factory stream @{stream.name} is not retained in P1")
            model = resolved.physical_binding.factory_model
            assert model is not None
            if stream.buffer_size != 1:
                raise ValueError(
                    f"factory stream @{stream.name} requires buffer_size=1 "
                    "for deterministic folded scheduling")
            lane_count = stream.region.capacity
            if lane_count is None or lane_count <= 0:
                raise ValueError(
                    f"factory stream @{stream.name} requires a positive "
                    "factory-region capacity")
            resources = resolved.physical_binding.resources
            if len(resources) != 1:
                raise ValueError(
                    f"factory stream @{stream.name} requires one dedicated "
                    "physical resource class")
            resource = resources[0]
            resource_physical_units = resource.count
            if resource.footprint is not None:
                resource_physical_units *= resource.footprint.units
            if resource_physical_units > 2**63 - 1:
                raise OverflowError(
                    f"factory resource class @{resource.name} physical "
                    "footprint exceeds signed 64-bit range")
            characterization = model.characterization
            if characterization is not None:
                source_identity = retained.attributes.get("producer_identity")
                source_semantics = retained.attributes.get(
                    "producer_semantics_sha256")
                if (source_identity is None or source_semantics is None or
                        _symbol_path(source_identity)[-1]
                        != characterization.source_provider or
                        _attr_text(source_semantics)
                        != characterization.source_provider_sha256):
                    raise ValueError(
                        f"factory stream @{stream.name} does not match "
                        "the compiled characterization's source producer")
                if characterization.resource_kind != stream.produces:
                    raise ValueError(
                        f"factory stream @{stream.name} and its compiled "
                        "characterization produce different resource kinds")
                unit_kind = (resource.kind if resource.footprint is None else
                             resource.footprint.unit_kind)
                if unit_kind != characterization.physical_unit_kind:
                    raise ValueError(
                        f"factory stream @{stream.name} uses physical unit "
                        f"{unit_kind!r}, but its characterization requires "
                        f"{characterization.physical_unit_kind!r}")
                if (characterization.timing_source
                        != self.device.operating_point.timing_source):
                    raise ValueError(
                        f"factory stream @{stream.name} timing source differs "
                        "from the compiled characterization")
                required_units = (lane_count * characterization.physical_units)
                if resource_physical_units < required_units:
                    raise ValueError(
                        f"factory stream @{stream.name} provides "
                        f"{resource_physical_units} {unit_kind}s, but its "
                        f"{lane_count} compiled lanes require at least "
                        f"{required_units}")
                for name, expected in characterization.timing_profile:
                    source_name = ("surface_cycle_ns"
                                   if name == "cycle_ns" else name)
                    try:
                        actual = float(timing[source_name])
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError(
                            f"factory stream @{stream.name} operating point "
                            f"lacks characterized timing {source_name!r}"
                        ) from error
                    if not math.isfinite(actual) or actual != expected:
                        raise ValueError(
                            f"factory stream @{stream.name} timing "
                            f"{source_name!r}={actual!r} differs from the "
                            f"compiled characterization value {expected!r}")
            producer = retained.attributes["produced_by"]
            digest = (retained.attributes["produced_by_sha256"]
                      if "produced_by_sha256" in retained.attributes else None)
            if digest is None:
                raise ValueError(
                    f"factory stream @{stream.name} lacks its producer digest")
            symbol = self.transaction.unique_symbol(
                f"{stream.name}_factory_model")
            with self.context:
                region = mlir_ir.SymbolRefAttr.get(
                    [self.device.logical.name, stream.region.name],
                    context=self.context,
                )
                stream_ref = mlir_ir.SymbolRefAttr.get(
                    [self.device.logical.name, stream.name],
                    context=self.context,
                )
                attributes = {
                    "sym_name":
                        mlir_ir.StringAttr.get(symbol, context=self.context),
                    "resource_kind":
                        mlir_ir.FlatSymbolRefAttr.get(stream.produces.name,
                                                      context=self.context),
                    "stream":
                        stream_ref,
                    "provider":
                        producer,
                    "provider_sha256":
                        digest,
                    "region":
                        region,
                    "qec_binding":
                        mlir_ir.FlatSymbolRefAttr.get(
                            resolved.physical_binding.name,
                            context=self.context,
                        ),
                    "physical_resource_class":
                        mlir_ir.FlatSymbolRefAttr.get(resource.name,
                                                      context=self.context),
                    "lane_count":
                        _i64(self.context, lane_count),
                    "buffer_capacity":
                        _i64(self.context, stream.buffer_size),
                    "physical_units":
                        _i64(self.context, resource_physical_units),
                    "startup_ns":
                        _f64(self.context, model.startup_cycles * cycle_ns),
                    "output_interval_ns":
                        _f64(
                            self.context,
                            model.output_interval_cycles * cycle_ns /
                            lane_count,
                        ),
                    "operating_point":
                        mlir_ir.FlatSymbolRefAttr.get(self.operating_point,
                                                      context=self.context),
                    "policy":
                        mlir_ir.StringAttr.get(model.policy,
                                               context=self.context),
                    "evidence":
                        mlir_ir.StringAttr.get(str(model.evidence),
                                               context=self.context),
                }
                if characterization is not None:
                    attributes.update({
                        "source_provider":
                            mlir_ir.StringAttr.get(
                                characterization.source_provider,
                                context=self.context,
                            ),
                        "source_provider_sha256":
                            mlir_ir.StringAttr.get(
                                characterization.source_provider_sha256,
                                context=self.context,
                            ),
                        "source_startup_cycles":
                            mlir_ir.StringAttr.get(
                                str(characterization.startup_cycles),
                                context=self.context,
                            ),
                        "source_output_interval_cycles":
                            mlir_ir.StringAttr.get(
                                str(characterization.output_interval_cycles),
                                context=self.context,
                            ),
                        "source_build_sha256":
                            mlir_ir.StringAttr.get(
                                characterization.build_sha256,
                                context=self.context,
                            ),
                        "source_schedule_sha256":
                            mlir_ir.StringAttr.get(
                                characterization.schedule_sha256,
                                context=self.context,
                            ),
                        "source_operating_point":
                            mlir_ir.StringAttr.get(
                                characterization.operating_point,
                                context=self.context,
                            ),
                        **({
                            "source_timing_source":
                                mlir_ir.StringAttr.get(
                                    characterization.timing_source,
                                    context=self.context,
                                )
                        } if characterization.timing_source is not None else {}),
                        "source_timing_profile":
                            _f64_dict(
                                self.context,
                                dict(characterization.timing_profile),
                            ),
                        "source_output_events":
                            mlir_ir.ArrayAttr.get(
                                [
                                    mlir_ir.StringAttr.get(event,
                                                           context=self.context)
                                    for event in characterization.output_events
                                ],
                                context=self.context,
                            ),
                        "source_selection_events":
                            mlir_ir.ArrayAttr.get(
                                [
                                    mlir_ir.StringAttr.get(event,
                                                           context=self.context)
                                    for event in
                                    characterization.selection_events
                                ],
                                context=self.context,
                            ),
                        "source_physical_units":
                            _i64(self.context, characterization.physical_units),
                        "source_physical_unit_kind":
                            mlir_ir.StringAttr.get(
                                characterization.physical_unit_kind,
                                context=self.context,
                            ),
                        "source_code_distances":
                            mlir_ir.DenseI64ArrayAttr.get(
                                characterization.code_distances,
                                self.context,
                            ),
                    })
            with self.location:
                operation = mlir_ir.Operation.create(
                    "phys.factory_model",
                    attributes=attributes,
                    loc=self.location,
                )
                self.transaction.module.body.append(operation)
            checkpoint("emit")
            models[path] = operation
            models[(stream.name,)] = operation
        self._factory_models = models

    def _factory_model_for(self, stream):
        path = _symbol_path(stream)
        model = self._factory_models.get(path)
        if model is None:
            model = self._factory_models.get((path[-1],)) if path else None
        if model is None:
            raise ValueError(
                f"resource stream @{self._region_leaf(stream)} has no closed "
                "physical factory model")
        return model

    def _materialize_component_models(self):
        """Materialize public compact P3 plans and channel models."""

        from cudaq.logical.compiler.component_identity import (
            channel_binding_sha256,
            channel_identity_sha256,
            logical_channel_sha256,
            physical_architecture_sha256,
            protocol_contract,
            selected_protocol_closure_sha256,
            selected_protocol_sha256,
            spacetime_model_commitment,
            spacetime_model_sha256,
            timing_profile,
            transport_model_commitment,
            transport_model_sha256,
        )
        from cudaq.logical.compiler.protocol_identity import (
            protocol_definition_payload,
            retained_protocol_matches,
        )

        self._spacetime_models = {}
        self._transport_models = {}
        if not self.device.spacetime_plans and not any(
                binding.transport_model is not None
                for binding in self.device.qec_channels_to_physical):
            return
        if self.device.operating_point is None or self.operating_point is None:
            raise ValueError(
                "compact physical component models require an operating point")
        timing = self.device.operating_point.timing
        raw_cycle = timing.get("surface_cycle_ns", timing.get("cycle_ns"))
        try:
            cycle_ns = float(raw_cycle)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "compact physical component models require a finite positive "
                "cycle") from error
        if not math.isfinite(cycle_ns) or cycle_ns <= 0.0:
            raise ValueError(
                "compact physical component models require a finite positive "
                "cycle")
        architecture_sha256 = physical_architecture_sha256(self.device.physical)

        factory_operations = {}
        logical_by_qec = {
            id(binding.qec_region): binding.logical_region
            for binding in self.device.logical_to_qec
        }
        for binding in self.device.qec_to_physical:
            model = binding.factory_model
            if model is None:
                continue
            logical = logical_by_qec.get(id(binding.qec_region))
            if logical is None:
                continue
            streams = tuple(stream for stream in self.device.logical.streams
                            if stream.region is not None and
                            stream.region.name == logical.name)
            for stream in streams:
                operation = self._factory_models.get(
                    (self.device.logical.name, stream.name))
                if operation is not None:
                    factory_operations[id(model)] = operation

        def claims(values):
            records = []
            for claim in values:
                with self.context:
                    resource = mlir_ir.SymbolRefAttr.get(
                        [self.architecture, claim.resource_class.name],
                        context=self.context,
                    )
                records.append(
                    mlir_ir.DictAttr.get(
                        {
                            "resource_class": resource,
                            "offset": _i64(self.context, claim.offset),
                            "count": _i64(self.context, claim.count),
                            "units": _i64(self.context, claim.units),
                        },
                        context=self.context))
            return mlir_ir.ArrayAttr.get(records, context=self.context)

        for model in self.device.spacetime_plans:
            source_symbol = model.protocol.name
            source_operation = self.transaction.find_symbol(
                source_symbol, "fabric.protocol")
            if source_operation is None:
                arity = getattr(model.protocol.implements, "arity", 0)
                if arity:
                    self.transaction.bind_protocol_payload_blocks(
                        model.protocol,
                        tuple(f"__qlx_component_block{index}"
                              for index in range(arity)),
                    )
                source = self.transaction.materialize(model.protocol)
                source_symbol = source.symbol
                source_operation = self.transaction.find_symbol(
                    source_symbol, "fabric.protocol")
            if source_operation is None:
                raise ValueError(
                    "compact spacetime source did not materialize a protocol")
            contract = protocol_contract(model.protocol)
            characterization = model.characterization
            arity = getattr(model.protocol.implements, "arity", 0)
            payload_blocks = tuple(
                f"__qlx_component_block{index}" for index in range(arity))
            if not retained_protocol_matches(
                    self.transaction,
                    source_operation,
                    source_symbol,
                    protocol_definition_payload(
                        model.protocol,
                        payload_blocks=payload_blocks or None,
                    ),
                    normalize_selection=True,
            ):
                raise ValueError(
                    "spacetime model selected P2 protocol differs from its "
                    "declared source")
            selected_digest = selected_protocol_closure_sha256(source_operation)
            if (characterization is not None and
                    characterization.selected_protocol_sha256
                    != selected_digest):
                raise ValueError(
                    "characterized spacetime model selected P2 protocol "
                    "differs from its compiler source")
            with self.context:
                source_operation.attributes["component_source_sha256"] = (
                    mlir_ir.StringAttr.get(contract["source_sha256"],
                                           context=self.context))
                source_operation.attributes["component_objective_sha256"] = (
                    mlir_ir.StringAttr.get(contract["objective_sha256"],
                                           context=self.context))
                source_operation.attributes["component_boundary_sha256"] = (
                    mlir_ir.StringAttr.get(contract["boundary_sha256"],
                                           context=self.context))
            model_sha256 = spacetime_model_sha256(model)
            attributes = {
                "sym_name":
                    mlir_ir.StringAttr.get(self.transaction.unique_symbol(
                        f"{source_symbol}_component_plan"),
                                           context=self.context),
                "architecture":
                    mlir_ir.FlatSymbolRefAttr.get(self.architecture,
                                                  context=self.context),
                "source_protocol":
                    mlir_ir.FlatSymbolRefAttr.get(source_symbol,
                                                  context=self.context),
                "operating_point":
                    mlir_ir.FlatSymbolRefAttr.get(self.operating_point,
                                                  context=self.context),
                "provider":
                    mlir_ir.StringAttr.get("qlx.component-model",
                                           context=self.context),
                "provider_version":
                    mlir_ir.StringAttr.get("1", context=self.context),
                "derivation":
                    mlir_ir.StringAttr.get("characterized" if characterization
                                           is not None else "asserted",
                                           context=self.context),
                "derivation_version":
                    _i64(self.context, 1),
                "evidence":
                    mlir_ir.StringAttr.get(str(model.evidence),
                                           context=self.context),
                "forwarding_latency_ns":
                    _f64(self.context, model.latency_cycles * cycle_ns),
                "initiation_interval_ns":
                    _f64(self.context,
                         model.initiation_interval_cycles * cycle_ns),
                "interval_semantics":
                    mlir_ir.StringAttr.get(model.interval_semantics.value,
                                           context=self.context),
                "policy":
                    mlir_ir.StringAttr.get(model.policy, context=self.context),
                "source_protocol_sha256":
                    mlir_ir.StringAttr.get(contract["source_sha256"],
                                           context=self.context),
                "source_selected_protocol_sha256":
                    mlir_ir.StringAttr.get(selected_digest,
                                           context=self.context),
                "source_objective_sha256":
                    mlir_ir.StringAttr.get(contract["objective_sha256"],
                                           context=self.context),
                "source_boundary_sha256":
                    mlir_ir.StringAttr.get(contract["boundary_sha256"],
                                           context=self.context),
                "source_architecture_sha256":
                    mlir_ir.StringAttr.get(architecture_sha256,
                                           context=self.context),
                "source_timing_profile":
                    _f64_dict(self.context,
                              dict(timing_profile(
                                  self.device.operating_point))),
                "source_code_distances":
                    mlir_ir.DenseI64ArrayAttr.get(model.code_distances,
                                                  self.context),
                "model_sha256":
                    mlir_ir.StringAttr.get(model_sha256, context=self.context),
                "model_commitment":
                    mlir_ir.StringAttr.get(spacetime_model_commitment(model),
                                           context=self.context),
            }
            if self.device.operating_point.timing_source is not None:
                attributes["source_timing_source"] = mlir_ir.StringAttr.get(
                    self.device.operating_point.timing_source,
                    context=self.context)
            if characterization is not None:
                attributes["source_build_sha256"] = mlir_ir.StringAttr.get(
                    characterization.build_sha256, context=self.context)
                attributes["source_schedule_sha256"] = mlir_ir.StringAttr.get(
                    characterization.schedule_sha256, context=self.context)
            with self.location:
                plan = mlir_ir.Operation.create("phys.spacetime_plan",
                                                attributes=attributes,
                                                regions=1,
                                                loc=self.location)
                self.transaction.module.body.append(plan)
                body = plan.regions[0].blocks.append()
            ip = mlir_ir.InsertionPoint(body)
            for phase in model.phases:
                factory_refs = []
                for factory in phase.factories:
                    operation = factory_operations.get(id(factory))
                    if operation is None:
                        raise ValueError(
                            "compact spacetime phase has an unbound factory "
                            "dependency")
                    factory_refs.append(
                        mlir_ir.FlatSymbolRefAttr.get(_symbol(operation),
                                                      context=self.context))
                phase_attributes = {
                    "sym_name":
                        mlir_ir.StringAttr.get(phase.name,
                                               context=self.context),
                    "steps":
                        _i64(self.context, phase.steps),
                    "step_duration_ns":
                        _f64(self.context,
                             phase.step_duration_cycles * cycle_ns),
                    "resource_classes":
                        mlir_ir.ArrayAttr.get([], context=self.context),
                    "resource_claims":
                        claims(phase.resources),
                    "factory_models":
                        mlir_ir.ArrayAttr.get(factory_refs,
                                              context=self.context),
                    "after":
                        mlir_ir.ArrayAttr.get([
                            mlir_ir.FlatSymbolRefAttr.get(value,
                                                          context=self.context)
                            for value in phase.after
                        ],
                                              context=self.context),
                }
                with self.location:
                    operation = mlir_ir.Operation.create(
                        "phys.spacetime_phase",
                        attributes=phase_attributes,
                        loc=self.location,
                    )
                    ip.insert(operation)
            reference = mlir_ir.FlatSymbolRefAttr.get(_symbol(plan),
                                                      context=self.context)
            self._spacetime_models[source_symbol] = reference
            self._spacetime_models[model.protocol.name] = reference

        for binding in self.device.qec_channels_to_physical:
            model = binding.transport_model
            if model is None:
                continue
            channel = binding.qec_channel
            if channel.protocol is None:
                raise ValueError(
                    "compact transport requires a typed delivery protocol")
            protocol_symbol = channel.protocol.name
            if self.transaction.find_symbol(protocol_symbol,
                                            "fabric.protocol") is None:
                protocol_symbol = self.transaction.materialize(
                    channel.protocol).symbol
            protocol_operation = self.transaction.find_symbol(
                protocol_symbol, "fabric.protocol")
            if protocol_operation is None:
                raise ValueError(
                    "compact transport protocol did not materialize")
            characterization = model.characterization
            if (characterization is not None and
                    characterization.selected_protocol_sha256
                    != selected_protocol_sha256(protocol_operation)):
                raise ValueError(
                    "characterized transport selected P2 protocol differs "
                    "from its compiler source")
            transport_contract = protocol_contract(channel.protocol)
            with self.context:
                protocol_operation.attributes["component_source_sha256"] = (
                    mlir_ir.StringAttr.get(transport_contract["source_sha256"],
                                           context=self.context))
            with self.context:
                qec_binding = mlir_ir.SymbolRefAttr.get(
                    [self.architecture, f"{channel.name}_physical"],
                    context=self.context,
                )
                qec_channel = mlir_ir.SymbolRefAttr.get(
                    [self.device.qec.name, channel.name],
                    context=self.context,
                )
            attributes = {
                "sym_name":
                    mlir_ir.StringAttr.get(self.transaction.unique_symbol(
                        f"{channel.name}_transport_model"),
                                           context=self.context),
                "architecture":
                    mlir_ir.FlatSymbolRefAttr.get(self.architecture,
                                                  context=self.context),
                "operating_point":
                    mlir_ir.FlatSymbolRefAttr.get(self.operating_point,
                                                  context=self.context),
                "qec_binding":
                    qec_binding,
                "qec_channel":
                    qec_channel,
                "protocol":
                    mlir_ir.FlatSymbolRefAttr.get(protocol_symbol,
                                                  context=self.context),
                "latency_ns":
                    _f64(self.context, model.latency_cycles * cycle_ns),
                "initiation_interval_ns":
                    _f64(self.context,
                         model.initiation_interval_cycles * cycle_ns),
                "interval_semantics":
                    mlir_ir.StringAttr.get(model.interval_semantics.value,
                                           context=self.context),
                "policy":
                    mlir_ir.StringAttr.get(model.policy, context=self.context),
                "source_endpoint_occupancy":
                    _i64(self.context, model.endpoint_occupancy[0]),
                "destination_endpoint_occupancy":
                    _i64(self.context, model.endpoint_occupancy[1]),
                "resource_claims":
                    claims(model.resources),
                "provider":
                    mlir_ir.StringAttr.get(
                        "qlx.compiler.transport" if characterization is not None
                        else "qlx.user.transport",
                        context=self.context),
                "provider_version":
                    mlir_ir.StringAttr.get("1", context=self.context),
                "evidence":
                    mlir_ir.StringAttr.get(str(model.evidence),
                                           context=self.context),
                "channel_sha256":
                    mlir_ir.StringAttr.get(logical_channel_sha256(channel),
                                           context=self.context),
                "realization_sha256":
                    mlir_ir.StringAttr.get(channel_identity_sha256(channel),
                                           context=self.context),
                "protocol_sha256":
                    mlir_ir.StringAttr.get(transport_contract["source_sha256"],
                                           context=self.context),
                "binding_sha256":
                    mlir_ir.StringAttr.get(channel_binding_sha256(binding),
                                           context=self.context),
                "architecture_sha256":
                    mlir_ir.StringAttr.get(architecture_sha256,
                                           context=self.context),
                "model_sha256":
                    mlir_ir.StringAttr.get(transport_model_sha256(model),
                                           context=self.context),
                "model_commitment":
                    mlir_ir.StringAttr.get(transport_model_commitment(model),
                                           context=self.context),
                "timing_profile":
                    _f64_dict(self.context,
                              dict(timing_profile(
                                  self.device.operating_point))),
            }
            if self.device.operating_point.timing_source is not None:
                attributes["timing_source"] = mlir_ir.StringAttr.get(
                    self.device.operating_point.timing_source,
                    context=self.context)
            if characterization is not None:
                attributes["selected_protocol_sha256"] = (
                    mlir_ir.StringAttr.get(
                        characterization.selected_protocol_sha256,
                        context=self.context))
                attributes["source_build_sha256"] = mlir_ir.StringAttr.get(
                    characterization.build_sha256, context=self.context)
                attributes["source_schedule_sha256"] = mlir_ir.StringAttr.get(
                    characterization.schedule_sha256, context=self.context)
            with self.location:
                operation = mlir_ir.Operation.create("phys.transport_model",
                                                     attributes=attributes,
                                                     loc=self.location)
                self.transaction.module.body.append(operation)
            self._transport_models[id(channel)] = operation
            self._transport_models[channel.name] = operation

    def _canonicalize_resource_requests(self):
        """Bind reusable P2 request placeholders to this selected device.

        A standalone protocol has no device from which to derive a nested
        stream symbol, so its request carries only a conventional leaf
        placeholder.  Selecting a device is the first point at which that
        request can be resolved.  Preserve already-qualified references only
        when they name the same uniquely selected stream; all other cases fail
        closed before P3 projection.
        """

        routes = self.transaction._resource_streams
        if routes is None:
            selected = self.transaction.find_symbol(self.device.name,
                                                    "qlx.device")
            if selected is None or "logical" not in selected.attributes:
                routes = {}
            else:
                domain = _symbol_path(selected.attributes["logical"])
                if len(domain) != 1:
                    raise ValueError(
                        f"device @{self.device.name} has an invalid logical "
                        f"machine reference: {domain!r}")
                grouped = {}
                for stream in self.device.logical.streams:
                    grouped.setdefault(stream.produces.name, []).append(
                        (domain[0], stream.name))
                routes = {kind: tuple(paths) for kind, paths in grouped.items()}

        requests = []
        specialized = []
        for operation in self.transaction._retained_resource_requests:
            requests.append(operation)
            kind = _text(operation.attributes["kind"])
            candidates = routes.get(kind, ())
            if not candidates:
                raise ValueError(
                    f"device @{self.device.name} has no stream producing "
                    f"{kind!r}")
            if len(candidates) != 1:
                raise ValueError(
                    f"device @{self.device.name} has ambiguous streams "
                    f"producing {kind!r}: {candidates!r}")
            expected = candidates[0]
            current = _symbol_path(operation.attributes["stream"])
            if len(current) > 1 and current != expected:
                raise ValueError(
                    f"resource request for {kind!r} names stream "
                    f"{current!r}, but selected device @{self.device.name} "
                    f"routes that resource through {expected!r}")
            # A published P2 has already verified a fully qualified stream
            # against this same selected device.  Do not rewrite that
            # identical immutable fact and then invoke the expensive
            # symbol-resolving verifier once per request.  Only a standalone
            # reusable protocol with a leaf placeholder needs an incremental
            # specialization check at this boundary.
            if current == expected:
                continue
            with self.context:
                operation.attributes["stream"] = mlir_ir.SymbolRefAttr.get(
                    list(expected), context=self.context)
            specialized.append(operation)
        return tuple(requests), tuple(specialized)

    def _validate_scheduled_macro_homes(self, requests):
        """Require one exact physical engine home per scheduled macro.

        ``phys.resource_request`` currently carries one physical binding, not
        a pooled engine instance or suballocation.  Accepting a larger carrier
        pool would therefore reserve and count an ambiguous home at P3.  Keep
        this check in the shared projection preparation so the native and
        reference paths fail closed identically.
        """

        if not requests:
            return
        streams = {
            stream.name: stream for stream in self.device.logical.streams
        }
        qec_by_logical = {
            binding.logical_region.name: binding.qec_region
            for binding in self.device.logical_to_qec
        }
        physical_by_qec = {
            id(binding.qec_region): binding
            for binding in self.device.qec_to_physical
        }
        for request in requests:
            stream_path = _symbol_path(request.attributes["stream"])
            stream = streams.get(stream_path[-1] if stream_path else "")
            if stream is None or stream.external or stream.region is None:
                continue
            producer = stream.produced_by
            metadata = {} if producer is None else dict(producer.metadata)
            if metadata.get("factory_mode") != "scheduled_macro":
                continue
            try:
                footprint = int(metadata["physical_qubits"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "scheduled resource provider has invalid or incomplete "
                    "physical-qubit factory evidence") from error
            if footprint <= 0:
                raise RuntimeError(
                    "scheduled resource provider physical_qubits must be "
                    "positive")
            qec_region = qec_by_logical.get(stream.region.name)
            binding = (None if qec_region is None else physical_by_qec.get(
                id(qec_region)))
            qubit_homes = () if binding is None else tuple(
                resource for resource in binding.resources
                if resource.kind == "qubit")
            if len(qubit_homes) != 1:
                raise RuntimeError(
                    "scheduled resource provider requires exactly one "
                    "physical qubit binding")
            capacity = qubit_homes[0].count
            if capacity != footprint:
                raise RuntimeError(
                    "scheduled resource provider physical binding has "
                    f"{capacity} qubits but requires {footprint} qubits "
                    "(exactly one engine footprint); pooled or multi-engine "
                    "homes are unsupported")

    def _resolve_root(self):
        root = self.symbols.get(self.source.root.symbol)
        if root is None:
            raise ValueError(f"missing P2 root @{self.source.root.symbol}")
        if root.name == "fabric.gadget_profile":
            root = self.symbols.get(_text(root.attributes["gadget"]))
        if root is None or root.name not in {
                "fabric.gadget", "fabric.protocol"
        }:
            raise TypeError(
                "physical projection requires an executable P2 root")
        return root

    def _create_graph(self, input_types=()):
        input_types = tuple(input_types)
        function_type = mlir_ir.FunctionType.get(input_types, (),
                                                 context=self.context)
        with self.context:
            function_type_attr = mlir_ir.TypeAttr.get(function_type)
        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(self.graph_symbol, context=self.context),
            "function_type":
                function_type_attr,
            "architecture":
                mlir_ir.FlatSymbolRefAttr.get(self.architecture,
                                              context=self.context),
            "source_protocol":
                mlir_ir.FlatSymbolRefAttr.get(_symbol(self.root),
                                              context=self.context),
        }
        if self.operating_point is not None:
            attrs["operating_point"] = mlir_ir.FlatSymbolRefAttr.get(
                self.operating_point, context=self.context)
        with self.location:
            self.graph = mlir_ir.Operation.create("phys.graph",
                                                  attributes=attrs,
                                                  regions=1,
                                                  loc=self.location)
            self.module.body.append(self.graph)
            self.block = self.graph.regions[0].blocks.append(*input_types)
        self.ip = mlir_ir.InsertionPoint(self.block)
        for model in dict.fromkeys(self._factory_models.values()):
            self._emit(
                "phys.factory_start",
                attributes={
                    "factory_model":
                        mlir_ir.FlatSymbolRefAttr.get(_symbol(model),
                                                      context=self.context),
                    "event_id":
                        mlir_ir.StringAttr.get(
                            self._event_id("factory_start"),
                            context=self.context,
                        ),
                },
            )

    def _physical_boundary_type(self, type_):
        text = str(type_)
        resource = "!fabric.resource<@"
        if text.startswith(resource):
            kind = text[len(resource):-1]
            return self._resource_payload_type(kind)
        event = "!event.handle<!fabric.resource<@"
        if text.startswith(event):
            kind = text[len(event):].split(">", 1)[0]
            return self._resource_payload_event_type(kind)
        if text in {"i1", "i8", "i16", "i32", "i64", "index", "f32", "f64"}:
            return type_
        raise TypeError(
            "P3 projection of this reusable P2 boundary requires an explicit "
            f"physical source for {type_}")

    def _emit(self, name, *, operands=(), results=(), attributes=None):
        with self.location:
            operation = mlir_ir.Operation.create(
                name,
                operands=list(operands),
                results=list(results),
                attributes=dict(attributes or {}),
                loc=self.location,
            )
            self.ip.insert(operation)
        return operation

    def _event_id(self, kind):
        result = f"{kind}{self._event}"
        self._event += 1
        return result

    def _action_event_id(self, kind):
        result = self._event_id(kind)
        if self._route_event_capture is not None:
            self._route_event_capture.append(result)
        return result

    @contextmanager
    def _route_event_scope(self, enabled=True):
        previous = getattr(self, "_route_event_capture", None)
        captured = [] if enabled else None
        if enabled:
            self._route_event_capture = captured
        try:
            yield captured if captured is not None else []
        finally:
            self._route_event_capture = previous

    @contextmanager
    def _exclusive_branch(self, control, branch):
        self._exclusive_path.append((control, branch))
        try:
            yield
        finally:
            self._exclusive_path.pop()

    def _resource_predecessors(self, resource_class, indices):
        current_path = tuple(self._exclusive_path)
        predecessors = []
        for index in indices:
            previous = self._released_index_events[resource_class].get(index)
            if previous is None:
                continue
            event, previous_path = previous
            if current_path[:len(previous_path)] != previous_path:
                return None
            if event not in predecessors:
                predecessors.append(event)
        return tuple(predecessors)

    def _record_release(self, allocation, event):
        path = tuple(self._exclusive_path)
        released = self._released_index_events[allocation.resource_class]
        for index in allocation.indices:
            released[index] = (event, path)

    def _resource_payload_type(self, kind):
        return mlir_ir.Type.parse(f"!phys.resource_payload<@{kind}>",
                                  context=self.context)

    def _resource_payload_event_type(self, kind):
        return mlir_ir.Type.parse(
            f'!event.handle<!phys.resource_payload<@{kind}>, "linear">',
            context=self.context,
        )

    @staticmethod
    def _region_leaf(region):
        value = getattr(region, "value", region)
        if isinstance(value, (tuple, list)):
            return str(value[-1])
        return _text(region).split("::")[-1].lstrip("@")

    def _region_symbol(self, region, kind):
        candidates = (_text(region), self._region_leaf(region))
        indexed_by_kind = hasattr(self.transaction,
                                  "_symbol_operations_by_kind")
        for candidate in candidates:
            operation = (self.transaction.find_symbol(
                candidate, kind, scan=False) if indexed_by_kind else
                         self.transaction.find_symbol(candidate, kind))
            if operation is not None:
                return operation
        return None

    def _binding_for_region(self, region, *, require_exact=True):
        if region is None:
            if not self.device.qec_to_physical:
                raise ValueError(
                    "physical architecture has no QEC-to-physical binding "
                    "for encoded carriers")
            physical = self.device.qec_to_physical[0]
            logical = next(
                (binding.logical_region.name
                 for binding in self.device.logical_to_qec
                 if binding.qec_region is physical.qec_region or
                 binding.qec_region.name == physical.qec_region.name),
                physical.qec_region.name,
            )
            return _RegionBinding(logical, physical.qec_region, physical)

        reference = region
        placement = self._region_symbol(reference, "lvm.placement")
        if placement is not None:
            reference = _text(placement.attributes["space"])
        else:
            space = self._region_symbol(reference, "lvm.space")
            if space is not None:
                reference = _symbol(space)
        logical_name = self._region_leaf(reference)
        logical = tuple(binding for binding in self.device.logical_to_qec
                        if binding.logical_region.name == logical_name)
        if len(logical) == 1:
            qec_region = logical[0].qec_region
        else:
            # Reusable library protocols may carry a QEC-region hint that is
            # not part of the concrete device's logical namespace. Preserve
            # the historical default-binding fallback for those ordinary
            # allocations, while communication endpoints opt into exact
            # resolution below.
            physical = tuple(binding for binding in self.device.qec_to_physical
                             if binding.qec_region.name == logical_name and
                             binding.resources)
            if len(physical) == 1:
                owners = tuple(
                    binding.logical_region.name
                    for binding in self.device.logical_to_qec
                    if (binding.qec_region is physical[0].qec_region or
                        binding.qec_region.name == physical[0].qec_region.name))
                return _RegionBinding(
                    owners[0] if len(owners) == 1 else logical_name,
                    physical[0].qec_region,
                    physical[0],
                )
            if require_exact:
                detail = "unresolved" if not logical and not physical else "ambiguous"
                raise ValueError(f"{detail} explicit logical region reference "
                                 f"@{_text(region)} during physical projection")
            fallback = next(
                (binding for binding in self.device.qec_to_physical
                 if binding.resources),
                None,
            )
            if fallback is None:
                raise ValueError(
                    "physical architecture has no QEC-to-physical binding "
                    "for encoded carriers")
            owner = next(
                (binding.logical_region.name
                 for binding in self.device.logical_to_qec
                 if binding.qec_region is fallback.qec_region or
                 binding.qec_region.name == fallback.qec_region.name),
                fallback.qec_region.name,
            )
            return _RegionBinding(owner, fallback.qec_region, fallback)
        physical = tuple(
            binding for binding in self.device.qec_to_physical
            if (binding.qec_region is qec_region or binding.qec_region.name ==
                qec_region.name) and binding.resources)
        if len(physical) != 1 and require_exact:
            detail = "unresolved" if not physical else "ambiguous"
            raise ValueError(
                f"{detail} QEC-to-physical binding for explicit logical region "
                f"@{logical_name}")
        if not physical:
            raise ValueError(
                f"unresolved QEC-to-physical binding for explicit logical "
                f"region @{logical_name}")
        return _RegionBinding(logical_name, qec_region, physical[0])

    def _space_for_region(self, region):
        return self._binding_for_region(region).qec_region

    def _resource_for_region(self, region):
        resolved = self._binding_for_region(region, require_exact=False)
        return resolved.physical_binding.resources[0], resolved

    def _communication_context(self, operation, patches, *, call_instance):
        # Local generated RPP calls carry only action_site provenance.  It is
        # also one member of a remote-call qualification, but cannot by itself
        # turn an ordinary local call into communication.
        present = tuple(key for key in _COMMUNICATION_CALL_ATTRIBUTES
                        if key != "action_site" and key in operation.attributes)
        if not present:
            return None
        missing = tuple(key for key in _COMMUNICATION_CALL_ATTRIBUTES
                        if key not in operation.attributes)
        if missing:
            raise ValueError(
                "fabric.call has incomplete communication qualification; "
                f"missing {missing!r}")
        endpoint_refs = tuple(operation.attributes["endpoints"])
        if len(endpoint_refs) != 2:
            raise ValueError(
                "communication-qualified fabric.call requires exactly two "
                "endpoints")
        endpoints = tuple(
            self._binding_for_region(endpoint,
                                     require_exact=True).logical_region
            for endpoint in endpoint_refs)
        if len(set(endpoints)) < 2:
            raise ValueError(
                "communication-qualified fabric.call endpoints must span "
                "logical regions")
        operand_regions = tuple(patch.region for patch in patches)
        if operand_regions != endpoints:
            raise ValueError(
                "communication-qualified fabric.call patch operands do not "
                "match its ordered declared endpoints")
        communication = {
            "channel": operation.attributes["channel"],
            "channel_capability": operation.attributes["channel_capability"],
            "endpoints": operation.attributes["endpoints"],
            "endpoint_names": endpoints,
            "action_site": operation.attributes["action_site"],
            "generated_by": operation.attributes["generated_by"],
            "source_call": call_instance,
        }
        return communication

    def _require_communication_bridge(self, before_bridges):
        if self._communication_bridges == before_bridges:
            raise ValueError(
                "communication-qualified fabric.call realization emitted no "
                "cross-region routed native action")

    def _topology_for(self, resource_class, space=None):
        for binding in self.device.qec_to_physical:
            if space is not None and binding.qec_region.name != space.name:
                continue
            if any(resource.name == resource_class
                   for resource in binding.resources):
                return binding.topology
        return None

    def _declare_resource(self, resource_class, resolved, index, hint):
        symbol = self.transaction.unique_symbol(f"{self.graph_symbol}_{hint}")
        carrier_capabilities = {
            value.key if isinstance(value, PhysicalCapability) else value
            for value in resource_class.capabilities
        }
        for binding in resource_class.capability_bindings:
            if index in binding.indices:
                carrier_capabilities.add(binding.capability.key)
        if (resource_class.erasure_indices is not None and
                index in resource_class.erasure_indices):
            carrier_capabilities.add("qlx.physical/heralded_erasure")
        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(symbol, context=self.context),
            "kind":
                mlir_ir.StringAttr.get(resource_class.kind,
                                       context=self.context),
            "architecture":
                mlir_ir.FlatSymbolRefAttr.get(
                    self.device.physical.name,
                    context=self.context,
                ),
            "resource_class":
                mlir_ir.FlatSymbolRefAttr.get(resource_class.name,
                                              context=self.context),
            "index":
                _i64(self.context, index),
        }
        if (resource_class.capability_bindings or resource_class.capabilities or
                resource_class.erasure_indices is not None):
            attrs["capabilities"] = mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(value, context=self.context)
                    for value in sorted(carrier_capabilities)
                ],
                context=self.context,
            )
        if resolved is not None:
            with self.context:
                attrs["qec_region"] = mlir_ir.SymbolRefAttr.get(
                    [self.device.qec.name, resolved.qec_region.name],
                    context=self.context,
                )
        if resource_class.erasure_indices is not None:
            attrs["metadata"] = _string_dict(
                self.context,
                {
                    # Preserve both sides of the selection. In particular,
                    # an explicit empty set must remain distinguishable from
                    # the legacy uniform-erasure model after serialization.
                    "erasure_capable":
                        ("true" if index in resource_class.erasure_indices else
                         "false")
                },
            )
        with self.location:
            self.module.body.append(
                mlir_ir.Operation.create("phys.resource",
                                         attributes=attrs,
                                         loc=self.location))
        self._resource_indices[symbol] = index
        if resource_class.kind == "atom" and "storage" in self._zone_roles:
            self._resource_zones[symbol] = self._zone_roles["storage"]
        return symbol

    def _patch_resources(self, resource_class, resolved, count):
        binding = resolved.physical_binding
        if not any(resource.name == resource_class.name
                   for resource in binding.resources):
            raise ValueError(
                f"QEC-to-physical binding @{binding.name} for logical region "
                f"@{resolved.logical_region} does not provide resource class "
                f"@{resource_class.name}")
        if binding is not None and binding.patch_topology is not None:
            active = self._slot_allocations[binding.name]
            ever_used = self._ever_slot_allocations[binding.name]
            candidates = tuple((slot, group)
                               for slot, group in enumerate(
                                   binding.patch_topology.carrier_groups)
                               if slot not in active and len(group) >= count)
            # Consume never-used capacity before recycling a released slot.
            # This preserves available parallelism while still allowing a
            # long sequence of non-overlapping invocation-local allocations
            # to fit in a finite architecture.
            candidates = tuple(
                sorted(candidates,
                       key=lambda item: (item[0] in ever_used, item[0])))
            candidates = tuple(
                (slot, group)
                for slot, group in candidates
                if (self._resource_predecessors(resource_class.name,
                                                group[:count]) is not None))
            if candidates:
                slot, group = candidates[0]
                active.add(slot)
                ever_used.add(slot)
                selected = tuple(group[:count])
                self._patch_allocated_indices[resource_class.name].update(group)
                return binding, slot, selected
            label = resolved.logical_region
            raise ValueError(
                f"carrier-derived patch topology for {label} has no free slot "
                f"with {count} @{resource_class.name} carriers")
        available = [
            index for index in range(resource_class.count)
            if index not in self._allocated_indices[resource_class.name] and
            (self._resource_predecessors(resource_class.name, (
                index,)) is not None)
        ]
        if len(available) < count:
            used = resource_class.count - len(available)
            raise ValueError(
                f"architecture resource class @{resource_class.name} has "
                f"capacity {resource_class.count}, but projection needs "
                f"{used + count} carriers")
        ever_used = self._ever_allocated_indices[resource_class.name]
        available.sort(key=lambda index: (index in ever_used, index))
        return binding, None, tuple(available[:count])

    def _allocate_patch(self, code, region, hint, *, encoding=None):
        try:
            partitions = self.codes[code]
        except KeyError as exc:
            raise ValueError(f"missing code definition @{code}") from exc
        resource_class, resolved = self._resource_for_region(region)
        count = sum(size for _, size in partitions)
        binding, slot, selected_indices = self._patch_resources(
            resource_class, resolved, count)
        after = self._resource_predecessors(resource_class.name,
                                            selected_indices)
        if after is None:
            raise ValueError(
                "physical allocation cannot prove resource reuse across "
                "exclusive structured-control boundaries")
        formal_roles = tuple((partition, local)
                             for partition, size in partitions
                             for local in range(size))
        index_by_role = dict(zip(formal_roles, selected_indices))
        patch_id = f"patch{self._patch_instance}"
        self._patch_instance += 1
        acquire_id = self._event_id("acquire")
        declarations = []
        partition_symbols = {}
        for partition, size in partitions:
            partition_symbols[partition] = []
            for local in range(size):
                index = index_by_role[(partition, local)]
                symbol = self._declare_resource(
                    resource_class,
                    resolved,
                    index,
                    f"{hint}_{partition}{local}",
                )
                declarations.append(symbol)
                partition_symbols[partition].append(symbol)
                self._allocated_indices[resource_class.name].add(index)
                self._ever_allocated_indices[resource_class.name].add(index)
                self._patch_allocated_indices[resource_class.name].add(index)
                self._mapping_initial.append({
                    "role": f"{patch_id}.{partition}[{local}]",
                    "patch": patch_id,
                    "partition": partition,
                    "index": local,
                    "resource": symbol,
                    "node": index,
                    "region": resolved.logical_region,
                    "qec_region": resolved.qec_region.name,
                    "physical_binding": binding.name,
                })
        types = [
            mlir_ir.Type.parse(f"!phys.state<@{symbol}>", context=self.context)
            for symbol in declarations
        ]
        acquire = self._emit(
            "phys.acquire",
            results=types,
            attributes={
                "resources":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.FlatSymbolRefAttr.get(symbol,
                                                          context=self.context)
                            for symbol in declarations
                        ],
                        context=self.context,
                    ),
                "event_id":
                    mlir_ir.StringAttr.get(acquire_id, context=self.context),
            },
        )
        values = iter(acquire.results)
        physical = {
            name: [
                _Carrier(
                    next(values),
                    symbol,
                    resource_class.name,
                    resolved.logical_region,
                    resolved.qec_region.name,
                    binding.name,
                ) for symbol in symbols
            ] for name, symbols in partition_symbols.items()
        }
        patch_topology = (self._patch_topology_symbols[binding.name]
                          if binding.patch_topology is not None else None)
        self._patch_nodes.append({
            "id": patch_id,
            "code": code,
            "encoding": encoding,
            "region": resolved.logical_region,
            "qec_region": resolved.qec_region.name,
            "physical_binding": binding.name,
            "slot": slot,
            "patch_topology": patch_topology,
        })
        if slot is not None:
            self._patch_assignments.append({
                "id": f"map.{patch_id}",
                "patch": patch_id,
                "slot": slot,
                "topology": patch_topology,
                "carriers": selected_indices,
                "region": resolved.logical_region,
                "qec_region": resolved.qec_region.name,
                "physical_binding": binding.name,
            })
        allocation = _AllocationBinding(
            allocation=patch_id,
            resource_class=resource_class.name,
            resources=tuple(declarations),
            indices=tuple(selected_indices),
            acquire=acquire_id,
            after=after,
            slot=slot,
            slot_binding=binding.name if binding is not None else None,
            scope=resolved.logical_region,
            qec_region=resolved.qec_region.name,
            physical_binding=binding.name,
        )
        self._allocation_bindings.append(allocation)
        self._allocation_by_patch[patch_id] = allocation
        return _Patch(
            code,
            encoding,
            physical,
            resolved.logical_region,
            patch_id,
            slot,
            patch_topology,
            resolved.qec_region.name,
            binding.name,
        )

    @staticmethod
    def _selected(operation, patch, partition):
        values = patch.selection(partition)
        if "indices" not in operation.attributes:
            return tuple(range(len(values)))
        return tuple(int(value) for value in operation.attributes["indices"])

    def _classes_for(self, carriers):
        classes = {
            item.name: item for item in self.device.physical.resource_classes
        }
        return tuple(classes[carrier.resource_class] for carrier in carriers)

    @staticmethod
    def _typed_native_action(resource_classes, action):
        action_name = getattr(action, "name", action)
        selected = []
        for resource_class in resource_classes:
            matches = tuple(
                candidate for candidate in resource_class.native_actions
                if (candidate == action if isinstance(action, PhysicalAction)
                    else getattr(candidate, "name", candidate) == action_name))
            if len(matches) != 1 or not isinstance(matches[0], PhysicalAction):
                return None
            selected.append(matches[0])
        if selected and all(candidate == selected[0] for candidate in selected):
            return selected[0]
        return None

    @staticmethod
    def _native_decomposition(resource_classes, action):
        action_name = getattr(action, "name", action)
        selected = []
        for resource_class in resource_classes:
            matches = tuple(
                candidate
                for candidate in resource_class.native_action_decompositions
                if candidate.source.name == action_name)
            if len(matches) != 1:
                return None
            selected.append(matches[0])
        if selected and all(candidate == selected[0] for candidate in selected):
            return selected[0]
        return None

    def _ensure_zone(self, carriers, role, *, trajectory=None):
        carriers = tuple(carriers)
        destination = self._zone_roles.get(role)
        if not carriers or destination is None:
            return carriers
        if any(resource_class.kind != "atom"
               for resource_class in self._classes_for(carriers)):
            return carriers

        # A move route has one source and destination. Preserve input ordering
        # while moving each current-zone group through its explicit route.
        result = list(carriers)
        by_source: dict[str, list[int]] = {}
        for ordinal, carrier in enumerate(carriers):
            source = self._resource_zones.get(carrier.resource)
            if source is None or source == destination:
                continue
            by_source.setdefault(source, []).append(ordinal)
        for source, ordinals in by_source.items():
            route = self._zone_routes.get((source, destination))
            if route is None:
                raise ValueError(
                    f"neutral-atom movement requires an explicit shuttle "
                    f"route {source!r} -> {destination!r}")
            moving = tuple(result[ordinal] for ordinal in ordinals)
            operation = self._emit(
                "phys.move",
                operands=[carrier.value for carrier in moving],
                results=[carrier.value.type for carrier in moving],
                attributes={
                    "route":
                        mlir_ir.FlatSymbolRefAttr.get(route.name,
                                                      context=self.context),
                    "trajectory":
                        mlir_ir.StringAttr.get(trajectory or f"to_{role}",
                                               context=self.context),
                    "event_id":
                        mlir_ir.StringAttr.get(self._event_id("move"),
                                               context=self.context),
                },
            )
            for ordinal, value, carrier in zip(ordinals, operation.results,
                                               moving):
                result[ordinal] = carrier.with_value(value)
                self._resource_zones[carrier.resource] = destination
            self._zoned_movements += 1
        return tuple(result)

    def _emit_native_action(
            self,
            carriers,
            action: PhysicalAction,
            *,
            reservations=(),
    ):
        carriers = tuple(carriers)
        blockade = (action.arity == 2 and
                    getattr(action.process, "name", "").lower() == "cz" and
                    "entangling" in self._zone_roles)
        if blockade:
            carriers = self._ensure_zone(carriers,
                                         "entangling",
                                         trajectory="activate")

        def emit(group):
            handle = self.transaction.lookup(action)
            action_symbol = action.name if handle is None else handle.symbol
            attrs = {
                "action":
                    mlir_ir.FlatSymbolRefAttr.get(action_symbol,
                                                  context=self.context),
                "event_id":
                    mlir_ir.StringAttr.get(self._action_event_id(action.name),
                                           context=self.context),
                "resources":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.FlatSymbolRefAttr.get(carrier.resource,
                                                          context=self.context)
                            for carrier in group
                        ],
                        context=self.context,
                    ),
            }
            if len(group) > 1 and len({item.resource_class for item in group
                                      }) == 1:
                topology = self._topology_for(group[0].resource_class)
                if topology is not None and topology.strict:
                    attrs["topology"] = mlir_ir.FlatSymbolRefAttr.get(
                        topology.name, context=self.context)
            if reservations:
                attrs["reservations"] = mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.StringAttr.get(str(value), context=self.context)
                        for value in reservations
                    ],
                    context=self.context,
                )
            operation = self._emit(
                "phys.apply",
                operands=[carrier.value for carrier in group],
                results=[carrier.value.type for carrier in group],
                attributes=attrs,
            )
            return tuple(
                carrier.with_value(result)
                for result, carrier in zip(operation.results, group))

        if action.arity == 1 and not action.broadcast:
            output = tuple(
                result for carrier in carriers for result in emit((carrier,)))
        else:
            if not action.broadcast and len(carriers) != action.arity:
                raise ValueError(
                    f"physical action {action.name!r} has arity {action.arity}, "
                    f"not {len(carriers)}")
            output = emit(carriers)
        if blockade:
            output = self._ensure_zone(output, "storage", trajectory="park")
        return output

    def _apply_carriers(self, carriers, action, *, reservations=()):
        carriers = tuple(carriers)
        if not carriers:
            return ()
        action_name = getattr(action, "name", action)
        resource_classes = self._classes_for(carriers)
        typed_action = self._typed_native_action(resource_classes, action)
        if typed_action is not None:
            return self._emit_native_action(carriers,
                                            typed_action,
                                            reservations=reservations)

        decomposition = self._native_decomposition(resource_classes, action)
        if decomposition is not None:
            if len(carriers) != decomposition.source.arity:
                raise ValueError(
                    f"physical action {action!r} has arity "
                    f"{decomposition.source.arity}, not {len(carriers)}")
            current = list(carriers)
            for step in decomposition.steps:
                selected = tuple(current[index] for index in step.operands)
                updated = self._emit_native_action(selected,
                                                   step.action,
                                                   reservations=reservations)
                for index, carrier in zip(step.operands, updated):
                    current[index] = carrier
            self._legalized_actions[action_name] = tuple(
                (step.action.name, step.operands)
                for step in decomposition.steps)
            return tuple(current)

        # Compatibility strings remain supported only when the class explicitly
        # advertises the exact name. They carry no decomposition information.
        if all(
                action_name in {
                    getattr(candidate, "name", candidate)
                    for candidate in resource_class.native_actions
                }
                for resource_class in resource_classes):
            attrs = {
                "action":
                    mlir_ir.FlatSymbolRefAttr.get(action_name,
                                                  context=self.context),
                "event_id":
                    mlir_ir.StringAttr.get(self._action_event_id(action_name),
                                           context=self.context),
                "resources":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.FlatSymbolRefAttr.get(carrier.resource,
                                                          context=self.context)
                            for carrier in carriers
                        ],
                        context=self.context,
                    ),
            }
            operation = self._emit(
                "phys.apply",
                operands=[carrier.value for carrier in carriers],
                results=[carrier.value.type for carrier in carriers],
                attributes=attrs,
            )
            return tuple(
                carrier.with_value(result)
                for result, carrier in zip(operation.results, carriers))

        names = sorted({item.name for item in resource_classes})
        raise ValueError(
            f"physical action {action_name!r} is neither native nor explicitly "
            f"decomposed by resource classes {names!r}")

    def _scratch_carrier(self, resource_class_name, index, template):
        key = (resource_class_name, index, template.physical_binding)
        if key in self._scratch:
            return self._scratch[key]
        resource_class = next(
            item for item in self.device.physical.resource_classes
            if item.name == resource_class_name)
        binding = next(item for item in self.device.qec_to_physical
                       if item.name == template.physical_binding)
        resolved = _RegionBinding(
            template.region,
            binding.qec_region,
            binding,
        )
        symbol = self._declare_resource(
            resource_class,
            resolved,
            index,
            f"route_scratch{index}",
        )
        state_type = mlir_ir.Type.parse(f"!phys.state<@{symbol}>",
                                        context=self.context)
        after = self._resource_predecessors(resource_class_name, (index,))
        if after is None:
            raise ValueError(
                "physical route scratch cannot prove resource reuse across "
                "exclusive structured-control boundaries")
        acquire_id = self._event_id("route_scratch_acquire")
        acquired = self._emit(
            "phys.acquire",
            results=[state_type],
            attributes={
                "resources":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.FlatSymbolRefAttr.get(symbol,
                                                          context=self.context)
                        ],
                        context=self.context,
                    ),
                "event_id":
                    mlir_ir.StringAttr.get(acquire_id, context=self.context),
            },
        )
        carrier = _Carrier(
            acquired.result,
            symbol,
            resource_class_name,
            template.region,
            template.qec_region,
            template.physical_binding,
        )
        self._scratch[key] = carrier
        self._allocated_indices[resource_class_name].add(index)
        self._ever_allocated_indices[resource_class_name].add(index)
        self._allocation_bindings.append(
            _AllocationBinding(
                allocation=(
                    f"route_scratch.{resource_class_name}.{index}.{acquire_id}"
                ),
                resource_class=resource_class_name,
                resources=(symbol,),
                indices=(index,),
                acquire=acquire_id,
                after=after,
                scope="routing",
                qec_region=template.qec_region,
                physical_binding=template.physical_binding,
            ))
        return carrier

    def _release_route_scratch(self, scratch):
        scratch_allocations = {
            allocation.resources[0]: allocation
            for allocation in self._allocation_bindings
            if allocation.allocation.startswith("route_scratch.") and
            len(allocation.resources) == 1 and allocation.release is None
        }
        for _, carrier in sorted(
                scratch.items(),
                key=lambda item: (item[0][0], item[0][1]),
        ):
            release_id = self._event_id("route_scratch_release")
            self._emit(
                "phys.release",
                operands=[carrier.value],
                attributes={
                    "event_id":
                        mlir_ir.StringAttr.get(
                            release_id,
                            context=self.context,
                        )
                },
            )
            allocation = scratch_allocations.get(carrier.resource)
            if allocation is None:
                raise ValueError(
                    "physical route scratch release has no allocation "
                    f"binding for @{carrier.resource}")
            allocation.release = release_id
            self._record_release(allocation, release_id)
            self._allocated_indices[
                allocation.resource_class].difference_update(allocation.indices)

    def _reset_carriers(self, carriers):
        carriers = tuple(carriers)
        operation = self._emit(
            "phys.reset",
            operands=[carrier.value for carrier in carriers],
            results=[carrier.value.type for carrier in carriers],
            attributes={
                "state":
                    mlir_ir.StringAttr.get("zero", context=self.context),
                "event_id":
                    mlir_ir.StringAttr.get(self._event_id("reset"),
                                           context=self.context),
            },
        )
        return tuple(
            carrier.with_value(result)
            for result, carrier in zip(operation.results, carriers))

    def _routed_pair(self, left, right, action):
        action_name = getattr(action, "name", action)
        cross_region = left.region != right.region
        if cross_region:
            if not self._communication_stack:
                raise ValueError(
                    f"bare cross-region physical action {action_name!r} "
                    "requires a communication-qualified fabric.call")
            if left.resource_class != right.resource_class:
                raise ValueError(
                    f"cross-region physical action {action_name!r} between "
                    f"@{left.region} and @{right.region} requires one shared "
                    "resource class")
            bindings = {
                binding.name: binding for binding in self.device.qec_to_physical
            }
            left_binding = bindings.get(left.physical_binding)
            right_binding = bindings.get(right.physical_binding)
            if left_binding is None or right_binding is None:
                raise ValueError(
                    f"cross-region physical action {action_name!r} lacks exact "
                    "QEC-to-physical binding provenance")
            left_topology = left_binding.topology
            right_topology = right_binding.topology
            if (left_topology is None or right_topology is None or
                    not left_topology.strict or not right_topology.strict or
                    left_topology != right_topology):
                raise ValueError(
                    f"cross-region physical action {action_name!r} between "
                    f"@{left.region} and @{right.region} requires a shared strict "
                    "carrier topology")
            topology = left_topology
        else:
            if left.resource_class != right.resource_class:
                return self._apply_carriers((left, right), action)
            space = next(
                (binding.qec_region
                 for binding in self.device.qec_to_physical
                 if binding.name == left.physical_binding),
                None,
            )
            topology = self._topology_for(left.resource_class, space)
            if topology is None or not topology.strict:
                return self._apply_carriers((left, right), action)
        source = self._resource_indices[left.resource]
        target = self._resource_indices[right.resource]
        if topology.edge(source, target) is not None:
            with self._route_event_scope(cross_region) as native_events:
                outputs = self._apply_carriers((left, right), action)
            if cross_region:
                self._record_route(action_name,
                                   topology, (source, target),
                                   left,
                                   right,
                                   bridge=True,
                                   native_events=native_events)
            return outputs

        blocked = self._patch_allocated_indices[left.resource_class] - {
            source,
            target,
        }
        allowed = set(topology.nodes) - blocked - {target}
        allowed.add(source)
        candidates = []
        for neighbor in topology.nodes:
            if neighbor == target or neighbor not in allowed:
                continue
            if topology.edge(neighbor, target) is None:
                continue
            path = topology.shortest_path(
                source,
                neighbor,
                allowed=allowed,
            )
            if path is not None:
                candidates.append(path)
        if not candidates:
            raise PlacementInfeasible(
                f"cannot route physical action {action_name!r} from node {source} "
                f"to node {target} on topology @{topology.name}; no path of "
                "unoccupied carriers reaches an adjacent interaction edge")
        path = min(candidates, key=lambda value: (len(value), value))
        with self._route_event_scope() as native_events:
            moving = left
            trail = []
            for destination in path[1:]:
                scratch = self._scratch_carrier(
                    left.resource_class,
                    destination,
                    left,
                )
                source_state, destination_state = self._apply_carriers(
                    (moving, scratch), "swap")
                source_state = scratch.with_value(source_state.value,
                                                  resource=moving.resource)
                trail.append((destination, source_state))
                moving = left.with_value(destination_state.value,
                                         resource=scratch.resource)
            moving, right = self._apply_carriers((moving, right), action)
            for destination, source_state in reversed(trail):
                restored, scratch_state = self._apply_carriers(
                    (source_state, moving), "swap")
                moving = left.with_value(restored.value,
                                         resource=source_state.resource)
                self._scratch[(
                    left.resource_class,
                    destination,
                    left.physical_binding,
                )] = source_state.with_value(scratch_state.value,
                                             resource=scratch_state.resource)
        full_path = (*path, target)
        self._record_route(
            action_name,
            topology,
            full_path,
            left,
            right,
            bridge=cross_region,
            native_events=native_events,
        )
        return moving, right

    def _record_route(
        self,
        action,
        topology,
        path,
        left,
        right,
        *,
        bridge,
        native_events,
    ):
        step = {
            "event": f"route{self._route}",
            "action": action,
            "topology": topology.name,
            "path": tuple(path),
            "native_events": tuple(native_events),
            "bridge": bridge,
            "source_region": left.region,
            "destination_region": right.region,
            "source_qec_region": left.qec_region,
            "destination_qec_region": right.qec_region,
            "source_binding": left.physical_binding,
            "destination_binding": right.physical_binding,
        }
        if bridge and self._communication_stack:
            step.update(self._communication_stack[-1])
        self._route_steps.append(step)
        self._route += 1
        if bridge:
            self._communication_bridges += 1

    def _bulk(self, operation, patch, action):
        result = patch.clone()
        apply = (self._reset_carriers if action == "reset" else
                 lambda carriers: self._apply_carriers(carriers, action))
        partition = _partition(operation.attributes["partition"])
        indices = self._selected(operation, patch, partition)
        selected = patch.selection(partition)
        entries = tuple(selected[index] for index in indices)
        updated = apply([carrier for _, _, carrier in entries])
        for (name, index, _), carrier in zip(entries, updated):
            result.partitions[name][index] = carrier
        return result

    def _feeds_communication_call(self, operation):
        """Return whether a patch result reaches a remote-observable call."""

        def reaches_remote(value, visiting):
            cached = self._communication_reachability.get(value)
            if cached is not None:
                return cached
            if value in visiting:
                return False
            visiting.add(value)
            for use in value.uses:
                owner = getattr(use.owner, "operation", use.owner)
                if (owner.name == "fabric.call" and
                        "channel_capability" in owner.attributes and
                        _capability_key(owner.attributes["channel_capability"])
                        == "qlx.machine/observable_remote"):
                    visiting.remove(value)
                    self._communication_reachability[value] = True
                    return True
                if any(
                        reaches_remote(result, visiting)
                        for result in owner.results
                        if _code_name(result.type) is not None):
                    visiting.remove(value)
                    self._communication_reachability[value] = True
                    return True
            visiting.remove(value)
            self._communication_reachability[value] = False
            return False

        return any(
            reaches_remote(result, set())
            for result in operation.results
            if _code_name(result.type) is not None)

    def _prepare(
        self,
        patch,
        state,
        partition="data",
        indices=None,
        *,
        encode=False,
    ):
        result = patch.clone()
        values = patch.selection(partition)
        indices = (tuple(range(len(values)))
                   if indices is None else tuple(indices))
        entries = tuple(values[index] for index in indices)
        carriers = tuple(carrier for _, _, carrier in entries)
        if not carriers:
            return result
        code = self.symbols.get(result.code) if encode else None
        encoded_plus = state == "plus" and encode
        if encoded_plus:

            def supports(name):
                if code is None or name not in code.attributes:
                    return ()
                return tuple(
                    frozenset(int(index)
                              for index in row)
                    for row in code.attributes[name])

            # Transversal H turns |0_L> into |+_L> only for a self-dual CSS
            # presentation.  The accepted interconnect provider uses Steane,
            # whose X/Z checks and logical representatives are identical.
            # Fail closed for other codes until they provide an explicit
            # encoded-state preparation gadget.
            if (not supports("hx") or
                    set(supports("hx")) != set(supports("hz")) or
                    set(supports("lx")) != set(supports("lz"))):
                raise ValueError(
                    f"encoded plus preparation for code @{result.code} "
                    "requires a self-dual CSS presentation or an explicit "
                    "physical preparation gadget")
        operation = self._emit(
            "phys.prepare",
            operands=[carrier.value for carrier in carriers],
            results=[carrier.value.type for carrier in carriers],
            attributes={
                "state":
                    mlir_ir.StringAttr.get(
                        "zero" if encoded_plus else state,
                        context=self.context,
                    ),
                "event_id":
                    mlir_ir.StringAttr.get(self._event_id("prepare"),
                                           context=self.context),
            },
        )
        for (name, index, carrier), value in zip(entries, operation.results):
            result.partitions[name][index] = carrier.with_value(value)
        if state in ("zero", "plus") and encode:
            data = result.partitions["data"]
            supports = (() if code is None or "hx" not in code.attributes else
                        tuple(
                            tuple(int(index)
                                  for index in row)
                            for row in code.attributes["hx"]))
            # Prepare the CSS code's |0_L> state, rather than interpreting an
            # encoded preparation as independent carrier resets.  RREF gives
            # one control/pivot per independent X stabilizer; H on those
            # controls followed by the row CNOTs creates the uniform
            # superposition over the X-stabilizer row space.  The untouched
            # Z-orthogonal complement fixes every logical Z to +1.
            rows = []
            for support in supports:
                if any(index < 0 or index >= len(data) for index in support):
                    raise ValueError(
                        f"code @{result.code} has an out-of-range X stabilizer")
                mask = sum(1 << index for index in support)
                if mask:
                    rows.append(mask)
            pivot_columns = []
            rank = 0
            columns = sorted(
                range(len(data)),
                key=lambda column: (
                    sum((row >> column) & 1 for row in rows),
                    column,
                ),
            )
            for column in columns:
                selected = next(
                    (index for index in range(rank, len(rows))
                     if (rows[index] >> column) & 1),
                    None,
                )
                if selected is None:
                    continue
                rows[rank], rows[selected] = rows[selected], rows[rank]
                for index in range(len(rows)):
                    if index != rank and ((rows[index] >> column) & 1):
                        rows[index] ^= rows[rank]
                pivot_columns.append(column)
                rank += 1
                if rank == len(rows):
                    break
            # Elimination continues after each pivot is chosen, so only the
            # final RREF rows are the generators that the circuit must
            # synthesize.  Capturing rows while elimination is in progress can
            # leave an earlier generator with a stale non-pivot tail.
            generators = tuple(zip(pivot_columns, rows[:rank]))
            for pivot, _ in generators:
                (data[pivot],) = self._apply_carriers((data[pivot],), "h")
            for pivot, row in generators:
                for target in range(len(data)):
                    if target == pivot or not ((row >> target) & 1):
                        continue
                    data[pivot], data[target] = self._routed_pair(
                        data[pivot], data[target], "cx")
            if encoded_plus:
                for index, carrier in enumerate(data):
                    (data[index],) = self._apply_carriers((carrier,), "h")
        return result

    def _append_projected_record(self, projected, record_id):
        """Retain the folded-repeat hierarchy of one physical record lane."""

        # ``phys.record_projection`` orders repeat evidence from the record's
        # immediate enclosing repeat outwards.  The active lowering stack is
        # naturally outermost-to-innermost, so freeze its reverse when the
        # measurement is created instead of trying to recover ancestry after
        # regions and call boundaries have been emitted.
        context = tuple(reversed(self._repeat_context))
        previous = self._record_projection_repeats.get(record_id)
        if previous is not None and previous != context:
            raise ValueError(
                f"physical record {record_id!r} has inconsistent folded-"
                "repeat projection context")
        self._record_projection_repeats[record_id] = context
        projected.append(record_id)

    def _measure(
        self,
        patch,
        partition,
        indices,
        record,
        source_symbol,
        *,
        basis="z",
        instance_symbol=None,
        lane_prefix=None,
        ordinal_offset=0,
        allow_legacy_z_fallback=True,
    ):
        result = patch.clone()
        records = []
        entries = tuple(result.selection(partition)[index] for index in indices)
        selected = tuple(carrier for _, _, carrier in entries)
        instrument = self._native_instrument(
            selected, "measure" if basis == "z" else "measure_x")
        decomposed_x_read = False
        if basis == "x" and instrument is None:
            z_instrument = self._native_instrument(selected, "measure")
            if (z_instrument is None or
                    not self._supports_native_action(selected, "h")):
                raise PlacementInfeasible(
                    "X-basis carrier measurement needs a native measure_x "
                    "instrument or a native H-measure_z-H realization")
            selected = tuple(self._apply_carriers(selected, "h"))
            for (name, index, _), carrier in zip(entries, selected):
                result.partitions[name][index] = carrier
            instrument = z_instrument
            decomposed_x_read = True
        if instrument is None and not allow_legacy_z_fallback:
            raise PlacementInfeasible(
                "Z-basis carrier measurement needs a declared native "
                "measure_z instrument")
        if instrument is None:
            # Preserve the established Z-read compatibility path for physical
            # models authored before native-instrument declarations became
            # mandatory. Basis-typed realizations disable this fallback.
            instrument = MZ
        measurement = self.transaction.materialize(instrument).symbol
        selected = self._ensure_zone(selected, "readout", trajectory="readout")
        for (name, index, _), carrier in zip(entries, selected):
            result.partitions[name][index] = carrier
        for ordinal, (name, index, _) in enumerate(entries):
            carrier = result.partitions[name][index]
            lane = lane_prefix or partition
            relative = f"{record}.{lane}{ordinal_offset + ordinal}"
            instance_symbol = instance_symbol or source_symbol
            source_record = f"{source_symbol}.{relative}"
            projection_key = (instance_symbol, source_record)
            projected = self._record_projection.setdefault(projection_key, [])
            record_id = f"{instance_symbol}.{relative}"
            if projected:
                record_id = f"{record_id}.occurrence{len(projected)}"
            record_type = mlir_ir.Type.parse("!phys.record<@bit>",
                                             context=self.context)
            operation = self._emit(
                "phys.measure",
                operands=[carrier.value],
                results=[carrier.value.type, record_type],
                attributes={
                    "measurement":
                        mlir_ir.FlatSymbolRefAttr.get(measurement,
                                                      context=self.context),
                    "record_id":
                        mlir_ir.StringAttr.get(record_id, context=self.context),
                    "event_id":
                        mlir_ir.StringAttr.get(self._event_id("measure"),
                                               context=self.context),
                },
            )
            result.partitions[name][index] = carrier.with_value(
                operation.results[0])
            records.append(operation.results[1])
            self._record_values[record_id] = operation.results[1]
            self._record_ids_by_value[operation.results[1]] = record_id
            self._append_projected_record(projected, record_id)
        if decomposed_x_read:
            restored = self._apply_carriers(
                tuple(result.partitions[name][index]
                      for name, index, _ in entries),
                "h",
            )
            for (name, index, _), carrier in zip(entries, restored):
                result.partitions[name][index] = carrier
        return result, tuple(records)

    def _supports_native_action(self, carriers, action):
        classes = {
            item.name: item for item in self.device.physical.resource_classes
        }
        if isinstance(action, PhysicalAction):
            return all(
                any(
                    isinstance(native, PhysicalAction) and native == action
                    for native in classes[
                        carrier.resource_class].native_actions)
                for carrier in carriers)
        return all(
            action in {
                getattr(native, "name", native)
                for native in classes[carrier.resource_class].native_actions
            }
            for carrier in carriers)

    def _supports_compat_action(self, carriers, action):
        """Whether every carrier explicitly advertises the legacy string name."""

        classes = {
            item.name: item for item in self.device.physical.resource_classes
        }
        return all(
            action in {
                native
                for native in classes[carrier.resource_class].native_actions
                if isinstance(native, str)
            }
            for carrier in carriers)

    def _supports_physical_capability(self, carriers, capability):
        """Whether every selected carrier owns one typed P3 capability."""

        if not isinstance(capability, PhysicalCapability):
            raise TypeError("physical capability checks require a typed value")
        classes = {
            item.name: item for item in self.device.physical.resource_classes
        }
        for carrier in carriers:
            resource_class = classes[carrier.resource_class]
            if any(
                (value.key if isinstance(value, PhysicalCapability) else value
                ) == capability.key for value in resource_class.capabilities):
                continue
            index = self._resource_indices.get(carrier.resource)
            if index is None or not any(
                    binding.capability.key == capability.key and
                    index in binding.indices
                    for binding in resource_class.capability_bindings):
                return False
        return True

    def _native_instrument(self, carriers, operation):
        classes = {
            item.name: item for item in self.device.physical.resource_classes
        }
        selected = []
        for carrier in carriers:
            matches = tuple(instrument for instrument in classes[
                carrier.resource_class].native_instruments
                            if instrument.operation == operation)
            if not matches:
                return None
            if len(matches) != 1:
                raise PlacementInfeasible(
                    f"resource class @{carrier.resource_class} advertises "
                    f"multiple {operation!r} instruments")
            instrument = matches[0]
            if operation in {"measure", "measure_x"}:
                expected_process = ("measure_x" if operation == "measure_x" else
                                    "measure_z")
                if (instrument.arity != 1 or
                        instrument.record_schema != "bit" or
                        not instrument.preserves_inputs or
                        not isinstance(instrument.process, QuantumProcess) or
                        instrument.process.name.lower() != expected_process):
                    raise PlacementInfeasible(
                        f"native instrument @{instrument.name} does not satisfy "
                        f"the unary, state-preserving {expected_process} -> bit "
                        "measurement contract")
            selected.append(matches[0])
        if not selected or any(
                instrument != selected[0] for instrument in selected[1:]):
            return None
        return selected[0]

    @staticmethod
    def _pauli_product(left, right):
        if left is None:
            return 1, right
        if left == right:
            return 1, None
        return {
            ("X", "Y"): (1j, "Z"),
            ("Y", "X"): (-1j, "Z"),
            ("X", "Z"): (-1j, "Y"),
            ("Z", "X"): (1j, "Y"),
            ("Y", "Z"): (1j, "X"),
            ("Z", "Y"): (-1j, "X"),
        }[(left, right)]

    def _physical_pauli_product(self, operation, patches):
        patch_indices = tuple(
            int(value) for value in operation.attributes["patch_indices"])
        logical_indices = tuple(
            int(value) for value in operation.attributes["logical_indices"])
        encoded = _text(operation.attributes["pauli_product"])
        invert = encoded.startswith("-")
        paulis = encoded.removeprefix("-")
        physical = {}
        phase = complex(-1 if invert else 1)

        def add(patch, index, pauli):
            nonlocal phase
            carrier = patch.partitions["data"][index]
            factor, updated = self._pauli_product(
                physical.get(carrier.resource), pauli)
            phase *= factor
            if updated is None:
                physical.pop(carrier.resource, None)
            else:
                physical[carrier.resource] = updated

        for patch_index, logical, pauli in zip(patch_indices, logical_indices,
                                               paulis):
            patch = patches[patch_index]
            code = self.symbols[patch.code]
            k = int(code.attributes["k"]) if "k" in code.attributes else 1
            if logical < k:
                basis_index = logical
                x_name, z_name = "lx", "lz"
            else:
                basis_index = logical - k
                x_name, z_name = "gx", "gz"
            try:
                x_support = tuple(
                    int(value)
                    for value in code.attributes[x_name][basis_index])
                z_support = tuple(
                    int(value)
                    for value in code.attributes[z_name][basis_index])
            except (KeyError, IndexError) as exc:
                raise ValueError(
                    f"code @{patch.code} has no representative for logical {logical}"
                ) from exc
            if pauli in {"X", "Y"}:
                if pauli == "Y":
                    phase *= 1j
                for index in x_support:
                    add(patch, index, "X")
            if pauli in {"Z", "Y"}:
                for index in z_support:
                    add(patch, index, "Z")

        carriers_by_resource = {
            carrier.resource: carrier for patch in patches
            for carrier in patch.partitions["data"]
        }
        ordered = tuple(
            (carriers_by_resource[resource], physical[resource])
            for resource in sorted(
                physical, key=lambda item: self._resource_indices[item]))
        if not ordered:
            raise ValueError("physical Pauli product reduced to identity")
        if abs(phase.imag) > 1e-9 or abs(abs(phase.real) - 1.0) > 1e-9:
            raise ValueError(
                "logical Pauli product did not project to a Hermitian physical operator"
            )
        return ordered, phase.real < 0

    def _measure_product(self, operation, patches, source_symbol,
                         instance_symbol):
        regions = {
            patch.region for patch in patches if patch.region is not None
        }
        if len(regions) > 1 and not self._communication_stack:
            raise ValueError(
                "cross-region fabric.measure_product requires an explicit "
                "communication-qualified fabric.call realization")
        ordered, invert = self._physical_pauli_product(operation, patches)
        carriers = tuple(item[0] for item in ordered)
        record = (_text(operation.attributes["record"])
                  if "record" in operation.attributes else "mpp")
        source_record = f"{source_symbol}.{record}"
        key = (instance_symbol, source_record)
        projected = self._record_projection.setdefault(key, [])
        # ``measure_product`` has one typed outcome field.  Keep the historic
        # bare record spelling as an alias while projecting the canonical
        # ``RecordRef`` spelling emitted by ``gadget.records.product(...)``.
        outcome_key = (instance_symbol, f"{source_record}.outcome")
        self._record_projection.setdefault(outcome_key, projected)
        record_id = f"{instance_symbol}.{record}"
        if projected:
            record_id = f"{record_id}.occurrence{len(projected)}"

        instrument = self._native_instrument(carriers, "measure_product")
        scalar_z_instrument = (self._native_instrument(carriers, "measure") if
                               (len(ordered) == 1 and ordered[0][1] == "Z" and
                                not invert) else None)
        if (instrument is None and scalar_z_instrument is not None):
            # A weight-one Z product is exactly scalar Z readout; requiring a
            # many-body MPP instrument here would reject devices whose honest
            # native realization is ordinary measurement. Keep the selected
            # typed instrument symbol so controller targets consume its linked
            # semantics rather than inferring behavior from a spelling.
            carrier, = self._ensure_zone(carriers,
                                         "readout",
                                         trajectory="readout")
            measurement = self.transaction.materialize(
                scalar_z_instrument).symbol
            record_type = mlir_ir.Type.parse("!phys.record<@bit>",
                                             context=self.context)
            event = self._emit(
                "phys.measure",
                operands=[carrier.value],
                results=[carrier.value.type, record_type],
                attributes={
                    "measurement":
                        mlir_ir.FlatSymbolRefAttr.get(measurement,
                                                      context=self.context),
                    "record_id":
                        mlir_ir.StringAttr.get(record_id, context=self.context),
                    "event_id":
                        mlir_ir.StringAttr.get(self._event_id("measure"),
                                               context=self.context),
                },
            )
            replacement = carrier.with_value(event.results[0])
            outputs = []
            for patch in patches:
                result = patch.clone()
                for partition in result.partitions.values():
                    for index, current in enumerate(partition):
                        if current.resource == carrier.resource:
                            partition[index] = replacement
                outputs.append(result)
            self._record_values[record_id] = event.results[-1]
            self._record_ids_by_value[event.results[-1]] = record_id
            self._append_projected_record(projected, record_id)
            return (*outputs, event.results[-1])

        if instrument is None:
            raise NotImplementedError(
                "logical Pauli-product intent requires a selected MPP protocol "
                "or physical resource classes advertising the typed MPP "
                "instrument")
        # The replayed module usually already carries the typed instrument
        # the architecture's resource classes advertise (materialized with
        # the device). Reuse that symbol: minting a fresh uniqued duplicate
        # (e.g. @mpp_1) would fail the native-instrument membership check
        # because no resource class advertises the duplicate.
        existing = self.transaction.find_symbol(instrument.name,
                                                "phys.instrument")
        if existing is not None:
            instrument_symbol = _symbol(existing)
        else:
            instrument_symbol = self.transaction.materialize(instrument).symbol
        record_type = mlir_ir.Type.parse(
            f"!phys.record<@{instrument.record_schema}>",
            context=self.context,
        )
        attrs = {
            "instrument":
                mlir_ir.FlatSymbolRefAttr.get(instrument_symbol,
                                              context=self.context),
            "paulis":
                mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.StringAttr.get(pauli, context=self.context)
                        for _, pauli in ordered
                    ],
                    context=self.context,
                ),
            "record_id":
                mlir_ir.StringAttr.get(record_id, context=self.context),
            "event_id":
                mlir_ir.StringAttr.get(self._event_id("mpp"),
                                       context=self.context),
        }
        if invert:
            attrs["invert"] = mlir_ir.UnitAttr.get(context=self.context)
        event = self._emit(
            "phys.measure_product",
            operands=[carrier.value for carrier in carriers],
            results=[
                *(carrier.value.type for carrier in carriers), record_type
            ],
            attributes=attrs,
        )
        replacements = {
            carrier.resource: carrier.with_value(result)
            for carrier, result in zip(carriers, event.results)
        }
        outputs = []
        for patch in patches:
            result = patch.clone()
            for partition in result.partitions.values():
                for index, carrier in enumerate(partition):
                    partition[index] = replacements.get(carrier.resource,
                                                        carrier)
            outputs.append(result)
        self._record_values[record_id] = event.results[-1]
        self._record_ids_by_value[event.results[-1]] = record_id
        self._append_projected_record(projected, record_id)
        return (*outputs, event.results[-1])

    def _rotate_product(self, operation, patches):
        ordered, invert = self._physical_pauli_product(operation, patches)
        carriers = tuple(item[0] for item in ordered)
        if not (self._supports_physical_capability(
                carriers, NATIVE_PAULI_PRODUCT_ROTATION) or
                self._supports_compat_action(carriers, "rpp")):
            raise NotImplementedError(
                "logical Pauli-product rotation intent requires a selected RPP "
                "protocol or carriers equipped with the typed native "
                "Pauli-product-rotation capability")
        angle = float(operation.attributes["angle"])
        if invert:
            angle = -angle
        with self.location:
            angle_attr = mlir_ir.FloatAttr.get(
                mlir_ir.F64Type.get(context=self.context), angle)
        event = self._emit(
            "phys.rotate_product",
            operands=[carrier.value for carrier in carriers],
            results=[carrier.value.type for carrier in carriers],
            attributes={
                "paulis":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.StringAttr.get(pauli, context=self.context)
                            for _, pauli in ordered
                        ],
                        context=self.context,
                    ),
                "angle":
                    angle_attr,
                "event_id":
                    mlir_ir.StringAttr.get(self._event_id("rpp"),
                                           context=self.context),
            },
        )
        replacements = {
            carrier.resource: carrier.with_value(result)
            for carrier, result in zip(carriers, event.results)
        }
        outputs = []
        for patch in patches:
            result = patch.clone()
            for partition in result.partitions.values():
                for index, carrier in enumerate(partition):
                    partition[index] = replacements.get(carrier.resource,
                                                        carrier)
            outputs.append(result)
        return tuple(outputs)

    def _resource_rotate_product(self, operation, resource, patches):
        ordered, invert = self._physical_pauli_product(operation, patches)
        carriers = tuple(item[0] for item in ordered)
        angle = float(operation.attributes["angle"])
        if invert:
            angle = -angle
        with self.location:
            angle_attr = mlir_ir.FloatAttr.get(
                mlir_ir.F64Type.get(context=self.context), angle)
        event = self._emit(
            "phys.resource_rotate_product",
            operands=[resource, *(carrier.value for carrier in carriers)],
            results=[carrier.value.type for carrier in carriers],
            attributes={
                "paulis":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.StringAttr.get(pauli, context=self.context)
                            for _, pauli in ordered
                        ],
                        context=self.context,
                    ),
                "angle":
                    angle_attr,
                "event_id":
                    mlir_ir.StringAttr.get(self._event_id("resource_rpp"),
                                           context=self.context),
            },
        )
        replacements = {
            carrier.resource: carrier.with_value(result)
            for carrier, result in zip(carriers, event.results)
        }
        outputs = []
        for patch in patches:
            result = patch.clone()
            for partition in result.partitions.values():
                for index, carrier in enumerate(partition):
                    partition[index] = replacements.get(carrier.resource,
                                                        carrier)
            outputs.append(result)
        return tuple(outputs)

    def _epoch_transition(self, operation, patch):
        carriers = tuple(patch.all())
        references = _patch_references(operation.operands[0].type)
        if len(references) < 3:
            raise ValueError(
                "fabric.epoch_transition source lacks explicit epoch qualification"
            )
        attrs = {
            "source_epoch":
                mlir_ir.FlatSymbolRefAttr.get(references[2],
                                              context=self.context),
            "destination_epoch":
                operation.attributes["to_epoch"],
            "evidence":
                operation.attributes["evidence"],
            "event_id":
                mlir_ir.StringAttr.get(self._event_id("epoch_transition"),
                                       context=self.context),
        }
        if "logical_map" in operation.attributes:
            attrs["logical_map"] = operation.attributes["logical_map"]
        event = self._emit(
            "phys.epoch_transition",
            operands=[carrier.value for carrier in carriers],
            results=[carrier.value.type for carrier in carriers],
            attributes=attrs,
        )
        replacements = {
            carrier.resource: carrier.with_value(result)
            for carrier, result in zip(carriers, event.results)
        }
        output = patch.clone()
        for partition in output.partitions.values():
            for index, carrier in enumerate(partition):
                partition[index] = replacements[carrier.resource]
        return output

    def _append_selection_sidecar(
        self,
        concrete,
        declaration,
        *,
        source_profile=None,
        source_kind=None,
        source_row=None,
        source_instance=None,
        source_records=None,
        input_syndromes=None,
        constant=None,
        metadata=None,
    ):
        if not concrete:
            raise ValueError(
                "P2-to-P3 projection cannot materialize a selection row "
                "without at least one concrete physical record")
        symbol = self.transaction.unique_symbol(
            f"{self.graph_symbol}_selection{self._sidecar}")
        self._sidecar += 1
        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(symbol, context=self.context),
            "graph":
                mlir_ir.FlatSymbolRefAttr.get(self.graph_symbol,
                                              context=self.context),
            "records":
                mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.StringAttr.get(record, context=self.context)
                        for record in concrete
                    ],
                    context=self.context,
                ),
        }
        if source_profile is not None:
            if source_instance is None:
                raise ValueError(
                    "detached physical sidecars require a source instance")
            attrs["source_profile"] = mlir_ir.FlatSymbolRefAttr.get(
                source_profile, context=self.context)
            attrs["source_instance"] = mlir_ir.StringAttr.get(
                source_instance, context=self.context)
            attrs["record_projection"] = mlir_ir.FlatSymbolRefAttr.get(
                self.record_projection_symbol, context=self.context)
        if source_kind is not None:
            attrs["source_kind"] = mlir_ir.StringAttr.get(source_kind,
                                                          context=self.context)
        if source_row is not None:
            attrs["source_row"] = _i64(self.context, source_row)
        if source_records is not None:
            attrs["source_records"] = mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(record, context=self.context)
                    for record in source_records
                ],
                context=self.context,
            )
        if input_syndromes is not None:
            attrs["input_syndromes"] = input_syndromes
        if constant is None:
            try:
                constant = declaration.attributes["constant"]
            except KeyError:
                constant = False
        if isinstance(constant, bool):
            constant = mlir_ir.BoolAttr.get(constant, context=self.context)
        attrs["constant"] = constant
        # Fabric success rows are canonical mismatch bits. Their accepted
        # value is always zero; retry's true-on-accept predicate is a separate
        # P2 control expression.
        attrs["expected"] = mlir_ir.BoolAttr.get(False, context=self.context)
        for name in (
                "expected",
                "scope",
        ):
            if name in declaration.attributes:
                attrs[name] = declaration.attributes[name]
        attrs.update(metadata or {})
        with self.location:
            sidecar = mlir_ir.Operation.create(
                "phys.selection_sidecar",
                attributes=attrs,
                loc=self.location,
            )
            self.module.body.append(sidecar)
        if source_profile is not None:
            self._provenance_sidecars.append(
                (sidecar, source_instance,
                 tuple(zip(source_records or (), concrete))))

    def _project_source_support(
        self,
        *,
        profile_symbol,
        gadget_symbol,
        instance_symbol,
        source_records,
    ):
        """Return concrete-record supports for one stable affine source row."""

        projected_support = []
        for source_record in source_records:
            prefix = f"{gadget_symbol}."
            if not source_record.startswith(prefix):
                raise ValueError(
                    f"profile @{profile_symbol} record {source_record!r} is "
                    f"not owned by @{gadget_symbol}")
            occurrences = self._record_projection.get(
                (instance_symbol, source_record), ())
            if not occurrences:
                raise ValueError(
                    f"profile @{profile_symbol} record {source_record!r} has "
                    "no concrete P3 measurement record")
            projected_support.append(tuple(occurrences))

        # An empty support is retained as one projection instance; success
        # callers reject it before materializing an invalid P3 row.
        multiplicity = max(map(len, projected_support), default=1)
        if any(
                len(items) not in {1, multiplicity}
                for items in projected_support):
            raise ValueError(
                f"profile @{profile_symbol} row has incompatible folded "
                "record multiplicities")
        return tuple(
            tuple(items[0] if len(items) == 1 else items[occurrence]
                  for items in projected_support)
            for occurrence in range(multiplicity))

    @staticmethod
    def _profile_source_rows(declaration):
        if "records" in declaration.attributes:
            return (tuple(
                _text(value) for value in declaration.attributes["records"]),)
        return ((),)

    def _emit_profile_sidecars(self, profile, instance_symbol):
        """Project selected success rows onto concrete P3 record IDs."""

        profile_symbol = _symbol(profile)
        gadget_symbol = _text(profile.attributes["gadget"])
        declarations = []
        for region in profile.regions:
            for block in region.blocks:
                for view in block.operations:
                    if view.operation.name == "fabric.success":
                        declarations.append(view.operation)

        declared_rows = []
        for source_row, declaration in enumerate(declarations):
            for source_records in self._profile_source_rows(declaration):
                input_syndromes = (declaration.attributes["input_syndromes"] if
                                   "input_syndromes" in declaration.attributes
                                   else _empty_array_attr(self.context))
                constant = (bool(declaration.attributes["constant"].value)
                            if "constant" in declaration.attributes else False)
                declared_rows.append(
                    (source_records, input_syndromes, constant))
                concrete_rows = self._project_source_support(
                    profile_symbol=profile_symbol,
                    gadget_symbol=gadget_symbol,
                    instance_symbol=instance_symbol,
                    source_records=source_records,
                )
                for concrete in concrete_rows:
                    self._append_selection_sidecar(
                        concrete,
                        declaration,
                        source_profile=profile_symbol,
                        source_kind="profile",
                        source_row=source_row,
                        source_instance=instance_symbol,
                        source_records=source_records,
                        input_syndromes=input_syndromes,
                    )

        # OutcomeMap is the authority for application and success results. A
        # selected profile may restate the complete success table; when it
        # omits success rows entirely, projection derives them.
        authoritative = self._outcome_map_role_rows(profile, role="success")
        if not declarations:
            self._emit_outcome_map_sidecars(
                profile,
                instance_symbol,
                role="success",
                rows=authoritative,
            )
        elif authoritative:
            if len(declared_rows) != len(authoritative):
                raise ValueError(
                    f"profile @{profile_symbol} declares {len(declared_rows)} "
                    "success row(s), but its GadgetSpec OutcomeMap tags "
                    f"{len(authoritative)} authoritative row(s)")
            for ordinal, (actual, expected) in enumerate(
                    zip(declared_rows, authoritative)):
                (_source_row, expected_records, expected_inputs,
                 _expected_constant) = expected
                actual_records, actual_inputs, _actual_constant = actual
                if (actual_records != expected_records or
                        str(actual_inputs) != str(expected_inputs)):
                    raise ValueError(
                        f"profile @{profile_symbol} success row {ordinal} "
                        "disagrees with its authoritative GadgetSpec "
                        "OutcomeMap row")

    def _outcome_map_role_rows(self, profile, *, role):
        """Return canonical role-tagged affine rows for one profile gadget."""

        gadget_symbol = _text(profile.attributes["gadget"])
        gadget = self.symbols.get(gadget_symbol)
        if gadget is None or "spec" not in gadget.attributes:
            return ()
        spec = self.symbols.get(_text(gadget.attributes["spec"]))
        if spec is None or "outcome_map" not in spec.attributes:
            return ()
        outcome = {
            str(named.name): named.attr
            for named in spec.attributes["outcome_map"]
        }
        records = tuple(_text(value) for value in outcome["records"])
        matrix = outcome["rows"]
        row_count, column_count = tuple(
            int(value) for value in matrix.type.shape)
        values = tuple(int(value) for value in matrix)
        constants = tuple(int(value) for value in outcome["constants"])
        syndrome_rows = (
            tuple(outcome["input_syndromes"])
            if "input_syndromes" in outcome else tuple(
                _empty_array_attr(self.context) for _ in range(row_count)))
        role_rows = (tuple(outcome["roles"]) if "roles" in outcome else None)
        result = []
        for source_row in range(row_count):
            roles = (("result",) if role_rows is None else tuple(
                _text(value) for value in role_rows[source_row]))
            if role not in roles:
                continue
            selected = values[source_row * column_count:(source_row + 1) *
                              column_count]
            source_records = tuple(
                f"{gadget_symbol}.{record}"
                for record, enabled in zip(records, selected)
                if enabled)
            result.append(
                (source_row, source_records, syndrome_rows[source_row],
                 bool(constants[source_row])))
        return tuple(result)

    def _emit_outcome_map_sidecars(self,
                                   profile,
                                   instance_symbol,
                                   *,
                                   role,
                                   rows=None):
        profile_symbol = _symbol(profile)
        gadget_symbol = _text(profile.attributes["gadget"])
        gadget = self.symbols.get(gadget_symbol)
        if gadget is None or "spec" not in gadget.attributes:
            return
        spec = self.symbols.get(_text(gadget.attributes["spec"]))
        if spec is None or "outcome_map" not in spec.attributes:
            return
        rows = self._outcome_map_role_rows(profile,
                                           role=role) if rows is None else rows
        for source_row, source_records, input_syndromes, constant in rows:
            concrete_rows = self._project_source_support(
                profile_symbol=profile_symbol,
                gadget_symbol=gadget_symbol,
                instance_symbol=instance_symbol,
                source_records=source_records,
            )
            for concrete in concrete_rows:
                self._append_selection_sidecar(
                    concrete,
                    spec,
                    source_profile=profile_symbol,
                    source_kind="outcome_map",
                    source_row=source_row,
                    source_instance=instance_symbol,
                    source_records=source_records,
                    input_syndromes=input_syndromes,
                    constant=constant,
                )

    def _unpack_hierarchy(self, operation, parent):
        if not parent.live:
            raise ValueError("hierarchy unpack requires a live parent patch")
        hierarchy_name = _text(operation.attributes["hierarchy"])
        hierarchy = self.symbols.get(hierarchy_name)
        if hierarchy is None or hierarchy.name != "fabric.encoding_hierarchy":
            raise ValueError(
                f"hierarchy unpack references missing encoding hierarchy "
                f"@{hierarchy_name}")
        child_encoding_name = _text(hierarchy.attributes["child"])
        child_encoding = self.symbols.get(child_encoding_name)
        if child_encoding is None or child_encoding.name != "fabric.encoding":
            raise ValueError(
                f"encoding hierarchy @{hierarchy_name} references missing child "
                f"encoding @{child_encoding_name}")
        child_code = _text(child_encoding.attributes["code"])
        try:
            child_partitions = dict(self.codes[child_code])
        except KeyError as exc:
            raise ValueError(
                f"encoding hierarchy @{hierarchy_name} references missing child "
                f"code @{child_code}") from exc
        multiplicity = int(hierarchy.attributes["multiplicity"])
        if multiplicity <= 0:
            raise ValueError("encoding hierarchy multiplicity must be positive")

        children = [
            _Patch(
                child_code,
                child_encoding_name,
                {},
                parent.region,
                parent.patch_id,
                parent.slot,
                parent.patch_topology,
                parent.qec_region,
                parent.physical_binding,
            ) for _ in range(multiplicity)
        ]
        remainder = {}
        entries = []
        for partition, parent_carriers in parent.partitions.items():
            child_size = child_partitions.get(partition, 0)
            consumed = child_size * multiplicity
            if consumed > len(parent_carriers):
                raise ValueError(
                    f"hierarchy @{hierarchy_name} needs {consumed} {partition} "
                    f"carriers for its children, but parent @{parent.code} has "
                    f"{len(parent_carriers)}")
            for child_index, child in enumerate(children):
                start = child_index * child_size
                carriers = list(parent_carriers[start:start + child_size])
                child.partitions[partition] = carriers
                entries.extend(
                    f"{_text(operation.attributes['slot_group'])}[{child_index}]."
                    f"{partition}[{local}]={carrier.resource}"
                    for local, carrier in enumerate(carriers))
            remainder[partition] = list(parent_carriers[consumed:])

        missing = set(child_partitions) - set(parent.partitions)
        if any(child_partitions[name] for name in missing):
            raise ValueError(f"parent @{parent.code} lacks child partitions "
                             f"{sorted(missing)!r} required by @{child_code}")
        parent.live = False
        self._emit_hierarchy_projection(hierarchy_name, entries)
        return _PatchBundle(
            parent.code,
            children,
            remainder,
            parent.region,
            hierarchy_name,
            _text(operation.attributes["slot_group"]),
        )

    def _map_hierarchy_children(self, operation, bundle, stack,
                                instance_symbol):
        if not bundle.live:
            raise ValueError("map_children requires a live patch bundle")
        callee_name = _text(operation.attributes["callee"])
        callee = self.symbols.get(callee_name)
        if callee is None or callee.name not in {
                "fabric.gadget", "fabric.protocol"
        }:
            raise ValueError(
                f"map_children references missing callable @{callee_name}")
        transformed = []
        for child_index, child in enumerate(bundle.children):
            child_instance = (
                f"{instance_symbol}.{bundle.slot_group}[{child_index}]."
                f"{callee_name}")
            outputs = self._emit_callable(
                callee,
                (child,),
                stack,
                instance_symbol=child_instance,
            )
            if len(outputs) != 1 or not isinstance(outputs[0], _Patch):
                raise ValueError(
                    f"folded map @{callee_name} must transform exactly one child "
                    "patch; use an explicit protocol when child records or "
                    "additional values cross the hierarchy boundary")
            if outputs[0].code != child.code:
                raise ValueError(
                    f"folded map @{callee_name} changed child code "
                    f"@{child.code} to @{outputs[0].code}")
            transformed.append(outputs[0])
        bundle.live = False
        return _PatchBundle(
            bundle.parent_code,
            transformed,
            {
                name: list(values) for name, values in bundle.remainder.items()
            },
            bundle.region,
            bundle.hierarchy,
            bundle.slot_group,
        )

    def _pack_hierarchy(self, operation, bundle):
        if not bundle.live:
            raise ValueError("encoding_pack requires a live patch bundle")
        parent_code = _code_name(operation.results[0].type)
        if parent_code != bundle.parent_code:
            raise ValueError(
                f"encoding_pack result @{parent_code} does not match unpacked "
                f"parent @{bundle.parent_code}")
        partitions = {}
        for partition, _ in self.codes[parent_code]:
            carriers = []
            for child in bundle.children:
                carriers.extend(child.partitions.get(partition, ()))
            carriers.extend(bundle.remainder.get(partition, ()))
            partitions[partition] = carriers
        bundle.live = False
        child = bundle.children[0]
        references = _patch_references(operation.results[0].type)
        encoding = references[1] if len(references) > 1 else None
        return _Patch(
            parent_code,
            encoding,
            partitions,
            bundle.region,
            child.patch_id,
            child.slot,
            child.patch_topology,
            child.qec_region,
            child.physical_binding,
        )

    def _emit_hierarchy_projection(self, hierarchy, entries):
        if not entries:
            return
        symbol = self.transaction.unique_symbol(
            f"{self.graph_symbol}_hierarchy{self._hierarchy_projection}")
        self._hierarchy_projection += 1
        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(symbol, context=self.context),
            "graph":
                mlir_ir.FlatSymbolRefAttr.get(self.graph_symbol,
                                              context=self.context),
            "source_hierarchy":
                mlir_ir.FlatSymbolRefAttr.get(hierarchy, context=self.context),
            "entries":
                mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.StringAttr.get(entry, context=self.context)
                        for entry in entries
                    ],
                    context=self.context,
                ),
        }
        with self.location:
            self.module.body.append(
                mlir_ir.Operation.create(
                    "phys.hierarchy_projection",
                    attributes=attrs,
                    loc=self.location,
                ))

    def _condition(self, value, *, expected=True):
        if value is None:
            raise ValueError(
                "physical control condition has no projected value")
        if str(value.type) == "i1":
            return value
        if not str(value.type).startswith("!phys.record<"):
            raise TypeError(
                "physical control requires an i1 or projected physical record")
        i1 = mlir_ir.IntegerType.get_signless(1, context=self.context)
        attrs = {
            "expected":
                mlir_ir.BoolAttr.get(expected, context=self.context),
            "event_id":
                mlir_ir.StringAttr.get(self._event_id("condition"),
                                       context=self.context),
        }
        return self._emit("phys.condition",
                          operands=[value],
                          results=[i1],
                          attributes=attrs).result

    @staticmethod
    def _flatten_projected(values):
        flattened = []
        for value in values:
            if isinstance(value, _Patch):
                flattened.extend(carrier.value for carrier in value.all())
            elif isinstance(value, (tuple, list)):
                flattened.extend(_P2ToP3._flatten_projected(value))
            elif value is not None:
                flattened.append(value)
        return tuple(flattened)

    @staticmethod
    def _patch_signature(patch):
        return tuple((name, tuple(carrier.resource
                                  for carrier in carriers))
                     for name, carriers in patch.partitions.items())

    def _trace_patch_template(self, value, local_values):
        mapped = local_values.get(value)
        if isinstance(mapped, _Patch):
            return mapped
        try:
            owner = value.owner
            result_number = value.result_number
        except (AttributeError, RuntimeError):
            owner = None
        if owner is not None:
            if owner.name == "cflow.if":
                alternatives = []
                for region in owner.regions:
                    terminator = region.blocks[0].operations[-1].operation
                    if result_number >= len(terminator.operands):
                        return None
                    candidate = self._trace_patch_template(
                        terminator.operands[result_number], local_values)
                    if candidate is None:
                        return None
                    alternatives.append(candidate)
                signatures = {
                    self._patch_signature(candidate)
                    for candidate in alternatives
                }
                return alternatives[0] if len(signatures) == 1 else None
            patch_results = [
                result for result in owner.results
                if _code_name(result.type) is not None
            ]
            patch_operands = [
                operand for operand in owner.operands
                if _code_name(operand.type) is not None
            ]
            try:
                ordinal = patch_results.index(value)
            except ValueError:
                ordinal = result_number
            if patch_operands:
                source = patch_operands[min(ordinal, len(patch_operands) - 1)]
                traced = self._trace_patch_template(source, local_values)
                if traced is not None:
                    return traced
        code = _code_name(value.type)
        candidates = []
        seen = set()
        for candidate in local_values.values():
            if not isinstance(candidate, _Patch) or candidate.code != code:
                continue
            signature = self._patch_signature(candidate)
            if signature not in seen:
                seen.add(signature)
                candidates.append(candidate)
        return candidates[0] if len(candidates) == 1 else None

    def _branch_templates(self, operation, local_values):
        then_block = operation.regions[0].blocks[0]
        terminator = then_block.operations[-1].operation
        template_values = dict(local_values)
        carries = tuple(
            local_values.get(value) for value in operation.operands[1:])
        template_values.update(zip(then_block.arguments, carries))
        templates = []
        for result, yielded in zip(operation.results, terminator.operands):
            if _code_name(result.type) is not None:
                patch = self._trace_patch_template(yielded, template_values)
                if patch is None:
                    raise ValueError(
                        "cannot determine the physical carrier boundary for a "
                        "conditional patch result")
                templates.append(patch)
                continue
            projected = template_values.get(yielded)
            if projected is None:
                raise NotImplementedError(
                    "conditional scalar results require an explicitly projected "
                    "physical value")
            templates.append(projected)
        return tuple(templates)

    def _rebuild_projected(self, templates, results):
        cursor = iter(results)

        def rebuild(template):
            if isinstance(template, _Patch):
                partitions = {}
                for name, carriers in template.partitions.items():
                    partitions[name] = [
                        carrier.with_value(next(cursor)) for carrier in carriers
                    ]
                return _Patch(
                    template.code,
                    template.encoding,
                    partitions,
                    template.region,
                    template.patch_id,
                    template.slot,
                    template.patch_topology,
                    template.qec_region,
                    template.physical_binding,
                )
            if isinstance(template, tuple):
                return tuple(rebuild(value) for value in template)
            if isinstance(template, list):
                return [rebuild(value) for value in template]
            return next(cursor)

        rebuilt = [rebuild(template) for template in templates]
        try:
            next(cursor)
        except StopIteration:
            return tuple(rebuilt)
        raise ValueError("structured physical result boundary has extra values")

    def _emit_conditional(self, operation, local_values, execute, call_stack,
                          instance_symbol):
        condition = self._condition(local_values.get(operation.operands[0]))
        templates = self._branch_templates(operation, local_values)
        result_types = [
            value.type for value in self._flatten_projected(templates)
        ]
        with self.location:
            physical_if = mlir_ir.Operation.create(
                "cflow.if",
                operands=[condition],
                results=result_types,
                attributes={
                    "event_id":
                        mlir_ir.StringAttr.get(self._event_id("if"),
                                               context=self.context)
                },
                regions=2,
                loc=self.location,
            )
        self.ip.insert(physical_if)
        parent_ip = self.ip
        branch_results = []
        control = self._exclusive_control
        self._exclusive_control += 1
        for branch, (source_region, target_region) in enumerate(
                zip(operation.regions, physical_if.regions)):
            target_block = target_region.blocks.append()
            self.ip = mlir_ir.InsertionPoint(target_block)
            with self._exclusive_branch(control, branch):
                source_block = source_region.blocks[0]
                branch_values = dict(local_values)
                branch_values.update(
                    zip(
                        source_block.arguments,
                        (local_values.get(value)
                         for value in operation.operands[1:]),
                    ))
                projected = execute(source_block, branch_values,
                                    call_stack) or ()
                flattened = self._flatten_projected(projected)
                if tuple(value.type
                         for value in flattened) != tuple(result_types):
                    raise ValueError(
                        "physical conditional branches must yield identical "
                        "state and record boundaries")
                self._emit("cflow.yield", operands=flattened)
            branch_results.append(projected)
        self.ip = parent_ip
        return self._rebuild_projected(templates, physical_if.results)

    def _emit_while(
        self,
        operation,
        operands,
        local_values,
        execute,
        call_stack,
    ):
        if any(value is None for value in operands):
            raise ValueError("cflow.while has an unprojected physical carry")
        templates = tuple(operands)
        states = self._flatten_projected(templates)
        result_types = [value.type for value in states]
        attrs = {
            "event_id":
                mlir_ir.StringAttr.get(self._event_id("while"),
                                       context=self.context)
        }
        if "max_iterations" in operation.attributes:
            attrs["max_iterations"] = operation.attributes["max_iterations"]
        with self.location:
            physical = mlir_ir.Operation.create(
                "cflow.while",
                operands=states,
                results=result_types,
                attributes=attrs,
                regions=2,
                loc=self.location,
            )
        self.ip.insert(physical)
        parent_ip = self.ip
        try:
            source_before = operation.regions[0].blocks[0]
            with self.location:
                target_before = physical.regions[0].blocks.append(*result_types)
            before_values = self._rebuild_projected(templates,
                                                    target_before.arguments)
            before_mapping = dict(local_values)
            before_mapping.update(zip(source_before.arguments, before_values))
            self.ip = mlir_ir.InsertionPoint(target_before)
            conditioned = execute(source_before, before_mapping, call_stack)
            if conditioned is None or len(conditioned) != len(templates) + 1:
                raise ValueError(
                    "cflow.while_condition must return a predicate and every carry"
                )
            condition = self._condition(conditioned[0])
            forwarded = tuple(conditioned[1:])
            forwarded_states = self._flatten_projected(forwarded)
            if tuple(value.type
                     for value in forwarded_states) != tuple(result_types):
                raise ValueError(
                    "physical while condition must preserve every carried state type"
                )
            self._emit(
                "cflow.while_condition",
                operands=[condition, *forwarded_states],
            )

            source_after = operation.regions[1].blocks[0]
            with self.location:
                target_after = physical.regions[1].blocks.append(*result_types)
            after_values = self._rebuild_projected(templates,
                                                   target_after.arguments)
            after_mapping = dict(local_values)
            after_mapping.update(zip(source_after.arguments, after_values))
            self.ip = mlir_ir.InsertionPoint(target_after)
            yielded = execute(source_after, after_mapping, call_stack)
            if yielded is None or len(yielded) != len(templates):
                raise ValueError(
                    "cflow.while body must yield every physical carry")
            yielded_states = self._flatten_projected(yielded)
            if tuple(value.type
                     for value in yielded_states) != tuple(result_types):
                raise ValueError(
                    "physical while body must preserve every carried state type"
                )
            self._emit("cflow.yield", operands=yielded_states)
        finally:
            self.ip = parent_ip
        return self._rebuild_projected(templates, physical.results)

    def _emit_repeat(
        self,
        operation,
        operands,
        local_values,
        execute,
        call_stack,
    ):
        if any(value is None for value in operands):
            raise ValueError("cflow.repeat has an unprojected physical carry")
        templates = tuple(operands)
        states = self._flatten_projected(templates)
        result_types = [value.type for value in states]
        repeat_event = self._event_id("repeat")
        repeat_count = int(operation.attributes["count"])
        attrs = {
            "count":
                operation.attributes["count"],
            "event_id":
                mlir_ir.StringAttr.get(repeat_event, context=self.context),
        }
        with self.location:
            physical = mlir_ir.Operation.create(
                "cflow.repeat",
                operands=states,
                results=result_types,
                attributes=attrs,
                regions=1,
                loc=self.location,
            )
        self.ip.insert(physical)
        parent_ip = self.ip
        try:
            source_body = operation.regions[0].blocks[0]
            with self.location:
                target_body = physical.regions[0].blocks.append(*result_types)
            body_values = self._rebuild_projected(templates,
                                                  target_body.arguments)
            body_mapping = dict(local_values)
            body_mapping.update(zip(source_body.arguments, body_values))
            self.ip = mlir_ir.InsertionPoint(target_body)
            self._repeat_context.append((repeat_event, repeat_count))
            try:
                yielded = execute(source_body, body_mapping, call_stack)
            finally:
                self._repeat_context.pop()
            if yielded is None or len(yielded) != len(templates):
                raise ValueError(
                    "cflow.repeat body must yield every physical carry")
            yielded_states = self._flatten_projected(yielded)
            if tuple(value.type
                     for value in yielded_states) != tuple(result_types):
                raise ValueError(
                    "physical repeat body must preserve every carried state type"
                )
            self._emit("cflow.yield", operands=yielded_states)
        finally:
            self.ip = parent_ip
        return self._rebuild_projected(templates, physical.results)

    def _emit_retry(self, operation, operands, patches):
        if not patches:
            raise ValueError(
                "physical retry requires at least one carried patch")
        if "attempt" not in operation.attributes:
            raise ValueError("physical retry has no selected attempt gadget")
        if "profile" not in operation.attributes:
            raise ValueError("physical retry has no selected attempt profile")
        attempt = operation.attributes["attempt"]
        attempt_definition = self.symbols.get(_text(attempt))
        if (attempt_definition is None or
                attempt_definition.name != "fabric.gadget"):
            raise ValueError(
                "physical retry attempt must resolve to fabric.gadget")
        success = self._condition(operands[-1])
        states = self._flatten_projected(patches)
        attempt_operations = {
            value.owner.operation
            for value in states
            if hasattr(value, "owner")
        }
        if (len(attempt_operations) != 1 or
                next(iter(attempt_operations)).name != "phys.call"):
            raise ValueError(
                "physical retry carries must originate from one projected "
                "attempt call")
        attempt_operation = next(iter(attempt_operations))
        decision_operation = success.owner.operation
        if "event_id" not in attempt_operation.attributes:
            raise ValueError("projected retry attempt call has no event ID")
        if "event_id" not in decision_operation.attributes:
            raise ValueError("projected retry success decision has no event ID")
        attrs = {
            "max_attempts":
                operation.attributes["max_attempts"],
            "attempt":
                attempt,
            "profile":
                operation.attributes["profile"],
            "attempt_event":
                attempt_operation.attributes["event_id"],
            "decision_event":
                decision_operation.attributes["event_id"],
            "event_id":
                mlir_ir.StringAttr.get(self._event_id("retry"),
                                       context=self.context),
        }
        for name in (
                "exhaustion",
                "commit_point",
                "success_probability",
                "success_probability_source",
                "success_probability_evidence",
                "attempt",
                "profile",
        ):
            if name in operation.attributes:
                attrs[name] = operation.attributes[name]
        event = self._emit(
            "phys.retry",
            operands=[*states, success],
            results=[value.type for value in states],
            attributes=attrs,
        )
        return self._rebuild_projected(tuple(patches), event.results)

    def _emit_event_try_take(self, operation, operands, patches, execute,
                             call_stack):
        if not operands or operands[0] is None:
            raise ValueError("event_try_take has no projected physical event")
        if any(value is None for value in operands[1:]):
            raise ValueError("event_try_take has an unprojected physical carry")
        # Carries are a typed structured-control boundary, not a patch-only
        # special case. Patch templates expand to their physical carrier
        # states; scalar records, predicates, statuses, and indices remain one
        # SSA value and are rebuilt in their original order after the join.
        templates = tuple(operands[1:])
        states = self._flatten_projected(templates)
        result_types = [value.type for value in states]
        attrs = {
            "event_id":
                mlir_ir.StringAttr.get(self._event_id("try_take"),
                                       context=self.context)
        }
        with self.location:
            physical = mlir_ir.Operation.create(
                "event.try_take",
                operands=[operands[0], *states],
                results=result_types,
                attributes=attrs,
                regions=3,
                loc=self.location,
            )
        self.ip.insert(physical)
        event_type = operands[0].type
        kind = _fabric_event_resource_kind(operation.operands[0].type)
        alternative_types = (
            self._resource_payload_type(kind),
            event_type,
            mlir_ir.IntegerType.get_signless(8, context=self.context),
        )
        parent_ip = self.ip
        control = self._exclusive_control
        self._exclusive_control += 1
        try:
            branches = zip(operation.regions, physical.regions,
                           alternative_types)
            for branch, (
                    source_region,
                    target_region,
                    alternative,
            ) in enumerate(branches):
                source_block = source_region.blocks[0]
                with self.location:
                    target_block = target_region.blocks.append(
                        alternative, *result_types)
                rebuilt = self._rebuild_projected(templates,
                                                  target_block.arguments[1:])
                nested = {source_block.arguments[0]: target_block.arguments[0]}
                nested.update(zip(source_block.arguments[1:], rebuilt))
                self.ip = mlir_ir.InsertionPoint(target_block)
                with self._exclusive_branch(control, branch):
                    projected = (execute(source_block, nested, call_stack) or
                                 rebuilt)
                    flattened = self._flatten_projected(projected)
                    if tuple(value.type
                             for value in flattened) != tuple(result_types):
                        raise ValueError(
                            "physical event_try_take branches must preserve "
                            "carry types")
                    self._emit("event.yield", operands=flattened)
        finally:
            self.ip = parent_ip
        return self._rebuild_projected(templates, physical.results)

    def _pair_indices(self, operation, patches):
        ctrl_name = _partition(operation.attributes["ctrl"])
        targ_name = _partition(operation.attributes["targ"])
        left, right = patches[0], patches[-1]
        if "pairs" in operation.attributes:
            pairs = _pairs(
                operation.attributes["pairs"],
                len(left.selection(ctrl_name)),
                len(right.selection(targ_name)),
            )
        elif "schedule" in operation.attributes:
            schedule = _text(operation.attributes["schedule"]).lower()
            code = self.symbols[left.code]
            rows = tuple(
                tuple(int(value)
                      for value in row)
                for row in code.attributes[schedule])
            if schedule == "hx":
                pairs = tuple((check, data)
                              for check, row in enumerate(rows)
                              for data in row)
                ctrl_name, targ_name = "sx", "data"
            elif schedule == "hz":
                pairs = tuple((data, check)
                              for check, row in enumerate(rows)
                              for data in row)
                ctrl_name, targ_name = "data", "sz"
            else:
                raise ValueError(
                    f"unknown CSS interaction schedule {schedule!r}")
        else:
            left_size = len(left.selection(ctrl_name))
            right_size = len(right.selection(targ_name))
            if left_size != right_size:
                raise ValueError(
                    "physical CX/CZ projection needs explicit interaction pairs"
                )
            pairs = tuple(zip(range(left_size), range(right_size)))
        return ctrl_name, targ_name, pairs

    def _native_actions_of(self, carrier):
        classes = {
            item.name: item for item in self.device.physical.resource_classes
        }
        return classes[carrier.resource_class].native_actions

    def _legalize_and_route(self, action, control, target):
        """Route and legalize a semantic two-qubit action to native actions.

        Device-owned recipes take precedence and run through ``_routed_pair``
        so routing-introduced SWAPs must also be explicitly decomposed. The
        compiler's generic typed library remains the fallback for devices that
        advertise its concrete entangler directly. Both paths fail closed.
        """
        carriers = (control, target)
        if self._native_decomposition(self._classes_for(carriers),
                                      action) is not None:
            return self._routed_pair(control, target, action)

        pre, entangler, post = self._legalize_pair(action, control, target)
        for name in pre:
            (target,) = self._apply_carriers((target,), name)
        control, target = self._routed_pair(control, target, entangler)
        for name in post:
            (target,) = self._apply_carriers((target,), name)
        return control, target

    def _legalize_pair(self, action, control, target):
        """Choose a native decomposition of ``action`` on ``(control, target)``.

        Returns typed ``(pre_target, entangler, post_target)`` actions. The
        premise is typed action identity -- membership in each participating
        resource class's ``native_actions`` -- never a target's output spelling.
        A rule applies only when its entangler is native on every participant and
        its single-qubit actions are native on the target.
        """
        carriers = (control, target)
        semantic_action = next(
            (candidate for candidate in SEMANTIC_ACTIONS
             if candidate.name == action),
            None,
        )
        if semantic_action is not None and self._supports_native_action(
                carriers, semantic_action):
            return (), semantic_action, ()
        # Keep alpha compatibility strings valid only as direct native actions;
        # they cannot discharge a typed decomposition rule.
        if self._supports_compat_action(carriers, action):
            return (), action, ()
        for rule in NATIVE_DECOMPOSITIONS:
            if rule.action.name != action:
                continue
            if not self._supports_native_action(carriers, rule.entangler):
                continue
            if not all(
                    self._supports_native_action((target,), native)
                    for native in (*rule.pre_target, *rule.post_target)):
                continue
            self._legalizations.append({
                "event": f"legalize{len(self._legalizations)}",
                "action": action,
                "rule": rule.name,
                "native": (
                    *(item.name for item in rule.pre_target),
                    rule.entangler.name,
                    *(item.name for item in rule.post_target),
                ),
                "cost": rule.cost,
            })
            return (
                tuple(rule.pre_target),
                rule.entangler,
                tuple(rule.post_target),
            )
        available = sorted({
            getattr(native, "name", native)
            for carrier in carriers
            for native in self._native_actions_of(carrier)
        })
        raise ValueError(
            f"no legal decomposition for semantic action @{action} on resource "
            f"class(es) {sorted({control.resource_class, target.resource_class})}; "
            f"native actions available: {available}. phys-legalize-native-actions "
            "fails closed rather than substitute action spelling or target printer "
            "behavior for instruction selection")

    def _two_patch(self, operation, patches, action):
        results = [patch.clone() for patch in patches]
        left_patch, right_patch = results[0], results[-1]
        ctrl_name, targ_name, pairs = self._pair_indices(operation, patches)
        right_index = len(patches) - 1
        left_selection = results[0].selection(ctrl_name)
        right_selection = results[right_index].selection(targ_name)
        for ctrl, targ in pairs:
            left_name, left_index, _ = left_selection[ctrl]
            right_name, right_index_in_partition, _ = (right_selection[targ])
            if right_index == 0 and (
                    left_name,
                    left_index,
            ) == (
                    right_name,
                    right_index_in_partition,
            ):
                raise PlacementInfeasible(
                    "physical CX/CZ projection rejects a self-aliasing "
                    "two-carrier interaction")
            left_carrier = results[0].partitions[left_name][left_index]
            right_carrier = results[right_index].partitions[right_name][
                right_index_in_partition]
            left_out, right_out = self._legalize_and_route(
                action, left_carrier, right_carrier)
            results[0].partitions[left_name][left_index] = left_out
            results[right_index].partitions[right_name][
                right_index_in_partition] = right_out
        interaction = {
            "id":
                f"interaction{len(self._patch_interactions)}",
            "action":
                action,
            "patches": (left_patch.patch_id, right_patch.patch_id),
            "slots": (left_patch.slot, right_patch.slot),
            "pairs":
                tuple(pairs),
            "regions": (left_patch.region, right_patch.region),
            "qec_regions": (left_patch.qec_region, right_patch.qec_region),
            "physical_bindings": (
                left_patch.physical_binding,
                right_patch.physical_binding,
            ),
        }
        if (left_patch.region != right_patch.region and
                self._communication_stack):
            interaction.update(self._communication_stack[-1])
        self._patch_interactions.append(interaction)
        return tuple(results)

    def _delay(self, patch, rounds):
        result = patch.clone()
        carriers = tuple(result.all())
        if not carriers:
            return result
        timing = ({} if self.device.operating_point is None else
                  self.device.operating_point.timing)
        cycle = float(timing.get("cycle_ns", 1.0))
        # FloatAttr.get requires an active location even for a detached
        # attribute.
        with mlir_ir.Location.unknown(self.context):
            duration = mlir_ir.FloatAttr.get(
                mlir_ir.F64Type.get(context=self.context), rounds * cycle)
        operation = self._emit(
            "phys.delay",
            operands=[carrier.value for carrier in carriers],
            results=[carrier.value.type for carrier in carriers],
            attributes={
                "duration_ns":
                    duration,
                "event_id":
                    mlir_ir.StringAttr.get(self._event_id("idle"),
                                           context=self.context),
            },
        )
        replacements = iter(operation.results)
        for partition in result.partitions.values():
            for index, carrier in enumerate(partition):
                partition[index] = carrier.with_value(next(replacements))
        return result

    def _moment_barrier(self, local_values):
        """Thread every live carrier through one authored moment boundary.

        ``fabric.tick`` has no SSA boundary of its own, but its P3 projection
        must still order the physical events on both sides.  Select the newest
        projected value for each live patch/bundle, then carry each distinct
        carrier state through one zero-duration ``phys.barrier``.  Subsequent
        events consume the barrier results, so the typed scheduler and every
        target observe the same local moment boundary without synchronizing
        unrelated physical resources elsewhere in the graph.
        """

        current = {}
        for projected in local_values.values():
            if isinstance(projected, _Patch) and projected.live:
                current[("patch", projected.patch_id)] = projected
            elif isinstance(projected, _PatchBundle) and projected.live:
                current[("bundle", projected.hierarchy)] = projected

        # resource -> (mutable carrier list, index, current carrier). Later
        # projected owners replace stale aliases retained in local_values.
        slots = {}

        def add_patch(patch):
            if not patch.live:
                return
            for carriers in patch.partitions.values():
                for index, carrier in enumerate(carriers):
                    slots[carrier.resource] = (carriers, index, carrier)

        for projected in current.values():
            if isinstance(projected, _Patch):
                add_patch(projected)
                continue
            for child in projected.children:
                add_patch(child)
            for carriers in projected.remainder.values():
                for index, carrier in enumerate(carriers):
                    slots[carrier.resource] = (carriers, index, carrier)

        if not slots:
            return
        ordered = [slots[resource] for resource in sorted(slots)]
        operation = self._emit(
            "phys.barrier",
            operands=[carrier.value for _, _, carrier in ordered],
            results=[carrier.value.type for _, _, carrier in ordered],
            attributes={
                "event_id":
                    mlir_ir.StringAttr.get(self._event_id("tick"),
                                           context=self.context),
            },
        )
        for result, (carriers, index, carrier) in zip(operation.results,
                                                      ordered):
            carriers[index] = carrier.with_value(result)

    def _release(self, patch):
        if not patch.live:
            raise ValueError("physical projection encountered a double release")
        carriers = tuple(patch.all())
        release_id = self._event_id("release")
        self._emit(
            "phys.release",
            operands=[carrier.value for carrier in carriers],
            attributes={
                "event_id":
                    mlir_ir.StringAttr.get(release_id, context=self.context)
            },
        )
        allocation = self._allocation_by_patch.get(patch.patch_id)
        if allocation is None:
            raise ValueError(
                f"physical release has no allocation binding for {patch.patch_id}"
            )
        allocation.release = release_id
        self._record_release(allocation, release_id)
        active = self._allocated_indices[allocation.resource_class]
        patch_active = self._patch_allocated_indices[allocation.resource_class]
        active.difference_update(allocation.indices)
        if allocation.slot is not None and allocation.slot_binding is not None:
            slot_binding = next(
                binding for binding in self.device.qec_to_physical
                if binding.name == allocation.slot_binding)
            patch_active.difference_update(
                slot_binding.patch_topology.carrier_groups[allocation.slot])
            self._slot_allocations[allocation.slot_binding].discard(
                allocation.slot)
        else:
            patch_active.difference_update(allocation.indices)
        patch.live = False

    def _assign(self, operation, values, outputs):
        if len(outputs) != len(operation.results):
            raise ValueError(
                f"{operation.name} physical projection returned {len(outputs)} "
                f"values for {len(operation.results)} SSA results")
        for result, output in zip(operation.results, outputs):
            values[result] = output

    @staticmethod
    def _replace_nested_operands(operation, replacements):
        for index, operand in enumerate(tuple(operation.operands)):
            replacement = replacements.get(operand)
            if replacement is not None:
                operation.operands[index] = replacement
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    _P2ToP3._replace_nested_operands(child.operation,
                                                     replacements)

    def _emit_folded_call(
        self,
        callee,
        arguments,
        call_stack,
        *,
        instance_symbol,
        analysis_profile=None,
        call_attributes=None,
    ):
        parent_block = self.ip.block
        before = len(parent_block.operations)
        outer_scratch = self._scratch
        self._scratch = {}
        try:
            outputs = self._emit_callable(
                callee,
                arguments,
                call_stack,
                instance_symbol=instance_symbol,
                analysis_profile=analysis_profile,
            )
            self._release_route_scratch(self._scratch)
        finally:
            self._scratch = outer_scratch
        body_operations = list(parent_block.operations)[before:]
        inputs = self._flatten_projected(arguments)
        yielded = self._flatten_projected(outputs)
        attrs = {
            "callee":
                mlir_ir.FlatSymbolRefAttr.get(_symbol(callee),
                                              context=self.context),
            "instance":
                mlir_ir.StringAttr.get(instance_symbol, context=self.context),
            "event_id":
                mlir_ir.StringAttr.get(self._event_id("call"),
                                       context=self.context),
        }
        if analysis_profile is not None:
            attrs["profile"] = mlir_ir.FlatSymbolRefAttr.get(
                _symbol(analysis_profile), context=self.context)
        if call_attributes:
            attrs.update(call_attributes)
        with self.location:
            folded = mlir_ir.Operation.create(
                "phys.call",
                operands=inputs,
                results=[value.type for value in yielded],
                attributes=attrs,
                regions=1,
                loc=self.location,
            )
        self.ip.insert(folded)
        with self.location:
            body = folded.regions[0].blocks.append(
                *(value.type for value in inputs))
        replacements = dict(zip(inputs, body.arguments))
        for view in body_operations:
            view.detach_from_parent()
            body.append(view)
            self._replace_nested_operands(view.operation, replacements)
        body_ip = mlir_ir.InsertionPoint(body)
        with self.location:
            terminator = mlir_ir.Operation.create(
                "phys.yield",
                operands=[replacements.get(value, value) for value in yielded],
                loc=self.location,
            )
        body_ip.insert(terminator)
        return self._rebuild_projected(outputs, folded.results)

    def _emit_callable(
            self,
            callable_op,
            arguments,
            stack=(),
            *,
            instance_symbol=None,
            analysis_profile=None,
    ):
        symbol = _symbol(callable_op)
        metadata = callable_op.attributes.get("metadata")
        if metadata is not None and "runtime_replay" in metadata:
            runtime_replay = _text(metadata["runtime_replay"])
            compiler = (_text(metadata["compiler"]) if "compiler" in metadata
                        else "architecture-selected protocol")
            if runtime_replay == "unsupported":
                raise NotImplementedError(
                    f"{compiler} selected P2 protocol gadget @{symbol} cannot "
                    "project to P3: runtime attempt replay is not implemented")
            if runtime_replay != "bounded_p3_retry":
                raise ValueError(
                    f"{compiler} selected P2 protocol gadget @{symbol} has "
                    f"unknown metadata.runtime_replay {runtime_replay!r}; "
                    "expected 'bounded_p3_retry'")
        instance_symbol = instance_symbol or symbol
        if symbol in stack:
            raise ValueError(f"recursive Fabric call graph through @{symbol}")
        body_owner = callable_op
        if (callable_op.name == "fabric.gadget" and
                "realization" in callable_op.attributes):
            realization = _text(callable_op.attributes["realization"])
            body_owner = self.symbols.get(realization)
            if body_owner is None or body_owner.name != "fabric.circuit":
                raise ValueError(
                    f"gadget @{symbol} references missing realization @{realization}"
                )
        block = body_owner.regions[0].blocks[0]
        if len(arguments) != len(block.arguments):
            raise ValueError(f"physical call boundary mismatch at @{symbol}")
        values = dict(zip(block.arguments, arguments))

        def execute(block_, local_values, call_stack):
            returned = None
            for view in block_.operations:
                operation = view.operation
                name = operation.name
                if name in {
                        "fabric.return",
                        "fabric.protocol_return",
                        # `cflow.yield` terminates cflow.if/repeat/while
                        # bodies; `event.yield` terminates
                        # event.try_take branches (which this
                        # generic walker also executes) -- both just
                        # capture the returned operands.
                        "cflow.yield",
                        "event.yield"
                }:
                    returned = tuple(
                        local_values.get(value) for value in operation.operands)
                    continue
                if name == "cflow.while_condition":
                    returned = tuple(
                        local_values.get(value) for value in operation.operands)
                    continue
                if name == "fabric.alloc":
                    code = _text(operation.attributes["code"])
                    region = _text(operation.attributes["region"])
                    references = _patch_references(operation.result.type)
                    encoding = references[1] if len(references) > 1 else None
                    self._assign(
                        operation,
                        local_values,
                        (self._allocate_patch(code,
                                              region,
                                              f"patch{self._event}",
                                              encoding=encoding),),
                    )
                    continue
                operands = [
                    local_values.get(value) for value in operation.operands
                ]
                patches = [
                    value for value in operands if isinstance(value, _Patch)
                ]
                if name == "fabric.produce_resource":
                    kind = _fabric_resource_kind(operation.result.type)
                    produced = self._emit(
                        "phys.produce_resource",
                        results=[self._resource_payload_type(kind)],
                        attributes={
                            "region":
                                operation.attributes["region"],
                            "resource_kind":
                                (operation.attributes["resource_kind"]
                                 if "resource_kind" in operation.attributes else
                                 mlir_ir.FlatSymbolRefAttr.get(
                                     kind, context=self.context)),
                            "protocol":
                                operation.attributes["protocol"],
                            "event_id":
                                mlir_ir.StringAttr.get(
                                    self._event_id("produce_resource"),
                                    context=self.context,
                                ),
                        },
                    )
                    self._assign(operation, local_values, (produced.result,))
                    continue
                if name == "fabric.transport":
                    attributes = {
                        "source":
                            operation.attributes["src_region"],
                        "destination":
                            operation.attributes["dst_region"],
                        "protocol":
                            operation.attributes["protocol"],
                        "event_id":
                            mlir_ir.StringAttr.get(
                                self._event_id("transport_resource"),
                                context=self.context,
                            ),
                    }
                    if "route" in operation.attributes:
                        route_value = getattr(operation.attributes["route"],
                                              "value", None)
                        route_name = (str(route_value[-1]) if isinstance(
                            route_value, (tuple, list)) else _text(
                                operation.attributes["route"]).split("::@")[-1])
                        routes = tuple(
                            channel for channel in self.device.qec.channels
                            if channel.name ==
                            route_name) if self.device.qec is not None else ()
                        if len(routes) != 1:
                            raise ValueError(
                                "fabric transport route is not selected by the device"
                            )
                        bindings = tuple(
                            binding
                            for binding in self.device.qec_channels_to_physical
                            if binding.qec_channel is routes[0])
                        if len(bindings) != 1:
                            raise ValueError(
                                "fabric transport route has no exact physical binding"
                            )
                        with self.context:
                            attributes["route"] = mlir_ir.SymbolRefAttr.get(
                                [
                                    self.architecture,
                                    f"{routes[0].name}_physical",
                                ],
                                context=self.context,
                            )
                            model = self._transport_models.get(id(routes[0]))
                            if model is not None:
                                attributes["model"] = (
                                    mlir_ir.FlatSymbolRefAttr.get(
                                        _symbol(model), context=self.context))
                    transported = self._emit(
                        "phys.transport_resource",
                        operands=operands,
                        results=[operands[0].type],
                        attributes=attributes,
                    )
                    self._assign(operation, local_values, (transported.result,))
                    continue
                if name == "fabric.resource_request":
                    kind = _text(operation.attributes["kind"])
                    attributes = {
                        "kind":
                            operation.attributes["kind"],
                        "stream":
                            operation.attributes["stream"],
                        "event_id":
                            mlir_ir.StringAttr.get(
                                self._event_id("resource_request"),
                                context=self.context,
                            ),
                    }
                    retained_stream = self._region_symbol(
                        operation.attributes["stream"], "lvm.stream")
                    if retained_stream is None:
                        raise ValueError(
                            "physical resource request stream is not retained")
                    if "external" in retained_stream.attributes:
                        attributes["external"] = mlir_ir.UnitAttr.get(
                            context=self.context)
                    else:
                        for required in ("produced_by", "backing_region"):
                            if required not in retained_stream.attributes:
                                raise ValueError(
                                    "backed physical resource stream lacks "
                                    f"{required}")
                        path = _symbol_path(operation.attributes["stream"])
                        model = self._factory_models.get(path)
                        if model is None and path:
                            model = self._factory_models.get((path[-1],))
                        if model is not None:
                            with self.context:
                                region_reference = mlir_ir.SymbolRefAttr.get(
                                    [
                                        path[0],
                                        _text(retained_stream.
                                              attributes["backing_region"]),
                                    ],
                                    context=self.context,
                                )
                                physical_binding = mlir_ir.SymbolRefAttr.get(
                                    [
                                        self.architecture,
                                        _text(model.attributes["qec_binding"]),
                                    ],
                                    context=self.context,
                                )
                            attributes.update({
                                "provider":
                                    retained_stream.attributes["produced_by"],
                                "region":
                                    region_reference,
                                "factory_model":
                                    mlir_ir.FlatSymbolRefAttr.get(
                                        _symbol(model), context=self.context),
                                "physical_binding":
                                    physical_binding,
                            })
                            if "transfer" in retained_stream.attributes:
                                attributes["transfer"] = (
                                    retained_stream.attributes["transfer"])
                    requested = self._emit(
                        "phys.resource_request",
                        results=[self._resource_payload_event_type(kind)],
                        attributes=attributes,
                    )
                    self._assign(operation, local_values, (requested.result,))
                    continue
                if name == "event.test":
                    tested = self._emit(
                        "event.test",
                        operands=operands,
                        results=[operation.result.type],
                        attributes={
                            "event_id":
                                mlir_ir.StringAttr.get(self._event_id("test"),
                                                       context=self.context)
                        },
                    )
                    self._assign(operation, local_values, (tested.result,))
                    continue
                if name == "event.poll":
                    polled = self._emit(
                        "event.poll",
                        operands=operands,
                        results=[operation.result.type],
                        attributes={
                            "event_id":
                                mlir_ir.StringAttr.get(self._event_id("poll"),
                                                       context=self.context)
                        },
                    )
                    self._assign(operation, local_values, (polled.result,))
                    continue
                if name == "event.is":
                    tested = self._emit(
                        "event.is",
                        operands=operands,
                        results=[operation.result.type],
                        attributes={
                            "state":
                                operation.attributes["state"],
                            "event_id":
                                mlir_ir.StringAttr.get(self._event_id("is"),
                                                       context=self.context),
                        },
                    )
                    self._assign(operation, local_values, (tested.result,))
                    continue
                if name == "event.select_ready":
                    selected_event = self._emit(
                        "event.select_ready",
                        operands=operands,
                        results=[operation.result.type],
                        attributes={
                            "policy":
                                operation.attributes["policy"],
                            "event_id":
                                mlir_ir.StringAttr.get(
                                    self._event_id("select_ready"),
                                    context=self.context,
                                ),
                        },
                    )
                    self._assign(operation, local_values,
                                 (selected_event.result,))
                    continue
                if name == "event.try_take":
                    self._assign(
                        operation,
                        local_values,
                        self._emit_event_try_take(
                            operation,
                            operands,
                            patches,
                            execute,
                            call_stack,
                        ),
                    )
                    continue
                if name == "event.cancel":
                    attrs = {
                        "event_id":
                            mlir_ir.StringAttr.get(self._event_id("cancel"),
                                                   context=self.context)
                    }
                    if "reason" in operation.attributes:
                        attrs["reason"] = operation.attributes["reason"]
                    cancelled = self._emit(
                        "event.cancel",
                        operands=operands,
                        results=[operation.result.type],
                        attributes=attrs,
                    )
                    self._assign(operation, local_values, (cancelled.result,))
                    continue
                if name == "event.await":
                    kind = _fabric_resource_kind(operation.result.type)
                    if kind is None:
                        kind = _fabric_event_resource_kind(
                            operation.operands[0].type)
                    awaited = self._emit(
                        "event.await",
                        operands=operands,
                        results=[self._resource_payload_type(kind)],
                        attributes={
                            "event_id":
                                mlir_ir.StringAttr.get(self._event_id("await"),
                                                       context=self.context)
                        },
                    )
                    self._assign(operation, local_values, (awaited.result,))
                    continue
                if name == "event.fence":
                    self._emit(
                        "event.fence",
                        attributes={
                            "effects":
                                operation.attributes["effects"],
                            "event_id":
                                mlir_ir.StringAttr.get(self._event_id("fence"),
                                                       context=self.context),
                        },
                    )
                    continue
                if name == "event.selection":
                    self._emit(
                        "event.selection",
                        operands=operands,
                        attributes={
                            "mode":
                                operation.attributes["mode"],
                            "accept_when":
                                operation.attributes["accept_when"],
                            "event_id":
                                mlir_ir.StringAttr.get(
                                    self._event_id("selection"),
                                    context=self.context),
                        },
                    )
                    continue
                if name == "fabric.unpack_resource":
                    if not patches:
                        raise ValueError(
                            "resource unpack requires at least one live anchor patch"
                        )
                    resource = next(
                        (value for value in operands
                         if value is not None and not isinstance(value, _Patch)
                        ),
                        None,
                    )
                    if resource is None:
                        raise ValueError(
                            "resource unpack has no resource payload")
                    count = len(patches)
                    if len(operation.results) != count * 2:
                        raise ValueError(
                            "resource unpack requires one successor and one "
                            "payload per distinct anchor patch")
                    payloads = []
                    encodings = []
                    carrier_groups = []
                    for index, anchor in enumerate(patches):
                        result_type = operation.results[count + index].type
                        payload_code = _code_name(result_type)
                        if payload_code is None:
                            raise ValueError(
                                "resource unpack payload has no concrete code")
                        references = _patch_references(result_type)
                        encoding = (references[1]
                                    if len(references) > 1 else None)
                        payload = self._allocate_patch(
                            payload_code,
                            anchor.region,
                            f"resource_payload{self._event}_{index}",
                            encoding=encoding,
                        )
                        payloads.append(payload)
                        encodings.append(encoding or references[0])
                        carrier_groups.append(tuple(payload.all()))
                    if len(set(encodings)) != 1:
                        raise ValueError(
                            "multi-patch resource unpack requires one common "
                            "payload encoding")
                    payload_carriers = tuple(carrier for group in carrier_groups
                                             for carrier in group)
                    segments = [0]
                    for group in carrier_groups:
                        segments.append(segments[-1] + len(group))
                    attributes = {
                        "encoding":
                            mlir_ir.FlatSymbolRefAttr.get(encodings[0],
                                                          context=self.context),
                        "payload_carrier_segments":
                            mlir_ir.DenseI64ArrayAttr.get(segments,
                                                          context=self.context),
                        "event_id":
                            mlir_ir.StringAttr.get(
                                self._event_id("unpack_resource"),
                                context=self.context,
                            ),
                    }
                    for attribute in (
                            "payload_action",
                            "payload_logical_block_ids",
                            "payload_logical_blocks",
                            "payload_logical_ports",
                            "payload_roles",
                    ):
                        if attribute in operation.attributes:
                            attributes[attribute] = operation.attributes[
                                attribute]
                    unpacked = self._emit(
                        "phys.unpack_resource",
                        operands=[
                            resource, *(item.value for item in payload_carriers)
                        ],
                        results=[item.value.type for item in payload_carriers],
                        attributes=attributes,
                    )
                    replacements = iter(unpacked.results)
                    for payload in payloads:
                        payload.partitions = {
                            partition: [
                                carrier.with_value(next(replacements))
                                for carrier in carriers
                            ] for partition, carriers in
                            payload.partitions.items()
                        }
                    self._assign(
                        operation,
                        local_values,
                        (*patches, *payloads),
                    )
                    continue
                if name == "fabric.pack_resource":
                    if not patches:
                        raise ValueError(
                            "resource pack requires encoded payload patches")
                    carrier_groups = tuple(
                        tuple(payload.all()) for payload in patches)
                    if any(not group for group in carrier_groups):
                        raise ValueError(
                            "resource pack payload patches must own carriers")
                    carriers = tuple(carrier for group in carrier_groups
                                     for carrier in group)
                    kind = _fabric_resource_kind(operation.result.type)
                    encodings = []
                    for operand in operation.operands:
                        references = _patch_references(operand.type)
                        encodings.append(references[1] if len(references) >
                                         1 else references[0])
                    if len(set(encodings)) != 1:
                        raise ValueError(
                            "multi-patch resource pack requires one common "
                            "payload encoding")
                    segments = [0]
                    for group in carrier_groups:
                        segments.append(segments[-1] + len(group))
                    attributes = {
                        "resource_kind":
                            mlir_ir.FlatSymbolRefAttr.get(kind,
                                                          context=self.context),
                        "encoding":
                            mlir_ir.FlatSymbolRefAttr.get(encodings[0],
                                                          context=self.context),
                        "payload_carrier_segments":
                            mlir_ir.DenseI64ArrayAttr.get(segments,
                                                          context=self.context),
                        "event_id":
                            mlir_ir.StringAttr.get(
                                self._event_id("pack_resource"),
                                context=self.context,
                            ),
                    }
                    if "payload_roles" in operation.attributes:
                        attributes["payload_roles"] = (
                            operation.attributes["payload_roles"])
                    packed = self._emit(
                        "phys.pack_resource",
                        operands=[item.value for item in carriers],
                        results=[self._resource_payload_type(kind)],
                        attributes=attributes,
                    )
                    self._assign(operation, local_values, (packed.result,))
                    continue
                if name == "fabric.inject":
                    raise ValueError(
                        "canonical P2 cannot project fabric.inject: select a "
                        "concrete protocol that unpacks the resource and "
                        "expresses its physical/QEC realization")
                if name == "fabric.discard_resource":
                    self._emit(
                        "phys.discard_resource_payload",
                        operands=operands,
                        attributes={
                            "event_id":
                                mlir_ir.StringAttr.get(
                                    self._event_id("discard_resource"),
                                    context=self.context,
                                )
                        },
                    )
                    continue
                if name == "fabric.dealloc":
                    self._release(patches[0])
                    continue
                if name == "fabric.call":
                    callee_name = _text(operation.attributes["callee"])
                    callee = self.symbols.get(callee_name)
                    if callee is None:
                        raise ValueError(
                            f"unresolved Fabric call @{callee_name}")
                    call_instance = (
                        f"{instance_symbol}.{callee_name}.call{self._call_instance}"
                    )
                    self._call_instance += 1
                    compact_plan = self._spacetime_models.get(callee_name)
                    if compact_plan is not None:
                        flattened = self._flatten_projected(tuple(operands))
                        if len(operation.results) != len(operation.operands):
                            raise ValueError(
                                "v1 compact spacetime plans require an exact "
                                "boundary-preserving selected protocol")
                        with self.context:
                            compact = mlir_ir.Operation.create(
                                "phys.spacetime_call",
                                operands=flattened,
                                results=[value.type for value in flattened],
                                attributes={
                                    "plan":
                                        compact_plan,
                                    "source_protocol":
                                        mlir_ir.FlatSymbolRefAttr.get(
                                            callee_name, context=self.context),
                                    "instance":
                                        mlir_ir.StringAttr.get(
                                            call_instance,
                                            context=self.context),
                                    "event_id":
                                        mlir_ir.StringAttr.get(
                                            self._event_id("spacetime_call"),
                                            context=self.context),
                                },
                                loc=self.location,
                            )
                            self.ip.insert(compact)
                        outputs = self._rebuild_projected(
                            tuple(operands), compact.results)
                        self._assign(operation, local_values, outputs)
                        continue
                    call_profile = None
                    if "profile" in operation.attributes:
                        call_profile = self.symbols.get(
                            _text(operation.attributes["profile"]))
                        if (call_profile is None or
                                call_profile.name != "fabric.gadget_profile"):
                            raise ValueError(
                                "Fabric call references a missing gadget profile"
                            )
                        if _text(call_profile.attributes["gadget"]
                                ) != callee_name:
                            raise ValueError(
                                "Fabric call profile analyzes a different gadget"
                            )
                    communication = self._communication_context(
                        operation,
                        patches,
                        call_instance=call_instance,
                    )
                    encode_zero = self._feeds_communication_call(operation)
                    if encode_zero:
                        self._encode_zero_stack.append(True)
                    try:
                        if communication is None:
                            resource_call_attributes = {
                                key: operation.attributes[key] for key in (
                                    "resource_action_site",
                                    "resource_objective",
                                ) if key in operation.attributes
                            }
                            outputs = self._emit_folded_call(
                                callee,
                                tuple(operands),
                                (*call_stack, symbol),
                                instance_symbol=call_instance,
                                analysis_profile=call_profile,
                                call_attributes=resource_call_attributes,
                            )
                        else:
                            before_bridges = self._communication_bridges
                            self._communication_stack.append(communication)
                            try:
                                outputs = self._emit_folded_call(
                                    callee,
                                    tuple(operands),
                                    (*call_stack, symbol),
                                    instance_symbol=call_instance,
                                    analysis_profile=call_profile,
                                    call_attributes={
                                        key: operation.attributes[key] for key in
                                        _COMMUNICATION_CALL_ATTRIBUTES
                                    },
                                )
                            finally:
                                self._communication_stack.pop()
                            self._require_communication_bridge(before_bridges)
                    finally:
                        if encode_zero:
                            self._encode_zero_stack.pop()
                    self._assign(operation, local_values, outputs)
                    continue
                if name == "fabric.relocate":
                    callee_name = _text(operation.attributes["callee"])
                    callee = self.symbols.get(callee_name)
                    if callee is None:
                        raise ValueError(
                            f"unresolved trajectory realization @{callee_name}")
                    relocation_instance = (
                        f"{instance_symbol}.{callee_name}.relocate"
                        f"{self._call_instance}")
                    self._call_instance += 1
                    outputs = self._emit_callable(
                        callee,
                        tuple(operands),
                        (*call_stack, symbol),
                        instance_symbol=relocation_instance,
                    )
                    self._assign(operation, local_values, outputs)
                    continue
                if name == "fabric.establish_support":
                    callee_name = _text(operation.attributes["callee"])
                    callee = self.symbols.get(callee_name)
                    if callee is None:
                        raise ValueError(
                            "unresolved distributed-support realization "
                            f"@{callee_name}")
                    support_instance = (
                        f"{instance_symbol}.{callee_name}.support"
                        f"{self._call_instance}")
                    self._call_instance += 1
                    outputs = self._emit_callable(
                        callee,
                        tuple(operands),
                        (*call_stack, symbol),
                        instance_symbol=support_instance,
                    )
                    self._assign(operation, local_values, outputs)
                    continue
                if name == "fabric.establish_topological_record":
                    callee_name = _text(operation.attributes["callee"])
                    callee = self.symbols.get(callee_name)
                    if callee is None:
                        raise ValueError(
                            "unresolved topological-record realization "
                            f"@{callee_name}")
                    record_instance = (
                        f"{instance_symbol}.{callee_name}.topological"
                        f"{self._call_instance}")
                    self._call_instance += 1
                    outputs = self._emit_callable(
                        callee,
                        tuple(operands),
                        (*call_stack, symbol),
                        instance_symbol=record_instance,
                    )
                    self._assign(operation, local_values, outputs)
                    continue
                if name == "cflow.repeat":
                    self._assign(
                        operation,
                        local_values,
                        self._emit_repeat(
                            operation,
                            tuple(operands),
                            local_values,
                            execute,
                            call_stack,
                        ),
                    )
                    continue
                if name == "cflow.if":
                    self._assign(
                        operation,
                        local_values,
                        self._emit_conditional(
                            operation,
                            local_values,
                            execute,
                            call_stack,
                            instance_symbol,
                        ),
                    )
                    continue
                if name == "cflow.while":
                    self._assign(
                        operation,
                        local_values,
                        self._emit_while(
                            operation,
                            tuple(operands),
                            local_values,
                            execute,
                            call_stack,
                        ),
                    )
                    continue
                if name == "fabric.retry":
                    self._assign(
                        operation,
                        local_values,
                        self._emit_retry(operation, operands, patches),
                    )
                    continue
                if name == "fabric.encoding_unpack":
                    if len(operands) != 1 or not isinstance(
                            operands[0], _Patch):
                        raise ValueError(
                            "encoding_unpack requires one physical patch")
                    self._assign(
                        operation,
                        local_values,
                        (self._unpack_hierarchy(operation, operands[0]),),
                    )
                    continue
                if name == "fabric.map_children":
                    if len(operands) != 1 or not isinstance(
                            operands[0], _PatchBundle):
                        raise ValueError(
                            "map_children requires one physical bundle")
                    self._assign(
                        operation,
                        local_values,
                        (self._map_hierarchy_children(
                            operation,
                            operands[0],
                            (*call_stack, symbol),
                            instance_symbol,
                        ),),
                    )
                    continue
                if name == "fabric.encoding_pack":
                    if len(operands) != 1 or not isinstance(
                            operands[0], _PatchBundle):
                        raise ValueError(
                            "encoding_pack requires one physical bundle")
                    self._assign(
                        operation,
                        local_values,
                        (self._pack_hierarchy(operation, operands[0]),),
                    )
                    continue
                if name == "fabric.epoch_transition":
                    if len(patches) != 1:
                        raise ValueError(
                            "epoch_transition requires one physical patch")
                    self._assign(
                        operation,
                        local_values,
                        (self._epoch_transition(operation, patches[0]),),
                    )
                    continue
                if name == "fabric.measure_product":
                    self._assign(
                        operation,
                        local_values,
                        self._measure_product(operation, patches, symbol,
                                              instance_symbol),
                    )
                    continue
                if name == "fabric.xor":
                    conditioned = [self._condition(value) for value in operands]
                    parity = self._emit(
                        "phys.xor",
                        operands=conditioned,
                        results=[
                            mlir_ir.IntegerType.get_signless(
                                1, context=self.context)
                        ],
                        attributes={
                            "event_id":
                                mlir_ir.StringAttr.get(self._event_id("xor"),
                                                       context=self.context)
                        },
                    )
                    self._assign(operation, local_values, (parity.result,))
                    continue
                if name == "fabric.all_false":
                    conditioned = [self._condition(value) for value in operands]
                    conjunction = self._emit(
                        "phys.all_false",
                        operands=conditioned,
                        results=[
                            mlir_ir.IntegerType.get_signless(
                                1, context=self.context)
                        ],
                        attributes={
                            "event_id":
                                mlir_ir.StringAttr.get(
                                    self._event_id("all_false"),
                                    context=self.context)
                        },
                    )
                    self._assign(operation, local_values, (conjunction.result,))
                    continue
                if name == "fabric.rotate_product":
                    self._assign(
                        operation,
                        local_values,
                        self._rotate_product(operation, patches),
                    )
                    continue
                if name == "fabric.resource_rotate_product":
                    resource = next(
                        (value for value in operands
                         if value is not None and not isinstance(value, _Patch)
                        ),
                        None,
                    )
                    if resource is None:
                        raise ValueError(
                            "resource-assisted rotation has no resource payload"
                        )
                    self._assign(
                        operation,
                        local_values,
                        self._resource_rotate_product(operation, resource,
                                                      patches),
                    )
                    continue
                if name in {"fabric.prep_z", "fabric.prep_x"}:
                    output = self._prepare(
                        patches[0],
                        "zero" if name == "fabric.prep_z" else "plus",
                        encode=(bool(self._communication_stack) or
                                bool(self._encode_zero_stack) or
                                self._feeds_communication_call(operation)),
                    )
                    self._assign(operation, local_values, (output,))
                    continue
                if name == "fabric.init_basis":
                    partition = _partition(operation.attributes["partition"])
                    indices = self._selected(operation, patches[0], partition)
                    basis = ("x" if "x" in str(
                        operation.attributes["basis"]).lower() else "z")
                    output = self._prepare(
                        patches[0],
                        "plus" if basis == "x" else "zero",
                        partition,
                        indices,
                    )
                    self._assign(operation, local_values, (output,))
                    continue
                bulk = {
                    "fabric.h": "h",
                    "fabric.s": "s",
                    "fabric.sdg": "sdg",
                    "fabric.x": "x",
                    "fabric.z": "z",
                    "fabric.t": "t",
                    "fabric.tdg": "tdg",
                    "fabric.reset": "reset",
                }
                if name in bulk:
                    self._assign(
                        operation,
                        local_values,
                        (self._bulk(operation, patches[0], bulk[name]),),
                    )
                    continue
                if name in {"fabric.cx", "fabric.cz"}:
                    self._assign(
                        operation,
                        local_values,
                        self._two_patch(operation, patches,
                                        "cx" if name == "fabric.cx" else "cz"),
                    )
                    continue
                if name == "fabric.transversal_cx":
                    left, right = patches
                    fake_attrs = {
                        "ctrl":
                            mlir_ir.Attribute.parse("#fabric.partition<data>",
                                                    context=self.context),
                        "targ":
                            mlir_ir.Attribute.parse("#fabric.partition<data>",
                                                    context=self.context),
                    }
                    if "perm" in operation.attributes:
                        permutation = tuple(
                            int(v) for v in operation.attributes["perm"])
                        fake_attrs["pairs"] = mlir_ir.StringAttr.get(
                            _pair_text(tuple(enumerate(permutation))),
                            context=self.context)
                    with self.location:
                        proxy = mlir_ir.Operation.create(
                            "fabric.cx",
                            operands=[value for value in operation.operands],
                            results=[value.type for value in operation.results],
                            attributes=fake_attrs,
                            loc=self.location,
                        )
                    self._assign(
                        operation,
                        local_values,
                        self._two_patch(proxy, (left, right), "cx"),
                    )
                    continue
                if name in {"fabric.mz", "fabric.measure_basis"}:
                    partition = _partition(operation.attributes["partition"])
                    indices = self._selected(operation, patches[0], partition)
                    record = _text(
                        operation.attributes["record"]
                    ) if "record" in operation.attributes else "mz"
                    patch, records = self._measure(
                        patches[0],
                        partition,
                        indices,
                        record,
                        symbol,
                        basis=("x" if name == "fabric.measure_basis" and "x"
                               in str(operation.attributes["basis"]).lower()
                               else "z"),
                        instance_symbol=instance_symbol,
                        allow_legacy_z_fallback=(name == "fabric.mz"),
                    )
                    self._assign(operation, local_values, (patch, records))
                    continue
                if name == "fabric.read_syndrome_ancillas":
                    patch = patches[0]
                    output = patch
                    records = []
                    record = _text(
                        operation.attributes["record"]
                    ) if "record" in operation.attributes else "syndrome"
                    ordinal_offset = 0
                    for partition in ("sx", "sz"):
                        indices = tuple(
                            range(len(output.partitions.get(partition, ()))))
                        output, measured = self._measure(
                            output,
                            partition,
                            indices,
                            record,
                            symbol,
                            instance_symbol=instance_symbol,
                            lane_prefix="s",
                            ordinal_offset=ordinal_offset,
                        )
                        records.extend(measured)
                        ordinal_offset += len(indices)
                    self._assign(operation, local_values,
                                 (output, tuple(records)))
                    continue
                if name == "fabric.assemble_syndrome":
                    sx_records = tuple(local_values[operation.operands[1]])
                    sz_records = tuple(local_values[operation.operands[2]])
                    records = (*sx_records, *sz_records)
                    record = _text(operation.attributes["record"])
                    for ordinal, value in enumerate(records):
                        record_id = self._record_ids_by_value.get(value)
                        if record_id is None:
                            raise ValueError(
                                "fabric.assemble_syndrome input has no "
                                "projected physical measurement record")
                        source_record = f"{symbol}.{record}.s{ordinal}"
                        projected = self._record_projection.setdefault(
                            (instance_symbol, source_record), [])
                        projected.append(record_id)
                    self._assign(
                        operation,
                        local_values,
                        (patches[0], records),
                    )
                    continue
                if name == "fabric.idle":
                    self._assign(
                        operation,
                        local_values,
                        (self._delay(patches[0],
                                     int(operation.attributes["rounds"])),),
                    )
                    continue
                if name == "fabric.permute":
                    result = patches[0].clone()
                    data = tuple(result.partitions.get("data", ()))
                    transformed = self._apply_carriers(
                        data,
                        "permute",
                        reservations=
                        (f"perm={tuple(int(v) for v in operation.attributes['perm'])}",
                        ),
                    )
                    permutation = tuple(
                        int(v) for v in operation.attributes["perm"])
                    reordered = [None] * len(transformed)
                    for source_index, destination in enumerate(permutation):
                        reordered[destination] = transformed[source_index]
                    result.partitions["data"] = reordered
                    self._assign(operation, local_values, (result,))
                    continue
                if name in {
                        "fabric.success",
                        "fabric.frame_update",
                        "fabric.frame_s",
                        "fabric.frame",
                }:
                    continue
                if name == "fabric.tick":
                    # Preserve the authored global moment boundary as a typed,
                    # state-carrying P3 dependency.  This is deliberately not an
                    # operand-free marker: every live carrier is threaded through
                    # the barrier, so cudaq.logical.schedule,
                    # cudaq.logical.ticks, idle insertion, and
                    # target serialization agree on the same order.
                    self._moment_barrier(local_values)
                    continue
                if name.startswith("fabric."):
                    raise NotImplementedError(
                        f"P2-to-P3 projection does not yet define {name}")
            return returned

        outputs = execute(block, values, stack) or ()
        if analysis_profile is not None:
            self._emit_profile_sidecars(analysis_profile, instance_symbol)
        return outputs

    def _emit_topology_facets(self):

        def strings(values):
            return mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(str(value), context=self.context)
                    for value in values
                ],
                context=self.context,
            )

        node_attrs = []
        for node in self._patch_nodes:
            fields = {
                "id":
                    mlir_ir.StringAttr.get(node["id"], context=self.context),
                "code":
                    mlir_ir.FlatSymbolRefAttr.get(node["code"],
                                                  context=self.context),
            }
            for key in ("encoding", "region"):
                if node[key] is not None:
                    fields[key] = mlir_ir.FlatSymbolRefAttr.get(
                        node[key], context=self.context)
            for key in ("qec_region", "physical_binding"):
                if node[key] is not None:
                    fields[key] = mlir_ir.StringAttr.get(node[key],
                                                         context=self.context)
            if node["slot"] is not None:
                fields["slot"] = _i64(self.context, node["slot"])
                with self.context:
                    fields["patch_topology"] = mlir_ir.SymbolRefAttr.get(
                        [self.architecture, node["patch_topology"]],
                        context=self.context,
                    )
            node_attrs.append(mlir_ir.DictAttr.get(fields,
                                                   context=self.context))
        interaction_attrs = []
        for interaction in self._patch_interactions:
            fields = {
                "id":
                    mlir_ir.StringAttr.get(interaction["id"],
                                           context=self.context),
                "action":
                    mlir_ir.StringAttr.get(interaction["action"],
                                           context=self.context),
                "patches":
                    strings(interaction["patches"]),
                "pairs":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.DenseI64ArrayAttr.get(pair, self.context)
                            for pair in interaction["pairs"]
                        ],
                        context=self.context,
                    ),
            }
            slots = tuple(
                slot for slot in interaction["slots"] if slot is not None)
            if len(slots) == len(interaction["slots"]):
                fields["slots"] = mlir_ir.DenseI64ArrayAttr.get(
                    slots, self.context)
            for key in ("regions", "qec_regions", "physical_bindings"):
                values = interaction.get(key)
                if values and all(value is not None for value in values):
                    fields[key] = strings(values)
            endpoint_names = interaction.get("endpoint_names")
            if endpoint_names:
                fields["endpoints"] = strings(endpoint_names)
            for key in ("channel", "action_site", "generated_by"):
                value = interaction.get(key)
                if value is not None:
                    fields[key] = mlir_ir.StringAttr.get(
                        self._region_leaf(value), context=self.context)
            if interaction.get("channel_capability") is not None:
                fields["channel_capability"] = mlir_ir.StringAttr.get(
                    _capability_key(interaction["channel_capability"]),
                    context=self.context,
                )
            interaction_attrs.append(
                mlir_ir.DictAttr.get(fields, context=self.context))
        with self.location:
            self.module.body.append(
                mlir_ir.Operation.create(
                    "fabric.patch_graph",
                    attributes={
                        "sym_name":
                            mlir_ir.StringAttr.get(self.patch_graph_symbol,
                                                   context=self.context),
                        "root":
                            mlir_ir.FlatSymbolRefAttr.get(_symbol(self.root),
                                                          context=self.context),
                        "nodes":
                            mlir_ir.ArrayAttr.get(node_attrs,
                                                  context=self.context),
                        "interactions":
                            mlir_ir.ArrayAttr.get(interaction_attrs,
                                                  context=self.context),
                    },
                    loc=self.location,
                ))

        if self._patch_assignments:
            assignment_attrs = []
            for assignment in self._patch_assignments:
                with self.context:
                    topology = mlir_ir.SymbolRefAttr.get(
                        [self.architecture, assignment["topology"]],
                        context=self.context,
                    )
                assignment_attrs.append(
                    mlir_ir.DictAttr.get(
                        {
                            "id":
                                mlir_ir.StringAttr.get(assignment["id"],
                                                       context=self.context),
                            "patch":
                                mlir_ir.StringAttr.get(assignment["patch"],
                                                       context=self.context),
                            "slot":
                                _i64(self.context, assignment["slot"]),
                            "topology":
                                topology,
                            "carriers":
                                mlir_ir.DenseI64ArrayAttr.get(
                                    assignment["carriers"], self.context),
                            "region":
                                mlir_ir.StringAttr.get(assignment["region"],
                                                       context=self.context),
                            "qec_region":
                                mlir_ir.StringAttr.get(assignment["qec_region"],
                                                       context=self.context),
                            "physical_binding":
                                mlir_ir.StringAttr.get(
                                    assignment["physical_binding"],
                                    context=self.context,
                                ),
                        },
                        context=self.context,
                    ))
            mapping_symbol = self.transaction.unique_symbol(
                f"{self.patch_graph_symbol}_mapping")
            with self.location:
                self.module.body.append(
                    mlir_ir.Operation.create(
                        "fabric.patch_mapping",
                        attributes={
                            "sym_name":
                                mlir_ir.StringAttr.get(mapping_symbol,
                                                       context=self.context),
                            "graph":
                                mlir_ir.FlatSymbolRefAttr.get(
                                    self.patch_graph_symbol,
                                    context=self.context),
                            "assignments":
                                mlir_ir.ArrayAttr.get(assignment_attrs,
                                                      context=self.context),
                        },
                        loc=self.location,
                    ))

        physical_entries = []
        for entry in self._mapping_initial:
            physical_entries.append(
                mlir_ir.DictAttr.get(
                    {
                        "role":
                            mlir_ir.StringAttr.get(entry["role"],
                                                   context=self.context),
                        "patch":
                            mlir_ir.StringAttr.get(entry["patch"],
                                                   context=self.context),
                        "partition":
                            mlir_ir.StringAttr.get(entry["partition"],
                                                   context=self.context),
                        "index":
                            _i64(self.context, entry["index"]),
                        "resource":
                            mlir_ir.FlatSymbolRefAttr.get(entry["resource"],
                                                          context=self.context),
                        "node":
                            _i64(self.context, entry["node"]),
                        "region":
                            mlir_ir.StringAttr.get(entry["region"],
                                                   context=self.context),
                        "qec_region":
                            mlir_ir.StringAttr.get(entry["qec_region"],
                                                   context=self.context),
                        "physical_binding":
                            mlir_ir.StringAttr.get(entry["physical_binding"],
                                                   context=self.context),
                    },
                    context=self.context,
                ))
        mapping_symbol = self.transaction.unique_symbol(
            f"{self.graph_symbol}_mapping")
        with self.location:
            self.module.body.append(
                mlir_ir.Operation.create(
                    "phys.mapping",
                    attributes={
                        "sym_name":
                            mlir_ir.StringAttr.get(mapping_symbol,
                                                   context=self.context),
                        "graph":
                            mlir_ir.FlatSymbolRefAttr.get(self.graph_symbol,
                                                          context=self.context),
                        "source_graph":
                            mlir_ir.FlatSymbolRefAttr.get(
                                self.patch_graph_symbol, context=self.context),
                        "initial":
                            mlir_ir.ArrayAttr.get(physical_entries,
                                                  context=self.context),
                        # The bounded fixed-qubit router restores every moved
                        # role after its interaction, so the final embedding
                        # equals the initial embedding by construction.
                        "final":
                            mlir_ir.ArrayAttr.get(physical_entries,
                                                  context=self.context),
                    },
                    loc=self.location,
                ))

        allocation_entries = []
        for binding in self._allocation_bindings:
            fields = {
                "allocation":
                    mlir_ir.StringAttr.get(binding.allocation,
                                           context=self.context),
                "resource_class":
                    mlir_ir.FlatSymbolRefAttr.get(binding.resource_class,
                                                  context=self.context),
                "resources":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.FlatSymbolRefAttr.get(resource,
                                                          context=self.context)
                            for resource in binding.resources
                        ],
                        context=self.context,
                    ),
                "indices":
                    mlir_ir.DenseI64ArrayAttr.get(binding.indices,
                                                  self.context),
                "acquire":
                    mlir_ir.StringAttr.get(binding.acquire,
                                           context=self.context),
            }
            if binding.release is not None:
                fields["release"] = mlir_ir.StringAttr.get(binding.release,
                                                           context=self.context)
            if binding.after:
                fields["after"] = mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.StringAttr.get(event, context=self.context)
                        for event in binding.after
                    ],
                    context=self.context,
                )
            if binding.slot is not None:
                fields["slot"] = _i64(self.context, binding.slot)
            if binding.scope is not None:
                fields["scope"] = mlir_ir.StringAttr.get(binding.scope,
                                                         context=self.context)
            if binding.qec_region is not None:
                fields["qec_region"] = mlir_ir.StringAttr.get(
                    binding.qec_region, context=self.context)
            if binding.physical_binding is not None:
                fields["physical_binding"] = mlir_ir.StringAttr.get(
                    binding.physical_binding, context=self.context)
            allocation_entries.append(
                mlir_ir.DictAttr.get(fields, context=self.context))
        allocation_symbol = self.transaction.unique_symbol(
            f"{self.graph_symbol}_allocations")
        with self.location:
            self.module.body.append(
                mlir_ir.Operation.create(
                    "phys.allocation_mapping",
                    attributes={
                        "sym_name":
                            mlir_ir.StringAttr.get(allocation_symbol,
                                                   context=self.context),
                        "graph":
                            mlir_ir.FlatSymbolRefAttr.get(self.graph_symbol,
                                                          context=self.context),
                        "entries":
                            mlir_ir.ArrayAttr.get(allocation_entries,
                                                  context=self.context),
                    },
                    loc=self.location,
                ))

        by_topology = {}
        for step in self._route_steps:
            by_topology.setdefault(step["topology"], []).append(step)
        for topology, steps in sorted(by_topology.items()):
            step_attrs = []
            for step in steps:
                fields = {
                    "event":
                        mlir_ir.StringAttr.get(step["event"],
                                               context=self.context),
                    "action":
                        mlir_ir.StringAttr.get(step["action"],
                                               context=self.context),
                    "path":
                        mlir_ir.DenseI64ArrayAttr.get(step["path"],
                                                      self.context),
                    "native_events":
                        strings(step["native_events"]),
                }
                if step.get("bridge"):
                    fields["communication_bridge"] = mlir_ir.UnitAttr.get(
                        context=self.context)
                for key in (
                        "source_region",
                        "destination_region",
                        "source_qec_region",
                        "destination_qec_region",
                        "source_binding",
                        "destination_binding",
                        "source_call",
                ):
                    value = step.get(key)
                    if value is not None:
                        fields[key] = mlir_ir.StringAttr.get(
                            value, context=self.context)
                for key in (
                        "channel",
                        "channel_capability",
                        "endpoints",
                        "action_site",
                        "generated_by",
                ):
                    value = step.get(key)
                    if value is not None:
                        fields[key] = value
                step_attrs.append(
                    mlir_ir.DictAttr.get(fields, context=self.context))
            symbol = self.transaction.unique_symbol(
                f"{self.graph_symbol}_{topology}_routing")
            with self.location:
                self.module.body.append(
                    mlir_ir.Operation.create(
                        "phys.routing",
                        attributes={
                            "sym_name":
                                mlir_ir.StringAttr.get(symbol,
                                                       context=self.context),
                            "graph":
                                mlir_ir.FlatSymbolRefAttr.get(
                                    self.graph_symbol, context=self.context),
                            "topology":
                                mlir_ir.FlatSymbolRefAttr.get(
                                    topology, context=self.context),
                            "steps":
                                mlir_ir.ArrayAttr.get(step_attrs,
                                                      context=self.context),
                        },
                        loc=self.location,
                    ))
        self._emit_record_projection()
        self._emit_legalization_record()

    def _emit_record_projection(self):
        """Retain exactly the stable-to-physical lanes claimed by sidecars."""

        entries = []
        pair_to_index = {}
        claimed = sorted({
            (instance, source_record, physical_record)
            for _sidecar, instance, pairs in self._provenance_sidecars
            for source_record, physical_record in pairs
        })
        for instance, source_record, physical_record in claimed:
            pair = (instance, source_record, physical_record)
            pair_to_index[pair] = len(entries)
            fields = {
                "instance":
                    mlir_ir.StringAttr.get(instance, context=self.context),
                "source_record":
                    mlir_ir.StringAttr.get(source_record, context=self.context),
                "physical_record":
                    mlir_ir.StringAttr.get(physical_record,
                                           context=self.context),
            }
            repeats = self._record_projection_repeats.get(physical_record, ())
            if repeats:
                fields["repeat_events"] = mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.StringAttr.get(event, context=self.context)
                        for event, _count in repeats
                    ],
                    context=self.context,
                )
                fields["repeat_counts"] = mlir_ir.DenseI64ArrayAttr.get(
                    [count for _event, count in repeats], self.context)
            entries.append(mlir_ir.DictAttr.get(fields, context=self.context))

        with self.location:
            self.module.body.append(
                mlir_ir.Operation.create(
                    "phys.record_projection",
                    attributes={
                        "sym_name":
                            mlir_ir.StringAttr.get(
                                self.record_projection_symbol,
                                context=self.context,
                            ),
                        "graph":
                            mlir_ir.FlatSymbolRefAttr.get(self.graph_symbol,
                                                          context=self.context),
                        "source_protocol":
                            mlir_ir.FlatSymbolRefAttr.get(_symbol(self.root),
                                                          context=self.context),
                        "entries":
                            mlir_ir.ArrayAttr.get(entries,
                                                  context=self.context),
                    },
                    loc=self.location,
                ))

        for sidecar, instance, pairs in self._provenance_sidecars:
            try:
                indices = tuple(pair_to_index[(instance, source_record,
                                               physical_record)]
                                for source_record, physical_record in pairs)
            except KeyError as exc:
                raise ValueError(
                    "physical sidecar references an unretained stable-to-"
                    "physical record projection") from exc
            sidecar.attributes["projection_indices"] = (
                mlir_ir.DenseI64ArrayAttr.get(indices, self.context))

    def _emit_legalization_record(self):
        """Record the native-action legalizations as durable IR evidence.

        ``phys-legalize-native-actions`` rewrote each unsupported semantic
        action into native actions using a typed decomposition rule; this
        stamps the chosen rule, the native-action sequence, and its cost onto
        the ``phys.graph`` so the selected decomposition and cost survive in the
        module -- the routing block's per-event ``steps`` record, one abstraction
        up. A device whose native set already covers every action legalizes
        nothing and carries no record.
        """
        if not self._legalizations:
            return
        records = [
            mlir_ir.DictAttr.get(
                {
                    "event":
                        mlir_ir.StringAttr.get(item["event"],
                                               context=self.context),
                    "action":
                        mlir_ir.StringAttr.get(item["action"],
                                               context=self.context),
                    "rule":
                        mlir_ir.StringAttr.get(item["rule"],
                                               context=self.context),
                    "native":
                        mlir_ir.ArrayAttr.get(
                            [
                                mlir_ir.StringAttr.get(name,
                                                       context=self.context)
                                for name in item["native"]
                            ],
                            context=self.context,
                        ),
                    "cost":
                        _i64(self.context, item["cost"]),
                },
                context=self.context,
            )
            for item in self._legalizations
        ]
        with self.context:
            self.graph.attributes["phys.legalizations"] = mlir_ir.ArrayAttr.get(
                records, context=self.context)

    def project(self, *, pipeline=None, experiment=None, _transient=False):
        root_arguments = tuple(self.root.regions[0].blocks[0].arguments)
        external_types = tuple(
            self._physical_boundary_type(argument.type)
            for argument in root_arguments
            if _code_name(argument.type) is None)
        self._create_graph(external_types)
        external_arguments = iter(self.block.arguments)
        arguments = []
        for index, argument in enumerate(root_arguments):
            code = _code_name(argument.type)
            if code is not None:
                references = _patch_references(argument.type)
                encoding = references[1] if len(references) > 1 else None
                arguments.append(
                    self._allocate_patch(code,
                                         None,
                                         f"arg{index}",
                                         encoding=encoding))
            else:
                arguments.append(next(external_arguments))
        try:
            returned = self._emit_callable(
                self.root,
                tuple(arguments),
                analysis_profile=self.root_profile,
            )
        except RecursionError as error:
            raise PhysicalProjectionCapacityError(
                "P2-to-P3 projection exhausted the host Python recursion "
                "capacity while traversing an acyclic Fabric composition; "
                "split the composition into a reusable circuit or a shallower "
                "typed protocol hierarchy") from error
        outputs = []
        seen_resources = set()
        for value in returned:
            if isinstance(value, _Patch):
                for carrier in value.all():
                    if carrier.resource not in seen_resources:
                        outputs.append(carrier.value)
                        seen_resources.add(carrier.resource)
            elif isinstance(value, tuple):
                outputs.extend(value)
            elif value is not None:
                outputs.append(value)
        self._release_route_scratch(self._scratch)
        self._emit("phys.return", operands=outputs)
        function_type = mlir_ir.FunctionType.get(
            external_types,
            tuple(value.type for value in outputs),
            context=self.context,
        )
        with self.context:
            self.graph.attributes["function_type"] = mlir_ir.TypeAttr.get(
                function_type)
        self._emit_topology_facets()
        self.transaction.add_profile("p3")
        unresolved = tuple(
            operation.name
            for operation in self.transaction.walk()
            if operation.name in {"fabric.inject", "phys.consume_resource"})
        if unresolved:
            raise ValueError(
                "canonical P3 cannot contain unresolved resource-backed "
                f"logical intent: {sorted(set(unresolved))!r}")
        if not self.module.operation.verify():
            raise ValueError(
                "P2-to-P3 physical projection produced invalid MLIR")
        # Linear single ownership of the projected physical states is
        # discharged by the real linear-use analysis (the same checker the
        # compile() paths run), never asserted unchecked. Deferred import:
        # compile.py imports this module lazily.
        from .compile import _verified_linearity_evidence

        linearity = _verified_linearity_evidence(
            self.module,
            obligation="linear-physical-state-ownership",
            subject=f"physical graph @{self.graph_symbol}",
        )
        return Build(
            context=self.context,
            module=self.module,
            root=DefinitionHandle(self.graph_symbol, "physical_graph", "p3"),
            profile="p3",
            facets=(
                *self.source.facets,
                "patch_graph",
                *(("patch_mapping",) if self._patch_assignments else ()),
                "native_legalization",
                "carrier_mapping",
                "physical_routing",
                *(("zoned_movement",) if self._zoned_movements else ()),
            ),
            pipeline=pipeline,
            evidence=(
                *self.source.evidence,
                EvidenceRecord(
                    kind="physical_projection_verification",
                    producer="cudaq-logical-python@0.3",
                    result="pass",
                    obligations=(
                        "selected-p2-realizations",
                        "deterministic-carrier-allocation",
                        "stable-record-projection",
                        "selected-success-sidecars",
                        "patch-graph-derivation",
                        ("implicit-slot-to-carrier-mapping"
                         if self._patch_assignments else
                         "explicit-unmapped-patch-compatibility"),
                        "carrier-topology-legality",
                        "explicit-swap-routing",
                        *(("cross-region-interconnect-bridge",)
                          if self._communication_bridges else ()),
                    ),
                ),
                EvidenceRecord(
                    kind="native_action_legalization",
                    producer="phys-legalize-native-actions@0.3",
                    result="pass",
                    obligations=(
                        "typed-action-identity-premise",
                        "support-on-all-participating-classes",
                        "recorded-decomposition-provenance",
                        "fail-closed-on-no-decomposition",
                        "all-phys.apply-actions-advertised-by-resource-class",
                        *tuple(f"{source}->" +
                               ",".join(f"{native}{operands}"
                                        for native, operands in steps)
                               for source, steps in sorted(
                                   self._legalized_actions.items())),
                        *((
                            "neutral-atom-blockade-pairs-isolated-by-zone-moves",
                            "destructive-readout-moved-to-readout-zone",
                        ) if self._zoned_movements else ()),
                    ),
                ),
                linearity,
            ),
            value_groups={
                name: len(group)
                for name, group in self.source.values._groups.items()
            },
            placement=self.source.placement,
            qec_selection=self.source.qec_selection,
            experiment=experiment or self.source.experiment,
            device=self.device,
            source_modules=self.source.source_modules,
            _transient=_transient,
        )


def _required_network_projector(source: Build):
    """Return the exact projector requirement retained by a network P2 root."""

    from ..qec import lattice_surgery

    # This is compiler-internal, read-only inspection of the Build authority.
    # Going through the public definitions view would defensively clone the
    # complete immutable P2 before projection creates its one mutable clone.
    root = next(
        (view.operation
         for view in source._module.body.operations
         if _symbol(view.operation) == source.root.symbol),
        None,
    )
    if root is None:
        raise ValueError(
            f"P2 root @{source.root.symbol} is missing from its Build authority"
        )
    try:
        serialized = _text(root.attributes["qlx.qec_network_plan"])
    except KeyError:
        return None
    plan = lattice_surgery.QECNetworkPlan.from_json(serialized)
    try:
        metadata = root.attributes["metadata"]
        if (_text(metadata["required_projector"]) != plan.required_projector_key
                or _text(metadata["required_projector_pipeline_sha256"])
                != plan.required_projector_pipeline_sha256):
            raise ValueError(
                "P2 network projector metadata differs from its exact plan")
    except KeyError as exc:
        raise ValueError(
            "P2 network plan omitted its exact projector requirement") from exc
    return plan


def _accepting_physical_projector(source: Build, device: Device):
    required = _required_network_projector(source)
    if required is not None:
        from ..qec import lattice_surgery

        _request, authenticated, _compiler, _context = (
            lattice_surgery._authenticate_network_replay(
                source,
                device=device,
            ))
        if authenticated.digest != required.digest:
            raise ValueError(
                "authenticated network plan differs from its projector "
                "requirement")
        matching = tuple(capability for capability in device.compilers
                         if isinstance(capability, PhysicalProjector) and
                         capability.key == required.required_projector_key)
        if not matching:
            raise LookupError(
                "P2 network plan requires unavailable physical projector "
                f"{required.required_projector_key!r}")
        if len(matching) != 1:
            raise LookupError(
                "P2 network plan projector identity is ambiguous: "
                f"{required.required_projector_key!r}")
        return matching[0]
    accepting = []
    for capability in device.compilers:
        accepts = getattr(capability, "accepts_projection", None)
        if not callable(accepts):
            continue
        verdict = accepts(source, device)
        key = getattr(capability, "key", type(capability).__name__)
        if type(verdict) is not bool:
            raise TypeError(
                f"physical projector {key!r} accepts_projection() must "
                "return bool")
        if verdict:
            accepting.append(capability)
    if len(accepting) > 1:
        raise LookupError("physical-projector selection is ambiguous: " +
                          ", ".join(value.key for value in accepting))
    return accepting[0] if accepting else None


def physical_projection_pipeline(source: Build, *, device: Device):
    """Return the exact device-contributed/default recipe for one P2 build."""

    if not isinstance(source, Build) or source.profile not in {"p2a", "p2n"}:
        raise TypeError(
            "physical projection requires a P2A/P2N cudaq.logical.Build")
    if not isinstance(device, Device):
        raise TypeError(
            "physical projection requires a typed cudaq.logical.Device")
    projector = _accepting_physical_projector(source, device)
    if projector is not None:
        return projector.projection_pipeline
    from .pipeline import pipelines

    return pipelines.physical()


def _projection_architecture_digest(projector, device: Device) -> str:
    from ..qec.lattice_surgery import _require_digest

    return _require_digest(
        projector.projection_architecture_digest(device),
        what=f"physical projector {projector.key!r} architecture digest",
    )


def _finish_projection_emission(
    source,
    device,
    projector,
    pipeline,
    emission,
    *,
    experiment,
):
    if not isinstance(emission, PhysicalProjectionEmission):
        raise TypeError(f"physical projector {projector.key!r} must return a "
                        "PhysicalProjectionEmission")
    return Build(
        context=emission.context,
        module=emission.module,
        root=emission.root,
        profile="p3",
        facets=source.facets,
        pipeline=pipeline,
        evidence=(
            *source.evidence,
            EvidenceRecord(
                kind="physical_projection_verification",
                producer="cudaq-logical-python@0.3",
                result="pass",
                obligations=(
                    f"exact-projector={projector.key}",
                    "core-owned-p3-build-finalization",
                    "retained-p2-provenance",
                ),
            ),
            *emission.evidence,
        ),
        value_groups={
            name: len(group) for name, group in source.values._groups.items()
        },
        placement=source.placement,
        qec_selection=source.qec_selection,
        experiment=experiment or source.experiment,
        device=device,
        source_modules=source.source_modules,
    )


def _validate_custom_projection(
    source: Build,
    device: Device,
    projector,
    pipeline,
    result,
    *,
    architecture_digest: str,
) -> Build:
    key = projector.key
    if not isinstance(result, Build):
        raise TypeError(
            f"physical projector {key!r} returned "
            f"{type(result).__name__}, expected cudaq.logical.Build")
    if (result.profile != "p3" or result.stage is not P3 or
            result.schedule is not None or not result.verify()):
        raise ValueError(
            f"physical projector {key!r} returned an invalid unscheduled "
            "P3 build")
    if result.pipeline != pipeline:
        raise ValueError(
            f"physical projector {key!r} returned a build with different "
            "pipeline provenance")
    try:
        graph = result.definitions[result.root.symbol]
        retained_source = _text(graph.op.attributes["source_protocol"])
    except KeyError as exc:
        raise ValueError(
            f"physical projector {key!r} did not retain the exact source "
            "P2 protocol") from exc
    if retained_source != source.root.symbol:
        raise ValueError(
            f"physical projector {key!r} changed the retained source P2 "
            "protocol")
    retained_definitions = result.definitions
    for symbol, source_definition in source.definitions.items():
        retained_definition = retained_definitions.get(symbol)
        if (retained_definition is None or
                retained_definition.kind != source_definition.kind or
                retained_definition.op.get_asm(
                    binary=True,
                    assume_verified=True,
                ) != source_definition.op.get_asm(
                    binary=True,
                    assume_verified=True,
                )):
            raise ValueError(
                f"physical projector {key!r} changed the retained source P2 "
                f"definition closure at @{symbol}")
    if result.evidence[:len(source.evidence)] != source.evidence:
        raise ValueError(
            f"physical projector {key!r} did not preserve P2 evidence")
    source_digest = sha256(source.serialize()).hexdigest()
    retained_architecture_digest = _projection_architecture_digest(
        projector,
        device,
    )
    if retained_architecture_digest != architecture_digest:
        raise ValueError(
            f"physical projector {key!r} returned an unstable projection "
            "architecture digest")
    obligations = {
        f"source-p2-digest={source_digest}",
        f"projection-architecture-digest={architecture_digest}",
    }
    if not any(evidence.producer == key and evidence.result == "pass" and
               obligations.issubset(evidence.obligations)
               for evidence in result.evidence):
        raise ValueError(
            f"physical projector {key!r} omitted source/architecture "
            "projection evidence")
    return result


def project_physical(source: Build,
                     *,
                     device,
                     pipeline=None,
                     experiment=None,
                     _transient=False) -> Build:
    if not isinstance(source, Build) or source.profile not in {"p2a", "p2n"}:
        raise TypeError(
            "physical projection requires a P2A/P2N cudaq.logical.Build")
    if not isinstance(device, Device):
        raise TypeError(
            "physical projection requires a typed cudaq.logical.Device")
    projector = _accepting_physical_projector(source, device)
    if projector is None:
        if pipeline is None:
            from .pipeline import pipelines

            pipeline = pipelines.physical()
        native = _project_physical_native(
            source,
            device=device,
            pipeline=pipeline,
            experiment=experiment,
            _transient=_transient,
        )
        return native
    selected_pipeline = (projector.projection_pipeline
                         if pipeline is None else pipeline)
    required = _required_network_projector(source)
    if required is not None:
        from ..qec import lattice_surgery

        if (lattice_surgery._pipeline_digest(selected_pipeline)
                != required.required_projector_pipeline_sha256):
            raise ValueError(
                "requested P3 pipeline differs from the exact network plan")
    verdict = projector.accepts_projection_pipeline(selected_pipeline)
    if type(verdict) is not bool:
        raise TypeError(f"physical projector {projector.key!r} "
                        "accepts_projection_pipeline() must return bool")
    if not verdict:
        raise ValueError(
            f"physical projector {projector.key!r} does not implement the "
            "requested P3 pipeline")
    architecture_digest = _projection_architecture_digest(projector, device)
    emit_projection = getattr(projector, "emit_projection", None)
    if required is not None:
        if not callable(emit_projection):
            raise TypeError(
                f"canonical network projector {projector.key!r} must "
                "implement emit_projection()")
        projection_architecture = getattr(projector, "projection_architecture",
                                          None)
        if not callable(projection_architecture):
            raise TypeError(
                f"canonical network projector {projector.key!r} must "
                "implement projection_architecture()")
        from ..qec import lattice_surgery

        projection = lattice_surgery.network_projection(source, device=device)
        prepared = PhysicalProjectionBuilder(
            f"{device.name}_qec_network_projection",
            architecture=projection_architecture(projection, device),
            projection=projection,
            device=device,
            pipeline=selected_pipeline,
        )
        emission = emit_projection(
            projection,
            device,
            builder=prepared,
            pipeline=selected_pipeline,
            experiment=experiment,
        )
        result = _finish_projection_emission(
            source,
            device,
            projector,
            selected_pipeline,
            emission,
            experiment=experiment,
        )
    else:
        result = projector.project_p3(
            source,
            device,
            pipeline=selected_pipeline,
            experiment=experiment,
        )
    return _validate_custom_projection(
        source,
        device,
        projector,
        selected_pipeline,
        result,
        architecture_digest=architecture_digest,
    )


def _project_physical_native(source: Build,
                             *,
                             device: Device,
                             pipeline,
                             experiment=None,
                             _transient=False):
    """Use the native progressive projector when its strict subset accepts.

    The native preflight pass is read-only, but preparing its canonical device
    closure is part of the live progressive transaction. An unsupported
    selected P2 closure therefore continues through the complete Python
    reference projector on this same prepared transaction. It must not create
    a second projector and duplicate device-owned symbols. Once preflight
    succeeds, native emission is authoritative and any later failure is
    surfaced rather than hidden by a semantic fallback.
    """

    profile = os.getenv("QLX_PROFILE_P2_TO_P3") is not None

    def timed(label, function):
        started = time.perf_counter()
        result = function()
        if profile:
            print(
                f"p2-to-p3 {label} {time.perf_counter() - started:.6f}s",
                flush=True,
            )
        return result

    prepared = timed("prepare", lambda: _P2ToP3(source, device))
    # Profile-selected success sidecars remain owned by the complete Python
    # reference projector. Native v1 does not yet consume the selected
    # root-profile option; accepting such a closure would silently publish a
    # P3 graph without its authenticated selection boundary.
    if prepared.root_profile is not None:
        return prepared.project(
            pipeline=pipeline,
            experiment=experiment,
            _transient=_transient,
        )
    root_symbol = _symbol(prepared.root)
    options = (f"root-symbol={root_symbol} device-symbol={device.name} "
               f"graph-symbol={prepared.graph_symbol}")
    if prepared.root_profile is not None:
        options += f" root-profile-symbol={_symbol(prepared.root_profile)}"
    from cudaq.logical._native import native

    try:
        timed(
            "native-preflight",
            lambda: native.run_pass(
                prepared.module,
                f"fabric-to-phys{{{options} preflight-only=true}}",
                verify=False,
            ),
        )
    except (mlir_ir.MLIRError, RuntimeError):
        # Preflight is read-only. Reuse the already cloned and prepared module
        # for the complete Python reference projection rather than replaying
        # the published P2 a second time.
        return prepared.project(
            pipeline=pipeline,
            experiment=experiment,
            _transient=_transient,
        )
    try:
        timed(
            "native-project",
            lambda: native.run_pass(
                prepared.module,
                f"fabric-to-phys{{{options} defer-output-verification=true}}",
                verify=False,
            ),
        )
    except RuntimeError as error:
        message = str(error)
        if "cannot satisfy an allocation that needs" in message:
            raise PlacementInfeasible(message) from error
        raise
    graph = prepared.transaction.find_symbol(prepared.graph_symbol,
                                             "phys.graph")
    if graph is None:
        raise ValueError("native P2-to-P3 projection omitted its phys.graph")

    # The native graph verifier is the ownership authority. The projector
    # records the count it computed in C++, and GraphOp independently recounts
    # and authenticates that attribute during the same verified increment.
    # Reading the retained scalar avoids a second Python walk over a
    # million-operation paper graph.
    try:
        linear_values_attr = graph.attributes["linear_value_count"]
    except KeyError as error:
        raise ValueError(
            "native physical projection omitted its verified linear-value count"
        ) from error
    linear_values = int(getattr(linear_values_attr, "value",
                                linear_values_attr))

    # The native pass recursively verified the new graph and every new P3
    # sidecar against the already authenticated P2/device input increment.
    # GraphOp's verifier owns physical-state linearity.
    linearity = EvidenceRecord(
        kind="linearity_verification",
        producer="phys.graph-verifier@0.3",
        result="pass",
        obligations=("linear-physical-state-ownership",),
        assumptions=(
            "checker:phys.graph-physical-state-linearity/v1",
            f"subject:physical graph @{prepared.graph_symbol}",
            f"linear-values:{linear_values}",
        ),
    )
    return Build._from_verified_increment(
        context=prepared.context,
        module=prepared.module,
        root=DefinitionHandle(prepared.graph_symbol, "physical_graph", "p3"),
        profile="p3",
        facets=(
            *source.facets,
            "patch_graph",
            "native_legalization",
            "carrier_mapping",
            "physical_routing",
        ),
        pipeline=pipeline,
        evidence=(
            *source.evidence,
            EvidenceRecord(
                kind="physical_projection_verification",
                producer="fabric-to-phys@0.3",
                result="pass",
                obligations=(
                    "selected-p2-realizations",
                    "deterministic-carrier-allocation",
                    "shared-physical-call-bodies",
                    "native-action-support",
                    "parallel-plan-deterministic-commit",
                ),
            ),
            linearity,
        ),
        value_groups={
            name: len(group) for name, group in source.values._groups.items()
        },
        placement=source.placement,
        qec_selection=source.qec_selection,
        experiment=experiment or source.experiment,
        device=device,
        source_modules=source.source_modules,
        _transient=_transient,
    )
