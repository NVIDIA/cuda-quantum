# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
from types import NoneType
from typing import get_args, get_origin, get_type_hints

import cudaq.mlir.ir as mlir_ir

from ..errors import UnsupportedCombination
from cudaq.logical.programs.context import (
    pop_trace,
    push_trace,
)
from cudaq.logical.types.values import (
    EventState,
    EventStatusValue,
    FabricEventValue,
    IndexValue,
    LogicalBool,
    MeasurementBits,
    PatchGaugeRef,
    PatchLogicalRef,
    PatchBundleValue,
    PatchValue,
    ResourceValue,
    SyndromeValue,
)
from cudaq.logical.algebra.angle import Angle
from cudaq.logical.codes import (
    Code,
    Encoding,
    EncodingEpoch,
)
from cudaq.logical.gadgets import (
    GadgetDefinition,
    GadgetProfile,
    patch,
)
from cudaq.logical.devices.builder import LogicalRegionBuilder
from cudaq.logical.devices.definition import QECRegion
from cudaq.logical.programs.definition import ProgramDefinition
from cudaq.logical.protocols.definition import ProtocolDefinition
from cudaq.logical.architecture.logical import Space
from cudaq.logical.types.semantic import (
    record,
    resource,
)
from cudaq.logical.gadgets import (
    CommitPointKind,
    InputSyndromeRef,
    ProfileParity,
    RecordRef,
    RetryPolicy,
    _PredicateProvenance,
)
from ..std import LogicalActionRef, LogicalInstrumentRef, ResourceFlowRef


class ProtocolBuilder:
    """Build a folded P2N graph while preserving linear patch ownership."""

    def __init__(self, transaction, definition: ProtocolDefinition) -> None:
        self.transaction = transaction
        self.definition = definition
        self.context = transaction.context
        self.location = transaction.location
        self._patches: list[PatchValue] = []
        self._bundles: list[PatchBundleValue] = []
        self._resources: list[ResourceValue] = []
        self._events: list[FabricEventValue] = []
        self._arguments = None
        self._finished = False
        hints = definition.type_hints
        self.input_boundaries = tuple(
            self._input_boundary(hints.get(name, parameter.annotation))
            for name, parameter in definition.signature.parameters.items())
        result_annotation = hints.get("return",
                                      definition.signature.return_annotation)
        self.result_boundaries = self._flatten_result_boundaries(
            result_annotation)
        self.input_types = tuple(
            self._boundary_type(kind, value)
            for kind, value in self.input_boundaries)
        self.result_types = tuple(
            self._boundary_type(kind, value)
            for kind, value in self.result_boundaries)
        self.function_type = mlir_ir.FunctionType.get(self.input_types,
                                                      self.result_types,
                                                      context=self.context)
        self.symbol = transaction.unique_symbol(definition.name)
        self.objective = self._materialize_objective(definition.implements)
        self._create_protocol()

    def _encoding_from_annotation(self, annotation):
        if get_origin(annotation) is not patch:
            raise TypeError("@cudaq.logical.protocol boundaries must use "
                            "cudaq.logical.patch[Code|Encoding]")
        (target,) = get_args(annotation)
        if isinstance(target, Code):
            return target.default_encoding
        if isinstance(target, Encoding):
            return target
        raise TypeError(
            "cudaq.logical.patch[...] requires a concrete Code or Encoding")

    @staticmethod
    def _encoding_target(target):
        if isinstance(target, Code):
            return target.default_encoding
        if isinstance(target, Encoding):
            return target
        raise TypeError("record schema requires a concrete Code or Encoding")

    @staticmethod
    def _kind_name(kind):
        return str(getattr(kind, "name", kind))

    @classmethod
    def _resource_payload_roles(cls, kind):
        from ..std import AUTO_CCZ_STATE

        if cls._kind_name(kind) == AUTO_CCZ_STATE.name:
            return AUTO_CCZ_STATE.payload_roles
        return tuple(getattr(kind, "payload_roles", ()))

    def _input_boundary(self, annotation):
        origin = get_origin(annotation)
        if origin is patch:
            return "patch", self._encoding_from_annotation(annotation)
        if origin is resource:
            (kind,) = get_args(annotation)
            return "resource", kind
        if origin is record:
            (schema,) = get_args(annotation)
            return "syndrome", self._encoding_target(schema)
        if annotation is bool:
            return "bool", None
        raise TypeError(
            "@cudaq.logical.protocol boundaries require patch, record, "
            "resource, or bool")

    @property
    def i1_type(self):
        return mlir_ir.IntegerType.get_signless(1, context=self.context)

    def _flatten_result_boundaries(self, annotation):
        if annotation in (None, NoneType):
            return ()
        if get_origin(annotation) in (tuple, list):
            result = []
            for item in get_args(annotation):
                result.extend(self._flatten_result_boundaries(item))
            return tuple(result)
        if annotation is bool:
            return (("bool", None),)
        if get_origin(annotation) is resource:
            (kind,) = get_args(annotation)
            return (("resource", kind),)
        if get_origin(annotation) is record:
            (schema,) = get_args(annotation)
            return (("syndrome", self._encoding_target(schema)),)
        return (("patch", self._encoding_from_annotation(annotation)),)

    def _patch_type(self, encoding, *, epoch=None):
        code = self.transaction.materialize(encoding.code)
        encoded = self.transaction.materialize(encoding)
        epoch = self.transaction.materialize(encoding.initial_epoch if epoch is
                                             None else epoch)
        return mlir_ir.Type.parse(
            f"!fabric.patch<@{code.symbol}, @{encoded.symbol}, @{epoch.symbol}>",
            context=self.context,
        )

    def _boundary_type(self, kind, value):
        if kind == "patch":
            return self._patch_type(value)
        if kind == "resource":
            return mlir_ir.Type.parse(
                f"!fabric.resource<@{self._kind_name(value)}>",
                context=self.context,
            )
        if kind == "syndrome":
            code = self.transaction.materialize(value.code)
            encoding = self.transaction.materialize(value)
            epoch = self.transaction.materialize(value.initial_epoch)
            return mlir_ir.Type.parse(
                f"!fabric.syndrome<@{code.symbol}, @{encoding.symbol}, "
                f"@{epoch.symbol}>",
                context=self.context,
            )
        if kind == "bool":
            return self.i1_type
        raise TypeError(f"unsupported protocol boundary kind {kind!r}")

    def _resource_type(self, kind):
        return mlir_ir.Type.parse(f"!fabric.resource<@{self._kind_name(kind)}>",
                                  context=self.context)

    def _materialize_objective(self, logical):
        if logical is None:
            return None
        if isinstance(logical, LogicalActionRef):
            return mlir_ir.Attribute.parse(f"#qlx.action<{logical.name}>",
                                           context=self.context)
        if isinstance(logical, LogicalInstrumentRef):
            if logical.name == "mpp":
                return mlir_ir.Attribute.parse("#qlx.instrument<mpp>",
                                               context=self.context)
            logical_type = mlir_ir.Type.parse("!qlx.logical_qubit",
                                              context=self.context)
            inputs = (logical_type,) * logical.arity
            results = ((logical_type,) * logical.result_arity
                       if logical.name.startswith("prepare_") else
                       (self.i1_type,) * logical.result_arity)
            symbol = self.transaction.objective(
                family="instrument",
                name=logical.name,
                inputs=inputs,
                results=results,
            )
            return mlir_ir.FlatSymbolRefAttr.get(symbol, context=self.context)
        if isinstance(logical, ResourceFlowRef):
            resource_type = self._resource_type(logical.resource)
            if logical.kind == "produce":
                inputs, results = (), (resource_type,)
            elif logical.kind == "transport":
                inputs = results = (resource_type,)
            else:
                raise ValueError(
                    f"unsupported resource-flow objective {logical.kind!r}")
            symbol = self.transaction.objective(
                family="action",
                name=logical.name,
                inputs=inputs,
                results=results,
            )
            return mlir_ir.FlatSymbolRefAttr.get(symbol, context=self.context)
        if isinstance(logical,
                      ProgramDefinition) and logical.kind == "objective":
            return mlir_ir.FlatSymbolRefAttr.get(
                self.transaction.materialize(logical).symbol,
                context=self.context,
            )
        raise TypeError(
            "@cudaq.logical.protocol implements= requires a logical objective")

    def _create_protocol(self):
        with self.context:
            function_type_attr = mlir_ir.TypeAttr.get(self.function_type)
        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(self.symbol, context=self.context),
            "function_type":
                function_type_attr,
        }
        if self.objective is not None:
            attrs["objective"] = self.objective
        if self.definition.metadata:
            attrs["metadata"] = mlir_ir.DictAttr.get(
                {
                    key:
                        mlir_ir.StringAttr.get(str(value), context=self.context)
                    for key, value in self.definition.metadata.items()
                },
                context=self.context,
            )
        with self.location:
            self.operation = mlir_ir.Operation.create("fabric.protocol",
                                                      attributes=attrs,
                                                      regions=1,
                                                      loc=self.location)
            self.transaction.module.body.append(self.operation)
            self.block = self.operation.regions[0].blocks.append(
                *self.input_types)
        self.insertion_point = mlir_ir.InsertionPoint(self.block)

    def _emit(self,
              name,
              *,
              operands=(),
              results=(),
              attributes=None,
              regions=0):
        with self.location:
            operation = mlir_ir.Operation.create(
                name,
                operands=list(operands),
                results=list(results),
                attributes=dict(attributes or {}),
                regions=regions,
                loc=self.location,
            )
            self.insertion_point.insert(operation)
        return operation

    def _new_patch(self, value, encoding, *, epoch=None):
        result = PatchValue(
            value,
            owner=self,
            encoding=encoding,
            epoch=epoch,
            semantic_ref=("patch", self.symbol, len(self._patches)),
            location=self.location,
        )
        self._patches.append(result)
        return result

    def _new_resource(self, value, kind):
        result = ResourceValue(value,
                               owner=self,
                               kind=kind,
                               location=self.location)
        self._resources.append(result)
        return result

    def _new_event(self, value, kind):
        result = FabricEventValue(value,
                                  owner=self,
                                  payload_kind=kind,
                                  location=self.location)
        self._events.append(result)
        return result

    def _wrap_boundary(self, value, boundary):
        kind, payload = boundary
        if kind == "patch":
            return self._new_patch(value, payload)
        if kind == "resource":
            return self._new_resource(value, payload)
        if kind == "syndrome":
            return SyndromeValue(
                value,
                owner=self,
                encoding=payload,
                record="boundary",
                location=self.location,
            )
        if kind == "bool":
            return LogicalBool(value, owner=self)
        raise TypeError(f"unsupported protocol boundary kind {kind!r}")

    def arguments(self):
        if self._arguments is None:
            self._arguments = tuple(
                self._wrap_boundary(value, boundary) for value, boundary in zip(
                    self.block.arguments, self.input_boundaries))
        return self._arguments

    def trace(self):
        args = self.arguments()
        token = push_trace(self)
        try:
            returned = self.definition.provider(*args)
        finally:
            pop_trace(token)
        self.finish(returned)

    def finish(self, returned):
        if self._finished:
            raise RuntimeError("ProtocolBuilder root is already finished")
        self._finished = True
        values = (() if returned is None else
                  returned if isinstance(returned, tuple) else (returned,))
        if len(values) != len(self.result_types):
            raise TypeError(
                "protocol return count does not match its annotation")
        operands = []
        returned_predicates = []
        predicate_contracts = []
        for result_index, (value, expected, boundary) in enumerate(
                zip(values, self.result_types, self.result_boundaries)):
            if boundary[0] == "patch":
                if not isinstance(value, PatchValue) or value.owner is not self:
                    raise TypeError(
                        "protocol patch results must be live patches")
                if value.type != expected:
                    raise TypeError(
                        "protocol returned a patch with the wrong boundary")
                value._consume("fabric.protocol_return")
                operands.append(value.mlir_value)
            elif boundary[0] == "bool":
                if not isinstance(value,
                                  LogicalBool) or value.owner is not self:
                    raise TypeError(
                        "protocol bool results must be traced values")
                operands.append(value.mlir_value)
                returned_predicates.append(value.producer if isinstance(
                    value.producer, _PredicateProvenance) else None)
                if (isinstance(value.producer, _PredicateProvenance) and
                        isinstance(value.producer.analysis, GadgetProfile) and
                        value.producer.profile is not None and
                        value.mlir_value.owner.name == "fabric.call"):
                    predicate_contracts.append((result_index, value.producer))
            elif boundary[0] == "resource":
                if not isinstance(value,
                                  ResourceValue) or value.owner is not self:
                    raise TypeError(
                        "protocol resource results must be live resources")
                if value.type != expected:
                    raise TypeError(
                        "protocol returned a resource with the wrong kind")
                value._consume("fabric.protocol_return")
                operands.append(value.mlir_value)
            elif boundary[0] == "syndrome":
                if not isinstance(value,
                                  SyndromeValue) or value.owner is not self:
                    raise TypeError(
                        "protocol syndrome results must be traced values")
                if value.type != expected:
                    raise TypeError(
                        "protocol returned a syndrome with the wrong schema")
                operands.append(value.mlir_value)
        self._emit("fabric.protocol_return", operands=operands)
        if any(value.is_live for value in self._patches):
            raise RuntimeError(
                "protocol leaves a patch owner live at its boundary")
        if any(value.is_live for value in self._bundles):
            raise RuntimeError(
                "protocol leaves a hierarchical patch bundle live")
        if any(value.is_live for value in self._resources):
            raise RuntimeError("protocol leaves a resource owner live")
        if any(value.is_live for value in self._events):
            raise RuntimeError("protocol leaves a resource event live")
        self.transaction._protocol_result_provenance[self.symbol] = tuple(
            returned_predicates)
        if len(predicate_contracts) == 1:
            result_index, provenance = predicate_contracts[0]
            gadget = self.transaction.materialize(provenance.analysis.gadget)
            self.operation.attributes["predicate_gadget"] = (
                mlir_ir.FlatSymbolRefAttr.get(gadget.symbol,
                                              context=self.context))
            self.operation.attributes["predicate_profile"] = (
                mlir_ir.FlatSymbolRefAttr.get(provenance.profile,
                                              context=self.context))
            self.operation.attributes["predicate_result"] = (
                mlir_ir.IntegerAttr.get(
                    mlir_ir.IntegerType.get_signless(64, context=self.context),
                    result_index,
                ))

    def request(self, kind):
        kind_name = self._kind_name(kind)
        routes = getattr(self.transaction, "_resource_streams", None)
        if routes is None:
            stream_path = (f"{kind_name}_stream",)
        else:
            candidates = routes.get(kind_name, ())
            if not candidates:
                raise ValueError(
                    f"selected device has no stream producing {kind_name!r}")
            if len(candidates) != 1:
                raise ValueError(
                    f"selected device has ambiguous streams producing "
                    f"{kind_name!r}: {candidates!r}")
            stream_path = candidates[0]
        resource_type = self._resource_type(kind_name)
        event_type = mlir_ir.Type.parse(
            f'!event.handle<{resource_type}, "linear">', context=self.context)
        with self.context:
            stream_reference = mlir_ir.SymbolRefAttr.get(list(stream_path),
                                                         context=self.context)
        operation = self._emit(
            "fabric.resource_request",
            results=[event_type],
            attributes={
                "kind": mlir_ir.StringAttr.get(kind_name, context=self.context),
                "stream": stream_reference,
            },
        )
        return self._new_event(operation.result, kind)

    def event_test(self, event):
        if not isinstance(event, FabricEventValue) or event.owner is not self:
            raise TypeError(
                "event_test expects a Fabric event from this protocol")
        if not event.is_live:
            event._consume("event.test")
        operation = self._emit(
            "event.test",
            operands=[event.mlir_value],
            results=[self.i1_type],
        )
        return LogicalBool(operation.result, owner=self)

    def event_poll(self, event):
        if not isinstance(event, FabricEventValue) or event.owner is not self:
            raise TypeError(
                "event_poll expects a Fabric event from this protocol")
        if not event.is_live:
            event._consume("event.poll")
        operation = self._emit(
            "event.poll",
            operands=[event.mlir_value],
            results=[mlir_ir.IntegerType.get_signless(8, context=self.context)],
        )
        return EventStatusValue(operation.result, owner=self)

    def event_is(self, status, state):
        if not isinstance(status, EventStatusValue) or status.owner is not self:
            raise TypeError(
                "event_is expects an event status from this protocol")
        try:
            state_name = EventState(state).value
        except ValueError as error:
            raise ValueError(f"unknown event state: {state!r}") from error
        operation = self._emit(
            "event.is",
            operands=[status.mlir_value],
            results=[self.i1_type],
            attributes={
                "state":
                    mlir_ir.StringAttr.get(state_name, context=self.context)
            },
        )
        return LogicalBool(operation.result, owner=self)

    def event_select_ready(self, events, *, policy):
        events = tuple(events)
        if not events:
            raise ValueError("event_select_ready requires at least one event")
        if any(not isinstance(event, FabricEventValue) or
               event.owner is not self for event in events):
            raise TypeError(
                "event_select_ready expects events from this protocol")
        if any(not event.is_live for event in events):
            raise ValueError(
                "event_select_ready cannot inspect a consumed event")
        if len({str(event.mlir_value.type) for event in events}) != 1:
            raise TypeError(
                "event_select_ready events must have one common type")
        if policy not in {"priority", "deterministic", "fair"}:
            raise ValueError(
                "event_select_ready policy must be priority, deterministic, or fair"
            )
        operation = self._emit(
            "event.select_ready",
            operands=[event.mlir_value for event in events],
            results=[mlir_ir.IndexType.get(context=self.context)],
            attributes={
                "policy": mlir_ir.StringAttr.get(policy, context=self.context)
            },
        )
        return IndexValue(operation.result, owner=self)

    def event_try_take(self, event, carries, *, ready, pending, failed):
        if not isinstance(event, FabricEventValue) or event.owner is not self:
            raise TypeError(
                "event_try_take expects a Fabric event from this protocol")
        event._consume("event.try_take")
        carries = tuple(carries)
        for value in carries:
            if getattr(value, "owner", None) is not self:
                raise TypeError(
                    "event_try_take carries must belong to this protocol")
            self._consume_branch_value(value, "event.try_take")
        result_types = tuple(value.mlir_value.type for value in carries)
        operation = self._emit(
            "event.try_take",
            operands=[
                event.mlir_value, *(value.mlir_value for value in carries)
            ],
            results=result_types,
            regions=3,
        )
        alternative_types = (
            self._resource_type(event.payload_kind),
            event.mlir_value.type,
            mlir_ir.IntegerType.get_signless(8, context=self.context),
        )
        callbacks = (ready, pending, failed)
        labels = ("ready", "pending", "failed")
        parent_ip = self.insertion_point
        try:
            for region, alternative_type, callback, label in zip(
                    operation.regions, alternative_types, callbacks, labels):
                with self.location:
                    block = region.blocks.append(alternative_type,
                                                 *result_types)
                self.insertion_point = mlir_ir.InsertionPoint(block)
                if label == "ready":
                    alternative = self._new_resource(block.arguments[0],
                                                     event.payload_kind)
                elif label == "pending":
                    alternative = self._new_event(block.arguments[0],
                                                  event.payload_kind)
                else:
                    alternative = EventStatusValue(block.arguments[0],
                                                   owner=self)
                branch_carries = tuple(
                    self._wrap_like(argument, prototype) for argument, prototype
                    in zip(block.arguments[1:], carries))
                returned = self._flatten_values(
                    callback(alternative, *branch_carries))
                if len(returned) != len(result_types):
                    raise TypeError(
                        "event_try_take branches must return one value per carry"
                    )
                operands = []
                for value, expected in zip(returned, result_types):
                    if (getattr(value, "owner", None) is not self or
                            value.mlir_value.type != expected):
                        raise TypeError(
                            "event_try_take branch result types must match carries"
                        )
                    self._consume_branch_value(value, "event.yield")
                    operands.append(value.mlir_value)
                self._emit("event.yield", operands=operands)
        finally:
            self.insertion_point = parent_ip
        return tuple(
            self._wrap_like(result, prototype)
            for result, prototype in zip(operation.results, carries))

    def event_cancel(self, event, *, reason=None):
        if not isinstance(event, FabricEventValue) or event.owner is not self:
            raise TypeError(
                "event_cancel expects a Fabric event from this protocol")
        if reason is not None and (not isinstance(reason, str) or not reason):
            raise TypeError("event_cancel reason must be a nonempty string")
        event._consume("event.cancel")
        attrs = {}
        if reason is not None:
            attrs["reason"] = mlir_ir.StringAttr.get(reason,
                                                     context=self.context)
        operation = self._emit(
            "event.cancel",
            operands=[event.mlir_value],
            results=[mlir_ir.IntegerType.get_signless(8, context=self.context)],
            attributes=attrs,
        )
        return EventStatusValue(operation.result, owner=self)

    def event_await(self, event):
        if not isinstance(event, FabricEventValue) or event.owner is not self:
            raise TypeError(
                "event_await expects a Fabric event from this protocol")
        event._consume("event.await")
        operation = self._emit(
            "event.await",
            operands=[event.mlir_value],
            results=[self._resource_type(event.payload_kind)],
        )
        return self._new_resource(operation.result, event.payload_kind)

    def fence(self, effects):
        effects = tuple(
            str(getattr(effect, "value", effect)) for effect in effects)
        allowed = {
            "all",
            "quantum",
            "classical",
            "resource",
            "event",
            "frame",
            "outcome",
            "selection",
        }
        if not effects:
            raise ValueError("fence requires at least one semantic effect")
        if len(set(effects)) != len(effects):
            raise ValueError("fence effects must be unique")
        unknown = set(effects) - allowed
        if unknown:
            raise ValueError(f"unknown fence effects: {sorted(unknown)!r}")
        if "all" in effects and len(effects) != 1:
            raise ValueError("fence effect 'all' cannot be combined")
        self._emit(
            "event.fence",
            attributes={
                "effects":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.StringAttr.get(effect, context=self.context)
                            for effect in effects
                        ],
                        context=self.context,
                    )
            },
        )

    def xor(self, lhs, rhs):
        if any(not isinstance(value, LogicalBool) or value.owner is not self
               for value in (lhs, rhs)):
            raise TypeError(
                "cudaq.logical.xor expects two Boolean values from this protocol"
            )
        operation = self._emit(
            "fabric.xor",
            operands=[lhs.mlir_value, rhs.mlir_value],
            results=[self.i1_type],
        )
        producer = None
        if (isinstance(lhs.producer, _PredicateProvenance) and
                lhs.producer.same_invocation(rhs.producer) and
                lhs.producer.parity is not None and
                rhs.producer.parity is not None):
            producer = replace(
                lhs.producer,
                parity=lhs.producer.parity ^ rhs.producer.parity,
                outcome_index=None,
                all_false_rows=None,
                all_false_indices=None,
            )
        return LogicalBool(operation.result, owner=self, producer=producer)

    def all_false(self, *events):
        if not events or any(
                not isinstance(value, LogicalBool) or value.owner is not self
                for value in events):
            raise TypeError(
                "cudaq.logical.all_false expects one or more Boolean events from this "
                "protocol")
        operation = self._emit(
            "fabric.all_false",
            operands=[value.mlir_value for value in events],
            results=[self.i1_type],
        )
        producer = events[0].producer
        if (not isinstance(producer, _PredicateProvenance) or
                producer.parity is None or producer.outcome_index is None or
                any(not producer.same_invocation(event.producer) or
                    event.producer.parity is None or
                    event.producer.outcome_index is None
                    for event in events[1:])):
            producer = None
        else:
            producer = replace(
                producer,
                parity=None,
                outcome_index=None,
                all_false_rows=tuple(event.producer.parity for event in events),
                all_false_indices=tuple(
                    event.producer.outcome_index for event in events),
            )
        return LogicalBool(operation.result, owner=self, producer=producer)

    @staticmethod
    def _protocol_attr(transaction, protocol, *, fallback):
        if isinstance(protocol, (GadgetDefinition, ProtocolDefinition)):
            handle = transaction.materialize(protocol)
            return mlir_ir.FlatSymbolRefAttr.get(handle.symbol,
                                                 context=transaction.context)
        name = fallback if protocol is None else str(protocol)
        return mlir_ir.Attribute.parse(f'#fabric.spec_only<"{name}">',
                                       context=transaction.context)

    def produce(self, kind, *, region=None, protocol=None):
        kind_name = self._kind_name(kind)
        if region is None:
            region = self.definition._factory_region
        if isinstance(region, LogicalRegionBuilder):
            region = region.space
        objective = self.definition.implements
        if (isinstance(objective, ResourceFlowRef) and
                objective.kind == "produce" and
                objective.resource.name == kind_name and
                self.definition._factory_region is not None):
            region = self.definition._factory_region
        if not isinstance(region, Space):
            raise TypeError(
                "cudaq.logical.produce region= requires a typed logical region")
        if not region.name:
            raise ValueError(
                "cudaq.logical.produce region= requires a named logical region")
        protocol_attr = (mlir_ir.FlatSymbolRefAttr.get(self.symbol,
                                                       context=self.context)
                         if protocol is None else self._protocol_attr(
                             self.transaction,
                             protocol,
                             fallback=f"inline_{kind_name}_production",
                         ))
        operation = self._emit(
            "fabric.produce_resource",
            results=[self._resource_type(kind_name)],
            attributes={
                "region":
                    mlir_ir.FlatSymbolRefAttr.get(region.name,
                                                  context=self.context),
                "resource_kind":
                    mlir_ir.FlatSymbolRefAttr.get(kind_name,
                                                  context=self.context),
                "protocol":
                    protocol_attr,
            },
        )
        return self._new_resource(operation.result, kind)

    def transport(self, resource_value, *, source, destination, protocol=None):
        if (not isinstance(resource_value, ResourceValue) or
                resource_value.owner is not self):
            raise TypeError("transport expects a live protocol resource")
        resource_value._consume("fabric.transport")
        source_name = getattr(source, "name", str(source))
        destination_name = getattr(destination, "name", str(destination))
        operation = self._emit(
            "fabric.transport",
            operands=[resource_value.mlir_value],
            results=[resource_value.type],
            attributes={
                "src_region":
                    mlir_ir.FlatSymbolRefAttr.get(source_name,
                                                  context=self.context),
                "dst_region":
                    mlir_ir.FlatSymbolRefAttr.get(destination_name,
                                                  context=self.context),
                "protocol":
                    self._protocol_attr(
                        self.transaction,
                        protocol,
                        fallback=
                        (f"inline_{self._kind_name(resource_value.kind)}_transport_"
                         f"{source_name}_to_{destination_name}"),
                    ),
            },
        )
        return self._new_resource(operation.result, resource_value.kind)

    def unpack_resource(
        self,
        resource_value,
        *,
        like,
        encoding=None,
        logical_ports=None,
    ):
        if (not isinstance(resource_value, ResourceValue) or
                resource_value.owner is not self):
            raise TypeError("unpack_resource expects one live resource owner")
        scalar = isinstance(like, PatchValue)
        anchors = (like,) if scalar else tuple(like)
        if not anchors or any(
                not isinstance(anchor, PatchValue) or anchor.owner is not self
                for anchor in anchors):
            raise TypeError(
                "unpack_resource like= expects one live patch or a nonempty "
                "collection of live patch owners")
        if len({id(anchor) for anchor in anchors}) != len(anchors):
            raise ValueError(
                "unpack_resource like= collection must contain distinct patch owners"
            )
        payload_roles = self._resource_payload_roles(resource_value.kind)
        if payload_roles and len(anchors) != len(payload_roles):
            raise ValueError(
                f"{self._kind_name(resource_value.kind)} requires exactly "
                f"{len(payload_roles)} payload roles")
        if encoding is None:
            payload_encodings = tuple(anchor.encoding for anchor in anchors)
        else:
            selected = self._encoding_target(encoding)
            payload_encodings = (selected,) * len(anchors)
        payload_epochs = tuple(
            anchor.epoch if payload_encoding is
            anchor.encoding else payload_encoding.initial_epoch
            for anchor, payload_encoding in zip(anchors, payload_encodings))
        from cudaq.logical.gadgets.builder import GadgetBuilder
        mapping = GadgetBuilder._resource_payload_logical_ports(
            resource_value,
            anchors,
            payload_encodings,
            logical_ports,
            require_explicit=not scalar,
        )
        resource_value._consume("fabric.unpack_resource")
        for anchor in anchors:
            anchor._consume("fabric.unpack_resource")
        attrs = {}
        if payload_roles:
            attrs["payload_roles"] = mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(role, context=self.context)
                    for role in payload_roles
                ],
                context=self.context,
            )
        if mapping is not None:
            block_indices, port_indices = mapping
            action = resource_value.kind.consume_action
            logical_blocks = self.transaction.protocol_payload_blocks(
                self.definition)
            if (logical_blocks is None and
                    isinstance(self.definition.implements, LogicalActionRef) and
                    self.definition.implements.name in {"ccz", "ccx"}):
                raise ValueError(
                    "generated three-qubit resource unpack is missing its selected "
                    "QEC block witness")
            attrs.update({
                "payload_action":
                    mlir_ir.Attribute.parse(f"#qlx.action<{action.name}>",
                                            context=self.context),
                "payload_logical_blocks":
                    mlir_ir.DenseI64ArrayAttr.get(block_indices, self.context),
                "payload_logical_ports":
                    mlir_ir.DenseI64ArrayAttr.get(port_indices, self.context),
            })
            if logical_blocks is not None:
                logical_blocks = tuple(logical_blocks)
                if any(not isinstance(block, str) or not block
                       for block in logical_blocks):
                    raise ValueError(
                        "generated payload block witness must contain nonempty "
                        "selected QEC block identities")
                if len(logical_blocks) == len(block_indices):
                    logical_block_rows = logical_blocks
                elif (len(logical_blocks) == len(anchors) and
                      len(set(logical_blocks)) == len(logical_blocks)):
                    logical_block_rows = tuple(
                        logical_blocks[index] for index in block_indices)
                else:
                    raise ValueError(
                        "generated payload block witness must contain either one "
                        "selected identity per action logical or one unique "
                        "identity per payload anchor")
                identity_by_payload = {}
                payload_by_identity = {}
                for payload_index, block_identity in zip(
                        block_indices, logical_block_rows):
                    if (payload_index in identity_by_payload and
                            identity_by_payload[payload_index]
                            != block_identity):
                        raise ValueError(
                            "one payload block maps to several selected QEC "
                            "block identities")
                    if (block_identity in payload_by_identity and
                            payload_by_identity[block_identity]
                            != payload_index):
                        raise ValueError(
                            "one selected QEC block identity maps to several "
                            "payload blocks")
                    identity_by_payload[payload_index] = block_identity
                    payload_by_identity[block_identity] = payload_index
                attrs["payload_logical_block_ids"] = mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.StringAttr.get(block, context=self.context)
                        for block in logical_block_rows
                    ],
                    context=self.context,
                )
        operation = self._emit(
            "fabric.unpack_resource",
            operands=[
                resource_value.mlir_value,
                *(anchor.mlir_value for anchor in anchors),
            ],
            results=[
                *(anchor.type for anchor in anchors),
                *(self._patch_type(payload_encoding, epoch=payload_epoch)
                  for payload_encoding, payload_epoch in zip(
                      payload_encodings, payload_epochs)),
            ],
            attributes=attrs,
        )
        successors = tuple(
            self._new_patch(result, anchor.encoding, epoch=anchor.epoch) for
            result, anchor in zip(operation.results[:len(anchors)], anchors))
        payloads = tuple(
            self._new_patch(result, payload_encoding, epoch=payload_epoch)
            for result, payload_encoding, payload_epoch in zip(
                operation.results[len(anchors):],
                payload_encodings,
                payload_epochs,
            ))
        return (successors[0], payloads[0]) if scalar else (successors,
                                                            payloads)

    def pack_resource(self, payload, *, kind):
        payloads = ((payload,)
                    if isinstance(payload, PatchValue) else tuple(payload))
        if not payloads or any(
                not isinstance(value, PatchValue) or value.owner is not self
                for value in payloads):
            raise TypeError(
                "pack_resource expects one live encoded patch or a nonempty "
                "collection of live encoded patches")
        if len({id(value) for value in payloads}) != len(payloads):
            raise ValueError("pack_resource payload patches must be distinct")
        kind_name = self._kind_name(kind)
        payload_roles = self._resource_payload_roles(kind)
        if payload_roles and len(payloads) != len(payload_roles):
            raise ValueError(
                f"{kind_name} requires exactly {len(payload_roles)} payload roles"
            )
        for value in payloads:
            value._consume("fabric.pack_resource")
        attributes = {
            "resource_kind":
                mlir_ir.FlatSymbolRefAttr.get(kind_name, context=self.context),
            "payload_encodings":
                mlir_ir.ArrayAttr.get([
                    mlir_ir.FlatSymbolRefAttr.get(
                        self.transaction.materialize(value.encoding).symbol,
                        context=self.context,
                    ) for value in payloads
                ],
                                      context=self.context),
        }
        if payload_roles:
            attributes["payload_roles"] = mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(role, context=self.context)
                    for role in payload_roles
                ],
                context=self.context,
            )
        operation = self._emit(
            "fabric.pack_resource",
            operands=[value.mlir_value for value in payloads],
            results=[self._resource_type(kind)],
            attributes=attributes,
        )
        return self._new_resource(operation.result, kind)

    def allocate_patch(self, target, *, region=None):
        if self.definition._factory_region is not None:
            region = self.definition._factory_region
        if region is None:
            raise TypeError(
                "protocol allocate_patch requires region= unless its "
                "producer is attached with logical.add_factory")
        encoding = self._encoding_target(target)
        code = self.transaction.materialize(encoding.code)
        region_name = getattr(region, "name", str(region))
        if not region_name:
            raise ValueError("allocate_patch region must be nonempty")
        attributes = {
            "code":
                mlir_ir.FlatSymbolRefAttr.get(code.symbol,
                                              context=self.context),
            "region":
                mlir_ir.FlatSymbolRefAttr.get(region_name,
                                              context=self.context),
        }
        if isinstance(region, QECRegion):
            attributes["strict_region"] = mlir_ir.UnitAttr.get(
                context=self.context)
        operation = self._emit(
            "fabric.alloc",
            results=[self._patch_type(encoding)],
            attributes=attributes,
        )
        return self._new_patch(operation.result, encoding)

    def prepare_patch(self, value, state: str):
        if not isinstance(value, PatchValue) or value.owner is not self:
            raise TypeError("encoded preparation expects one live patch")
        if state not in {"zero", "plus"}:
            raise ValueError("encoded preparation supports zero or plus")
        value._consume(f"fabric.prep_{'z' if state == 'zero' else 'x'}")
        operation = self._emit(
            "fabric.prep_z" if state == "zero" else "fabric.prep_x",
            operands=[value.mlir_value],
            results=[value.type],
        )
        return self._new_patch(operation.result,
                               value.encoding,
                               epoch=value.epoch)

    @staticmethod
    def _validate_selected_predicate(provenance, *, operation: str) -> None:
        analysis = provenance.analysis
        if not isinstance(analysis, GadgetProfile):
            raise UnsupportedCombination(
                f"{operation} requires a selected GadgetProfile with one or "
                "more success predicates")
        success_parities = analysis._effective_role_parities("success")
        if not success_parities:
            raise UnsupportedCombination(
                f"{operation} requires a selected GadgetProfile with one or "
                "more success predicates")
        if not provenance.outcome_rows:
            raise UnsupportedCombination(
                f"{operation} attempt has no total GadgetSpec OutcomeMap")

        bindings = []
        claimed = set()
        for mismatch in success_parities:
            support = tuple(record.name for record in mismatch.records)
            matches = tuple(
                index for index, row in enumerate(provenance.outcome_rows)
                if "success" in provenance.outcome_roles[index] and tuple(
                    record.name for record in row.records) == support and
                row.input_syndromes == mismatch.input_syndromes)
            if len(matches) != 1:
                raise UnsupportedCombination(
                    f"{operation} success semantics must match one exact "
                    "ordered OutcomeMap row")
            index = matches[0]
            if index in claimed:
                raise UnsupportedCombination(
                    f"{operation} success rows must map one-to-one to "
                    "distinct OutcomeMap rows")
            claimed.add(index)
            bindings.append((index, mismatch))

        binding_indices = tuple(index for index, _ in bindings)
        if provenance.all_false_indices is not None:
            if provenance.all_false_indices != binding_indices:
                raise UnsupportedCombination(
                    f"{operation} all_false predicate must preserve every "
                    "selected success-row identity and order")
            for index, mismatch in bindings:
                if provenance.outcome_rows[index] != mismatch:
                    raise UnsupportedCombination(
                        f"{operation} predicate polarity contradicts the "
                        "selected profile success semantics")
            return

        if len(bindings) != 1 or provenance.outcome_index != bindings[0][0]:
            raise UnsupportedCombination(
                f"{operation} multi-row success requires all_false over the "
                "complete ordered success table")
        index, mismatch = bindings[0]
        outcome = provenance.outcome_rows[index]
        if (outcome.records != mismatch.records or
                outcome.input_syndromes != mismatch.input_syndromes or
                outcome.constant == mismatch.constant):
            raise UnsupportedCombination(
                f"{operation} direct predicate polarity contradicts the "
                "selected profile success semantics")

    def postselect(self, predicate, *, expected=False):
        if not isinstance(predicate,
                          LogicalBool) or predicate.owner is not self:
            raise TypeError("postselect expects one Boolean from this protocol")
        if not isinstance(expected, bool):
            raise TypeError("postselect expected= must be a Python bool")
        attrs = {
            "mode":
                mlir_ir.StringAttr.get(
                    "require" if expected else "abort_on",
                    context=self.context,
                ),
            "accept_when":
                mlir_ir.BoolAttr.get(expected, context=self.context),
        }
        if isinstance(predicate.producer, _PredicateProvenance):
            provenance = predicate.producer
            # Postselection is protocol policy and may condition any exact
            # Boolean produced by the selected attempt.  When the call also
            # selects a GadgetProfile, verify its stronger success-table
            # contract; do not require a profile merely to authorize policy.
            if provenance.analysis is not None:
                self._validate_selected_predicate(provenance,
                                                  operation="postselect")
            attempt = provenance.attempt
            profile = provenance.profile
            attrs["attempt"] = mlir_ir.FlatSymbolRefAttr.get(
                attempt, context=self.context)
            if profile is not None:
                attrs["profile"] = mlir_ir.FlatSymbolRefAttr.get(
                    profile, context=self.context)
        self._emit(
            "event.selection",
            operands=[predicate.mlir_value],
            attributes=attrs,
        )

    def discard(self, values, *, reason=None):
        del reason
        values = values if isinstance(values, (tuple, list)) else (values,)
        for value in values:
            if isinstance(value, ResourceValue) and value.owner is self:
                value._consume("fabric.discard_resource")
                self._emit("fabric.discard_resource",
                           operands=[value.mlir_value])
            elif isinstance(value, PatchValue) and value.owner is self:
                value._consume("fabric.dealloc")
                self._emit("fabric.dealloc", operands=[value.mlir_value])
            else:
                raise TypeError(
                    "protocol discard expects live patches or resources")

    def _product_terms(self, product, operation):
        from cudaq.logical.algebra import PauliProduct

        if not isinstance(product, PauliProduct):
            raise TypeError(f"{operation} expects a cudaq.logical.PauliProduct")
        patches = []
        patch_positions = {}
        patch_indices = []
        logical_indices = []
        paulis = []
        for factor in product.factors:
            reference = factor.operand
            if not isinstance(reference, (PatchLogicalRef, PatchGaugeRef)):
                raise TypeError(
                    f"{operation} factors must reference patch[i] or "
                    "patch.gauge[i]")
            patch = reference.patch
            if patch.owner is not self:
                raise ValueError(
                    f"{operation} received a patch from another trace")
            key = patch.semantic_ref
            if key not in patch_positions:
                patch_positions[key] = len(patches)
                patches.append(patch)
            patch_indices.append(patch_positions[key])
            logical_indices.append(reference.index +
                                   (patch.encoding.code.k if isinstance(
                                       reference, PatchGaugeRef) else 0))
            paulis.append(factor.pauli)
        for patch in patches:
            patch._consume(operation)
        return patches, {
            "patch_indices":
                mlir_ir.DenseI64ArrayAttr.get(patch_indices, self.context),
            "logical_indices":
                mlir_ir.DenseI64ArrayAttr.get(logical_indices, self.context),
            "pauli_product":
                mlir_ir.StringAttr.get(
                    ("-" if product.sign < 0 else "") + "".join(paulis),
                    context=self.context,
                ),
        }

    def mpp(self, product):
        patches, attrs = self._product_terms(product, "fabric.measure_product")
        i1 = mlir_ir.IntegerType.get_signless(1, context=self.context)
        operation = self._emit(
            "fabric.measure_product",
            operands=[patch.mlir_value for patch in patches],
            results=[*(patch.type for patch in patches), i1],
            attributes=attrs,
        )
        successors = [
            self._new_patch(result, patch.encoding, epoch=patch.epoch)
            for result, patch in zip(operation.results[:-1], patches)
        ]
        return (*successors, LogicalBool(operation.results[-1], owner=self))

    def rotate(self, product, *, angle):
        patches, attrs = self._product_terms(product, "fabric.rotate_product")
        if isinstance(angle, Angle):
            angle = float(angle)
        if not isinstance(angle, (int, float)) or isinstance(angle, bool):
            raise TypeError(
                "P2 product rotations currently require a numeric angle")
        with self.location:
            attrs["angle"] = mlir_ir.FloatAttr.get(
                mlir_ir.F64Type.get(context=self.context), float(angle))
        attrs["synthesis"] = mlir_ir.StringAttr.get("auto",
                                                    context=self.context)
        operation = self._emit(
            "fabric.rotate_product",
            operands=[patch.mlir_value for patch in patches],
            results=[patch.type for patch in patches],
            attributes=attrs,
        )
        return [
            self._new_patch(result, patch.encoding, epoch=patch.epoch)
            for result, patch in zip(operation.results, patches)
        ]

    def resource_rotate(self, resource_value, product, *, angle):
        """Consume one typed resource in a protocol-level product rotation."""

        if (not isinstance(resource_value, ResourceValue) or
                resource_value.owner is not self):
            raise TypeError("resource_rotate expects one live resource owner")
        patches, attrs = self._product_terms(product,
                                             "fabric.resource_rotate_product")
        if isinstance(angle, Angle):
            angle = float(angle)
        if not isinstance(angle, (int, float)) or isinstance(angle, bool):
            raise TypeError("resource rotations require a numeric angle")
        resource_value._consume("fabric.resource_rotate_product")
        with self.location:
            attrs["angle"] = mlir_ir.FloatAttr.get(
                mlir_ir.F64Type.get(context=self.context), float(angle))
        operation = self._emit(
            "fabric.resource_rotate_product",
            operands=[
                resource_value.mlir_value,
                *(patch.mlir_value for patch in patches),
            ],
            results=[patch.type for patch in patches],
            attributes=attrs,
        )
        return [
            self._new_patch(result, patch.encoding, epoch=patch.epoch)
            for result, patch in zip(operation.results, patches)
        ]

    def _new_bundle(self, value, hierarchy, slot_group):
        result = PatchBundleValue(
            value,
            owner=self,
            hierarchy=hierarchy,
            slot_group=slot_group,
            location=self.location,
        )
        self._bundles.append(result)
        return result

    def _bundle_type(self, hierarchy, slot_group):
        hierarchy_handle = self.transaction.materialize(hierarchy)
        return mlir_ir.Type.parse(
            f"!fabric.patch_bundle<@{hierarchy_handle.symbol}, @{slot_group}>",
            context=self.context,
        )

    def encoding_unpack(self, parent, *, hierarchy=None, slot_group=None):
        if not isinstance(parent, PatchValue) or parent.owner is not self:
            raise TypeError("encoding_unpack requires a live protocol patch")
        hierarchy = hierarchy or parent.encoding.hierarchy
        if hierarchy is None or parent.encoding.hierarchy is not hierarchy:
            raise ValueError(
                "patch encoding does not own the requested hierarchy")
        slot_group = slot_group or hierarchy.child.name
        if not isinstance(slot_group, str) or not slot_group:
            raise TypeError("slot_group must be a nonempty string")
        hierarchy_handle = self.transaction.materialize(hierarchy)
        bundle_type = self._bundle_type(hierarchy, slot_group)
        parent._consume("fabric.encoding_unpack")
        operation = self._emit(
            "fabric.encoding_unpack",
            operands=[parent.mlir_value],
            results=[bundle_type],
            attributes={
                "hierarchy":
                    mlir_ir.FlatSymbolRefAttr.get(hierarchy_handle.symbol,
                                                  context=self.context),
                "slot_group":
                    mlir_ir.FlatSymbolRefAttr.get(slot_group,
                                                  context=self.context),
            },
        )
        return self._new_bundle(operation.result, hierarchy, slot_group)

    def map_children(self, callee, children):
        if not isinstance(children,
                          PatchBundleValue) or children.owner is not self:
            raise TypeError(
                "map_children requires a live protocol patch bundle")
        if not isinstance(callee, (GadgetDefinition, ProtocolDefinition)):
            raise TypeError("map_children callee must be a gadget or protocol")
        children._consume("fabric.map_children")
        handle = self.transaction.materialize(callee)
        operation = self._emit(
            "fabric.map_children",
            operands=[children.mlir_value],
            results=[children.type],
            attributes={
                "callee":
                    mlir_ir.FlatSymbolRefAttr.get(handle.symbol,
                                                  context=self.context),
                "slot_group":
                    mlir_ir.FlatSymbolRefAttr.get(children.slot_group,
                                                  context=self.context),
            },
        )
        return self._new_bundle(operation.result, children.hierarchy,
                                children.slot_group)

    def encoding_pack(self, children, *, encoding=None):
        if not isinstance(children,
                          PatchBundleValue) or children.owner is not self:
            raise TypeError(
                "encoding_pack requires a live protocol patch bundle")
        hierarchy = children.hierarchy
        if encoding is None:
            raise TypeError(
                "encoding_pack requires encoding= for the parent view")
        if encoding is None or encoding.hierarchy is not hierarchy:
            raise ValueError(
                "encoding_pack requires the hierarchy's structural encoding")
        children._consume("fabric.encoding_pack")
        encoding_handle = self.transaction.materialize(encoding)
        patch_type = self._patch_type(encoding)
        operation = self._emit(
            "fabric.encoding_pack",
            operands=[children.mlir_value],
            results=[patch_type],
            attributes={
                "encoding":
                    mlir_ir.FlatSymbolRefAttr.get(encoding_handle.symbol,
                                                  context=self.context)
            },
        )
        return self._new_patch(operation.result, encoding)

    def transition_epoch(self, patch_value, *, to, evidence, logical_map=None):
        if not isinstance(patch_value,
                          PatchValue) or patch_value.owner is not self:
            raise TypeError("transition_epoch expects one live protocol patch")
        if not isinstance(to, EncodingEpoch):
            raise TypeError("transition_epoch to= must be an EncodingEpoch")
        if to.encoding is not patch_value.encoding:
            raise ValueError("epoch transition cannot change encoding identity")
        if not isinstance(evidence, str) or not evidence:
            raise TypeError(
                "epoch transition evidence must be a nonempty string")
        source_phase = patch_value.epoch.phase
        edge = f"{source_phase}->{to.phase}"
        epoch_schema = patch_value.encoding.epoch_schema
        if (source_phase, to.phase) not in epoch_schema.transitions:
            raise ValueError(
                f"epoch transition {edge!r} is not declared by the encoding")
        authoritative_map = epoch_schema.logical_maps.get(edge)
        if patch_value.encoding.profile.dynamic_phases:
            if not isinstance(authoritative_map, Mapping):
                raise ValueError(
                    "a dynamic profile transition requires its exact declared "
                    "logical map")
            authoritative_map = dict(authoritative_map)
            if (logical_map is not None and
                    dict(logical_map) != authoritative_map):
                raise ValueError(
                    "epoch transition logical_map contradicts the dynamic "
                    "profile")
            logical_map = authoritative_map
        elif authoritative_map is not None:
            authoritative_map = dict(authoritative_map)
            if (logical_map is not None and
                    dict(logical_map) != authoritative_map):
                raise ValueError(
                    "epoch transition logical_map contradicts the encoding "
                    "schema")
            logical_map = authoritative_map
        else:
            logical_map = dict(logical_map or {})
        destination = self.transaction.materialize(to)
        attrs = {
            "to_epoch":
                mlir_ir.FlatSymbolRefAttr.get(destination.symbol,
                                              context=self.context),
            "evidence":
                mlir_ir.StringAttr.get(evidence, context=self.context),
        }
        if logical_map:
            attrs["logical_map"] = mlir_ir.DictAttr.get(
                {
                    str(key):
                        mlir_ir.StringAttr.get(str(value), context=self.context)
                    for key, value in logical_map.items()
                },
                context=self.context,
            )
        patch_value._consume("fabric.epoch_transition")
        operation = self._emit(
            "fabric.epoch_transition",
            operands=[patch_value.mlir_value],
            results=[self._patch_type(patch_value.encoding, epoch=to)],
            attributes=attrs,
        )
        return self._new_patch(operation.result, patch_value.encoding, epoch=to)

    def _callee_boundaries(self, definition):
        hints = definition.type_hints
        inputs = tuple(
            self._input_boundary(hints.get(name, parameter.annotation))
            for name, parameter in definition.signature.parameters.items())
        result = hints.get("return", definition.signature.return_annotation)
        return inputs, self._flatten_result_boundaries(result)

    def call(self, definition, args, kwargs):
        analysis = kwargs.pop("analysis", None)
        profile = None
        if kwargs:
            raise TypeError(
                f"unsupported protocol call keyword arguments: {sorted(kwargs)}"
            )
        if not isinstance(definition, (GadgetDefinition, ProtocolDefinition)):
            raise TypeError("protocols may call @cudaq.logical.gadget or "
                            "@cudaq.logical.protocol values")
        if analysis is not None:
            if not isinstance(analysis, GadgetProfile):
                raise TypeError(
                    "analysis= requires a cudaq.logical.GadgetProfile")
            if not isinstance(
                    definition,
                    GadgetDefinition) or analysis.gadget is not definition:
                raise ValueError(
                    "analysis profile does not describe this gadget")
        input_boundaries, result_boundaries = self._callee_boundaries(
            definition)
        if len(args) != len(input_boundaries):
            raise TypeError(
                "protocol call argument count does not match callee")
        operands = []
        for value, (kind, payload) in zip(args, input_boundaries):
            if kind == "patch":
                if not isinstance(value, PatchValue) or value.owner is not self:
                    raise TypeError("protocol call requires a live patch")
                if value.encoding is not payload:
                    raise TypeError(
                        "protocol call patch encoding does not match callee")
                value._consume(f"fabric.call @{definition.name}")
            elif kind == "resource":
                if not isinstance(value,
                                  ResourceValue) or value.owner is not self:
                    raise TypeError("protocol call requires a live resource")
                if value.kind != payload:
                    raise TypeError(
                        "protocol call resource kind does not match callee")
                value._consume(f"fabric.call @{definition.name}")
            elif kind == "syndrome":
                if not isinstance(value,
                                  SyndromeValue) or value.owner is not self:
                    raise TypeError("protocol call requires a syndrome record")
                if value.encoding is not payload:
                    raise TypeError(
                        "protocol call syndrome schema does not match callee")
            elif kind == "bool":
                if not isinstance(value,
                                  LogicalBool) or value.owner is not self:
                    raise TypeError("protocol call requires a traced bool")
            operands.append(value.mlir_value)
        callee = self.transaction.materialize(definition)
        inherited_predicates = (
            self.transaction._protocol_result_provenance.get(callee.symbol, ())
            if isinstance(definition, ProtocolDefinition) else ())
        has_boolean_results = any(
            kind == "bool" for kind, _ in result_boundaries)
        outcome_parities, outcome_roles = (
            self._compiled_outcome_parities(definition)
            if has_boolean_results else ((), ()))
        if analysis is not None:
            resolved_roles = [list(roles) for roles in outcome_roles]
            role = "success"
            if not any(role in roles for roles in outcome_roles):
                for parity in analysis._effective_role_parities(role):
                    matches = tuple(
                        index
                        for index, candidate in enumerate(outcome_parities)
                        if role not in outcome_roles[index] and
                        candidate.records == parity.records and
                        candidate.input_syndromes == parity.input_syndromes)
                    if len(matches) > 1:
                        raise UnsupportedCombination(
                            "selected GadgetProfile success semantics "
                            "ambiguously match multiple OutcomeMap rows")
                    if not matches:
                        # A detached success family may contain rows that are
                        # not application results. Only an exact unique match
                        # can classify a returned Boolean.
                        continue
                    resolved_roles[matches[0]].append(role)
            outcome_roles = tuple(tuple(roles) for roles in resolved_roles)
        result_types = tuple(
            self._boundary_type(kind, value)
            for kind, value in result_boundaries)
        attrs = {
            "callee":
                mlir_ir.FlatSymbolRefAttr.get(callee.symbol,
                                              context=self.context)
        }
        if analysis is not None:
            profile = self.transaction.materialize(analysis)
            attrs["profile"] = mlir_ir.FlatSymbolRefAttr.get(
                profile.symbol, context=self.context)
        else:
            inherited_profiles = {
                predicate.profile
                for predicate in inherited_predicates
                if isinstance(predicate, _PredicateProvenance) and
                predicate.profile is not None
            }
            if len(inherited_profiles) == 1:
                attrs["profile"] = mlir_ir.FlatSymbolRefAttr.get(
                    inherited_profiles.pop(), context=self.context)
        operation = self._emit(
            "fabric.call",
            operands=operands,
            results=result_types,
            attributes=attrs,
        )
        values = tuple(
            self._wrap_boundary(result, boundary)
            for result, boundary in zip(operation.results, result_boundaries))
        probability_metadata = None
        probability_source = None
        definition_metadata = (definition._authoritative_spec_metadata()
                               if isinstance(definition, GadgetDefinition) else
                               definition.metadata)
        if analysis is not None and "success_probability" in analysis.metadata:
            probability_metadata = analysis.metadata
            probability_source = profile.symbol
        elif "success_probability" in definition_metadata:
            probability_metadata = definition_metadata
            probability_source = callee.symbol
        bool_index = 0
        for value, boundary in zip(values, result_boundaries):
            if boundary[0] == "bool":
                inherited = (inherited_predicates[bool_index] if bool_index
                             < len(inherited_predicates) else None)
                if isinstance(inherited, _PredicateProvenance):
                    value.producer = _PredicateProvenance(
                        definition=definition,
                        attempt=callee.symbol,
                        analysis=inherited.analysis,
                        profile=inherited.profile,
                        invocation=id(operation),
                        parity=inherited.parity,
                        outcome_rows=inherited.outcome_rows,
                        outcome_roles=inherited.outcome_roles,
                        outcome_index=inherited.outcome_index,
                        all_false_rows=inherited.all_false_rows,
                        all_false_indices=inherited.all_false_indices,
                        patch_results=tuple(result for result in values
                                            if isinstance(result, PatchValue)),
                        probability_source=(probability_source
                                            if probability_metadata is not None
                                            else inherited.probability_source),
                        success_probability=(
                            probability_metadata["success_probability"]
                            if probability_metadata is not None else
                            inherited.success_probability),
                        probability_evidence=(probability_metadata.get(
                            "success_probability_evidence")
                                              if probability_metadata
                                              is not None else
                                              inherited.probability_evidence),
                    )
                    bool_index += 1
                    continue
                parity = (outcome_parities[bool_index]
                          if bool_index < len(outcome_parities) else None)
                value.producer = _PredicateProvenance(
                    definition=definition,
                    attempt=callee.symbol,
                    analysis=analysis,
                    profile=None if profile is None else profile.symbol,
                    invocation=id(operation),
                    parity=parity,
                    outcome_rows=outcome_parities,
                    outcome_roles=outcome_roles,
                    outcome_index=(bool_index if parity is not None else None),
                    all_false_rows=None,
                    all_false_indices=None,
                    patch_results=tuple(result for result in values
                                        if isinstance(result, PatchValue)),
                    probability_source=probability_source,
                    success_probability=(
                        None if probability_metadata is None else
                        probability_metadata["success_probability"]),
                    probability_evidence=(None if probability_metadata is None
                                          else probability_metadata.get(
                                              "success_probability_evidence")),
                )
                bool_index += 1
        return values[0] if len(values) == 1 else values

    def _compiled_outcome_parities(self, definition):
        """Read the gadget's canonical typed OutcomeMap without IR parsing."""

        if not isinstance(definition, GadgetDefinition):
            return (), ()
        outcome_map = definition.outcome_map
        if outcome_map is None:
            return (), ()
        input_endpoints = tuple(definition.interface.inputs)
        rows = tuple(
            ProfileParity(
                records=tuple(
                    RecordRef(definition, name)
                    for selected, name in zip(outcome_map.matrix.rows[row],
                                              outcome_map.records)
                    if selected),
                input_syndromes=tuple(
                    InputSyndromeRef(input_endpoints[term.port], term.index)
                    for term in outcome_map.input_syndromes[row]),
                constant=bool(outcome_map.constants[row]),
            )
            for row in range(outcome_map.outcome_count))
        return rows, outcome_map.roles

    def retry(
        self,
        carries,
        *,
        until,
        max_attempts,
        exhaustion,
        commit_point,
    ):
        carries = tuple(carries)
        if not isinstance(until, LogicalBool) or until.owner is not self:
            raise TypeError(
                "retry until= requires a Boolean from this protocol")
        if not isinstance(until.producer, _PredicateProvenance):
            raise UnsupportedCombination(
                "retry requires a success predicate derived from one gadget "
                "attempt")
        provenance = until.producer
        attempt = provenance.attempt
        profile = provenance.profile
        self._validate_selected_predicate(provenance, operation="retry")
        policy = RetryPolicy(max_attempts, exhaustion, commit_point)
        encodings = []
        epochs = []
        operands = []
        result_types = []
        for value in carries:
            if not isinstance(value, PatchValue) or value.owner is not self:
                raise TypeError("retry carries must be live protocol patches")
            value._validate_consume("fabric.retry")
            encodings.append(value.encoding)
            epochs.append(value.epoch)
            operands.append(value.mlir_value)
            result_types.append(value.type)
        expected_carries = provenance.patch_results
        if carries != expected_carries:
            raise UnsupportedCombination(
                "retry must carry every linear patch result from the selected "
                "attempt exactly once and in result order; attempt-scoped sibling cleanup is not "
                "a closed replay boundary")
        attrs = {
            "max_attempts":
                mlir_ir.IntegerAttr.get(
                    mlir_ir.IntegerType.get_signless(64, context=self.context),
                    policy.max_attempts,
                ),
            "exhaustion":
                mlir_ir.StringAttr.get(policy.exhaustion.value,
                                       context=self.context),
        }
        if policy.commit_point is not None:
            serialized_commit = policy.commit_point.to_ir()
            if (policy.commit_point.kind is CommitPointKind.BEFORE_OUTPUT and
                    policy.commit_point.output is not None):
                endpoint = policy.commit_point.output.name
                outputs = tuple(provenance.definition.interface.outputs)
                if (policy.commit_point.output.gadget
                        is not provenance.definition or
                        endpoint not in {output.name for output in outputs}):
                    raise ValueError(
                        "retry commit point names no output endpoint in the "
                        "selected attempt specification")
            attrs["commit_point"] = mlir_ir.StringAttr.get(serialized_commit,
                                                           context=self.context)
        attrs["attempt"] = mlir_ir.FlatSymbolRefAttr.get(attempt,
                                                         context=self.context)
        if profile is not None:
            attrs["profile"] = mlir_ir.FlatSymbolRefAttr.get(
                profile, context=self.context)
        if provenance.success_probability is not None:
            try:
                probability = float(provenance.success_probability)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "retry certificate success_probability must be numeric"
                ) from exc
            if (not math.isfinite(probability) or not 0.0 < probability <= 1.0):
                raise ValueError(
                    "retry certificate success_probability must be finite and lie in (0, 1]"
                )
            if (provenance.probability_source is None or
                    not isinstance(provenance.probability_evidence, str) or
                    not provenance.probability_evidence):
                raise ValueError(
                    "retry certificate requires a source and nonempty evidence")
            with self.location:
                attrs["success_probability"] = mlir_ir.FloatAttr.get(
                    mlir_ir.F64Type.get(context=self.context), probability)
            attrs["success_probability_source"] = (
                mlir_ir.FlatSymbolRefAttr.get(provenance.probability_source,
                                              context=self.context))
            attrs["success_probability_evidence"] = mlir_ir.StringAttr.get(
                provenance.probability_evidence, context=self.context)
        for value in carries:
            value._consume("fabric.retry")
        operation = self._emit(
            "fabric.retry",
            operands=[*operands, until.mlir_value],
            results=result_types,
            attributes=attrs,
        )
        return tuple(
            self._new_patch(result, encoding, epoch=epoch) for result, encoding,
            epoch in zip(operation.results, encodings, epochs))

    @staticmethod
    def _flatten_values(value):
        if isinstance(value, (tuple, list)):
            result = []
            for item in value:
                result.extend(ProtocolBuilder._flatten_values(item))
            return tuple(result)
        return (value,)

    def _branch_clone(self, value):
        if isinstance(value, PatchValue):
            return self._new_patch(value.mlir_value,
                                   value.encoding,
                                   epoch=value.epoch)
        if isinstance(value, ResourceValue):
            return self._new_resource(value.mlir_value, value.kind)
        if isinstance(value, FabricEventValue):
            return self._new_event(value.mlir_value, value.payload_kind)
        if isinstance(value, SyndromeValue):
            return SyndromeValue(
                value.mlir_value,
                owner=self,
                encoding=value.encoding,
                record=value.record,
                location=self.location,
            )
        if isinstance(value, LogicalBool):
            return LogicalBool(value.mlir_value, owner=self)
        raise TypeError(
            "cudaq.logical.cond protocol carries must be patch, record, resource, or bool"
        )

    def _wrap_like(self, value, prototype):
        if isinstance(prototype, PatchValue):
            return self._new_patch(value,
                                   prototype.encoding,
                                   epoch=prototype.epoch)
        if isinstance(prototype, ResourceValue):
            return self._new_resource(value, prototype.kind)
        if isinstance(prototype, FabricEventValue):
            return self._new_event(value, prototype.payload_kind)
        if isinstance(prototype, SyndromeValue):
            return SyndromeValue(
                value,
                owner=self,
                encoding=prototype.encoding,
                record=prototype.record,
                location=self.location,
            )
        if isinstance(prototype, LogicalBool):
            return LogicalBool(value, owner=self)
        if isinstance(prototype, EventStatusValue):
            return EventStatusValue(value, owner=self)
        if isinstance(prototype, IndexValue):
            return IndexValue(value, owner=self)
        raise TypeError(f"unsupported structured carry {prototype!r}")

    @staticmethod
    def _consume_branch_value(value, operation):
        if isinstance(value, (PatchValue, ResourceValue, FabricEventValue)):
            value._consume(operation)

    def cond(self, condition, then, else_, carries):
        if not isinstance(condition,
                          LogicalBool) or condition.owner is not self:
            raise TypeError(
                "cudaq.logical.cond condition must be a bool from this protocol"
            )
        carries = tuple(carries)
        result_types = tuple(value.mlir_value.type for value in carries)
        for value in carries:
            if getattr(value, "owner", None) is not self:
                raise TypeError(
                    "cudaq.logical.cond carries must belong to this protocol")
            self._consume_branch_value(value, "cflow.if")
        operation = self._emit(
            "cflow.if",
            operands=[condition.mlir_value],
            results=result_types,
            regions=2,
        )
        parent_ip = self.insertion_point
        try:
            for region, callback in zip(operation.regions, (then, else_)):
                block = region.blocks.append()
                self.insertion_point = mlir_ir.InsertionPoint(block)
                branch_values = tuple(
                    self._branch_clone(value) for value in carries)
                returned = self._flatten_values(callback(*branch_values))
                if len(returned) != len(result_types):
                    raise TypeError(
                        "cudaq.logical.cond branches must return one value per carry"
                    )
                operands = []
                for value, expected in zip(returned, result_types):
                    if (getattr(value, "owner", None) is not self or
                            value.mlir_value.type != expected):
                        raise TypeError(
                            "cudaq.logical.cond branch results must match carries"
                        )
                    self._consume_branch_value(value, "cflow.yield")
                    operands.append(value.mlir_value)
                self._emit("cflow.yield", operands=operands)
        finally:
            self.insertion_point = parent_ip

        results = []
        for result, prototype in zip(operation.results, carries):
            if isinstance(prototype, PatchValue):
                results.append(
                    self._new_patch(result,
                                    prototype.encoding,
                                    epoch=prototype.epoch))
            elif isinstance(prototype, ResourceValue):
                results.append(self._new_resource(result, prototype.kind))
            elif isinstance(prototype, SyndromeValue):
                results.append(
                    SyndromeValue(
                        result,
                        owner=self,
                        encoding=prototype.encoding,
                        record=prototype.record,
                        location=self.location,
                    ))
            else:
                results.append(LogicalBool(result, owner=self))
        return tuple(results)

    def explicit_if(self, condition, carries):
        return _ExplicitProtocolIf(self, condition, carries)

    def repeat(self, count, carries, body):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise TypeError("protocol repeat count must be a nonnegative int")
        if not callable(body):
            raise TypeError("protocol repeat body must be callable")
        encodings = []
        epochs = []
        operands = []
        for value in carries:
            if not isinstance(value, PatchValue) or value.owner is not self:
                raise TypeError("protocol repeat carries must be live patches")
            value._consume("cflow.repeat")
            encodings.append(value.encoding)
            epochs.append(value.epoch)
            operands.append(value.mlir_value)
        attrs = {
            "count":
                mlir_ir.IntegerAttr.get(
                    mlir_ir.IntegerType.get_signless(64, context=self.context),
                    count)
        }
        with self.location:
            operation = mlir_ir.Operation.create(
                "cflow.repeat",
                operands=operands,
                results=[operand.type for operand in operands],
                attributes=attrs,
                regions=1,
                loc=self.location,
            )
            self.insertion_point.insert(operation)
            block = operation.regions[0].blocks.append(
                *(operand.type for operand in operands))
        outer_ip = self.insertion_point
        self.insertion_point = mlir_ir.InsertionPoint(block)
        nested = tuple(
            self._new_patch(value, encoding, epoch=epoch) for value, encoding,
            epoch in zip(block.arguments, encodings, epochs))
        returned = body(*nested)
        values = returned if isinstance(returned, tuple) else (returned,)
        if len(values) != len(nested):
            raise TypeError(
                "protocol repeat body must return every carried patch")
        yielded = []
        for value, encoding, epoch in zip(values, encodings, epochs):
            if not isinstance(value, PatchValue) or value.owner is not self:
                raise TypeError("protocol repeat returned a foreign patch")
            if value.encoding is not encoding:
                raise TypeError("protocol repeat changed a carry encoding")
            if value.epoch is not epoch:
                raise TypeError("protocol repeat changed a carry epoch")
            value._consume("cflow.yield")
            yielded.append(value.mlir_value)
        self._emit("cflow.yield", operands=yielded)
        self.insertion_point = outer_ip
        results = tuple(
            self._new_patch(value, encoding, epoch=epoch) for value, encoding,
            epoch in zip(operation.results, encodings, epochs))
        return results[0] if len(results) == 1 else results

    def while_loop(self, condition, carries, body, *, max_iterations=None):
        if not callable(condition) or not callable(body):
            raise TypeError(
                "cudaq.logical.while_ condition and body must be callable")
        if max_iterations is not None and (
                not isinstance(max_iterations, int) or
                isinstance(max_iterations, bool) or max_iterations <= 0):
            raise TypeError(
                "cudaq.logical.while_ max_iterations must be a positive int")
        carries = tuple(carries)
        result_types = tuple(value.mlir_value.type for value in carries)
        for value in carries:
            if getattr(value, "owner", None) is not self:
                raise TypeError(
                    "protocol cudaq.logical.while_ carries must belong to this protocol"
                )
            self._consume_branch_value(value, "cflow.while")
        attrs = {}
        if max_iterations is not None:
            attrs["max_iterations"] = mlir_ir.IntegerAttr.get(
                mlir_ir.IntegerType.get_signless(64, context=self.context),
                max_iterations,
            )
        operation = self._emit(
            "cflow.while",
            operands=[value.mlir_value for value in carries],
            results=result_types,
            attributes=attrs,
            regions=2,
        )
        with self.location:
            before = operation.regions[0].blocks.append(*result_types)
            after = operation.regions[1].blocks.append(*result_types)
        parent_ip = self.insertion_point
        try:
            self.insertion_point = mlir_ir.InsertionPoint(before)
            before_values = tuple(
                self._wrap_like(value, prototype)
                for value, prototype in zip(before.arguments, carries))
            condition_result = self._flatten_values(condition(*before_values))
            if len(condition_result) == 1:
                predicate = condition_result[0]
                forwarded = before_values
            elif len(condition_result) == len(carries) + 1:
                predicate, *forwarded = condition_result
                forwarded = tuple(forwarded)
            else:
                raise TypeError(
                    "cudaq.logical.while_ condition must return a bool, or "
                    "(bool, *forwarded_carries)")
            if not isinstance(predicate,
                              LogicalBool) or predicate.owner is not self:
                raise TypeError(
                    "cudaq.logical.while_ condition must return a traced bool")
            forwarded_operands = []
            for value, expected in zip(forwarded, result_types):
                if (getattr(value, "owner", None) is not self or
                        value.mlir_value.type != expected):
                    raise TypeError(
                        "protocol cudaq.logical.while_ forwarded values must match carries"
                    )
                self._consume_branch_value(value, "cflow.while_condition")
                forwarded_operands.append(value.mlir_value)
            self._emit(
                "cflow.while_condition",
                operands=[predicate.mlir_value, *forwarded_operands],
            )

            self.insertion_point = mlir_ir.InsertionPoint(after)
            after_values = tuple(
                self._wrap_like(value, prototype)
                for value, prototype in zip(after.arguments, carries))
            returned = self._flatten_values(body(*after_values))
            if len(returned) != len(carries):
                raise TypeError(
                    "protocol cudaq.logical.while_ body must return every carried value"
                )
            yielded = []
            for value, expected in zip(returned, result_types):
                if (getattr(value, "owner", None) is not self or
                        value.mlir_value.type != expected):
                    raise TypeError(
                        "protocol cudaq.logical.while_ body results must match carries"
                    )
                self._consume_branch_value(value, "cflow.yield")
                yielded.append(value.mlir_value)
            self._emit("cflow.yield", operands=yielded)
        finally:
            self.insertion_point = parent_ip
        results = tuple(
            self._wrap_like(value, prototype)
            for value, prototype in zip(operation.results, carries))
        return results[0] if len(results) == 1 else results


class _ExplicitProtocolIfRegion:

    def __init__(self, branch: "_ExplicitProtocolIf", name: str) -> None:
        self.branch = branch
        self.name = name

    def __enter__(self):
        self.branch._enter_region(self.name)
        return self.branch

    def __exit__(self, exc_type, exc, traceback):
        self.branch._exit_region(self.name, exc_type is None)
        return False


class _ExplicitProtocolIf:
    """Explicit-region spelling of ``ProtocolBuilder.cond``."""

    def __init__(self, builder: ProtocolBuilder, condition, carries) -> None:
        if not isinstance(condition,
                          LogicalBool) or condition.owner is not builder:
            raise TypeError(
                "cudaq.logical.if_ condition must be a bool from this protocol")
        self.builder = builder
        self.carries = tuple(carries)
        self.result_types = tuple(
            value.mlir_value.type for value in self.carries)
        for value in self.carries:
            if getattr(value, "owner", None) is not builder:
                raise TypeError(
                    "cudaq.logical.if_ carries must belong to this protocol")
            if isinstance(value, (PatchValue, ResourceValue)):
                value._consume("cflow.if")
            elif not isinstance(value, (SyndromeValue, LogicalBool)):
                raise TypeError(
                    "protocol cudaq.logical.if_ carries must be patch, record, resource, or bool"
                )
        self.operation = builder._emit(
            "cflow.if",
            operands=[condition.mlir_value],
            results=self.result_types,
            regions=2,
        )
        self.blocks = {
            "then": self.operation.regions[0].blocks.append(),
            "else": self.operation.regions[1].blocks.append(),
        }
        self.parent_ip = builder.insertion_point
        self.active = None
        self.entered = set()
        self.yielded = set()
        self.results = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.builder.insertion_point = self.parent_ip
        if exc_type is not None:
            return False
        missing = {"then", "else"} - self.yielded
        if missing:
            raise RuntimeError(
                "cudaq.logical.if_ requires explicit yields from both protocol branches; "
                "missing " + ", ".join(sorted(missing)))
        values = []
        for result, prototype in zip(self.operation.results, self.carries):
            if isinstance(prototype, PatchValue):
                values.append(
                    self.builder._new_patch(result,
                                            prototype.encoding,
                                            epoch=prototype.epoch))
            elif isinstance(prototype, ResourceValue):
                values.append(self.builder._new_resource(
                    result, prototype.kind))
            elif isinstance(prototype, SyndromeValue):
                values.append(
                    SyndromeValue(
                        result,
                        owner=self.builder,
                        encoding=prototype.encoding,
                        record=prototype.record,
                        location=self.builder.location,
                    ))
            else:
                values.append(LogicalBool(result, owner=self.builder))
        self.results = tuple(values)
        return False

    def then(self):
        return _ExplicitProtocolIfRegion(self, "then")

    def else_(self):
        return _ExplicitProtocolIfRegion(self, "else")

    def _enter_region(self, name):
        if self.active is not None:
            raise RuntimeError(
                "cudaq.logical.if_ protocol branch regions cannot overlap")
        if name in self.entered:
            raise RuntimeError(
                f"cudaq.logical.if_ {name} protocol region was already authored"
            )
        self.entered.add(name)
        self.active = name
        for value in self.carries:
            if isinstance(value, (PatchValue, ResourceValue)):
                value._live = True
        self.builder.insertion_point = mlir_ir.InsertionPoint(self.blocks[name])

    def _exit_region(self, name, successful):
        try:
            if successful and name not in self.yielded:
                raise RuntimeError(
                    f"cudaq.logical.if_ {name} protocol region must call branch.yield_()"
                )
            if successful:
                leaked = [
                    str(value.type)
                    for value in self.carries
                    if isinstance(value, (PatchValue,
                                          ResourceValue)) and value.is_live
                ]
                if leaked:
                    raise RuntimeError(
                        f"cudaq.logical.if_ {name} protocol region did not consume every "
                        f"linear carry: {leaked!r}")
        finally:
            for value in self.carries:
                if isinstance(value, (PatchValue, ResourceValue)):
                    value._live = False
            self.active = None
            self.builder.insertion_point = self.parent_ip

    def yield_(self, *values):
        if self.active is None:
            raise RuntimeError(
                "branch.yield_() must appear inside then()/else_()")
        if self.active in self.yielded:
            raise RuntimeError(
                f"cudaq.logical.if_ {self.active} protocol region already yielded"
            )
        flattened = ProtocolBuilder._flatten_values(values)
        if len(flattened) != len(self.result_types):
            raise TypeError(
                "cudaq.logical.if_ branches must yield one value per carry")
        operands = []
        for value, expected in zip(flattened, self.result_types):
            if getattr(value, "owner", None) is not self.builder:
                raise TypeError(
                    "cudaq.logical.if_ protocol yields must be traced values")
            if value.mlir_value.type != expected:
                raise TypeError(
                    "cudaq.logical.if_ protocol result types must match carries"
                )
            if isinstance(value, (PatchValue, ResourceValue)):
                value._consume("cflow.yield")
            operands.append(value.mlir_value)
        self.builder._emit("cflow.yield", operands=operands)
        self.yielded.add(self.active)
