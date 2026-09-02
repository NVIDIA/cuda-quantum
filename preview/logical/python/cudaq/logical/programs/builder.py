# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import math
from types import NoneType
from typing import Any, get_args, get_origin, get_type_hints

from cudaq.mlir import ir as mlir_ir

from ..algebra.angle import Angle
from ..programs.context import (
    pop_trace,
    push_trace,
)
from ..types.values import (
    Float64Value,
    EventState,
    EventStatusValue,
    IndexValue,
    LogicalBool,
    LogicalEventValue,
    LogicalFrameValue,
    LogicalRegister,
    LogicalResourceValue,
    _LogicalLinearValue,
    _SSAProxy,
    logical_qubit,
)
from ..algebra.pauli import PauliProduct
from ..programs.definition import ProgramDefinition
from ..types.semantic import (
    float64,
    index,
    logical_event,
    logical_frame,
    logical_resource,
)
from ..std import LogicalActionRef


def _flatten_annotation(annotation: Any) -> tuple[Any, ...]:
    if annotation in (None, NoneType):
        return ()
    origin = get_origin(annotation)
    if origin in (tuple, list):
        flattened: list[Any] = []
        for item in get_args(annotation):
            if item is Ellipsis:
                raise TypeError(
                    "variadic tuple annotations are not canonical CUDA-Q Logical signatures"
                )
            flattened.extend(_flatten_annotation(item))
        return tuple(flattened)
    return (annotation,)


def _flatten_values(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, LogicalRegister):
        return value.as_tuple()
    if isinstance(value, (tuple, list)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten_values(item))
        return tuple(flattened)
    return (value,)


def _static_attribute(context, value):
    if isinstance(value, mlir_ir.Attribute):
        return value
    if isinstance(value, bool):
        return mlir_ir.BoolAttr.get(value, context=context)
    if isinstance(value, int):
        return mlir_ir.IntegerAttr.get(
            mlir_ir.IntegerType.get_signless(64, context=context), value)
    if isinstance(value, float):
        # FloatAttr.get requires an active location even for a detached
        # attribute.
        with mlir_ir.Location.unknown(context):
            return mlir_ir.FloatAttr.get(mlir_ir.F64Type.get(context=context),
                                         value)
    if isinstance(value, Angle):
        # Specialization provenance only: the substituted rotation op already
        # carries the exact angle numerator and denominator; record the numeric
        # value here.
        with mlir_ir.Location.unknown(context):
            return mlir_ir.FloatAttr.get(mlir_ir.F64Type.get(context=context),
                                         float(value))
    if isinstance(value, str):
        return mlir_ir.StringAttr.get(value, context=context)
    if isinstance(value, (tuple, list)):
        return mlir_ir.ArrayAttr.get(
            [_static_attribute(context, item) for item in value],
            context=context,
        )
    if isinstance(value, dict):
        return mlir_ir.DictAttr.get(
            {
                str(name): _static_attribute(context, item)
                for name, item in value.items()
            },
            context=context,
        )
    raise TypeError(
        f"static CUDA-Q Logical parameter {value!r} has no canonical MLIR attribute"
    )


def _builtin_action(context, name: str):
    return mlir_ir.Attribute.parse(f"#qlx.action<{name}>", context=context)


def _builtin_instrument(context, name: str):
    return mlir_ir.Attribute.parse(f"#qlx.instrument<{name}>", context=context)


def _pauli_basis(context, basis: str):
    return mlir_ir.Attribute.parse(f"#qlx.pauli<{basis.upper()}>",
                                   context=context)


class UnplacedBuilder:
    """Direct P0 builder and decorator-trace backend."""

    def __init__(self, transaction, definition: ProgramDefinition) -> None:
        self.transaction = transaction
        self.definition = definition
        self.context = transaction.context
        self.location = transaction.location
        self._all_qubits: list[logical_qubit] = []
        self._all_linear: list[_LogicalLinearValue] = []
        self._generation = 0
        self._alloc_group = 0
        self._arguments = None
        self._finished = False
        self.value_groups: dict[str, int] = {}
        self._type_hints = definition.type_hints
        self._input_annotations = tuple(
            self._type_hints.get(name, parameter.annotation)
            for name, parameter in definition.signature.parameters.items())
        return_annotation = self._type_hints.get(
            "return", definition.signature.return_annotation)
        self._result_annotations = _flatten_annotation(return_annotation)
        self.input_types = tuple(
            self._mlir_type(item) for item in self._input_annotations)
        self.result_types = tuple(
            self._mlir_type(item) for item in self._result_annotations)
        self.function_type = mlir_ir.FunctionType.get(self.input_types,
                                                      self.result_types,
                                                      context=self.context)
        requested = (definition.name if definition.kind == "program" else
                     f"{definition.name}_objective_body")
        self.symbol = transaction.unique_symbol(requested)
        self._create_program()

    def _mlir_type(self, annotation: Any):
        if annotation is logical_qubit:
            return mlir_ir.Type.parse("!qlx.logical_qubit",
                                      context=self.context)
        if annotation is bool:
            return mlir_ir.IntegerType.get_signless(1, context=self.context)
        if annotation in (index, int):
            return mlir_ir.IndexType.get(context=self.context)
        if annotation in (float64, float):
            return mlir_ir.F64Type.get(context=self.context)
        origin = get_origin(annotation)
        arguments = get_args(annotation)
        if origin is logical_resource:
            kind = self._semantic_name(arguments[0])
            return mlir_ir.Type.parse(f'!qlx.logical_resource<"{kind}">',
                                      context=self.context)
        if origin is logical_event:
            payload = self._mlir_type(arguments[0])
            return mlir_ir.Type.parse(
                f'!qlx.logical_event<{payload}, "linear">',
                context=self.context)
        if origin is logical_frame:
            domain = self._semantic_name(arguments[0])
            return mlir_ir.Type.parse(f'!qlx.logical_frame<"{domain}">',
                                      context=self.context)
        raise TypeError(
            f"unsupported CUDA-Q Logical P0 annotation: {annotation!r}")

    @staticmethod
    def _semantic_name(value) -> str:
        name = getattr(value, "name", value)
        name = str(name)
        if not name or any(
                not (char.isalnum() or char in "_.$-") for char in name):
            raise ValueError(
                f"logical type name is not an MLIR keyword: {name!r}")
        return name

    @property
    def logical_type(self):
        return mlir_ir.Type.parse("!qlx.logical_qubit", context=self.context)

    @property
    def i1_type(self):
        return mlir_ir.IntegerType.get_signless(1, context=self.context)

    def _create_program(self) -> None:
        self.transaction.add_profile("p0")
        with self.context:
            function_type_attr = mlir_ir.TypeAttr.get(self.function_type)
        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(self.symbol, context=self.context),
            "function_type":
                function_type_attr,
            "qlx.profile":
                mlir_ir.StringAttr.get("p0", context=self.context),
            "qlx.stage":
                mlir_ir.StringAttr.get("p0", context=self.context),
        }
        if self.definition.specialization:
            attrs["specialization"] = _static_attribute(
                self.context, dict(self.definition.specialization))
        if self.definition.estimate_only:
            attrs["estimate_only"] = mlir_ir.UnitAttr.get(context=self.context)
        with self.location:
            self.operation = mlir_ir.Operation.create(
                ("qlx.program" if self.definition.kind == "program" else
                 "qlx.objective_body"),
                results=[],
                operands=[],
                attributes=attrs,
                regions=1,
                loc=self.location,
            )
            self.transaction.module.body.append(self.operation)
            self.block = self.operation.regions[0].blocks.append(
                *self.input_types)
        self.insertion_point = mlir_ir.InsertionPoint(self.block)

    def _emit(self, name: str, *, operands=(), results=(), attributes=None):
        with self.location:
            operation = mlir_ir.Operation.create(
                name,
                operands=list(operands),
                results=list(results),
                attributes=dict(attributes or {}),
                loc=self.location,
            )
            self.insertion_point.insert(operation)
        return operation

    def _wrap_argument(self, value, annotation, index_: int):
        if annotation is logical_qubit:
            return self._new_qubit(value, semantic_ref=("arg", index_, 0))
        if annotation is bool:
            return LogicalBool(value, owner=self)
        if annotation in (index, int):
            return IndexValue(value, owner=self)
        if annotation in (float64, float):
            return Float64Value(value, owner=self)
        origin = get_origin(annotation)
        arguments = get_args(annotation)
        if origin is logical_resource:
            return self._new_resource(value, arguments[0])
        if origin is logical_event:
            payload = arguments[0]
            payload_kind = (get_args(payload)[0] if get_origin(payload)
                            is logical_resource else None)
            return self._new_event(value,
                                   self._mlir_type(payload),
                                   payload_kind=payload_kind)
        if origin is logical_frame:
            return self._new_frame(value, self._semantic_name(arguments[0]))
        raise TypeError(f"unsupported argument annotation: {annotation!r}")

    def _new_qubit(self, value, *, semantic_ref=None) -> logical_qubit:
        self._generation += 1
        ref = semantic_ref or ("value", self._generation)
        result = logical_qubit(value,
                               owner=self,
                               semantic_ref=ref,
                               location=self.location)
        self._all_qubits.append(result)
        return result

    def _canonical_qubit_for_ssa(self, mlir_value) -> logical_qubit | None:
        """Return this builder's canonical proxy for one logical SSA value.

        ``_all_qubits`` is the builder's existing minting and liveness record.
        Looking through it by MLIR value equality authenticates the SSA against
        that record without trusting the proxy's public ``owner`` field or
        creating a second identity registry.
        """

        for candidate in reversed(self._all_qubits):
            try:
                if candidate.mlir_value == mlir_value:
                    return candidate
            except (TypeError, ValueError):
                # Values from an incompatible MLIR context/domain do not belong
                # to this builder.  Some bindings report that as inequality;
                # others reject the comparison.
                continue
        return None

    def _new_resource(self, value, kind):
        result = LogicalResourceValue(value,
                                      owner=self,
                                      kind=kind,
                                      location=self.location)
        self._all_linear.append(result)
        return result

    def _new_event(self, value, payload_type, *, payload_kind=None):
        result = LogicalEventValue(
            value,
            owner=self,
            payload_type=payload_type,
            payload_kind=payload_kind,
            location=self.location,
        )
        self._all_linear.append(result)
        return result

    def _new_frame(self, value, domain: str):
        result = LogicalFrameValue(value,
                                   owner=self,
                                   domain=domain,
                                   location=self.location)
        self._all_linear.append(result)
        return result

    def _constant_index(self, value: int):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise TypeError(
                "logical iteration counts must be nonnegative integers")
        attr = mlir_ir.IntegerAttr.get(
            mlir_ir.IndexType.get(context=self.context), value)
        return self._emit("arith.constant",
                          results=[mlir_ir.IndexType.get(context=self.context)],
                          attributes={
                              "value": attr
                          }).result

    def _constant_f64(self, value: float):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                "rotation angles must be numeric or traced float64 values")
        f64 = mlir_ir.F64Type.get(context=self.context)
        with self.location:
            attr = mlir_ir.FloatAttr.get(f64, float(value))
        return self._emit("arith.constant",
                          results=[f64],
                          attributes={
                              "value": attr
                          }).result

    def arguments(self):
        if self._arguments is None:
            self._arguments = tuple(
                self._wrap_argument(value, annotation, i)
                for i, (value, annotation) in enumerate(
                    zip(self.block.arguments, self._input_annotations)))
        return self._arguments

    def trace(self):
        args = self.arguments()
        token = push_trace(self)
        try:
            returned = self.definition.provider(*args)
            self._emit_selection(returned)
        finally:
            pop_trace(token)
        self.finish(*_flatten_values(returned))

    def objective_family(self) -> str:
        """Infer the internal action/instrument family of an ideal objective."""

        if self.definition.kind != "objective":
            raise TypeError(
                "objective-family inference requires @cudaq.logical.objective")

        operations = []

        def walk(operation):
            for region in operation.regions:
                for block in region.blocks:
                    for child in block.operations:
                        operations.append(child.operation.name)
                        walk(child.operation)

        walk(self.operation)
        forbidden = {
            "qlx.resource_request",
            "qlx.event_await",
            "qlx.event_poll",
            "qlx.event_test",
            "qlx.event_is",
            "qlx.event_select_ready",
            "qlx.event_try_take",
            "qlx.event_cancel",
            "qlx.consume_resource",
            "qlx.fence",
        }
        used_forbidden = sorted(forbidden.intersection(operations))
        if used_forbidden:
            raise TypeError(
                "@cudaq.logical.objective describes closed ideal behavior and cannot "
                f"contain runtime/resource operations: {used_forbidden!r}")

        instrument_ops = {
            "qlx.prepare",
            "qlx.measure",
            "qlx.instrument",
            "qlx.discard",
            "qlx.selection",
        }
        inferred = ("instrument" if instrument_ops.intersection(operations) or
                    any(str(result) == "i1" for result in self.result_types) or
                    sum(
                        str(value) == "!qlx.logical_qubit"
                        for value in self.input_types) != sum(
                            str(value) == "!qlx.logical_qubit"
                            for value in self.result_types) else "action")
        claimed = self.definition.objective_kind
        if claimed == "action" and inferred != "action":
            raise TypeError(
                "@cudaq.logical.objective(kind='action') contains outcome-bearing or "
                "nonunitary instrument semantics")
        family = inferred if claimed == "auto" else claimed
        self.operation.attributes["objective_kind"] = mlir_ir.StringAttr.get(
            family, context=self.context)
        return family

    def _emit_selection(self, returned):
        selection = self.definition.selection
        if selection is None:
            return
        predicate = selection.predicate(returned)
        if isinstance(predicate, LogicalBool) and predicate.owner is self:
            value = predicate.mlir_value
        elif isinstance(predicate, bool):
            value = self._emit(
                "arith.constant",
                results=[self.i1_type],
                attributes={
                    "value":
                        mlir_ir.IntegerAttr.get(self.i1_type, int(predicate))
                },
            ).result
        else:
            raise TypeError(
                "logical selection predicates must return a P0 Boolean outcome")
        self._emit(
            "qlx.selection",
            operands=[value],
            attributes={
                "mode":
                    mlir_ir.StringAttr.get(selection.mode,
                                           context=self.context),
                "accept_when":
                    mlir_ir.BoolAttr.get(selection.accept_when,
                                         context=self.context),
            },
        )

    def finish(self, *returned):
        if self._finished:
            raise RuntimeError("UnplacedBuilder root is already finished")
        self._finished = True
        results = _flatten_values(returned)
        if len(results) != len(self.result_types):
            raise TypeError(
                f"{self.definition.name} returned {len(results)} values, but its "
                f"annotation declares {len(self.result_types)}")
        mlir_results = []
        for value, expected in zip(results, self.result_types):
            if isinstance(value, logical_qubit):
                value._consume("qlx.return")
            elif isinstance(value, _LogicalLinearValue):
                value._consume("qlx.return")
            if isinstance(value, _SSAProxy):
                mlir_value = value.mlir_value
            elif isinstance(value, bool):
                attr = mlir_ir.IntegerAttr.get(self.i1_type, int(value))
                mlir_value = self._emit("arith.constant",
                                        results=[self.i1_type],
                                        attributes={
                                            "value": attr
                                        }).result
            else:
                raise TypeError(
                    f"unsupported CUDA-Q Logical return value: {value!r}")
            if mlir_value.type != expected:
                raise TypeError(
                    f"return type {mlir_value.type} does not match annotated {expected}"
                )
            mlir_results.append(mlir_value)
        self._emit("qlx.return", operands=mlir_results)
        leaked = [q.semantic_ref for q in self._all_qubits if q.is_live]
        leaked_linear = [
            str(value.type) for value in self._all_linear if value.is_live
        ]
        if leaked or leaked_linear:
            raise RuntimeError(
                "logical values remain live at the program boundary; return, "
                f"measure, await/consume, or discard them: {leaked!r} "
                f"{leaked_linear!r}")

    def call(self, definition, args, kwargs):
        if definition.kind == "objective":
            if kwargs:
                raise TypeError(
                    "composite objective calls do not accept keyword operands")
            return self.apply_definition(definition, args, {})

        bound = definition.signature.bind(*args, **kwargs)
        values = tuple(
            bound.arguments[name] for name in definition.signature.parameters)
        handle = self.transaction.materialize(definition)
        input_types, result_types, _ = self.transaction.signature_of(definition)
        mlir_inputs = self._prepare_call_inputs(values, input_types, "qlx.call")
        operation = self._emit(
            "qlx.call",
            operands=mlir_inputs,
            results=result_types,
            attributes={
                "callee":
                    mlir_ir.FlatSymbolRefAttr.get(handle.symbol,
                                                  context=self.context)
            },
        )
        return self._shape_call_results(definition, operation.results)

    def allocate(self, count: int, *, state: str, name: str | None):
        allocation = self._alloc_group
        self._alloc_group += 1
        group = name or f"alloc{allocation}"
        if group in self.value_groups:
            raise ValueError(f"duplicate logical value group name {group!r}")
        self.value_groups[group] = count
        values = [
            self.prepare(
                state,
                semantic_ref=("allocation", allocation, group, i, 0),
            ) for i in range(count)
        ]
        return LogicalRegister(values, name=name)

    def prepare(self, state: str, *, semantic_ref=None):
        if semantic_ref is None:
            allocation = self._alloc_group
            self._alloc_group += 1
            group = f"alloc{allocation}"
            self.value_groups[group] = 1
            semantic_ref = ("allocation", allocation, group, 0, 0)
        attributes = {
            "state": mlir_ir.StringAttr.get(str(state), context=self.context)
        }
        if semantic_ref[0] == "allocation":
            attributes["allocation"] = mlir_ir.IntegerAttr.get(
                mlir_ir.IntegerType.get_signless(64, context=self.context),
                semantic_ref[1],
            )
            attributes["value_index"] = mlir_ir.IntegerAttr.get(
                mlir_ir.IntegerType.get_signless(64, context=self.context),
                semantic_ref[3],
            )
        op = self._emit(
            "qlx.prepare",
            results=[self.logical_type],
            attributes=attributes,
        )
        return self._new_qubit(op.result, semantic_ref=semantic_ref)

    def _consume_qubits(self, values, operation: str):
        values = tuple(values)
        for value in values:
            if not isinstance(value, logical_qubit):
                raise TypeError(f"{operation} expects logical_qubit operands")
            if value.owner is not self:
                raise ValueError(
                    f"{operation} received a value from another builder")
            value._consume(operation)
        return values

    def apply_standard(self, name: str, values, **options):
        if any(value is not None for value in options.values()):
            raise TypeError(
                f"P0 qlx.{name} does not accept machine-specific options")
        values = self._consume_qubits(values, f"qlx.{name}")
        inputs = tuple(value.mlir_value for value in values)
        results = (self.logical_type,) * len(values)
        op = self._emit(
            "qlx.apply",
            operands=inputs,
            results=results,
            attributes={"action": _builtin_action(self.context, name)},
        )
        return [
            self._new_qubit(result,
                            semantic_ref=value.semantic_ref[:-1] +
                            (self._generation + i + 1,))
            for i, (result, value) in enumerate(zip(op.results, values))
        ]

    def apply_definition(self, action, values, parameters):
        if not isinstance(action,
                          ProgramDefinition) or action.kind != "objective":
            raise TypeError(
                "qlx.apply expects an @cudaq.logical.objective definition")
        handle = self.transaction.materialize(action)
        family = handle.kind
        if family not in {"action", "instrument"}:
            raise TypeError(
                "materialized objective has no action/instrument family")
        input_types, result_types, _ = self.transaction.signature_of(action)
        mlir_inputs = self._prepare_call_inputs(values, input_types,
                                                f"qlx.{family}")
        attrs = {
            family:
                mlir_ir.FlatSymbolRefAttr.get(handle.symbol,
                                              context=self.context)
        }
        if parameters:
            attrs["parameters"] = _static_attribute(self.context,
                                                    dict(parameters))
        operation = self._emit(
            "qlx.apply" if family == "action" else "qlx.instrument",
            operands=mlir_inputs,
            results=result_types,
            attributes=attrs,
        )
        return self._shape_call_results(action, operation.results)

    def _prepare_call_inputs(self, values, expected_types, operation: str):
        if len(values) != len(expected_types):
            raise TypeError(
                f"{operation} expects {len(expected_types)} operands, got {len(values)}"
            )
        mlir_inputs = []
        for value, expected in zip(values, expected_types):
            if isinstance(value, logical_qubit):
                (value,) = self._consume_qubits((value,), operation)
                mlir_value = value.mlir_value
            elif isinstance(value, _SSAProxy):
                if value.owner is not self:
                    raise ValueError(
                        f"{operation} received a value from another builder")
                if isinstance(value, _LogicalLinearValue):
                    value._consume(operation)
                mlir_value = value.mlir_value
            else:
                raise TypeError(
                    f"{operation} received unsupported operand {value!r}")
            if mlir_value.type != expected:
                raise TypeError(
                    f"{operation} operand type {mlir_value.type} does not match {expected}"
                )
            mlir_inputs.append(mlir_value)
        return tuple(mlir_inputs)

    def _shape_call_results(self, definition, results):
        shaped = [
            self._wrap_dynamic_result(value, annotation)
            for value, annotation in zip(
                results, self._result_annotations_for(definition))
        ]
        if len(shaped) == 0:
            return None
        if len(shaped) == 1:
            return shaped[0]
        return tuple(shaped)

    def _result_annotations_for(self, definition):
        hints = definition.type_hints
        annotation = hints.get("return", definition.signature.return_annotation)
        return _flatten_annotation(annotation)

    def _wrap_dynamic_result(self, value, annotation):
        if annotation is logical_qubit:
            return self._new_qubit(value)
        if annotation is bool:
            return LogicalBool(value, owner=self)
        if annotation in (index, int):
            return IndexValue(value, owner=self)
        if annotation in (float64, float):
            return Float64Value(value, owner=self)
        origin = get_origin(annotation)
        arguments = get_args(annotation)
        if origin is logical_resource:
            return self._new_resource(value, arguments[0])
        if origin is logical_event:
            payload = arguments[0]
            payload_kind = (get_args(payload)[0] if get_origin(payload)
                            is logical_resource else None)
            return self._new_event(value,
                                   self._mlir_type(payload),
                                   payload_kind=payload_kind)
        if origin is logical_frame:
            return self._new_frame(value, self._semantic_name(arguments[0]))
        raise TypeError(f"unsupported dynamic result annotation {annotation!r}")

    def _pauli_parameters(self, product: PauliProduct, *, operands=None):
        operands = product.operands if operands is None else tuple(operands)
        x_mask, z_mask = product.symplectic_for(operands)
        i64 = mlir_ir.IntegerType.get_signless(64, context=self.context)
        return operands, mlir_ir.DictAttr.get(
            {
                "x_mask": mlir_ir.IntegerAttr.get(i64, x_mask),
                "z_mask": mlir_ir.IntegerAttr.get(i64, z_mask),
                "sign": mlir_ir.IntegerAttr.get(i64, product.sign),
            },
            context=self.context,
        )

    def rotate(self, product: PauliProduct, *, angle, precision=None):
        operands, parameters = self._pauli_parameters(product)
        i64 = mlir_ir.IntegerType.get_signless(64, context=self.context)
        # Canonical "full operator sign": a rotation's Pauli-product sign and
        # its angle sign are the same degree of freedom (R_{-P}(t) = R_P(-t)).
        # Fold both into the sign field and keep the angle as a non-negative
        # magnitude, so R_{-Z}(pi/4) and R_Z(-pi/4) produce identical IR. The
        # sign field is the single canonical carrier (uniform with
        # ``mpp``), and the effective rotation is sign * angle -- consumers
        # MUST read both.
        net_sign = product.sign
        if isinstance(angle, Angle):
            if angle.pi_fraction[0] < 0:
                net_sign = -net_sign
                angle = -angle
        elif isinstance(angle, Float64Value):
            pass  # runtime sign stays in the traced value; field holds the static sign
        else:
            if angle < 0:
                net_sign = -net_sign
                angle = -angle
        extra = {"sign": mlir_ir.IntegerAttr.get(i64, net_sign)}
        if precision is not None:
            if (not isinstance(precision,
                               (int, float)) or isinstance(precision, bool) or
                    not math.isfinite(float(precision)) or precision <= 0):
                raise TypeError(
                    "rotation precision must be a finite positive number")
            with self.location:
                extra["precision"] = mlir_ir.FloatAttr.get(
                    mlir_ir.F64Type.get(context=self.context),
                    float(precision),
                )
        if isinstance(angle, Angle):
            # Record the exact rational-pi coefficient alongside the f64 value
            # so downstream classification is authoritative, not tolerance-
            # inferred. Product rotations are projective, so first reduce the
            # exact coefficient modulo two. This keeps arbitrarily large source
            # numerators away from the fixed-width IR boundary and makes the
            # f64 operand and exact metadata describe the same representative.
            numer, denom = angle.pi_fraction
            angle = Angle(numer % (2 * denom), denom)
            numer, denom = angle.pi_fraction
            i64_max = (1 << 63) - 1
            if numer > i64_max or denom > i64_max:
                raise ValueError(
                    "canonical exact rotation coefficient does not fit the "
                    "CUDA-Q Logical i64 rational-angle metadata")
            extra["angle_pi_numer"] = mlir_ir.IntegerAttr.get(i64, numer)
            extra["angle_pi_denom"] = mlir_ir.IntegerAttr.get(i64, denom)
            angle = float(angle)
        if extra:
            merged = {item.name: item.attr for item in parameters}
            merged.update(extra)
            parameters = mlir_ir.DictAttr.get(merged, context=self.context)
        values = self._consume_qubits(operands, "cudaq.logical.rotate")
        angle_value = angle.mlir_value if isinstance(
            angle, Float64Value) else self._constant_f64(angle)
        input_values = tuple(
            value.mlir_value for value in values) + (angle_value,)
        results = (self.logical_type,) * len(values)
        op = self._emit(
            "qlx.apply",
            operands=input_values,
            results=results,
            attributes={
                "action": _builtin_action(self.context, "pauli_rotation"),
                "parameters": parameters,
            },
        )
        return [self._new_qubit(result) for result in op.results]

    def mpp(self, product: PauliProduct):
        operands, parameters = self._pauli_parameters(product)
        values = self._consume_qubits(operands, "qlx.mpp")
        inputs = tuple(value.mlir_value for value in values)
        results = (self.logical_type,) * len(values) + (self.i1_type,)
        op = self._emit(
            "qlx.instrument",
            operands=inputs,
            results=results,
            attributes={
                "instrument": _builtin_instrument(self.context, "mpp"),
                "parameters": parameters,
            },
        )
        qubits = [self._new_qubit(result) for result in op.results[:-1]]
        return (*qubits, LogicalBool(op.results[-1], owner=self))

    def readout(self, product: PauliProduct):
        """Destructive product readout: consumes every covered operand.

        Identity-covered operands (``cudaq.logical.I``) contribute no mask bits but are
        still owned and destroyed by the readout. The native builtin instrument
        vocabulary has an inline form only for ``mpp``, so the destructive
        product instrument is declared as the standard
        ``qlx_standard_readout`` objective symbol.
        """
        if len(product.operands) > 63:
            raise ValueError(
                "cudaq.logical.readout currently supports at most 63 non-identity operands"
            )
        # Mask positions describe the measured support, not the ownership
        # closure. Put non-identity factors first so arbitrarily many explicit
        # I-covered operands cannot push a live factor outside the finite i64
        # mask. Identity operands remain inputs and are consumed below.
        covered_operands = (*product.operands, *product.identity_operands)
        operands, parameters = self._pauli_parameters(product,
                                                      operands=covered_operands)
        values = self._consume_qubits(operands, "cudaq.logical.readout")
        symbol = self.transaction.objective(
            family="instrument",
            name="readout",
            inputs=(self.logical_type,) * len(values),
            results=(self.i1_type,),
        )
        op = self._emit(
            "qlx.instrument",
            operands=tuple(value.mlir_value for value in values),
            results=(self.i1_type,),
            attributes={
                "instrument":
                    mlir_ir.FlatSymbolRefAttr.get(symbol, context=self.context),
                "parameters":
                    parameters,
            },
        )
        return LogicalBool(op.results[0], owner=self)

    def measure(self, basis: str, value):
        (value,) = self._consume_qubits((value,), f"qlx.measure_{basis}")
        op = self._emit(
            "qlx.measure",
            operands=[value.mlir_value],
            results=[self.i1_type],
            attributes={"basis": _pauli_basis(self.context, basis)},
        )
        return LogicalBool(op.result, owner=self)

    def xor(self, lhs, rhs):
        if any(not isinstance(value, LogicalBool) or value.owner is not self
               for value in (lhs, rhs)):
            raise TypeError(
                "qlx.xor expects two Boolean values from this trace")
        operation = self._emit(
            "qlx.xor",
            operands=[lhs.mlir_value, rhs.mlir_value],
            results=[self.i1_type],
        )
        return LogicalBool(operation.result, owner=self)

    def idle(self, values, *, rounds):
        values = self._consume_qubits(values, "qlx.idle")
        rounds_value = rounds.mlir_value if isinstance(
            rounds, IndexValue) else self._constant_index(rounds)
        op = self._emit(
            "qlx.idle",
            operands=[*(value.mlir_value for value in values), rounds_value],
            results=[self.logical_type] * len(values),
        )
        return [
            self._new_qubit(result,
                            semantic_ref=("idle", self._generation + i + 1))
            for i, result in enumerate(op.results)
        ]

    def discard(self, values, *, reason: str | None):
        values = self._consume_qubits(values, "qlx.discard")
        attrs = {}
        if reason is not None:
            attrs["reason"] = mlir_ir.StringAttr.get(reason,
                                                     context=self.context)
        self._emit(
            "qlx.discard",
            operands=[value.mlir_value for value in values],
            attributes=attrs,
        )

    def request(self, kind):
        kind_name = self._semantic_name(kind)
        payload_type = mlir_ir.Type.parse(
            f'!qlx.logical_resource<"{kind_name}">', context=self.context)
        event_type = mlir_ir.Type.parse(
            f'!qlx.logical_event<{payload_type}, "linear">',
            context=self.context)
        operation = self._emit(
            "qlx.resource_request",
            results=[event_type],
            attributes={
                "kind": mlir_ir.StringAttr.get(kind_name, context=self.context)
            },
        )
        return self._new_event(operation.result,
                               payload_type,
                               payload_kind=kind)

    def event_test(self, event):
        if not isinstance(event, LogicalEventValue) or event.owner is not self:
            raise TypeError(
                "event_test expects a logical event from this trace")
        if not event.is_live:
            event._consume("qlx.event_test")
        operation = self._emit(
            "qlx.event_test",
            operands=[event.mlir_value],
            results=[self.i1_type],
        )
        return LogicalBool(operation.result, owner=self)

    def event_poll(self, event):
        if not isinstance(event, LogicalEventValue) or event.owner is not self:
            raise TypeError(
                "event_poll expects a logical event from this trace")
        if not event.is_live:
            event._consume("qlx.event_poll")
        status_type = mlir_ir.IntegerType.get_signless(8, context=self.context)
        operation = self._emit(
            "qlx.event_poll",
            operands=[event.mlir_value],
            results=[status_type],
        )
        return EventStatusValue(operation.result, owner=self)

    def event_is(self, status, state):
        if not isinstance(status, EventStatusValue) or status.owner is not self:
            raise TypeError("event_is expects an event status from this trace")
        try:
            state_name = EventState(state).value
        except ValueError as error:
            raise ValueError(f"unknown event state: {state!r}") from error
        operation = self._emit(
            "qlx.event_is",
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
        if any(not isinstance(event, LogicalEventValue) or
               event.owner is not self for event in events):
            raise TypeError(
                "event_select_ready expects logical events from this trace")
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
            "qlx.event_select_ready",
            operands=[event.mlir_value for event in events],
            results=[mlir_ir.IndexType.get(context=self.context)],
            attributes={
                "policy": mlir_ir.StringAttr.get(policy, context=self.context)
            },
        )
        return IndexValue(operation.result, owner=self)

    def event_try_take(self, event, carries, *, ready, pending, failed):
        if not isinstance(event, LogicalEventValue) or event.owner is not self:
            raise TypeError(
                "event_try_take expects a logical event from this trace")
        event._consume("qlx.event_try_take")
        carries = tuple(carries)
        result_types = tuple(value.mlir_value.type for value in carries)
        for value in carries:
            if not isinstance(value, _SSAProxy):
                raise TypeError("event_try_take carries must be traced values")
            if isinstance(value, logical_qubit):
                value._consume("qlx.event_try_take")
            elif isinstance(value, _LogicalLinearValue):
                value._consume("qlx.event_try_take")
        status_type = mlir_ir.IntegerType.get_signless(8, context=self.context)
        with self.location:
            operation = mlir_ir.Operation.create(
                "qlx.event_try_take",
                operands=[
                    event.mlir_value, *(value.mlir_value for value in carries)
                ],
                results=result_types,
                regions=3,
                loc=self.location,
            )
            self.insertion_point.insert(operation)
            ready_block = operation.regions[0].blocks.append(
                event.payload_type, *result_types)
            pending_block = operation.regions[1].blocks.append(
                event.mlir_value.type, *result_types)
            failed_block = operation.regions[2].blocks.append(
                status_type, *result_types)
        parent_ip = self.insertion_point
        self._trace_event_take_branch(
            ready_block,
            "ready",
            ready,
            result_types,
            payload_type=event.payload_type,
            payload_kind=event.payload_kind,
        )
        self._trace_event_take_branch(
            pending_block,
            "pending",
            pending,
            result_types,
            payload_type=event.payload_type,
            payload_kind=event.payload_kind,
        )
        self._trace_event_take_branch(failed_block, "failed", failed,
                                      result_types)
        self.insertion_point = parent_ip
        return tuple(
            self._wrap_result_by_type(result) for result in operation.results)

    def event_cancel(self, event, *, reason=None):
        if not isinstance(event, LogicalEventValue) or event.owner is not self:
            raise TypeError(
                "event_cancel expects a logical event from this trace")
        if reason is not None and (not isinstance(reason, str) or not reason):
            raise TypeError("event_cancel reason must be a nonempty string")
        event._consume("qlx.event_cancel")
        attrs = {}
        if reason is not None:
            attrs["reason"] = mlir_ir.StringAttr.get(reason,
                                                     context=self.context)
        operation = self._emit(
            "qlx.event_cancel",
            operands=[event.mlir_value],
            results=[mlir_ir.IntegerType.get_signless(8, context=self.context)],
            attributes=attrs,
        )
        return EventStatusValue(operation.result, owner=self)

    def _trace_event_take_branch(
        self,
        block,
        branch_name,
        callback,
        result_types,
        *,
        payload_type=None,
        payload_kind=None,
    ):
        self.insertion_point = mlir_ir.InsertionPoint(block)
        if branch_name == "ready" and payload_kind is not None:
            alternative = self._new_resource(block.arguments[0], payload_kind)
        elif branch_name == "pending" and payload_kind is not None:
            alternative = self._new_event(
                block.arguments[0],
                payload_type,
                payload_kind=payload_kind,
            )
        else:
            alternative = self._wrap_result_by_type(block.arguments[0])
        branch_carries = tuple(
            self._wrap_result_by_type(value) for value in block.arguments[1:])
        returned = _flatten_values(callback(alternative, *branch_carries))
        if len(returned) != len(result_types):
            raise TypeError(
                "event_try_take branches must return one value per carry")
        operands = []
        for value, expected in zip(returned, result_types):
            if not isinstance(value,
                              _SSAProxy) or value.mlir_value.type != expected:
                raise TypeError(
                    "event_try_take branch result types must match carries")
            if isinstance(value, logical_qubit):
                value._consume("qlx.yield")
            elif isinstance(value, _LogicalLinearValue):
                value._consume("qlx.yield")
            operands.append(value.mlir_value)
        self._emit("qlx.yield", operands=operands)

    def event_await(self, event):
        if not isinstance(event, LogicalEventValue) or event.owner is not self:
            raise TypeError(
                "event_await expects a logical event from this trace")
        event._consume("qlx.event_await")
        operation = self._emit(
            "qlx.event_await",
            operands=[event.mlir_value],
            results=[event.payload_type],
        )
        payload = str(event.payload_type)
        prefix = "!qlx.logical_resource<"
        if not payload.startswith(prefix):
            raise NotImplementedError(
                "this resource frontend currently awaits logical-resource payloads"
            )
        kind = event.payload_kind
        if kind is None:
            kind = payload[len(prefix):-1].strip('"')
        return self._new_resource(operation.result, kind)

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
            "qlx.fence",
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

    def consume_resource(self, resource, values, *, action=None):
        if not isinstance(resource,
                          LogicalResourceValue) or resource.owner is not self:
            raise TypeError(
                "consume expects a logical resource from this trace")
        resource._consume("qlx.consume_resource")
        values = self._consume_qubits(values, "qlx.consume_resource")
        inferred = getattr(resource.kind, "consume_action", None)
        if action is None:
            action = inferred
            if action is None:
                raise TypeError(
                    f"resource kind {self._semantic_name(resource.kind)!r} "
                    "does not define a consumption action; pass action=")
        elif inferred is not None and action != inferred:
            raise ValueError(
                f"resource kind {self._semantic_name(resource.kind)!r} "
                f"consumes as {inferred.name!r}, not "
                f"{getattr(action, 'name', action)!r}")
        if isinstance(action, LogicalActionRef):
            if action.arity != len(values):
                raise TypeError(
                    f"logical action {action.name!r} expects {action.arity} operands"
                )
            objective = _builtin_action(self.context, action.name)
        elif isinstance(action,
                        ProgramDefinition) and action.kind == "objective":
            handle = self.transaction.materialize(action)
            if handle.kind != "action":
                raise TypeError("consume action= requires an action objective")
            objective = mlir_ir.FlatSymbolRefAttr.get(handle.symbol,
                                                      context=self.context)
        else:
            raise TypeError("consume action= expects a logical action")
        operation = self._emit(
            "qlx.consume_resource",
            operands=[
                resource.mlir_value, *(value.mlir_value for value in values)
            ],
            results=[self.logical_type] * len(values),
            attributes={"action": objective},
        )
        return tuple(
            self._new_qubit(result,
                            semantic_ref=("resource", resource.kind, index))
            for index, result in enumerate(operation.results))

    def frame(self, domain):
        domain_name = self._semantic_name(domain)
        frame_type = mlir_ir.Type.parse(f'!qlx.logical_frame<"{domain_name}">',
                                        context=self.context)
        operation = self._emit(
            "qlx.frame_init",
            results=[frame_type],
            attributes={
                "domain":
                    mlir_ir.StringAttr.get(domain_name, context=self.context)
            },
        )
        return self._new_frame(operation.result, domain_name)

    def frame_update(self, frame, source, *, update="outcome"):
        if not isinstance(frame, LogicalFrameValue) or frame.owner is not self:
            raise TypeError("frame_update expects a frame from this trace")
        if not isinstance(source, LogicalBool) or source.owner is not self:
            raise TypeError(
                "frame_update source must be a bool from this trace")
        frame._consume("qlx.frame_update")
        operation = self._emit(
            "qlx.frame_update",
            operands=[frame.mlir_value, source.mlir_value],
            results=[frame.mlir_value.type],
            attributes={
                "update":
                    mlir_ir.StringAttr.get(str(update), context=self.context)
            },
        )
        return self._new_frame(operation.result, frame.domain)

    def frame_transform(self, frame, transform):
        if not isinstance(frame, LogicalFrameValue) or frame.owner is not self:
            raise TypeError("frame transform expects a frame from this trace")
        frame._consume("qlx.frame_transform")
        operation = self._emit(
            "qlx.frame_transform",
            operands=[frame.mlir_value],
            results=[frame.mlir_value.type],
            attributes={
                "transform":
                    mlir_ir.StringAttr.get(str(transform), context=self.context)
            },
        )
        return self._new_frame(operation.result, frame.domain)

    def cond(self, condition, then, else_, carries):
        if not isinstance(condition,
                          LogicalBool) or condition.owner is not self:
            raise TypeError(
                "cudaq.logical.cond condition must be a bool result from this trace"
            )
        result_types = tuple(value.mlir_value.type for value in carries)
        for value in carries:
            if isinstance(value, logical_qubit):
                value._consume("cudaq.logical.cond")
            elif isinstance(value, _LogicalLinearValue):
                value._consume("cudaq.logical.cond")
            elif not isinstance(value, _SSAProxy):
                raise TypeError(
                    "cudaq.logical.cond carries must be traced values")
        with self.location:
            operation = mlir_ir.Operation.create(
                "qlx.if",
                operands=[condition.mlir_value],
                results=result_types,
                regions=2,
                loc=self.location,
            )
            self.insertion_point.insert(operation)
            then_block = operation.regions[0].blocks.append()
            else_block = operation.regions[1].blocks.append()

        parent_ip = self.insertion_point
        self._trace_cond_branch(then_block, "then", then, carries, result_types)
        self._trace_cond_branch(else_block, "else", else_, carries,
                                result_types)
        self.insertion_point = parent_ip
        return tuple(
            self._wrap_result_by_type(result) for result in operation.results)

    def explicit_if(self, condition, carries):
        return _ExplicitIf(self, condition, carries)

    def _clone_branch_value(self, value, branch: str):
        if isinstance(value, logical_qubit):
            return self._new_qubit(
                value.mlir_value,
                semantic_ref=("branch", branch, *value.semantic_ref),
            )
        if isinstance(value, LogicalBool):
            return LogicalBool(value.mlir_value, owner=self)
        if isinstance(value, IndexValue):
            return IndexValue(value.mlir_value, owner=self)
        if isinstance(value, Float64Value):
            return Float64Value(value.mlir_value, owner=self)
        if isinstance(value, LogicalResourceValue):
            return self._new_resource(value.mlir_value, value.kind)
        if isinstance(value, LogicalEventValue):
            return self._new_event(
                value.mlir_value,
                value.payload_type,
                payload_kind=value.payload_kind,
            )
        if isinstance(value, LogicalFrameValue):
            return self._new_frame(value.mlir_value, value.domain)
        raise TypeError(f"unsupported branch carry {value!r}")

    def _trace_cond_branch(self, block, branch_name, callback, carries,
                           result_types):
        self.insertion_point = mlir_ir.InsertionPoint(block)
        branch_values = tuple(
            self._clone_branch_value(value, branch_name) for value in carries)
        returned = _flatten_values(callback(*branch_values))
        if len(returned) != len(result_types):
            raise TypeError(
                "cudaq.logical.cond branches must return one value per carry")
        operands = []
        for value, expected in zip(returned, result_types):
            if not isinstance(value,
                              _SSAProxy) or value.mlir_value.type != expected:
                raise TypeError(
                    "cudaq.logical.cond branch result types must match carries")
            if isinstance(value, logical_qubit):
                value._consume("qlx.yield")
            elif isinstance(value, _LogicalLinearValue):
                value._consume("qlx.yield")
            operands.append(value.mlir_value)
        self._emit("qlx.yield", operands=operands)

    def _wrap_result_by_type(self, value):
        if value.type == self.logical_type:
            return self._new_qubit(value)
        if value.type == self.i1_type:
            return LogicalBool(value, owner=self)
        if value.type == mlir_ir.IndexType.get(context=self.context):
            return IndexValue(value, owner=self)
        if value.type == mlir_ir.IntegerType.get_signless(8,
                                                          context=self.context):
            return EventStatusValue(value, owner=self)
        if value.type == mlir_ir.F64Type.get(context=self.context):
            return Float64Value(value, owner=self)
        text = str(value.type)
        resource_prefix = "!qlx.logical_resource<"
        frame_prefix = "!qlx.logical_frame<"
        event_prefix = "!qlx.logical_event<"
        if text.startswith(resource_prefix):
            return self._new_resource(value,
                                      text[len(resource_prefix):-1].strip('"'))
        if text.startswith(frame_prefix):
            return self._new_frame(value, text[len(frame_prefix):-1].strip('"'))
        if text.startswith(event_prefix):
            payload = text[len(event_prefix):-1].rsplit(",", 1)[0].strip()
            return self._new_event(
                value, mlir_ir.Type.parse(payload, context=self.context))
        raise TypeError(
            f"unsupported structured-control result type {value.type}")

    def repeat(self, count, carries, body):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise NotImplementedError(
                "this slice supports static nonnegative qlx.repeat counts; "
                "symbolic index counts are pending the final repeat op")
        result_types = tuple(value.mlir_value.type for value in carries)
        init_values = []
        for value in carries:
            if not isinstance(value, _SSAProxy):
                raise TypeError("qlx.repeat carries must be traced values")
            if isinstance(value, logical_qubit):
                value._consume("qlx.repeat")
            elif isinstance(value, _LogicalLinearValue):
                value._consume("qlx.repeat")
            init_values.append(value.mlir_value)
        with self.location:
            operation = mlir_ir.Operation.create(
                "qlx.repeat",
                operands=init_values,
                results=result_types,
                attributes={
                    "count":
                        mlir_ir.IntegerAttr.get(
                            mlir_ir.IntegerType.get_signless(
                                64, context=self.context),
                            count,
                        )
                },
                regions=1,
                loc=self.location,
            )
            self.insertion_point.insert(operation)
            block = operation.regions[0].blocks.append(*result_types)
        parent_ip = self.insertion_point
        self.insertion_point = mlir_ir.InsertionPoint(block)
        args = tuple(
            self._wrap_result_by_type(value) for value in block.arguments)
        returned = _flatten_values(body(0, *args))
        if len(returned) != len(result_types):
            raise TypeError("qlx.repeat body must return one value per carry")
        yielded = []
        for value, expected in zip(returned, result_types):
            if not isinstance(value,
                              _SSAProxy) or value.mlir_value.type != expected:
                raise TypeError(
                    "qlx.repeat body result types must match carries")
            if isinstance(value, logical_qubit):
                value._consume("qlx.yield")
            elif isinstance(value, _LogicalLinearValue):
                value._consume("qlx.yield")
            yielded.append(value.mlir_value)
        self._emit("qlx.yield", operands=yielded)
        self.insertion_point = parent_ip
        return tuple(
            self._wrap_result_by_type(value) for value in operation.results)

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
        init_values = []
        for value in carries:
            if not isinstance(value, _SSAProxy) or value.owner is not self:
                raise TypeError(
                    "cudaq.logical.while_ carries must be traced values from this trace"
                )
            if isinstance(value, (logical_qubit, _LogicalLinearValue)):
                value._consume("qlx.while")
            init_values.append(value.mlir_value)
        attrs = {}
        if max_iterations is not None:
            attrs["max_iterations"] = mlir_ir.IntegerAttr.get(
                mlir_ir.IntegerType.get_signless(64, context=self.context),
                max_iterations,
            )
        with self.location:
            operation = mlir_ir.Operation.create(
                "qlx.while",
                operands=init_values,
                results=result_types,
                attributes=attrs,
                regions=2,
                loc=self.location,
            )
            self.insertion_point.insert(operation)
            before = operation.regions[0].blocks.append(*result_types)
            after = operation.regions[1].blocks.append(*result_types)

        parent_ip = self.insertion_point
        try:
            self.insertion_point = mlir_ir.InsertionPoint(before)
            before_values = tuple(
                self._wrap_result_by_type(value) for value in before.arguments)
            condition_result = _flatten_values(condition(*before_values))
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
                if (not isinstance(value, _SSAProxy) or
                        value.owner is not self or
                        value.mlir_value.type != expected):
                    raise TypeError(
                        "cudaq.logical.while_ forwarded values must match the carries"
                    )
                if isinstance(value, (logical_qubit, _LogicalLinearValue)):
                    value._consume("qlx.while_condition")
                forwarded_operands.append(value.mlir_value)
            self._emit(
                "qlx.while_condition",
                operands=[predicate.mlir_value, *forwarded_operands],
            )

            self.insertion_point = mlir_ir.InsertionPoint(after)
            after_values = tuple(
                self._wrap_result_by_type(value) for value in after.arguments)
            returned = _flatten_values(body(*after_values))
            if len(returned) != len(result_types):
                raise TypeError(
                    "cudaq.logical.while_ body must return one value per carry")
            yielded = []
            for value, expected in zip(returned, result_types):
                if (not isinstance(value, _SSAProxy) or
                        value.owner is not self or
                        value.mlir_value.type != expected):
                    raise TypeError(
                        "cudaq.logical.while_ body result types must match carries"
                    )
                if isinstance(value, (logical_qubit, _LogicalLinearValue)):
                    value._consume("qlx.yield")
                yielded.append(value.mlir_value)
            self._emit("qlx.yield", operands=yielded)
        finally:
            self.insertion_point = parent_ip
        return tuple(
            self._wrap_result_by_type(value) for value in operation.results)


class _ExplicitIfRegion:

    def __init__(self, branch: "_ExplicitIf", name: str) -> None:
        self.branch = branch
        self.name = name

    def __enter__(self):
        self.branch._enter_region(self.name)
        return self.branch

    def __exit__(self, exc_type, exc, traceback):
        self.branch._exit_region(self.name, exc_type is None)
        return False


class _ExplicitIf:
    """One in-progress explicit P0 branch owned by a UnplacedBuilder."""

    def __init__(self, builder: UnplacedBuilder, condition, carries) -> None:
        if not isinstance(condition,
                          LogicalBool) or condition.owner is not builder:
            raise TypeError(
                "cudaq.logical.if_ condition must be a bool result from this trace"
            )
        self.builder = builder
        self.carries = tuple(carries)
        self.result_types = tuple(
            value.mlir_value.type for value in self.carries)
        for value in self.carries:
            if isinstance(value, logical_qubit):
                value._consume("cudaq.logical.if_")
            elif isinstance(value, _LogicalLinearValue):
                value._consume("cudaq.logical.if_")
            elif not isinstance(value, _SSAProxy) or value.owner is not builder:
                raise TypeError(
                    "cudaq.logical.if_ carries must be traced values from this trace"
                )
        with builder.location:
            self.operation = mlir_ir.Operation.create(
                "qlx.if",
                operands=[condition.mlir_value],
                results=self.result_types,
                regions=2,
                loc=builder.location,
            )
            builder.insertion_point.insert(self.operation)
            self.blocks = {
                "then": self.operation.regions[0].blocks.append(),
                "else": self.operation.regions[1].blocks.append(),
            }
        self.parent_ip = builder.insertion_point
        self.active: str | None = None
        self.entered: set[str] = set()
        self.yielded: set[str] = set()
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
                "cudaq.logical.if_ requires explicit yields from both branches; missing "
                + ", ".join(sorted(missing)))
        self.results = tuple(
            self.builder._wrap_result_by_type(value)
            for value in self.operation.results)
        return False

    def then(self):
        return _ExplicitIfRegion(self, "then")

    def else_(self):
        return _ExplicitIfRegion(self, "else")

    def _enter_region(self, name: str) -> None:
        if self.active is not None:
            raise RuntimeError(
                "cudaq.logical.if_ branch regions cannot overlap")
        if name in self.entered:
            raise RuntimeError(
                f"cudaq.logical.if_ {name} region was already authored")
        self.entered.add(name)
        self.active = name
        # The same lexical carry denotes an independent linear owner inside
        # each region.  Both regions capture the pre-branch SSA value, while
        # the Python proxy is re-armed only for the duration of that region.
        for value in self.carries:
            if isinstance(value, (logical_qubit, _LogicalLinearValue)):
                value._live = True
        self.builder.insertion_point = mlir_ir.InsertionPoint(self.blocks[name])

    def _exit_region(self, name: str, successful: bool) -> None:
        try:
            if successful and name not in self.yielded:
                raise RuntimeError(
                    f"cudaq.logical.if_ {name} region must call branch.yield_()"
                )
            if successful:
                leaked = [
                    getattr(value, "semantic_ref", str(value.type))
                    for value in self.carries
                    if isinstance(value, (
                        logical_qubit, _LogicalLinearValue)) and value.is_live
                ]
                if leaked:
                    raise RuntimeError(
                        f"cudaq.logical.if_ {name} region did not consume every linear carry: "
                        f"{leaked!r}")
        finally:
            for value in self.carries:
                if isinstance(value, (logical_qubit, _LogicalLinearValue)):
                    value._live = False
            self.active = None
            self.builder.insertion_point = self.parent_ip

    def yield_(self, *values) -> None:
        if self.active is None:
            raise RuntimeError(
                "branch.yield_() must appear inside then()/else_()")
        if self.active in self.yielded:
            raise RuntimeError(
                f"cudaq.logical.if_ {self.active} region already yielded")
        flattened = _flatten_values(values)
        if len(flattened) != len(self.result_types):
            raise TypeError(
                "cudaq.logical.if_ branches must yield one value per carry")
        operands = []
        for value, expected in zip(flattened, self.result_types):
            if not isinstance(value,
                              _SSAProxy) or value.owner is not self.builder:
                raise TypeError(
                    "cudaq.logical.if_ branch yields must be traced values")
            if value.mlir_value.type != expected:
                raise TypeError(
                    "cudaq.logical.if_ branch result types must match carries")
            if isinstance(value, logical_qubit):
                value._consume("qlx.yield")
            elif isinstance(value, _LogicalLinearValue):
                value._consume("qlx.yield")
            operands.append(value.mlir_value)
        self.builder._emit("qlx.yield", operands=operands)
        self.yielded.add(self.active)
