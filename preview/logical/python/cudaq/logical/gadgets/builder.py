# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from collections.abc import Mapping
import inspect
from itertools import permutations
from types import NoneType
from typing import get_args, get_origin, get_type_hints

from cudaq.mlir import ir as mlir_ir

from ..algebra.angle import Angle
from ..errors import InvalidSyndromeSchedule, ObjectiveMismatch
from ..programs.context import (
    pop_trace,
    push_trace,
)
from ..types.values import (
    GaugeRecordsValue,
    MeasurementBits,
    LogicalBool,
    PartitionSelection,
    PartitionView,
    PatchGaugeRef,
    PatchLogicalRef,
    PatchValue,
    ResourceValue,
    SyndromeValue,
    _BBSyndromeContinuation,
)


class _ExcludedBBSyndromeSchedule:
    """Sentinel preserving fail-closed dispatch after BB research removal."""


BBSyndromeSchedule = _ExcludedBBSyndromeSchedule
from ..codes import (
    Code,
    Encoding,
    EncodingEpoch,
    GaugeMeasurementMap,
    PatchTransform,
)
from ..gadgets.definition import GadgetDefinition
from ..gadgets.interface import patch
from ..gadgets.records import ProfileParity, RecordRef
from ..gadgets.semantics import OutcomeMap, OutcomeRole, OutcomeSyndromeTerm
from ..algebra.gf2 import GF2Matrix
from ..algebra.pauli import PauliProduct
from ..devices.definition import QECRegion
from ..programs.definition import ProgramDefinition
from ..qec.objectives import SubsystemFragmentObjective
from ..types.semantic import (
    record,
    resource,
)
from ..gadgets.specification import _inferred_port_table
from ..std import (
    LogicalActionRef,
    LogicalInstrumentRef,
    SyndromeExtractionObjective,
)


def _normalize_pairs(pairs, control_width: int, target_width: int):
    """Return one bounded matching in Fabric's partition-local coordinates."""

    if isinstance(pairs, str):
        if pairs != "index":
            raise ValueError(
                "pairs= string input only accepts the canonical 'index' relation"
            )
        entries = tuple(range(min(control_width, target_width)))
        if not entries:
            raise ValueError(
                "pairs='index' must select at least one interaction")
        return tuple((index, index) for index in entries)
    normalized = []
    try:
        entries = tuple(pairs)
    except TypeError as exc:
        raise TypeError(
            "pairs= must be an iterable of (control, target) indices") from exc
    for entry in entries:
        try:
            control, target = entry
        except (TypeError, ValueError) as exc:
            raise TypeError("pairs= entries must contain two indices") from exc
        if (isinstance(control, bool) or isinstance(target, bool) or
                not isinstance(control, int) or not isinstance(target, int)):
            raise TypeError("pairs= indices must be Python ints")
        if control < 0 or target < 0:
            raise ValueError("pairs= indices must be nonnegative")
        if control >= control_width:
            raise ValueError(
                f"pairs= control index {control} is outside partition width "
                f"{control_width}")
        if target >= target_width:
            raise ValueError(
                f"pairs= target index {target} is outside partition width "
                f"{target_width}")
        normalized.append((control, target))
    if not normalized:
        raise ValueError("pairs= must contain at least one interaction")
    return tuple(normalized)


def _pair_attribute(pairs) -> str:
    """Serialize a validated interaction relation to Fabric's ``c:t`` grammar."""

    return ",".join(f"{control}:{target}" for control, target in pairs)


def _carrier_offset(block, partition: str, index: int) -> int:
    """Map a partition-local index to Fabric's canonical whole-patch order."""

    if partition == "all":
        return index
    partitions = block.partitions
    order = [name for name in ("data", "sx", "sz") if name in partitions]
    order.extend(sorted(name for name in partitions if name not in order))
    offset = 0
    for name in order:
        if name == partition:
            return offset + index
        offset += partitions[name]
    raise KeyError(partition)


class GadgetBuilder:
    """P2A physical/QEC realization builder over typed patch boundaries."""

    def __init__(self, transaction, definition: GadgetDefinition) -> None:
        self.transaction = transaction
        self.definition = definition
        self.context = transaction.context
        self.location = transaction.location
        self._patches: list[PatchValue] = []
        self._resources: list[ResourceValue] = []
        self._record_counters: dict[str, int] = {}
        self._produced_record_bases: set[str] = set()
        self._produced_records: set[str] = set()
        self._produced_record_order: list[str] = []
        self._arguments = None
        self._finished = False
        self._bb_pending_continuation = None
        self._automorphism = None
        self._derive_subsystem_fragment = False
        self._subsystem_steps: list[dict[str, object]] = []
        self._subsystem_outcomes: dict[int, int] = {}
        self.expected_clifford_action = None
        hints = definition.type_hints
        self.input_boundaries = tuple(
            self._input_boundary(hints.get(name, parameter.annotation))
            for name, parameter in definition.signature.parameters.items())
        result_annotation = hints.get("return",
                                      definition.signature.return_annotation)
        self.result_boundaries = self._flatten_result_boundaries(
            result_annotation)
        patch_inputs = tuple(
            value for kind, value in self.input_boundaries if kind == "patch")
        patch_results = tuple(
            value for kind, value in self.result_boundaries if kind == "patch")
        self.patch_transform = definition.transform
        if self.patch_transform is not None:
            terminal_input_only = len(patch_inputs) == 1 and not patch_results
            if (len(patch_inputs) != 1 or
                (len(patch_results) != 1 and not terminal_input_only)):
                raise TypeError(
                    "gadget transform= requires one input and one output "
                    "patch, or one consumed input for terminal analysis")
            if (self.patch_transform.source is not patch_inputs[0] or
                (patch_results and
                 self.patch_transform.destination is not patch_results[0])):
                raise TypeError(
                    "gadget transform= source/destination must match its patch "
                    "annotations")
        elif (len(patch_inputs) == 1 and len(patch_results) == 1 and
              patch_inputs[0] is not patch_results[0]):
            self.patch_transform = PatchTransform.infer(patch_inputs[0],
                                                        patch_results[0])
        self.transform_handle = None
        self.patch_frame_type = None
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
        self._materialize_dependencies()
        self._create_gadget()

    def _entry_binding(self, region=None):
        device = self.definition.device
        if device is None:
            raise TypeError(
                "implicit entry context requires "
                "@cudaq.logical.gadget(device=<cudaq.logical.Device>)")
        if region is None:
            qec_region = device.default_qec_region
            return next(binding for binding in device.logical_to_qec
                        if binding.qec_region is qec_region)
        region_name = getattr(region, "name", str(region))
        binding = next(
            (item for item in device.logical_to_qec
             if item.logical_region.name == region_name or
             item.qec_region.name == region_name),
            None,
        )
        if binding is None:
            raise ValueError(
                f"region {region_name!r} is not a QEC region of device "
                f"@{device.name}")
        return binding

    def _encoding_from_annotation(self, annotation):
        if get_origin(annotation) is not patch:
            raise TypeError(
                "@cudaq.logical.gadget parameters must use cudaq.logical.patch[Code|Encoding]"
            )
        (target,) = get_args(annotation)
        if isinstance(target, Code):
            return target.default_encoding
        if isinstance(target, Encoding):
            return target
        raise TypeError(
            "cudaq.logical.patch[...] requires a concrete Code or Encoding")

    @staticmethod
    def _kind_name(kind):
        return str(getattr(kind, "name", kind))

    def _input_boundary(self, annotation):
        if get_origin(annotation) is patch:
            return "patch", self._encoding_from_annotation(annotation)
        if get_origin(annotation) is resource:
            (kind,) = get_args(annotation)
            return "resource", kind
        if get_origin(annotation) is record:
            (schema,) = get_args(annotation)
            return "syndrome", self._encoding_target(schema)
        raise TypeError(
            "@cudaq.logical.gadget parameters must use cudaq.logical.patch[...] or cudaq.logical.resource[...]"
        )

    @property
    def i1_type(self):
        return mlir_ir.IntegerType.get_signless(1, context=self.context)

    def _i64(self, value):
        return mlir_ir.IntegerAttr.get(
            mlir_ir.IntegerType.get_signless(64, context=self.context), value)

    def _flatten_result_boundaries(self, annotation):
        if annotation in (None, NoneType):
            return ()
        origin = get_origin(annotation)
        if origin in (tuple, list):
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

    @staticmethod
    def _encoding_target(target):
        if isinstance(target, Code):
            return target.default_encoding
        if isinstance(target, Encoding):
            return target
        raise TypeError(
            "cudaq.logical.record[...] requires a concrete Code or Encoding")

    def _resource_type(self, kind):
        return mlir_ir.Type.parse(f"!fabric.resource<@{self._kind_name(kind)}>",
                                  context=self.context)

    def _boundary_type(self, kind, value):
        if kind == "patch":
            return self._patch_type(value)
        if kind == "resource":
            return self._resource_type(value)
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
        raise TypeError(f"unsupported gadget boundary kind {kind!r}")

    def _patch_type(self, encoding, *, epoch=None):
        code_handle = self.transaction.materialize(encoding.code)
        encoding_handle = self.transaction.materialize(encoding)
        epoch_handle = self.transaction.materialize(
            encoding.initial_epoch if epoch is None else epoch)
        return mlir_ir.Type.parse(
            f"!fabric.patch<@{code_handle.symbol}, @{encoding_handle.symbol}, "
            f"@{epoch_handle.symbol}>",
            context=self.context,
        )

    def _materialize_dependencies(self):
        encodings = []
        for kind, value in (*self.input_boundaries, *self.result_boundaries):
            if kind == "patch" and value not in encodings:
                encodings.append(value)
        encoding_handles = [
            self.transaction.materialize(value) for value in encodings
        ]
        if self.patch_transform is not None:
            self.transform_handle = self.transaction.materialize(
                self.patch_transform)
            self.patch_frame_type = mlir_ir.Type.parse(
                f"!fabric.patch_frame<@{self.transform_handle.symbol}>",
                context=self.context,
            )
        logical = self.definition.implements
        self.logical_objective = logical
        logical_type = mlir_ir.Type.parse("!qlx.logical_qubit",
                                          context=self.context)
        if isinstance(logical, LogicalActionRef):
            logical_symbol = self.transaction.objective(
                family="action",
                name=logical.name,
                inputs=(logical_type,) * logical.arity,
                results=(logical_type,) * logical.arity,
            )
            logical_function_type = mlir_ir.FunctionType.get(
                (logical_type,) * logical.arity,
                (logical_type,) * logical.arity,
                context=self.context,
            )
            self.objective_parameter_names = self._standard_parameter_names(
                logical)
            self.logical_objective_symbol = logical_symbol
        elif isinstance(logical, SyndromeExtractionObjective):
            logical_inputs = (logical_type,) * logical.arity
            logical_symbol = self.transaction.objective(
                family="instrument",
                name=logical.name,
                inputs=logical_inputs,
                results=(),
            )
            logical_function_type = mlir_ir.FunctionType.get(
                logical_inputs, (), context=self.context)
            self.objective_parameter_names = tuple(
                f"q{index}" for index in range(logical.arity))
            self.logical_objective_symbol = logical_symbol
        elif isinstance(logical, LogicalInstrumentRef):
            logical_inputs = (logical_type,) * logical.arity
            logical_results = ((logical_type,) * logical.result_arity
                               if logical.name.startswith("prepare_") else
                               (self.i1_type,) * logical.result_arity)
            logical_symbol = self.transaction.objective(
                family="instrument",
                name=logical.name,
                inputs=logical_inputs,
                results=logical_results,
            )
            logical_function_type = mlir_ir.FunctionType.get(
                logical_inputs, logical_results, context=self.context)
            self.objective_parameter_names = tuple(
                f"q{index}" for index in range(logical.arity))
            self.logical_objective_symbol = logical_symbol
        elif isinstance(logical,
                        ProgramDefinition) and logical.kind == "objective":
            logical_handle = self.transaction.materialize(logical)
            logical_symbol = logical_handle.symbol
            inputs, results, _ = self.transaction.signature_of(logical)
            logical_function_type = mlir_ir.FunctionType.get(
                inputs, results, context=self.context)
            self.objective_parameter_names = tuple(logical.signature.parameters)
            self.logical_objective_symbol = logical_handle.symbol
        elif isinstance(logical, SubsystemFragmentObjective):
            patch_inputs = sum(
                kind == "patch" for kind, _ in self.input_boundaries)
            patch_results = sum(
                kind == "patch" for kind, _ in self.result_boundaries)
            if patch_inputs != 1 or patch_results != 1:
                raise TypeError(
                    "cudaq.logical.qec.subsystem_fragment derivation currently requires "
                    "one inout encoded patch; use an explicit ObjectiveGraph for "
                    "multi-block, split, merge, or terminal fragments")
            logical_symbol = None
            logical_function_type = self.function_type
            self.objective_parameter_names = ()
            self.logical_objective_symbol = None
            self._derive_subsystem_fragment = True
        elif logical is None:
            # A root P2 gadget may be the user-facing entry point instead of a
            # selectable realization of a separate logical objective.  Keep
            # its inferred boundary contract behind an internal entry point
            # declaration and gadget spec, but do not claim equivalence to a
            # user-authored logical objective.
            logical_symbol = self.transaction.objective(
                family="instrument",
                name=f"{self.definition.name}_entrypoint",
                inputs=self.input_types,
                results=self.result_types,
            )
            logical_function_type = self.function_type
            self.objective_parameter_names = ()
            self.logical_objective_symbol = logical_symbol
        else:
            raise TypeError(
                "@cudaq.logical.gadget implements= requires cudaq.logical.std.<objective>, "
                "cudaq.logical.qec.subsystem_fragment, or a "
                "@cudaq.logical.objective definition")
        with self.context:
            logical_type_attr = mlir_ir.TypeAttr.get(logical_function_type)
            gadget_type_attr = mlir_ir.TypeAttr.get(self.function_type)
        self.objective_symbol = self.transaction.unique_symbol(
            f"{self.definition.name}_objective")
        objective_attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(self.objective_symbol,
                                       context=self.context),
            "function_type":
                logical_type_attr,
        }
        if logical_symbol is not None:
            objective_attrs["logical"] = mlir_ir.FlatSymbolRefAttr.get(
                logical_symbol, context=self.context)
        self.objective_operation = self._append_module_op(
            "fabric.objective", objective_attrs)
        self.spec_symbol = self.transaction.unique_symbol(
            f"{self.definition.name}_spec")
        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(self.spec_symbol, context=self.context),
            "objective":
                mlir_ir.FlatSymbolRefAttr.get(self.objective_symbol,
                                              context=self.context),
            "function_type":
                gadget_type_attr,
            "encodings":
                mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.FlatSymbolRefAttr.get(handle.symbol,
                                                      context=self.context)
                        for handle in encoding_handles
                    ],
                    context=self.context,
                ),
        }
        if self.definition.implements is None:
            attrs["entrypoint"] = mlir_ir.UnitAttr.get(context=self.context)
        explicit_spec = self.definition.spec
        if explicit_spec is not None:
            # The explicit spec was already verified against the inferred
            # boundary (same count, directions, and encodings); it supplies
            # the port artifacts and the semantic-map attributes.
            ports = [
                self._explicit_port_attr(port) for port in explicit_spec.ports
            ]
        else:
            ports = []
            for name, direction, encoding in _inferred_port_table(
                    self.definition.interface):
                has_input = direction in {"input", "inout"}
                has_output = direction in {"output", "inout"}
                encoding_handle = self.transaction.materialize(encoding)
                code = encoding.code
                fields = {
                    "name":
                        mlir_ir.StringAttr.get(name, context=self.context),
                    "direction":
                        mlir_ir.StringAttr.get(direction, context=self.context),
                    "ownership":
                        mlir_ir.StringAttr.get(
                            "borrow" if direction == "inout" else
                            ("consume" if direction == "input" else "produce"),
                            context=self.context,
                        ),
                    "encoding":
                        mlir_ir.FlatSymbolRefAttr.get(encoding_handle.symbol,
                                                      context=self.context),
                    "input_state":
                        mlir_ir.StringAttr.get(
                            "initialized" if has_input else "uninitialized",
                            context=self.context,
                        ),
                    "output_state":
                        mlir_ir.StringAttr.get(
                            "initialized" if has_output else "absent",
                            context=self.context,
                        ),
                    "logical_arity":
                        self._i64(code.k),
                    "data_width":
                        self._i64(code.n),
                    "scratch_width":
                        self._i64(max(0, code.block.size - code.n)),
                }
                ports.append(mlir_ir.DictAttr.get(fields, context=self.context))
        if ports:
            attrs["ports"] = mlir_ir.ArrayAttr.get(ports, context=self.context)
        if self.definition.interface.flows:
            attrs["flows"] = mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.DictAttr.get(
                        {
                            "kind":
                                mlir_ir.StringAttr.get(flow.kind,
                                                       context=self.context),
                            "inputs":
                                mlir_ir.DenseI64ArrayAttr.get(
                                    tuple(endpoint.index
                                          for endpoint in flow.inputs),
                                    context=self.context,
                                ),
                            "outputs":
                                mlir_ir.DenseI64ArrayAttr.get(
                                    tuple(endpoint.index
                                          for endpoint in flow.outputs),
                                    context=self.context,
                                ),
                        },
                        context=self.context,
                    )
                    for flow in self.definition.interface.flows
                ],
                context=self.context,
            )
        if self.definition.logical_ports:
            attrs["logical_ports"] = mlir_ir.DictAttr.get(
                {
                    key: mlir_ir.StringAttr.get(value, context=self.context)
                    for key, value in self.definition.logical_ports.items()
                },
                context=self.context,
            )
        if self.transform_handle is not None:
            attrs["transform"] = mlir_ir.FlatSymbolRefAttr.get(
                self.transform_handle.symbol, context=self.context)
        # Preserve the boundary inferred from the callable independently of an
        # explicit GadgetSpec. Certificate consumers compare this witness with
        # the authoritative spec before trusting its metadata.
        self.realization_boundary = mlir_ir.DictAttr.get(
            {name: attrs[name] for name in ("ports", "flows") if name in attrs},
            context=self.context,
        )
        if explicit_spec is not None:
            self._append_explicit_spec_attrs(attrs, explicit_spec)
        metadata_values = self.definition._authoritative_spec_metadata()
        if metadata_values:
            attrs["metadata"] = mlir_ir.DictAttr.get(
                {
                    key:
                        mlir_ir.StringAttr.get(str(value), context=self.context)
                    for key, value in metadata_values.items()
                },
                context=self.context,
            )
        self.spec_operation = self._append_module_op("fabric.gadget_spec",
                                                     attrs)

    def _explicit_port_attr(self, port):
        """Serialize one explicit cudaq.logical.Port to the normalized port dictionary."""

        encoding_handle = self.transaction.materialize(port.encoding)
        code = port.encoding.code
        # The canonical lifecycles mirror GadgetSpecOp::verify: inout borrows
        # initialized-to-initialized, input consumes initialized-to-absent,
        # output produces initialized state from uninitialized carriers.  A
        # "measured" wire is a consumed input refined for terminal readout.
        output_state = "absent" if port.wire_state == "measured" else port.wire_state
        fields = {
            "name":
                mlir_ir.StringAttr.get(port.name, context=self.context),
            "direction":
                mlir_ir.StringAttr.get(port.direction, context=self.context),
            "ownership":
                mlir_ir.StringAttr.get(port.ownership, context=self.context),
            "encoding":
                mlir_ir.FlatSymbolRefAttr.get(encoding_handle.symbol,
                                              context=self.context),
            "input_state":
                mlir_ir.StringAttr.get(
                    "uninitialized"
                    if port.direction == "output" else "initialized",
                    context=self.context,
                ),
            "output_state":
                mlir_ir.StringAttr.get(output_state, context=self.context),
            "logical_arity":
                self._i64(code.k),
            "data_width":
                self._i64(code.n),
            "scratch_width":
                self._i64(max(0, code.block.size - code.n)),
        }
        if port.wire_state == "measured":
            fields["wire_state"] = mlir_ir.StringAttr.get("measured",
                                                          context=self.context)
        if port.logical_ports:
            fields["logical_port_map"] = mlir_ir.DictAttr.get(
                {
                    operand:
                        mlir_ir.StringAttr.get(target, context=self.context)
                    for operand, target in port.logical_ports.items()
                },
                context=self.context,
            )
        if port.data_view is not None:
            fields["data_view"] = mlir_ir.DenseI64ArrayAttr.get(
                port.data_view, context=self.context)
        if port.ancilla_view is not None:
            fields["ancilla_view"] = mlir_ir.DenseI64ArrayAttr.get(
                port.ancilla_view, context=self.context)
        return mlir_ir.DictAttr.get(fields, context=self.context)

    def _append_explicit_spec_attrs(self, attrs, spec):
        """Serialize explicit outcome/parameter/frame maps onto the spec op."""

        def strings(values):
            return mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(value, context=self.context)
                    for value in values
                ],
                context=self.context,
            )

        if spec.record_schema:
            attrs["record_schema"] = strings(spec.record_schema)
        if spec.outcome_map is not None:
            attrs["outcome_map"] = self._outcome_map_attr(spec.outcome_map)
        if spec.parameter_map is not None:
            attrs["parameter_map"] = mlir_ir.DictAttr.get(
                {
                    realization:
                        mlir_ir.StringAttr.get(objective, context=self.context)
                    for realization, objective in spec.parameter_map.pairs
                },
                context=self.context,
            )

    def _outcome_map_attr(self, outcome_map: OutcomeMap):
        """Serialize the complete typed affine outcome table."""

        def strings(values):
            return mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(value, context=self.context)
                    for value in values
                ],
                context=self.context,
            )

        matrix = outcome_map.matrix
        literal = ("" if matrix.nrows == 0 or matrix.ncols == 0 else str(
            [list(row) for row in matrix.rows]).replace(" ", ""))
        rows = mlir_ir.Attribute.parse(
            f"dense<{literal}> : tensor<{matrix.nrows}x{matrix.ncols}xi1>",
            context=self.context,
        )
        input_syndromes = mlir_ir.ArrayAttr.get(
            [
                mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.DictAttr.get(
                            {
                                "port": self._i64(term.port),
                                "index": self._i64(term.index),
                            },
                            context=self.context,
                        ) for term in row
                    ],
                    context=self.context,
                ) for row in outcome_map.input_syndromes
            ],
            context=self.context,
        )
        roles = mlir_ir.ArrayAttr.get(
            [strings(row) for row in outcome_map.roles], context=self.context)
        return mlir_ir.DictAttr.get(
            {
                "records":
                    strings(outcome_map.records),
                "rows":
                    rows,
                "constants":
                    mlir_ir.DenseI64ArrayAttr.get(outcome_map.constants,
                                                  context=self.context),
                "input_syndromes":
                    input_syndromes,
                "roles":
                    roles,
            },
            context=self.context,
        )

    @staticmethod
    def _standard_parameter_names(logical):
        names = {
            "cx": ("control", "target"),
            "cz": ("left", "right"),
            "ccz": ("a", "b", "c"),
        }.get(logical.name)
        return names or tuple(f"q{index}" for index in range(logical.arity))

    @staticmethod
    def _symbol_name(attribute):
        raw = getattr(attribute, "value", attribute)
        if isinstance(raw, (tuple, list)):
            raw = raw[-1]
        return str(raw).strip('"').removeprefix("@").split("::@")[-1]

    @staticmethod
    def _identity_clifford(arity):
        return [
            *((1 << index, 0, 0) for index in range(arity)),
            *((0, 1 << index, 0) for index in range(arity)),
        ]

    @staticmethod
    def _conjugate_clifford(images, name, wires):
        if name == "idle":
            return
        if name in {"t", "tdg", "ccz"}:
            raise TypeError(
                f"logical objective contains non-Clifford action {name!r}")
        if name in {"x", "y", "z", "h", "s", "sdg"}:
            if len(wires) != 1:
                raise TypeError(f"logical action {name!r} requires one operand")
            qubit = wires[0]
            bit = 1 << qubit
            for index, (x_mask, z_mask, phase) in enumerate(images):
                x_bit = bool(x_mask & bit)
                z_bit = bool(z_mask & bit)
                if name == "x":
                    phase ^= int(z_bit)
                elif name == "z":
                    phase ^= int(x_bit)
                elif name == "y":
                    phase ^= int(x_bit ^ z_bit)
                elif name == "h":
                    phase ^= int(x_bit and z_bit)
                    if x_bit != z_bit:
                        x_mask ^= bit
                        z_mask ^= bit
                elif name in {"s", "sdg"}:
                    phase ^= int(x_bit and
                                 (z_bit if name == "s" else not z_bit))
                    if x_bit:
                        z_mask ^= bit
                images[index] = (x_mask, z_mask, phase)
            return
        if name in {"cx", "cz"}:
            if len(wires) != 2 or wires[0] == wires[1]:
                raise TypeError(
                    f"logical action {name!r} requires two operands")
            control, target = wires
            if name == "cz":
                GadgetBuilder._conjugate_clifford(images, "h", (target,))
                GadgetBuilder._conjugate_clifford(images, "cx", wires)
                GadgetBuilder._conjugate_clifford(images, "h", (target,))
                return
            control_bit = 1 << control
            target_bit = 1 << target
            for index, (x_mask, z_mask, phase) in enumerate(images):
                x_control = bool(x_mask & control_bit)
                x_target = bool(x_mask & target_bit)
                z_control = bool(z_mask & control_bit)
                z_target = bool(z_mask & target_bit)
                phase ^= int(x_control and z_target and
                             (x_target ^ z_control ^ True))
                if x_control:
                    x_mask ^= target_bit
                if z_target:
                    z_mask ^= control_bit
                images[index] = (x_mask, z_mask, phase)
            return
        raise TypeError(f"unsupported Clifford objective action {name!r}")

    @classmethod
    def _standard_clifford_action(cls, name, arity):
        from ..algebra.clifford import CliffordAction

        return CliffordAction.standard(name, arity).images

    def _program_clifford_action(self, action_symbol, arity):
        from ..algebra.clifford import CliffordAction

        action = self.transaction.find_symbol(action_symbol, "qlx.action")
        if action is None or "semantics" not in action.attributes:
            raise ValueError(
                "logical action is missing its canonical semantics body")
        semantics = self._symbol_name(action.attributes["semantics"])
        program = self.transaction.find_symbol(semantics, "qlx.objective_body")
        if program is None:
            raise ValueError(
                "logical action semantics objective body is missing")
        ports = tuple(self.objective_parameter_names)
        if len(ports) != arity:
            ports = tuple(range(arity))
        return CliffordAction.from_mlir_program(program, ports=ports).images

    def _append_module_op(self, name, attributes):
        with self.location:
            operation = mlir_ir.Operation.create(name,
                                                 attributes=attributes,
                                                 loc=self.location)
            self.transaction.module.body.append(operation)
        return operation

    def _create_gadget(self):
        with self.context:
            function_type_attr = mlir_ir.TypeAttr.get(self.function_type)
        reusable = self.definition.metadata.get("realization_form") == "circuit"
        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(self.symbol, context=self.context),
            "function_type":
                function_type_attr,
            "spec":
                mlir_ir.FlatSymbolRefAttr.get(self.spec_symbol,
                                              context=self.context),
            "realization_boundary":
                self.realization_boundary,
            "realization_kind":
                mlir_ir.StringAttr.get("circuit" if reusable else "inline",
                                       context=self.context),
        }
        if self.transform_handle is not None:
            attrs["transform"] = mlir_ir.FlatSymbolRefAttr.get(
                self.transform_handle.symbol, context=self.context)
        with self.location:
            if reusable:
                self.circuit_symbol = self.transaction.unique_symbol(
                    f"{self.definition.name}_body")
                circuit = mlir_ir.Operation.create(
                    "fabric.circuit",
                    attributes={
                        "sym_name":
                            mlir_ir.StringAttr.get(self.circuit_symbol,
                                                   context=self.context),
                        "function_type":
                            function_type_attr,
                    },
                    regions=1,
                    loc=self.location,
                )
                self.transaction.module.body.append(circuit)
                self.block = circuit.regions[0].blocks.append(*self.input_types)
                attrs["realization"] = mlir_ir.FlatSymbolRefAttr.get(
                    self.circuit_symbol, context=self.context)
            self.operation = mlir_ir.Operation.create("fabric.gadget",
                                                      attributes=attrs,
                                                      regions=1,
                                                      loc=self.location)
            self.transaction.module.body.append(self.operation)
            gadget_block = self.operation.regions[0].blocks.append(
                *self.input_types)
            if reusable:
                stub_ip = mlir_ir.InsertionPoint(gadget_block)
                stub_ip.insert(
                    mlir_ir.Operation.create("fabric.return",
                                             loc=self.location))
            else:
                self.block = gadget_block
        self.insertion_point = mlir_ir.InsertionPoint(self.block)

    def _emit(self, name, *, operands=(), results=(), attributes=None):
        self._before_operation(name)
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

    def _new_patch(
        self,
        value,
        encoding,
        *,
        epoch=None,
        encoded_state_live=True,
        terminal_only_reason=None,
        bb_syndrome_continuation=None,
        carrier_frame=None,
    ):
        if (carrier_frame is None and self.patch_transform is not None and
                self.patch_frame_type is not None and
                value.type == self.patch_frame_type):
            carrier_frame = self.patch_transform
        result = PatchValue(
            value,
            owner=self,
            encoding=encoding,
            epoch=epoch,
            semantic_ref=("patch", self.symbol, len(self._patches)),
            encoded_state_live=encoded_state_live,
            terminal_only_reason=terminal_only_reason,
            bb_syndrome_continuation=bb_syndrome_continuation,
            carrier_frame=carrier_frame,
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

    def _before_operation(self, operation):
        """Reserve the next authored operation for a pending BB continuation."""

        if self._bb_pending_continuation is not None:
            raise InvalidSyndromeSchedule(
                "a nonterminal BB depth-8 cycle must be continued immediately "
                "with prime=False; no other authored operation may intervene, "
                f"not {operation}")

    def arguments(self):
        if self._arguments is None:
            arguments = []
            for value, (kind, boundary) in zip(self.block.arguments,
                                               self.input_boundaries):
                if kind == "patch":
                    if (self.patch_transform is not None and
                            boundary is self.patch_transform.source):
                        operation = self._emit(
                            "fabric.transform_begin",
                            operands=[value],
                            results=[self.patch_frame_type],
                            attributes={
                                "transform":
                                    mlir_ir.FlatSymbolRefAttr.get(
                                        self.transform_handle.symbol,
                                        context=self.context,
                                    )
                            },
                        )
                        arguments.append(
                            self._new_patch(
                                operation.result,
                                boundary,
                                carrier_frame=self.patch_transform,
                            ))
                    else:
                        arguments.append(self._new_patch(value, boundary))
                elif kind == "syndrome":
                    arguments.append(
                        SyndromeValue(
                            value,
                            owner=self,
                            encoding=boundary,
                            record="input",
                            location=self.location,
                        ))
                else:
                    arguments.append(self._new_resource(value, boundary))
            self._arguments = tuple(arguments)
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
            raise RuntimeError("GadgetBuilder root is already finished")
        self._before_operation("fabric.return")
        self._finished = True
        values = (() if returned is None else
                  returned if isinstance(returned, tuple) else (returned,))
        if len(values) != len(self.result_types):
            raise TypeError(
                "gadget return count does not match its patch annotation")
        if self._derive_subsystem_fragment:
            for result_index, value in enumerate(values):
                step_index = self._subsystem_outcomes.get(id(value))
                if step_index is not None:
                    self._subsystem_steps[step_index][
                        "outcome_result"] = result_index
        operands = []
        for value, expected, boundary in zip(values, self.result_types,
                                             self.result_boundaries):
            kind, _ = boundary
            if kind == "patch":
                if not isinstance(value, PatchValue) or value.owner is not self:
                    raise TypeError(
                        "gadget patch results must be live PatchValue objects")
                if value.type == expected:
                    value._consume("fabric.return")
                    operands.append(value.mlir_value)
                elif (self.patch_transform is not None and
                      value.carrier_frame is self.patch_transform and
                      value.type == self.patch_frame_type and
                      boundary[1] is self.patch_transform.destination):
                    value._consume("fabric.transform_end")
                    operation = self._emit(
                        "fabric.transform_end",
                        operands=[value.mlir_value],
                        results=[expected],
                        attributes={
                            "transform":
                                mlir_ir.FlatSymbolRefAttr.get(
                                    self.transform_handle.symbol,
                                    context=self.context,
                                )
                        },
                    )
                    operands.append(operation.result)
                else:
                    raise TypeError(
                        "gadget returned a patch with the wrong boundary")
            elif kind == "bool":
                if not isinstance(value,
                                  LogicalBool) or value.owner is not self:
                    raise TypeError(
                        "gadget bool results must be traced measurement values")
                operands.append(value.mlir_value)
            elif kind == "resource":
                if not isinstance(value,
                                  ResourceValue) or value.owner is not self:
                    raise TypeError(
                        "gadget resource results must be live ResourceValue objects"
                    )
                if value.type != expected:
                    raise TypeError(
                        "gadget returned a resource with the wrong kind")
                value._consume("fabric.return")
                operands.append(value.mlir_value)
            elif kind == "syndrome":
                if not isinstance(value,
                                  SyndromeValue) or value.owner is not self:
                    raise TypeError(
                        "gadget syndrome results must be traced values")
                if value.type != expected:
                    raise TypeError(
                        "gadget returned a syndrome with the wrong schema")
                operands.append(value.mlir_value)
        self._derive_inferred_outcome_map(values)
        self._emit("fabric.return", operands=operands)
        if self._derive_subsystem_fragment:
            self._finish_subsystem_fragment_objective()
        elif (any(child.operation.name == "fabric.epoch_transition"
                  for child in self.block.operations) and
              any(payload.profile.dynamic_phases
                  for kind, payload in self.input_boundaries
                  if kind == "patch")):
            raise TypeError("dynamic profile period gadgets require the exact "
                            "cudaq.logical.qec.subsystem_fragment objective")
        if self._automorphism is not None:
            semantic_ops = [
                child.operation.name
                for child in self.block.operations
                if child.operation.name != "fabric.return"
            ]
            if semantic_ops != ["fabric.permute"]:
                raise ValueError(
                    "automatic code-automorphism action matching requires a "
                    "single typed cudaq.logical.permute realization")
        if any(value.is_live for value in self._patches):
            raise RuntimeError(
                "gadget leaves a patch owner live at its boundary")
        if any(value.is_live for value in self._resources):
            raise RuntimeError(
                "gadget leaves a resource owner live at its boundary")
        self._validate_explicit_spec_records()

    def _derive_inferred_outcome_map(self, values) -> None:
        """Materialize canonical affine result semantics when Python can prove it."""

        if (self.definition.spec is not None or
                self.definition.implements is None):
            return
        outcomes = tuple(
            value for value, boundary in zip(values, self.result_boundaries)
            if boundary[0] == "bool")
        if outcomes:
            parities = tuple(value.producer for value in outcomes)
            if any(not isinstance(parity, ProfileParity)
                   for parity in parities):
                raise ObjectiveMismatch(
                    f"gadget {self.definition.name!r} returns a Boolean whose "
                    "affine stable-record OutcomeMap cannot be derived; "
                    "declare an explicit GadgetSpec or return supported "
                    "affine record expressions")
        else:
            return

        records = []
        for parity in parities:
            for record in parity.records:
                if record not in records:
                    records.append(record)
        record_names = tuple(record.name for record in records)
        rows = tuple(
            tuple(int(record in parity.records)
                  for record in records)
            for parity in parities)

        def strings(names):
            return mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(name, context=self.context)
                    for name in names
                ],
                context=self.context,
            )

        # The schema is the complete stable record manifest in production
        # order; the OutcomeMap names only the columns its returned values use.
        if self._produced_record_order:
            self.spec_operation.attributes["record_schema"] = strings(
                self._produced_record_order)
        input_endpoints = tuple(self.definition.interface.inputs)
        input_syndromes = []
        for parity in parities:
            terms = []
            for reference in parity.input_syndromes:
                try:
                    port = input_endpoints.index(reference.port)
                except ValueError as exc:
                    raise ObjectiveMismatch(
                        "OutcomeMap input-syndrome term belongs to a different "
                        "gadget boundary") from exc
                terms.append(OutcomeSyndromeTerm(port, reference.index))
            input_syndromes.append(tuple(terms))
        outcome_map = OutcomeMap(
            records=record_names,
            matrix=GF2Matrix.from_rows(rows, ncols=len(record_names)),
            constants=tuple(int(parity.constant) for parity in parities),
            input_syndromes=tuple(input_syndromes),
            roles=tuple((OutcomeRole.RESULT,) for _ in parities),
        )
        self.definition._attach_inferred_outcome_map(outcome_map)
        self.spec_operation.attributes["outcome_map"] = self._outcome_map_attr(
            outcome_map)

    def _finish_subsystem_fragment_objective(self):
        semantic_ops = [
            child.operation.name
            for child in self.block.operations
            if child.operation.name != "fabric.return"
        ]
        if len(semantic_ops) != len(self._subsystem_steps):
            unsupported = sorted(
                set(semantic_ops) - {
                    "fabric.measure_product", "fabric.rotate_product",
                    "fabric.x", "fabric.z", "fabric.measure_gauges",
                    "fabric.epoch_transition"
                })
            detail = ", ".join(
                unsupported) if unsupported else "untyped carrier operation"
            raise TypeError(
                "cannot derive an exact subsystem-fragment objective from "
                f"{detail}; use a supported protected/gauge Pauli fragment or "
                "an explicit ObjectiveGraph")

        protected_kinds = {
            kind for step in self._subsystem_steps
            for kind in step.get("port_kinds", ()) if kind == "protected"
        }
        gauge_kinds = {
            kind for step in self._subsystem_steps
            for kind in step.get("port_kinds", ()) if kind == "gauge"
        }
        has_epoch_transition = any(step["operation"] == "epoch_transition"
                                   for step in self._subsystem_steps)
        has_gauge_measurement = any(step["operation"] == "measure_gauges"
                                    for step in self._subsystem_steps)
        dynamic_encoding = None
        if has_epoch_transition:
            input_encodings = tuple(
                value for kind, value in self.input_boundaries
                if kind == "patch")
            if len(input_encodings) != 1:
                raise TypeError(
                    "a dynamic subsystem period requires exactly one patch "
                    "input")
            dynamic_encoding = input_encodings[0]
            profile = dynamic_encoding.profile
            expected_dynamic_steps = []
            for phase in profile.dynamic_phases:
                expected_dynamic_steps.extend((
                    ("measure_gauges", phase.name, phase.input_epoch,
                     phase.measured_gauges),
                    ("epoch_transition", phase.input_epoch, phase.output_epoch,
                     dict(phase.logical_map)),
                ))
            actual_dynamic_steps = []
            for step in self._subsystem_steps:
                if step["operation"] == "measure_gauges":
                    actual_dynamic_steps.append((
                        "measure_gauges",
                        step["phase"],
                        step["input_epoch"],
                        step["operators"],
                    ))
                elif step["operation"] == "epoch_transition":
                    actual_dynamic_steps.append((
                        "epoch_transition",
                        step["from_epoch"],
                        step["to_epoch"],
                        dict(step["logical_map"]),
                    ))
                else:
                    raise TypeError(
                        "a dynamic subsystem period may contain only the "
                        "profile's exact gauge measurements and epoch "
                        "transitions")
            if actual_dynamic_steps != expected_dynamic_steps:
                raise ValueError(
                    "a dynamic subsystem period must realize the complete "
                    "declared measurement/epoch cycle in profile order")
        protected_effect = ("unitary"
                            if has_epoch_transition else "instrument" if any(
                                step["operation"] == "measure" and
                                "protected" in step["port_kinds"]
                                for step in self._subsystem_steps) else
                            "unitary" if protected_kinds else "identity")
        gauge_effect = ("instrument" if has_gauge_measurement or any(
            step["operation"] == "measure" and "gauge" in step["port_kinds"]
            for step in self._subsystem_steps) else
                        "unitary" if gauge_kinds else "identity")

        def string(value):
            return mlir_ir.StringAttr.get(str(value), context=self.context)

        def step_attr(step):
            operation = step["operation"]
            values = {"operation": string(operation)}
            if operation in {"measure", "pauli", "rotate"}:
                values.update({
                    "patch_indices":
                        mlir_ir.DenseI64ArrayAttr.get(step["patch_indices"],
                                                      self.context),
                    "port_kinds":
                        mlir_ir.ArrayAttr.get(
                            [string(value) for value in step["port_kinds"]],
                            context=self.context,
                        ),
                    "port_indices":
                        mlir_ir.DenseI64ArrayAttr.get(step["port_indices"],
                                                      self.context),
                    "paulis":
                        string(step["paulis"]),
                    "sign":
                        self._i64(step["sign"]),
                })
            elif operation == "measure_gauges":
                values.update({
                    "operators": self._gf2_matrix_attr(step["operators"]),
                    "phase": string(step["phase"]),
                    "record": string(step["record"]),
                    "input_epoch": string(step["input_epoch"]),
                })
            elif operation == "epoch_transition":
                values.update({
                    "from_epoch":
                        string(step["from_epoch"]),
                    "to_epoch":
                        string(step["to_epoch"]),
                    "logical_map":
                        mlir_ir.DictAttr.get(
                            {
                                string_key: string(string_value) for string_key,
                                string_value in step["logical_map"].items()
                            },
                            context=self.context,
                        ),
                    "evidence":
                        string(step["evidence"]),
                })
                if step.get("period_closure") is not None:
                    values["period_closure"] = self._gf2_matrix_attr(
                        step["period_closure"])
            if "record" in step:
                values["record"] = string(step["record"])
            if "angle" in step:
                with self.context:
                    values["angle"] = mlir_ir.FloatAttr.get(
                        mlir_ir.F64Type.get(context=self.context),
                        step["angle"])
            if "outcome_result" in step:
                values["outcome_result"] = self._i64(step["outcome_result"])
            return mlir_ir.DictAttr.get(values, context=self.context)

        fragment_values = {
            "derivation":
                string("body_exact"),
            "protected_effect":
                string(protected_effect),
            "gauge_effect":
                string(gauge_effect),
            "epoch_effect":
                string("cycle" if has_epoch_transition else "preserve"),
            "steps":
                mlir_ir.ArrayAttr.get(
                    [step_attr(step) for step in self._subsystem_steps],
                    context=self.context,
                ),
        }
        if has_epoch_transition:
            profile = self.transaction.materialize(dynamic_encoding.profile)
            fragment_values["profile"] = mlir_ir.FlatSymbolRefAttr.get(
                profile.symbol, context=self.context)
            fragment_values["period_closure"] = self._gf2_matrix_attr(
                dynamic_encoding.profile.period_closure)
        self.objective_operation.attributes["subsystem_fragment"] = (
            mlir_ir.DictAttr.get(fragment_values, context=self.context))

    def call(self, definition, args, kwargs):
        if kwargs:
            raise TypeError(
                f"unsupported gadget call keyword arguments: {sorted(kwargs)}")
        if not isinstance(definition, GadgetDefinition):
            raise TypeError(
                "gadgets may call only other @cudaq.logical.gadget values")
        hints = definition.type_hints
        input_boundaries = tuple(
            self._input_boundary(hints.get(name, parameter.annotation))
            for name, parameter in definition.signature.parameters.items())
        result_boundaries = self._flatten_result_boundaries(
            hints.get("return", definition.signature.return_annotation))
        if len(args) != len(input_boundaries):
            raise TypeError("gadget call argument count does not match callee")

        operation_name = f"fabric.call @{definition.name}"
        operands = []
        for value, (kind, payload) in zip(args, input_boundaries):
            if kind == "patch":
                if not isinstance(value, PatchValue) or value.owner is not self:
                    raise TypeError("gadget call requires a live patch")
                if value.encoding is not payload:
                    raise TypeError(
                        "gadget call patch encoding does not match callee")
                value._validate_consume(operation_name)
            elif kind == "resource":
                if not isinstance(value,
                                  ResourceValue) or value.owner is not self:
                    raise TypeError("gadget call requires a live resource")
                if value.kind != payload:
                    raise TypeError(
                        "gadget call resource kind does not match callee")
                value._validate_consume(operation_name)
            elif kind == "syndrome":
                if not isinstance(value,
                                  SyndromeValue) or value.owner is not self:
                    raise TypeError("gadget call requires a syndrome record")
                if value.encoding is not payload:
                    raise TypeError(
                        "gadget call syndrome schema does not match callee")
            elif kind == "bool":
                if not isinstance(value,
                                  LogicalBool) or value.owner is not self:
                    raise TypeError("gadget call requires a traced bool")
            operands.append(value.mlir_value)

        # Calls are authored scheduling boundaries even when their signature
        # contains no patch operand. Reject a pending BB continuation before
        # consuming any linear resource.
        self._before_operation(operation_name)
        for value, (kind, _) in zip(args, input_boundaries):
            if kind in {"patch", "resource"}:
                value._consume(operation_name)

        callee = self.transaction.materialize(definition)
        result_types = tuple(
            self._boundary_type(kind, value)
            for kind, value in result_boundaries)
        attrs = {
            "callee":
                mlir_ir.FlatSymbolRefAttr.get(callee.symbol,
                                              context=self.context)
        }
        operation = self._emit(
            "fabric.call",
            operands=operands,
            results=result_types,
            attributes=attrs,
        )
        wrapped = []
        for result, (kind, payload) in zip(operation.results,
                                           result_boundaries):
            if kind == "patch":
                wrapped.append(self._new_patch(result, payload))
            elif kind == "resource":
                wrapped.append(self._new_resource(result, payload))
            elif kind == "syndrome":
                wrapped.append(
                    SyndromeValue(
                        result,
                        owner=self,
                        encoding=payload,
                        record=f"{definition.name}.result",
                        location=self.location,
                    ))
            elif kind == "bool":
                wrapped.append(
                    LogicalBool(result, owner=self, location=self.location))
        return wrapped[0] if len(wrapped) == 1 else tuple(wrapped)

    def _partition_attr(self, name: str):
        return mlir_ir.Attribute.parse(f"#fabric.partition<{name}>",
                                       context=self.context)

    def _prep_attr(self, basis: str):
        if basis not in {"x", "z"}:
            raise ValueError(
                "carrier initialization/measurement basis must be x or z")
        return mlir_ir.Attribute.parse(f"#fabric.prep<{basis}>",
                                       context=self.context)

    def _view(self, value, default="all"):
        if isinstance(value, PartitionSelection):
            return value.owner, value.partition, value.indices
        if isinstance(value, PartitionView):
            return value.owner, value.partition, None
        if isinstance(value, PatchValue):
            return value, default, None
        raise TypeError("Fabric operations require a patch or partition view")

    @staticmethod
    def _partition_width(patch_value, partition):
        if partition == "all":
            return patch_value.carrier_block.size
        return patch_value.carrier_block.partitions[partition]

    def apply_standard(self, name: str, values, **options):
        if name in {"cx", "cz"}:
            return self._two_partition(name, values, **options)
        if len(values) != 1:
            raise TypeError(f"fabric {name} expects one patch view")
        if isinstance(values[0], (PatchLogicalRef, PatchGaugeRef)):
            if name not in {"x", "z"}:
                raise TypeError(
                    "direct logical/gauge-port actions currently support Pauli X/Z"
                )
            reference = values[0]
            patch_value = reference.patch
            if patch_value.owner is not self:
                raise ValueError("logical-port action received a foreign patch")
            code = patch_value.encoding.code
            if isinstance(reference, PatchGaugeRef):
                operators = code.gx if name == "x" else code.gz
            else:
                operators = code.lx if name == "x" else code.lz
            try:
                support = operators[reference.index]
            except IndexError as exc:
                raise ValueError(
                    f"code {code.name!r} has no {name.upper()} representative "
                    f"for the selected port") from exc
            patch_value._consume(f"fabric.{name}")
            operation = self._emit(
                f"fabric.{name}",
                operands=[patch_value.mlir_value],
                results=[patch_value.type],
                attributes={
                    "partition":
                        self._partition_attr("data"),
                    "indices":
                        mlir_ir.DenseI64ArrayAttr.get(support, self.context),
                    "subsystem_kind":
                        mlir_ir.StringAttr.get(
                            "gauge" if isinstance(reference, PatchGaugeRef) else
                            "protected",
                            context=self.context,
                        ),
                    "subsystem_index":
                        self._i64(reference.index),
                },
            )
            if self._derive_subsystem_fragment:
                self._subsystem_steps.append({
                    "operation": "pauli",
                    "patch_indices": (0,),
                    "port_kinds": ("gauge" if isinstance(
                        reference, PatchGaugeRef) else "protected",),
                    "port_indices": (reference.index,),
                    "paulis": name.upper(),
                    "sign": 1,
                })
            return [
                self._new_patch(
                    operation.result,
                    patch_value.encoding,
                    epoch=patch_value.epoch,
                )
            ]
        patch_value, partition, indices = self._view(values[0])
        patch_value._consume(f"fabric.{name}")
        attrs = {"partition": self._partition_attr(partition)}
        if indices is not None:
            attrs["indices"] = mlir_ir.DenseI64ArrayAttr.get(
                indices, self.context)
        operation = self._emit(
            f"fabric.{name}",
            operands=[patch_value.mlir_value],
            results=[patch_value.type],
            attributes=attrs,
        )
        return [
            self._new_patch(operation.result,
                            patch_value.encoding,
                            epoch=patch_value.epoch)
        ]

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

    def allocate_patch(self, target=None, *, region=None):
        # Reject a pending BB continuation before resolving or materializing a
        # different encoding into the transaction.
        self._before_operation("fabric.alloc")
        binding = None
        if target is None or region is None:
            binding = self._entry_binding(region)
        if target is None:
            target = binding.qec_region.encoding
        elif binding is not None:
            selected = self._encoding_target(target)
            expected = binding.qec_region.encoding
            selected_handle = self.transaction.materialize(selected)
            expected_handle = self.transaction.materialize(expected)
            if selected_handle.symbol != expected_handle.symbol:
                raise ValueError(
                    f"allocation encoding {selected.name!r} conflicts "
                    "semantically with device region "
                    f"@{binding.qec_region.name} encoding {expected.name!r}")
        if binding is not None:
            # An entry binding resolves logical allocation intent to one
            # concrete QEC pool.  Keep detached reusable provider hints
            # non-strict, but never emit an unresolved logical-space name for
            # a device-bound allocation.
            region = binding.qec_region
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

    def consume_resource(self, resource_value, values, *, action=None):
        del resource_value, values, action
        raise TypeError(
            "cudaq.logical.consume is portable P0/P1 intent and cannot appear inside a "
            "P2 gadget; use cudaq.logical.unpack_resource(...) and explicitly author "
            "the code-specific injection circuit")

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
        if encoding is None:
            payload_encodings = tuple(anchor.encoding for anchor in anchors)
        else:
            selected = self._encoding_target(encoding)
            payload_encodings = (selected,) * len(anchors)
        payload_epochs = tuple(
            anchor.epoch if payload_encoding is
            anchor.encoding else payload_encoding.initial_epoch
            for anchor, payload_encoding in zip(anchors, payload_encodings))
        mapping = self._resource_payload_logical_ports(
            resource_value,
            anchors,
            payload_encodings,
            logical_ports,
            require_explicit=not scalar,
        )
        resource_value._validate_consume("fabric.unpack_resource")
        for anchor in anchors:
            anchor._validate_consume("fabric.unpack_resource")
        resource_value._consume("fabric.unpack_resource")
        for anchor in anchors:
            anchor._consume("fabric.unpack_resource")
        attrs = {}
        if mapping is not None:
            block_indices, port_indices = mapping
            action = resource_value.kind.consume_action
            attrs = {
                "payload_action":
                    mlir_ir.Attribute.parse(f"#qlx.action<{action.name}>",
                                            context=self.context),
                "payload_logical_blocks":
                    mlir_ir.DenseI64ArrayAttr.get(block_indices, self.context),
                "payload_logical_ports":
                    mlir_ir.DenseI64ArrayAttr.get(port_indices, self.context),
            }
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

    @staticmethod
    def _resource_payload_logical_ports(
        resource_value,
        anchors,
        payload_encodings,
        logical_ports,
        *,
        require_explicit,
    ):
        if logical_ports is None:
            if require_explicit:
                raise TypeError(
                    "multi-patch unpack_resource requires logical_ports= to map "
                    "resource logicals onto distinct payload blocks")
            return None
        groups = tuple(tuple(group) for group in logical_ports)
        if len(groups) != len(anchors):
            raise ValueError(
                "unpack_resource logical_ports= must provide one group per anchor"
            )
        flattened = tuple((block, port)
                          for block, group in enumerate(groups)
                          for port in group)
        if not flattened:
            raise ValueError(
                "unpack_resource logical_ports= must map at least one logical")
        if any(
                isinstance(port, bool) or not isinstance(port, int) or port < 0
                for _block, port in flattened):
            raise TypeError(
                "unpack_resource logical port indices must be nonnegative ints")
        if len(set(flattened)) != len(flattened):
            raise ValueError(
                "unpack_resource logical_ports= cannot alias one payload port")
        for block, port in flattened:
            if (port >= anchors[block].encoding.code.k or
                    port >= payload_encodings[block].code.k):
                raise ValueError(
                    "unpack_resource logical port exceeds its anchor or payload "
                    "encoding capacity")
        action = getattr(resource_value.kind, "consume_action", None)
        if action is None:
            raise ValueError(
                "mapped resource payload requires a typed consume action")
        if action.arity != len(flattened):
            raise ValueError(
                f"resource kind {resource_value.kind.name!r} carries "
                f"{action.arity} logical operand(s), but logical_ports= maps "
                f"{len(flattened)}")
        return (
            tuple(block for block, _port in flattened),
            tuple(port for _block, port in flattened),
        )

    def pack_resource(self, payload, *, kind):
        if not isinstance(payload, PatchValue) or payload.owner is not self:
            raise TypeError("pack_resource expects one live encoded patch")
        kind_name = self._kind_name(kind)
        payload._consume("fabric.pack_resource")
        operation = self._emit(
            "fabric.pack_resource",
            operands=[payload.mlir_value],
            results=[self._resource_type(kind)],
            attributes={
                "resource_kind":
                    mlir_ir.FlatSymbolRefAttr.get(kind_name,
                                                  context=self.context),
                "payload_encoding":
                    mlir_ir.FlatSymbolRefAttr.get(
                        self.transaction.materialize(payload.encoding).symbol,
                        context=self.context,
                    ),
            },
        )
        return self._new_resource(operation.result, kind)

    def discard(self, values, *, reason=None):
        values = tuple(values)
        if not values or any(
                not isinstance(value, PatchValue) or value.owner is not self
                for value in values):
            raise TypeError("gadget qlx.discard expects live whole patches")
        for value in values:
            operand = value.mlir_value
            if (self.patch_transform is not None and
                    value.carrier_frame is self.patch_transform and
                    value.type == self.patch_frame_type):
                value._consume("fabric.transform_end")
                transformed = self._emit(
                    "fabric.transform_end",
                    operands=[operand],
                    results=[
                        self._patch_type(self.patch_transform.destination)
                    ],
                    attributes={
                        "transform":
                            mlir_ir.FlatSymbolRefAttr.get(
                                self.transform_handle.symbol,
                                context=self.context,
                            )
                    },
                )
                operand = transformed.result
            else:
                value._consume("fabric.dealloc")
            self._emit(
                "fabric.dealloc",
                operands=[operand],
                attributes=({
                    "reason":
                        mlir_ir.StringAttr.get(reason, context=self.context)
                } if reason is not None else None),
            )

    def _two_partition(self, name, values, *, schedule=None, pairs=None, **_):
        if len(values) != 2:
            raise TypeError(f"fabric {name} expects control and target views")
        control, ctrl_partition, ctrl_indices = self._view(values[0])
        target, targ_partition, targ_indices = self._view(values[1])
        owners = [control] if control is target else [control, target]
        attrs = {
            "ctrl": self._partition_attr(ctrl_partition),
            "targ": self._partition_attr(targ_partition),
        }
        if schedule is not None:
            attrs["schedule"] = mlir_ir.StringAttr.get(
                str(getattr(schedule, "value", schedule)).lower(),
                context=self.context,
            )
        if control is not target and schedule is not None:
            raise ValueError(
                "cross-patch CX/CZ uses explicit pairs= or equal-width "
                "pairwise collection semantics, not a CSS schedule")
        control_width = self._partition_width(control, ctrl_partition)
        target_width = self._partition_width(target, targ_partition)
        if ctrl_indices is not None or targ_indices is not None:
            if schedule is not None:
                raise ValueError(
                    "selected carrier views use their explicit pairwise order, "
                    "not a CSS schedule")
            if pairs is not None:
                raise ValueError(
                    "selected carrier views already define the interaction pairs"
                )
            control_values = (ctrl_indices if ctrl_indices is not None else
                              tuple(range(control_width)))
            target_values = (targ_indices if targ_indices is not None else
                             tuple(range(target_width)))
            if len(control_values) != len(target_values):
                raise ValueError(
                    "selected CX/CZ carrier views must have equal width")
            pairs = tuple(zip(control_values, target_values))
        elif control is not target and pairs is None:
            if control_width != target_width:
                raise ValueError(
                    "cross-patch CX/CZ partitions have different widths; "
                    "provide pairs= explicitly")
            pairs = tuple(zip(range(control_width), range(target_width)))
        if pairs is not None:
            pairs = _normalize_pairs(pairs, control_width, target_width)
            if control is target:
                aliases = tuple(
                    (control_index, target_index)
                    for control_index, target_index in pairs
                    if _carrier_offset(control.carrier_block, ctrl_partition,
                                       control_index) == _carrier_offset(
                                           target.carrier_block, targ_partition,
                                           target_index))
                if aliases:
                    raise ValueError(
                        "CX/CZ interaction pairs must select two distinct "
                        f"carriers; self-aliasing pair {aliases[0]!r} maps both "
                        "endpoints to one patch carrier")
            attrs["pairs"] = mlir_ir.StringAttr.get(_pair_attribute(pairs),
                                                    context=self.context)
        operation_name = f"fabric.{name}"
        # Validate the complete operation before mutating any linear owner.
        # A caught authoring diagnostic must leave both input patches usable.
        for owner in owners:
            owner._validate_consume(operation_name)
        for owner in owners:
            owner._consume(operation_name)
        operation = self._emit(
            operation_name,
            operands=[owner.mlir_value for owner in owners],
            results=[owner.type for owner in owners],
            attributes=attrs,
        )
        return [
            self._new_patch(result, owner.encoding, epoch=owner.epoch)
            for result, owner in zip(operation.results, owners)
        ]

    @staticmethod
    def _validate_record(requested: str | None) -> str | None:
        if requested is not None:
            if not isinstance(requested, str) or not requested:
                raise TypeError(
                    "record= must be a nonempty relative path string")
        return requested

    def _record(self, requested: str | None, kind: str) -> str:
        requested = self._validate_record(requested)
        # Keep automatic record allocation transactional with respect to the
        # operation-global BB continuation guard.  Explicit record names also
        # represent an authored operation and must not bypass that guard.
        self._before_operation(f"{kind} record allocation")
        if requested is not None:
            result = requested
        else:
            index = self._record_counters.get(kind, 0)
            self._record_counters[kind] = index + 1
            result = f"{kind}{index}"
        if result in self._produced_record_bases:
            raise ValueError(
                f"record base {result!r} already has a producer in gadget "
                f"{self.definition.name!r}")
        self._produced_record_bases.add(result)
        return result

    def _register_record_family(self, record: str, field: str,
                                width: int) -> None:
        for index in range(width):
            self._register_record(f"{record}.{field}{index}")

    def _register_record(self, name: str) -> None:
        if name not in self._produced_records:
            self._produced_records.add(name)
            self._produced_record_order.append(name)

    def _measurement_parity(
        self,
        bits: MeasurementBits,
        *,
        indices=None,
        constant: bool = False,
    ) -> ProfileParity | None:
        prefix = f"{bits.record}."
        names = tuple(name for name in self._produced_record_order
                      if name.startswith(prefix))
        if indices is not None:
            indices = tuple(indices)
            if any(not 0 <= index < len(names) for index in indices):
                return None
            names = tuple(names[index] for index in indices)
        if not names:
            return None
        return ProfileParity(
            records=tuple(RecordRef(self.definition, name) for name in names),
            constant=constant,
        )

    @staticmethod
    def _syndrome_record_width(encoding) -> int:
        return encoding.profile.effective_stabilizers.nrows

    def _validate_explicit_spec_records(self) -> None:
        spec = self.definition.spec
        if spec is None or spec.record_schema is None:
            return
        missing = sorted(set(spec.record_schema) - self._produced_records)
        if not missing:
            return
        map_records = (set(spec.outcome_map.records)
                       if spec.outcome_map is not None else set())
        missing_map_records = sorted(map_records & set(missing))
        detail = (f"; semantic maps also reference {missing_map_records}"
                  if missing_map_records else "")
        raise ObjectiveMismatch(
            f"gadget {self.definition.name!r} record_schema names record(s) "
            f"the concrete realization does not produce: {missing}{detail}")

    @staticmethod
    def _basis_name(basis):
        if basis is None:
            return None
        value = getattr(basis, "value", basis)
        value = str(value).lower()
        if value not in {"x", "y", "z"}:
            raise TypeError("basis must be cudaq.logical.Basis.X, Y, or Z")
        return value

    def mz(self, value, *, record=None):
        patch_value, partition, indices = self._view(value)
        width = (len(indices) if indices is not None else self._partition_width(
            patch_value, partition))
        record = self._record(record, "mz")
        self._register_record_family(record, partition, width)
        with self.context:
            i1 = mlir_ir.IntegerType.get_signless(1, context=self.context)
            bits_type = mlir_ir.RankedTensorType.get([width],
                                                     i1,
                                                     loc=self.location)
        attrs = {
            "partition": self._partition_attr(partition),
            "record": mlir_ir.StringAttr.get(record, context=self.context),
        }
        if indices is not None:
            attrs["indices"] = mlir_ir.DenseI64ArrayAttr.get(
                indices, self.context)
        patch_value._consume("fabric.mz")
        operation = self._emit(
            "fabric.mz",
            operands=[patch_value.mlir_value],
            results=[patch_value.type, bits_type],
            attributes=attrs,
        )
        return (
            self._new_patch(
                operation.results[0],
                patch_value.encoding,
                epoch=patch_value.epoch,
                encoded_state_live=not (partition == "data" and
                                        indices is None),
            ),
            MeasurementBits(operation.results[1],
                            owner=self,
                            record=record,
                            location=self.location),
        )

    def _init_basis(self, value, *, basis):
        patch_value, partition, indices = self._view(value)
        operation_name = "fabric.init_basis"
        patch_value._validate_consume(operation_name)
        attrs = {
            "partition": self._partition_attr(partition),
            "basis": self._prep_attr(basis),
        }
        if indices is not None:
            attrs["indices"] = mlir_ir.DenseI64ArrayAttr.get(
                indices, self.context)
        patch_value._consume(operation_name)
        operation = self._emit(
            operation_name,
            operands=[patch_value.mlir_value],
            results=[patch_value.type],
            attributes=attrs,
        )
        return self._new_patch(
            operation.result,
            patch_value.encoding,
            epoch=patch_value.epoch,
        )

    def _measure_basis(self, value, *, basis, record):
        patch_value, partition, indices = self._view(value)
        width = (len(indices) if indices is not None else self._partition_width(
            patch_value, partition))
        record = self._record(record, f"m{basis}")
        self._register_record_family(record, partition, width)
        with self.context:
            bits_type = mlir_ir.RankedTensorType.get([width],
                                                     self.i1_type,
                                                     loc=self.location)
        attrs = {
            "partition": self._partition_attr(partition),
            "basis": self._prep_attr(basis),
            "record": mlir_ir.StringAttr.get(record, context=self.context),
        }
        if indices is not None:
            attrs["indices"] = mlir_ir.DenseI64ArrayAttr.get(
                indices, self.context)
        operation_name = "fabric.measure_basis"
        patch_value._validate_consume(operation_name)
        patch_value._consume(operation_name)
        operation = self._emit(
            operation_name,
            operands=[patch_value.mlir_value],
            results=[patch_value.type, bits_type],
            attributes=attrs,
        )
        return (
            self._new_patch(
                operation.results[0],
                patch_value.encoding,
                epoch=patch_value.epoch,
                encoded_state_live=not (partition == "data" and
                                        indices is None),
            ),
            MeasurementBits(
                operation.results[1],
                owner=self,
                record=record,
                location=self.location,
            ),
        )

    @staticmethod
    def _extraction_support(schedule, basis, check_index, support):
        """Resolve and validate one user-authored hook-error ordering."""

        support = tuple(support)
        if schedule is None:
            return support
        selected = schedule
        if isinstance(selected, Mapping):
            if basis in selected or basis.upper() in selected:
                selected = selected.get(basis, selected.get(basis.upper()))
            else:
                try:
                    selected = selected[check_index]
                except KeyError as exc:
                    raise ValueError(
                        f"syndrome schedule has no {basis.upper()} check "
                        f"{check_index}") from exc
        if callable(selected):
            try:
                parameters = inspect.signature(selected).parameters
            except (TypeError, ValueError):
                parameters = {}
            ordered = (selected(basis, check_index, support) if len(parameters)
                       >= 3 else selected(check_index, support))
        else:
            try:
                ordered = tuple(selected[check_index])
            except (TypeError, IndexError) as exc:
                raise TypeError(
                    "syndrome schedule must be a callable, basis mapping, "
                    "or per-check sequence") from exc
        try:
            ordered = tuple(ordered)
        except TypeError as exc:
            raise TypeError("scheduled check support must be iterable") from exc
        if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in ordered):
            raise TypeError("scheduled check support must contain Python ints")
        if len(ordered) != len(support) or set(ordered) != set(support):
            raise ValueError(
                f"syndrome schedule for {basis.upper()} check {check_index} "
                f"must permute canonical support {support}, got {ordered}")
        return ordered

    def extract_syndrome(
        self,
        patch_value,
        *,
        record=None,
        schedule=None,
        cx_schedule=None,
        prime=None,
        final_cycle=False,
    ):
        """Expand the standard CSS ancilla round into explicit Fabric gates.

        ``schedule`` reorders each check's canonical support without adding
        time layers. ``cx_schedule`` accepts the layered output of
        ``code.colored_schedule(...)`` and emits those CX layers in order. The
        two representations are intentionally mutually exclusive.
        """

        if schedule is not None and cx_schedule is not None:
            raise ValueError(
                "extract_syndrome accepts either schedule or cx_schedule, "
                "not both")

        if not isinstance(patch_value,
                          PatchValue) or patch_value.owner is not self:
            raise TypeError("extract_syndrome expects one live gadget patch")
        code = patch_value.encoding.code
        if isinstance(schedule, BBSyndromeSchedule):
            if not schedule._matches_code(code):
                raise InvalidSyndromeSchedule(
                    "BB depth-8 schedule belongs to a semantically different "
                    "code")
            if prime is None:
                raise InvalidSyndromeSchedule(
                    "BB depth-8 extraction requires prime=True for the first "
                    "cycle or prime=False when q(Z) was initialized by the "
                    "previous cycle")
            if not isinstance(prime, bool):
                raise TypeError("BB depth-8 prime= must be bool")
            if not isinstance(final_cycle, bool):
                raise TypeError("BB depth-8 final_cycle= must be bool")
            partitions = patch_value.carrier_block.partitions
            half = code.group.l * code.group.m
            if (partitions.get("data") != 2 * half or
                    partitions.get("sx") != half or
                    partitions.get("sz") != half):
                raise InvalidSyndromeSchedule(
                    "BB depth-8 extraction requires data=2*l*m and "
                    "sx=sz=l*m carrier partitions")
            validated_record = self._validate_record(record)
            continuation = patch_value._bb_syndrome_continuation
            pending = self._bb_pending_continuation
            if prime and (continuation is not None or pending is not None):
                raise InvalidSyndromeSchedule(
                    "prime=True cannot reinitialize a pending BB continuation; "
                    "the next cycle must use prime=False")
            # A rejected retry on a terminal or already-consumed patch must not
            # reserve an automatic record number. Check the patch-local linear
            # state before clearing a continuation or calling ``_record``;
            # owner-level continuation validation still runs transactionally
            # through the first authored operation below.
            patch_value._validate_consume_state("fabric.init_basis")
            if not prime:
                if (continuation is None or
                        not schedule._matches_code(continuation.code) or
                        pending is not continuation):
                    raise InvalidSyndromeSchedule(
                        "prime=False requires the immediate output of a "
                        "matching nonterminal BB depth-8 cycle")
                # Every user-controlled validation has completed. Consume the
                # local and builder-global proofs together before the first
                # operation of the continued cycle.
                patch_value._bb_syndrome_continuation = None
                self._bb_pending_continuation = None
            normalized_record = self._record(validated_record, "syndrome")
            return self._extract_bb_syndrome(
                patch_value,
                schedule=schedule,
                record=normalized_record,
                prime=prime,
                final_cycle=final_cycle,
            )
        if prime is not None:
            raise TypeError(
                "prime= is available only with cudaq.logical.BBSyndromeSchedule"
            )
        if final_cycle:
            raise TypeError(
                "final_cycle= is available only with cudaq.logical.BBSyndromeSchedule"
            )
        partitions = patch_value.carrier_block.partitions
        sx_width = partitions.get("sx", 0)
        sz_width = partitions.get("sz", 0)
        if sx_width != len(code.hx) or sz_width != len(code.hz):
            raise ValueError(
                "standard CSS extraction requires one sx ancilla per hx row "
                "and one sz ancilla per hz row; author an explicit extraction "
                "gadget for a different check or gauge schedule")
        if not sx_width and not sz_width:
            raise ValueError(
                "standard CSS extraction requires hx or hz checks; use "
                "measure_gauges/infer_syndrome or an explicit MPP gadget")

        hx_incidences = tuple(
            (check, data)
            for check, support in enumerate(code.hx)
            for data in self._extraction_support(schedule, "x", check, support))
        hz_incidences = tuple(
            (check, data)
            for check, support in enumerate(code.hz)
            for data in self._extraction_support(schedule, "z", check, support))
        x_layers, z_layers = self._resolve_cx_layers(cx_schedule, hx_incidences,
                                                     hz_incidences)

        current = patch_value
        if sx_width:
            current = self.apply_standard("reset", (current.sx,))[0]
            current = self.apply_standard("h", (current.sx,))[0]
            for layer in x_layers:
                current = self.apply_standard(
                    "cx",
                    (current.sx, current.data),
                    pairs=tuple((check, data) for check, data in layer),
                )[0]
            current = self.apply_standard("h", (current.sx,))[0]
        if sz_width:
            current = self.apply_standard("reset", (current.sz,))[0]
            for layer in z_layers:
                current = self.apply_standard(
                    "cx",
                    (current.data, current.sz),
                    pairs=tuple((data, check) for check, data in layer),
                )[0]
        return self.read_syndrome_ancillas(current, record=record)

    def _extract_bb_syndrome(
        self,
        patch_value,
        *,
        schedule,
        record,
        prime,
        final_cycle,
    ):
        current = patch_value
        if prime:
            current = self._init_basis(current.sz, basis="z")

        sx_bits = None
        sz_bits = None
        for moment in schedule.moments:
            if moment.initialize_x:
                current = self._init_basis(current.sx, basis="x")
            if moment.x_cx:
                current = self.apply_standard(
                    "cx",
                    (current.sx, current.data),
                    pairs=moment.x_cx,
                )[0]
            if moment.z_cx:
                current = self.apply_standard(
                    "cx",
                    (current.data, current.sz),
                    pairs=moment.z_cx,
                )[0]
            if moment.measure_z:
                current, sz_bits = self._measure_basis(current.sz,
                                                       basis="z",
                                                       record=f"{record}.sz")
            if moment.measure_x:
                current, sx_bits = self._measure_basis(current.sx,
                                                       basis="x",
                                                       record=f"{record}.sx")
            if moment.initialize_z and not final_cycle:
                current = self._init_basis(current.sz, basis="z")
        if sx_bits is None or sz_bits is None:
            raise InvalidSyndromeSchedule(
                "BB depth-8 schedule did not produce both syndrome bundles")
        return self._assemble_syndrome(
            current,
            sx_bits=sx_bits,
            sz_bits=sz_bits,
            record=record,
            terminal_cycle=final_cycle,
        )

    @staticmethod
    def _resolve_cx_layers(cx_schedule, hx_incidences, hz_incidences):
        """Normalize a schedule to per-basis layers of ``(check, data)`` pairs.

        ``None`` collapses to a single all-pairs layer per basis (the default,
        unscheduled emission). Otherwise ``cx_schedule`` is the plain
        ``(x_layers, z_layers)`` from ``code.colored_schedule(...)``; it must
        cover exactly the code's incidences, once each, with no qubit reused
        within a layer.
        """
        if cx_schedule is None:
            return (hx_incidences,), (hz_incidences,)
        raw_x, raw_z = cx_schedule[0], cx_schedule[1]
        x_layers = tuple(tuple(map(tuple, layer)) for layer in raw_x)
        z_layers = tuple(tuple(map(tuple, layer)) for layer in raw_z)
        for name, layers, incidences in (
            ("hx", x_layers, hx_incidences),
            ("hz", z_layers, hz_incidences),
        ):
            flat = [pair for layer in layers for pair in layer]
            if sorted(flat) != sorted(incidences):
                raise ValueError(
                    f"cx_schedule does not cover the {name} incidences exactly once"
                )
            for layer in layers:
                checks = [check for check, _ in layer]
                data = [data for _, data in layer]
                if len(set(checks)) != len(checks) or len(
                        set(data)) != len(data):
                    raise ValueError(
                        f"cx_schedule {name} layer reuses a qubit within one step"
                    )
        return x_layers, z_layers

    def read_syndrome_ancillas(self, patch_value, *, record=None):
        if not isinstance(patch_value,
                          PatchValue) or patch_value.owner is not self:
            raise TypeError(
                "read_syndrome_ancillas expects one live gadget patch")
        record = self._record(record, "syndrome")
        patch_value._consume("fabric.read_syndrome_ancillas")
        code = self.transaction.materialize(patch_value.encoding.code)
        encoding = self.transaction.materialize(patch_value.encoding)
        epoch = self.transaction.materialize(patch_value.epoch)
        syndrome_type = mlir_ir.Type.parse(
            f"!fabric.syndrome<@{code.symbol}, @{encoding.symbol}, "
            f"@{epoch.symbol}>",
            context=self.context,
        )
        operation = self._emit(
            "fabric.read_syndrome_ancillas",
            operands=[patch_value.mlir_value],
            results=[patch_value.type, syndrome_type],
            attributes={
                "record": mlir_ir.StringAttr.get(record, context=self.context)
            },
        )
        self._register_record_family(
            record,
            "s",
            self._syndrome_record_width(patch_value.encoding),
        )
        return (
            self._new_patch(operation.results[0],
                            patch_value.encoding,
                            epoch=patch_value.epoch),
            SyndromeValue(
                operation.results[1],
                owner=self,
                encoding=patch_value.encoding,
                record=record,
                location=self.location,
            ),
        )

    def _assemble_syndrome(
        self,
        patch_value,
        *,
        sx_bits,
        sz_bits,
        record,
        terminal_cycle=False,
    ):
        if not isinstance(patch_value,
                          PatchValue) or patch_value.owner is not self:
            raise TypeError("assemble_syndrome expects one live gadget patch")
        if (not isinstance(sx_bits, MeasurementBits) or
                sx_bits.owner is not self or
                not isinstance(sz_bits, MeasurementBits) or
                sz_bits.owner is not self):
            raise TypeError(
                "assemble_syndrome expects sx/sz measurement bundles from "
                "the active gadget")
        patch_value._consume("fabric.assemble_syndrome")
        code = self.transaction.materialize(patch_value.encoding.code)
        encoding = self.transaction.materialize(patch_value.encoding)
        epoch = self.transaction.materialize(patch_value.epoch)
        syndrome_type = mlir_ir.Type.parse(
            f"!fabric.syndrome<@{code.symbol}, @{encoding.symbol}, "
            f"@{epoch.symbol}>",
            context=self.context,
        )
        operation = self._emit(
            "fabric.assemble_syndrome",
            operands=[
                patch_value.mlir_value,
                sx_bits.mlir_value,
                sz_bits.mlir_value,
            ],
            results=[patch_value.type, syndrome_type],
            attributes={
                "record": mlir_ir.StringAttr.get(record, context=self.context)
            },
        )
        self._register_record_family(
            record,
            "s",
            self._syndrome_record_width(patch_value.encoding),
        )
        continuation = (None if terminal_cycle else _BBSyndromeContinuation(
            patch_value.encoding.code))
        patch_out = self._new_patch(
            operation.results[0],
            patch_value.encoding,
            epoch=patch_value.epoch,
            bb_syndrome_continuation=continuation,
            terminal_only_reason=(
                "a BB extraction with final_cycle=True omits the "
                "next-cycle ancilla initialization"
                if terminal_cycle else None),
        )
        if continuation is not None:
            self._bb_pending_continuation = continuation
        return (
            patch_out,
            SyndromeValue(
                operation.results[1],
                owner=self,
                encoding=patch_value.encoding,
                record=record,
                location=self.location,
            ),
        )

    def _gf2_matrix_attr(self, matrix):
        if not isinstance(matrix, GF2Matrix):
            raise TypeError("expected a cudaq.logical.GF2Matrix")
        literal = str([list(row) for row in matrix.rows]).replace(" ", "")
        return mlir_ir.Attribute.parse(
            f"dense<{literal}> : tensor<{matrix.nrows}x{matrix.ncols}xi1>",
            context=self.context,
        )

    def measure_gauges(self,
                       patch_value,
                       *,
                       operators,
                       record=None,
                       phase=None):
        if not isinstance(patch_value,
                          PatchValue) or patch_value.owner is not self:
            raise TypeError("measure_gauges expects one live gadget patch")
        if isinstance(operators, GaugeMeasurementMap):
            operator_matrix = operators.operators
            stabilizer_map = operators.stabilizer_map
        elif isinstance(operators, GF2Matrix) and phase is not None:
            operator_matrix = operators
            stabilizer_map = None
        else:
            raise TypeError(
                "operators= must be GaugeMeasurementMap, or GF2Matrix with "
                "an explicit dynamic-code phase=")
        if operator_matrix.ncols != 2 * patch_value.encoding.code.n:
            raise ValueError("gauge operators must have symplectic width 2n")
        if phase is not None and (not isinstance(phase, str) or not phase):
            raise TypeError("phase= must be a nonempty string")
        self._before_operation("fabric.measure_gauges")
        code = self.transaction.materialize(patch_value.encoding.code)
        encoding = self.transaction.materialize(patch_value.encoding)
        epoch = self.transaction.materialize(patch_value.epoch)
        records_type = mlir_ir.Type.parse(
            f"!fabric.gauge_records<@{code.symbol}, @{encoding.symbol}, "
            f"@{epoch.symbol}>",
            context=self.context,
        )
        record = self._record(record, "gauge")
        attrs = {
            "operators": self._gf2_matrix_attr(operator_matrix),
            "record": mlir_ir.StringAttr.get(record, context=self.context),
        }
        if stabilizer_map is not None:
            attrs["stabilizer_map"] = self._gf2_matrix_attr(stabilizer_map)
        if phase is not None:
            attrs["phase"] = mlir_ir.StringAttr.get(phase, context=self.context)
        patch_value._consume("fabric.measure_gauges")
        operation = self._emit(
            "fabric.measure_gauges",
            operands=[patch_value.mlir_value],
            results=[patch_value.type, records_type],
            attributes=attrs,
        )
        self._register_record_family(record, "g", operator_matrix.nrows)
        if self._derive_subsystem_fragment:
            if phase is None:
                raise TypeError(
                    "subsystem-fragment gauge measurement requires a typed "
                    "dynamic phase")
            self._subsystem_steps.append({
                "operation": "measure_gauges",
                "operators": operator_matrix,
                "phase": phase,
                "record": record,
                "input_epoch": patch_value.epoch.phase,
            })
        return (
            self._new_patch(operation.results[0],
                            patch_value.encoding,
                            epoch=patch_value.epoch),
            GaugeRecordsValue(
                operation.results[1],
                owner=self,
                encoding=patch_value.encoding,
                epoch=patch_value.epoch,
                record=record,
                measurement_map=operators,
                location=self.location,
            ),
        )

    def transition_epoch(self, patch_value, *, to, evidence, logical_map=None):
        if not isinstance(patch_value,
                          PatchValue) or patch_value.owner is not self:
            raise TypeError("transition_epoch expects one live gadget patch")
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
        self._before_operation("fabric.epoch_transition")
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
        if self._derive_subsystem_fragment:
            step = {
                "operation": "epoch_transition",
                "from_epoch": source_phase,
                "to_epoch": to.phase,
                "logical_map": logical_map,
                "evidence": evidence,
            }
            if to.phase == patch_value.encoding.epoch_schema.initial:
                step[
                    "period_closure"] = patch_value.encoding.profile.period_closure
            self._subsystem_steps.append(step)
        return self._new_patch(operation.result, patch_value.encoding, epoch=to)

    def xor(self, lhs, rhs):
        if any(not isinstance(value, LogicalBool) or value.owner is not self
               for value in (lhs, rhs)):
            raise TypeError(
                "qlx.xor expects two Boolean values from this gadget")
        operation = self._emit(
            "fabric.xor",
            operands=[lhs.mlir_value, rhs.mlir_value],
            results=[self.i1_type],
        )
        producer = None
        if isinstance(lhs.producer, ProfileParity) and isinstance(
                rhs.producer, ProfileParity):
            producer = lhs.producer ^ rhs.producer
        return LogicalBool(operation.result, owner=self, producer=producer)

    def all_zero(self, bits):
        if not isinstance(bits, MeasurementBits) or bits.owner is not self:
            raise TypeError(
                "qlx.all_zero expects measurement bits from this gadget")
        operation = self._emit(
            "fabric.all_zero",
            operands=[bits.mlir_value],
            results=[self.i1_type],
        )
        producer = self._measurement_parity(bits)
        if producer is not None and len(producer.records) == 1:
            producer = producer ^ True
        else:
            producer = None
        return LogicalBool(operation.result, owner=self, producer=producer)

    def measure_pauli(self, selection, *, paulis, record=None):
        owner, partition, indices = self._view(selection)
        if owner.owner is not self:
            raise TypeError("qlx.measure_pauli expects a view from this gadget")
        if indices is None:
            indices = tuple(range(self._partition_width(owner, partition)))
        else:
            indices = tuple(indices)
        if len(set(indices)) != len(indices):
            raise ValueError(
                "measure_pauli requires distinct physical carrier indices")
        if not isinstance(paulis, str):
            raise TypeError(
                "measure_pauli paulis must match the selected width")
        paulis = paulis.upper()
        # One optional leading '-' measures the negated product: identical
        # projectors and a complemented record (the inverted-target MPP form
        # used by Stim).
        labels = paulis.removeprefix("-")
        if len(labels) != len(indices):
            raise TypeError(
                "measure_pauli paulis must match the selected width")
        if not labels or any(value not in "XYZ" for value in labels):
            raise ValueError(
                "measure_pauli paulis must contain only X, Y, and Z")
        record_name = self._record(record, "mpp")
        self._register_record(f"{record_name}.outcome")
        owner._consume("fabric.mpp")
        with self.context:
            bits_type = mlir_ir.RankedTensorType.get((1,),
                                                     self.i1_type,
                                                     loc=self.location)
        operation = self._emit(
            "fabric.mpp",
            operands=[owner.mlir_value],
            results=[owner.type, bits_type],
            attributes={
                "partition":
                    self._partition_attr(partition),
                "indices":
                    mlir_ir.DenseI64ArrayAttr.get(indices, self.context),
                "paulis":
                    mlir_ir.StringAttr.get(paulis, context=self.context),
                "record":
                    mlir_ir.StringAttr.get(record_name, context=self.context),
            },
        )
        return (
            self._new_patch(
                operation.results[0],
                owner.encoding,
                epoch=owner.epoch,
                carrier_frame=owner.carrier_frame,
            ),
            MeasurementBits(operation.results[1],
                            owner=self,
                            record=record_name),
        )

    def parity(self, *bits):
        if not bits or any(not isinstance(value, MeasurementBits) or
                           value.owner is not self for value in bits):
            raise TypeError(
                "qlx.parity expects one or more measurement bundles from this gadget"
            )
        operation = self._emit(
            "fabric.parity",
            operands=[value.mlir_value for value in bits],
            results=[self.i1_type],
        )
        producer = ProfileParity()
        for value in bits:
            parity = self._measurement_parity(value)
            if parity is None:
                producer = None
                break
            producer = producer ^ parity
        return LogicalBool(operation.result, owner=self, producer=producer)

    def all_false(self, *events):
        if not events or any(
                not isinstance(value, LogicalBool) or value.owner is not self
                for value in events):
            raise TypeError(
                "qlx.all_false expects one or more Boolean events from this gadget"
            )
        operation = self._emit(
            "fabric.all_false",
            operands=[value.mlir_value for value in events],
            results=[self.i1_type],
        )
        producer = events[0].producer if len(events) == 1 else None
        if isinstance(producer, ProfileParity):
            producer = producer ^ True
        return LogicalBool(operation.result, owner=self, producer=producer)

    @staticmethod
    def _mask(row):
        result = 0
        for index in row:
            result ^= 1 << index
        return result

    @classmethod
    def _span(cls, rows):
        span = {0}
        for row in rows:
            mask = cls._mask(row)
            span |= {value ^ mask for value in tuple(span)}
        return span

    @classmethod
    def _logical_coordinates(cls, support, stabilizers, logicals):
        """Reduce one support modulo stabilizers onto an ordered logical basis."""

        rows = tuple(stabilizers) + tuple(logicals)
        width = len(rows)
        pivots = {}
        for index, row in enumerate(rows):
            value = cls._mask(row)
            coefficients = 1 << index
            while value:
                pivot = value.bit_length() - 1
                if pivot not in pivots:
                    pivots[pivot] = (value, coefficients)
                    break
                basis, basis_coefficients = pivots[pivot]
                value ^= basis
                coefficients ^= basis_coefficients
        value = cls._mask(support)
        coefficients = 0
        while value:
            pivot = value.bit_length() - 1
            if pivot not in pivots:
                raise ValueError(
                    "permuted logical support left the code normalizer")
            basis, basis_coefficients = pivots[pivot]
            value ^= basis
            coefficients ^= basis_coefficients
        offset = len(stabilizers)
        return tuple(logical for logical in range(len(logicals))
                     if coefficients & (1 << (offset + logical)))

    def permute(self, patch_value, permutation):
        if not isinstance(patch_value,
                          PatchValue) or patch_value.owner is not self:
            raise TypeError(
                "cudaq.logical.permute expects one live encoded patch")
        code = patch_value.encoding.code
        permutation = tuple(permutation)
        if len(permutation) != code.n:
            raise ValueError(
                f"permutation for {code.name} must contain exactly {code.n} entries"
            )
        if any(not isinstance(index, int) or isinstance(index, bool)
               for index in permutation):
            raise TypeError("permutation entries must be Python ints")
        if set(permutation) != set(range(code.n)):
            raise ValueError(
                "permutation must contain every carrier index exactly once")

        transform = lambda row: tuple(
            sorted(permutation[index] for index in row))
        for basis, rows in (("X", code.hx), ("Z", code.hz)):
            span = self._span(rows)
            if any(self._mask(transform(row)) not in span for row in rows):
                raise ValueError(
                    f"permutation does not preserve the {basis}-stabilizer group"
                )
        for basis, stabilizers, gauges in (
            ("X", code.hx, code.gx),
            ("Z", code.hz, code.gz),
        ):
            span = self._span((*stabilizers, *gauges))
            if any(self._mask(transform(row)) not in span for row in gauges):
                raise ValueError(
                    f"permutation does not preserve the {basis}-gauge group")

        x_basis = (*code.lx, *code.gx)
        z_basis = (*code.lz, *code.gz)
        x_action = tuple(
            self._logical_coordinates(transform(row), code.hx, x_basis)
            for row in x_basis)
        z_action = tuple(
            self._logical_coordinates(transform(row), code.hz, z_basis)
            for row in z_basis)
        self._before_operation("fabric.permute")
        binding = self._match_automorphism_action(patch_value.encoding,
                                                  x_action, z_action)
        patch_value._consume("fabric.permute")
        operation = self._emit(
            "fabric.permute",
            operands=[patch_value.mlir_value],
            results=[patch_value.type],
            attributes={
                "perm":
                    mlir_ir.DenseI64ArrayAttr.get(permutation,
                                                  context=self.context),
                "logical_x_action":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.DenseI64ArrayAttr.get(row,
                                                          context=self.context)
                            for row in x_action
                        ],
                        context=self.context,
                    ),
                "logical_z_action":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.DenseI64ArrayAttr.get(row,
                                                          context=self.context)
                            for row in z_action
                        ],
                        context=self.context,
                    ),
                "protected_logicals":
                    mlir_ir.IntegerAttr.get(
                        mlir_ir.IntegerType.get_signless(64,
                                                         context=self.context),
                        code.k,
                    ),
                "gauge_qubits":
                    mlir_ir.IntegerAttr.get(
                        mlir_ir.IntegerType.get_signless(64,
                                                         context=self.context),
                        code.r,
                    ),
                "derivation":
                    mlir_ir.StringAttr.get("verified_code_automorphism",
                                           context=self.context),
                "objective_binding":
                    mlir_ir.DictAttr.get(
                        {
                            name:
                                mlir_ir.StringAttr.get(port,
                                                       context=self.context)
                            for name, port in binding.items()
                        },
                        context=self.context,
                    ),
            },
        )
        self._automorphism = (x_action, z_action, binding)
        return self._new_patch(operation.result,
                               patch_value.encoding,
                               epoch=patch_value.epoch)

    @staticmethod
    def _embed_mask(mask, mapping):
        result = 0
        for objective_port, code_port in enumerate(mapping):
            if mask & (1 << objective_port):
                result |= 1 << code_port
        return result

    def _match_automorphism_action(self, encoding, x_action, z_action):
        code = encoding.code
        arity = len(self.objective_parameter_names)
        if self.expected_clifford_action is None:
            if isinstance(self.logical_objective, LogicalActionRef):
                self.expected_clifford_action = self._standard_clifford_action(
                    self.logical_objective.name, self.logical_objective.arity)
            elif (isinstance(self.logical_objective, ProgramDefinition) and
                  self.transaction.materialize(
                      self.logical_objective).kind == "action"):
                self.expected_clifford_action = self._program_clifford_action(
                    self.logical_objective_symbol, arity)
            else:
                raise TypeError(
                    "a code-automorphism realization requires a Clifford "
                    "action-like @cudaq.logical.objective")
        if arity > code.k:
            raise ValueError(
                "logical objective has more operands than the encoding exposes")
        requested = dict(self.definition.logical_ports)
        unknown = set(requested) - set(self.objective_parameter_names)
        if unknown:
            raise ValueError(
                f"logical_ports contains unknown objective operands: {sorted(unknown)}"
            )
        constrained = {}
        for operand, port_name in requested.items():
            leaf_name = str(port_name).split(".")[-1]
            try:
                constrained[self.objective_parameter_names.index(operand)] = (
                    encoding.logical_port_indices[leaf_name])
            except KeyError as exc:
                raise ValueError(
                    f"encoding {encoding.name!r} has no logical port {leaf_name!r}"
                ) from exc

        physical = (
            *((sum(1 << item for item in row), 0, 0) for row in x_action),
            *((0, sum(1 << item for item in row), 0) for row in z_action),
        )

        def matches(mapping):
            if any(mapping[index] != port
                   for index, port in constrained.items()):
                return False
            objective_for_code = {
                code_port: objective
                for objective, code_port in enumerate(mapping)
            }
            for code_port in range(code.k):
                objective = objective_for_code.get(code_port)
                for basis in range(2):
                    actual = physical[basis * (code.k + code.r) + code_port]
                    if objective is None:
                        expected = (
                            1 << code_port if basis == 0 else 0,
                            1 << code_port if basis == 1 else 0,
                            0,
                        )
                    else:
                        source = self.expected_clifford_action[basis * arity +
                                                               objective]
                        expected = (
                            self._embed_mask(source[0], mapping),
                            self._embed_mask(source[1], mapping),
                            source[2],
                        )
                    if actual != expected:
                        return False
            # A protected objective cannot be hidden in a gauge output.
            protected_mask = (1 << code.k) - 1
            for gauge_index in range(code.k, code.k + code.r):
                for basis in range(2):
                    actual = physical[basis * (code.k + code.r) + gauge_index]
                    if actual[0] & protected_mask or actual[1] & protected_mask:
                        return False
            return True

        candidates = tuple(
            mapping for mapping in permutations(range(code.k), arity)
            if matches(mapping))
        if not candidates:
            raise ValueError(
                "code automorphism logical action does not implement the "
                "declared objective under any logical-port binding")
        if len(candidates) != 1:
            from ..errors import AmbiguousLogicalPortMap

            raise AmbiguousLogicalPortMap(
                "code automorphism matches the "
                "declared objective under multiple bindings; specify only "
                "logical_ports={objective.operands.name: encoding.ports.name}")
        mapping = candidates[0]
        binding = {
            name: encoding.logical_ports[mapping[index]]
            for index, name in enumerate(self.objective_parameter_names)
        }
        self.spec_operation.attributes["logical_ports"] = mlir_ir.DictAttr.get(
            {
                key: mlir_ir.StringAttr.get(value, context=self.context)
                for key, value in binding.items()
            },
            context=self.context,
        )
        self.spec_operation.attributes[
            "action_equivalence"] = mlir_ir.StringAttr.get(
                "derived_exact_signed_symplectic_match", context=self.context)
        self.spec_operation.attributes["epoch_map"] = mlir_ir.StringAttr.get(
            "preserve", context=self.context)
        return binding

    def _product_terms(self, product, operation):
        from ..algebra.pauli import PauliProduct

        if not isinstance(product, PauliProduct):
            raise TypeError(f"{operation} expects a cudaq.logical.PauliProduct")
        patches = []
        patch_positions = {}
        patch_indices = []
        logical_indices = []
        subsystem_kinds = []
        subsystem_indices = []
        paulis = []
        for factor in product.factors:
            reference = factor.operand
            if not isinstance(reference, (PatchLogicalRef, PatchGaugeRef)):
                raise TypeError(
                    f"{operation} factors must reference patch[i] or patch.gauge[i]"
                )
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
            subsystem_kinds.append("gauge" if isinstance(
                reference, PatchGaugeRef) else "protected")
            subsystem_indices.append(reference.index)
            paulis.append(factor.pauli)
        for patch in patches:
            patch._consume(operation)
        attrs = {
            "patch_indices":
                mlir_ir.DenseI64ArrayAttr.get(patch_indices, self.context),
            "logical_indices":
                mlir_ir.DenseI64ArrayAttr.get(logical_indices, self.context),
            "subsystem_kinds":
                mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.StringAttr.get(value, context=self.context)
                        for value in subsystem_kinds
                    ],
                    context=self.context,
                ),
            "subsystem_indices":
                mlir_ir.DenseI64ArrayAttr.get(subsystem_indices, self.context),
            "pauli_product":
                mlir_ir.StringAttr.get(
                    ("-" if product.sign < 0 else "") + "".join(paulis),
                    context=self.context,
                ),
        }
        semantics = {
            "patch_indices": tuple(patch_indices),
            "port_kinds": tuple(subsystem_kinds),
            "port_indices": tuple(subsystem_indices),
            "paulis": "".join(paulis),
            "sign": product.sign,
        }
        return patches, attrs, semantics

    def mpp(self, product):
        patches, attrs, semantics = self._product_terms(
            product, "fabric.measure_product")
        record = self._record(None, "mpp")
        self._register_record(f"{record}.outcome")
        attrs["record"] = mlir_ir.StringAttr.get(record, context=self.context)
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
        outcome = LogicalBool(
            operation.results[-1],
            owner=self,
            producer=ProfileParity(
                records=(RecordRef(self.definition, f"{record}.outcome"),)),
        )
        if self._derive_subsystem_fragment:
            step_index = len(self._subsystem_steps)
            self._subsystem_steps.append({
                "operation": "measure",
                "record": record,
                **semantics
            })
            self._subsystem_outcomes[id(outcome)] = step_index
        return (*successors, outcome)

    def rotate(self, product, *, angle):
        patches, attrs, semantics = self._product_terms(
            product, "fabric.rotate_product")
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
        if self._derive_subsystem_fragment:
            self._subsystem_steps.append({
                "operation": "rotate",
                "angle": float(angle),
                **semantics
            })
        return [
            self._new_patch(result, patch.encoding, epoch=patch.epoch)
            for result, patch in zip(operation.results, patches)
        ]

    def resource_rotate(self, resource_value, product, *, angle):
        if (not isinstance(resource_value, ResourceValue) or
                resource_value.owner is not self):
            raise TypeError("resource_rotate expects one live resource owner")
        patches, attrs, _ = self._product_terms(
            product, "fabric.resource_rotate_product")
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
                *(patch.mlir_value for patch in patches)
            ],
            results=[patch.type for patch in patches],
            attributes=attrs,
        )
        return [
            self._new_patch(result, patch.encoding, epoch=patch.epoch)
            for result, patch in zip(operation.results, patches)
        ]
