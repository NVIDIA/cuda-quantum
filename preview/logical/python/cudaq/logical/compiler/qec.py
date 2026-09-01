# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from typing import Any, Mapping

from .. import ir as mlir_ir

from ..codes import (
    Code,
    CodeProfile,
    Encoding,
    EncodingEpoch,
    EncodingEpochSchema,
    EncodingHierarchy,
    EncodingProjection,
    PatchTransform,
)
from ..programs.definition import DefinitionHandle
from ..algebra.gf2 import GF2Matrix
from ..codes import _materialized_code_metadata


def _i64(context, value: int):
    return mlir_ir.IntegerAttr.get(
        mlir_ir.IntegerType.get_signless(64, context=context), value)


def _rows(context, rows):
    return mlir_ir.ArrayAttr.get(
        [mlir_ir.DenseI64ArrayAttr.get(list(row), context) for row in rows],
        context=context,
    )


def _gf2_matrix(context, matrix):
    literal = ("" if matrix.nrows == 0 or matrix.ncols == 0 else str(
        [list(row) for row in matrix.rows]).replace(" ", ""))
    return mlir_ir.Attribute.parse(
        f"dense<{literal}> : tensor<{matrix.nrows}x{matrix.ncols}xi1>",
        context=context,
    )


def _distance_metadata(evidence, prefix: str = "distance") -> dict[str, Any]:
    """Flatten typed distance evidence into serializable metadata entries."""

    entries: dict[str, Any] = {f"{prefix}_status": evidence.status}
    if evidence.reason:
        entries[f"{prefix}_reason"] = evidence.reason
    if evidence.method:
        entries[f"{prefix}_method"] = evidence.method
    if evidence.provenance is not None:
        entries[f"{prefix}_provenance"] = str(evidence.provenance)
    if evidence.certificate is not None:
        entries[f"{prefix}_certificate"] = str(evidence.certificate)
    if evidence.scope:
        entries[f"{prefix}_scope"] = evidence.scope
    if evidence.status == "asymmetric":
        entries.update(_distance_metadata(evidence.x, f"{prefix}_x"))
        entries.update(_distance_metadata(evidence.z, f"{prefix}_z"))
        if evidence.x.value is not None:
            entries[f"{prefix}_x_value"] = evidence.x.value
        if evidence.z.value is not None:
            entries[f"{prefix}_z_value"] = evidence.z.value
    return entries


def _metadata(context, values: Mapping[str, Any]):
    attrs = {}
    for key, value in values.items():
        if isinstance(value, Mapping):
            attrs[str(key)] = _metadata(context, value)
        elif isinstance(value, (tuple, list)):
            attrs[str(key)] = mlir_ir.ArrayAttr.get(
                [_metadata_value(context, item) for item in value],
                context=context,
            )
        else:
            attrs[str(key)] = _metadata_value(context, value)
    return mlir_ir.DictAttr.get(attrs, context=context)


def _metadata_value(context, value):
    if isinstance(value, Mapping):
        return _metadata(context, value)
    if isinstance(value, (tuple, list)):
        return mlir_ir.ArrayAttr.get(
            [_metadata_value(context, item) for item in value],
            context=context,
        )
    if isinstance(value, bool):
        return mlir_ir.BoolAttr.get(value, context=context)
    if isinstance(value, int):
        return _i64(context, value)
    if isinstance(value, float):
        return mlir_ir.FloatAttr.get(mlir_ir.F64Type.get(context=context),
                                     value)
    return mlir_ir.StringAttr.get(str(value), context=context)


def _append_profile(transaction, profile: str) -> None:
    existing = transaction.module.operation.attributes["qlx.profiles"]
    values = [str(getattr(item, "value", item)).strip('"') for item in existing]
    if profile not in values:
        values.append(profile)
    transaction.module.operation.attributes[
        "qlx.profiles"] = mlir_ir.ArrayAttr.get(
            [
                mlir_ir.StringAttr.get(item, context=transaction.context)
                for item in values
            ],
            context=transaction.context,
        )


def _insert(transaction, name: str, attributes):
    with transaction.location:
        operation = mlir_ir.Operation.create(name,
                                             attributes=attributes,
                                             loc=transaction.location)
        transaction.module.body.append(operation)
    return operation


def _operation_symbol(operation) -> str:
    attribute = operation.attributes["sym_name"]
    return str(getattr(attribute, "value", attribute)).strip('"')


def _equivalent_declaration(
        transaction,
        operation_name: str,
        attributes,
        *,
        ignored=("sym_name",),
):
    """Find a transaction-wide declaration with the same semantic payload.

    Authored symbol names are labels, not semantic identity. Dependencies in
    ``attributes`` have already been materialized to their canonical handles,
    so comparing every non-identity attribute implements structural closure
    coalescing even after an earlier same-name collision allocated a suffix.
    """

    ignored = frozenset(ignored)
    expected_names = tuple(
        sorted(name for name in attributes if name not in ignored))
    for operation in transaction.walk():
        if operation.name != operation_name:
            continue
        names = tuple(
            sorted(
                name for name in operation.attributes if name not in ignored))
        if names != expected_names:
            continue
        if all(
                str(operation.attributes[name]) == str(attributes[name])
                for name in expected_names):
            return operation
    return None


def materialize_qec(transaction, definition):
    existing = transaction.lookup(definition)
    if existing is not None:
        return existing
    # A symbol name is an authored label, not semantic identity: independently
    # constructed codes may share a default name while carrying different
    # algebra (and therefore different extraction schedules). Exact
    # transaction-wide declaration payloads coalesce; ``unique_symbol``
    # preserves every structurally distinct definition.
    _append_profile(transaction, "p2s")
    context = transaction.context

    if isinstance(definition, EncodingEpochSchema):
        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(definition.name, context=context),
            "phases":
                mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.StringAttr.get(phase, context=context)
                        for phase in definition.phases
                    ],
                    context=context,
                ),
            "initial":
                mlir_ir.StringAttr.get(definition.initial, context=context),
            "transitions":
                mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.StringAttr.get(f"{source}->{target}",
                                               context=context)
                        for source, target in definition.transitions
                    ],
                    context=context,
                ),
        }
        if definition.logical_maps:
            attrs["logical_maps"] = _metadata(context, definition.logical_maps)
        if definition.is_periodic:
            attrs["periodic"] = mlir_ir.UnitAttr.get(context=context)
        if definition.closure is not None:
            attrs["closure"] = _gf2_matrix(context, definition.closure)
        equivalent = _equivalent_declaration(
            transaction,
            "fabric.encoding_epoch_schema",
            attrs,
        )
        if equivalent is not None:
            symbol = _operation_symbol(equivalent)
            handle = DefinitionHandle(symbol, EncodingEpochSchema, "p2s")
            transaction.bind(definition, handle)
            return handle
        symbol = transaction.unique_symbol(definition.name)
        attrs["sym_name"] = mlir_ir.StringAttr.get(symbol, context=context)
        _insert(transaction, "fabric.encoding_epoch_schema", attrs)
        handle = DefinitionHandle(symbol, EncodingEpochSchema, "p2s")
        transaction.bind(definition, handle)
        return handle

    if isinstance(definition, EncodingEpoch):
        encoding = materialize_qec(transaction, definition.encoding)
        existing = transaction.lookup(definition)
        if existing is not None:
            return existing
        schema = materialize_qec(transaction, definition.schema)
        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(definition.name, context=context),
            "encoding":
                mlir_ir.FlatSymbolRefAttr.get(encoding.symbol, context=context),
            "schema":
                mlir_ir.FlatSymbolRefAttr.get(schema.symbol, context=context),
            "phase":
                mlir_ir.StringAttr.get(definition.phase, context=context),
            "index":
                _i64(context, definition.index),
        }
        equivalent = _equivalent_declaration(
            transaction,
            "fabric.encoding_epoch",
            attrs,
        )
        if equivalent is not None:
            symbol = _operation_symbol(equivalent)
            handle = DefinitionHandle(symbol, EncodingEpoch, "p2s")
            transaction.bind(definition, handle)
            return handle
        symbol = transaction.unique_symbol(definition.name)
        attrs["sym_name"] = mlir_ir.StringAttr.get(symbol, context=context)
        _insert(transaction, "fabric.encoding_epoch", attrs)
        handle = DefinitionHandle(symbol, EncodingEpoch, "p2s")
        transaction.bind(definition, handle)
        return handle

    if isinstance(definition, Code):
        partitions = mlir_ir.DictAttr.get(
            {
                name: _i64(context, count)
                for name, count in definition.block.partitions.items()
            },
            context=context,
        )
        attrs: dict[str, Any] = {
            "sym_name":
                mlir_ir.StringAttr.get(definition.name, context=context),
            "distance":
                _i64(context, definition.d.conservative_value or 0),
            "partitions":
                partitions,
            "n":
                _i64(context, definition.n),
            "k":
                _i64(context, definition.k),
            "r":
                _i64(context, definition.r),
        }
        for name in ("hx", "hz", "gx", "gz", "lx", "lz"):
            rows = getattr(definition, name)
            if not rows:
                continue
            attrs[name] = _rows(context, rows)
        if definition.stabilizers:
            attrs["stabilizers"] = mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(str(item), context=context)
                    for item in definition.stabilizers
                ],
                context=context,
            )
        if definition.gauges:
            attrs["gauges"] = mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(str(item), context=context)
                    for item in definition.gauges
                ],
                context=context,
            )
        attrs["stabilizer_basis"] = _gf2_matrix(context,
                                                definition.stabilizer_basis)
        attrs["logical_x_basis"] = _gf2_matrix(context,
                                               definition.logical_x_basis)
        attrs["logical_z_basis"] = _gf2_matrix(context,
                                               definition.logical_z_basis)
        attrs["gauge_x_basis"] = _gf2_matrix(context, definition.gauge_x_basis)
        attrs["gauge_z_basis"] = _gf2_matrix(context, definition.gauge_z_basis)
        attrs["anti_stabilizers"] = _gf2_matrix(context,
                                                definition.anti_stabilizers)
        attrs["encoding_clifford"] = _gf2_matrix(context,
                                                 definition.encoding_clifford)
        attrs["metadata"] = _metadata(
            context,
            _materialized_code_metadata(definition),
        )
        equivalent = _equivalent_declaration(transaction, "fabric.code", attrs)
        if equivalent is not None:
            handle = DefinitionHandle(_operation_symbol(equivalent), Code,
                                      "p2s")
            transaction.bind(definition, handle)
            return handle
        code_symbol = transaction.unique_symbol(definition.name)
        attrs["sym_name"] = mlir_ir.StringAttr.get(code_symbol, context=context)
        _insert(transaction, "fabric.code", attrs)
        handle = DefinitionHandle(code_symbol, Code, "p2s")
        transaction.bind(definition, handle)

        # Every concrete code receives one generated default profile and
        # encoding. They are ordinary symbols and can be superseded by explicit
        # non-default views without changing code algebra.
        materialize_qec(transaction, definition.default_profile)
        materialize_qec(transaction, definition.default_encoding)
        return handle

    if isinstance(definition, CodeProfile):
        code_handle = materialize_qec(transaction, definition.code)
        existing = transaction.lookup(definition)
        if existing is not None:
            return existing
        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(definition.name, context=context),
            "code":
                mlir_ir.FlatSymbolRefAttr.get(code_handle.symbol,
                                              context=context),
            "distance_status":
                mlir_ir.StringAttr.get(definition.distance.status,
                                       context=context),
        }
        if definition.distance.conservative_value is not None:
            attrs["distance_claim"] = _i64(
                context, definition.distance.conservative_value)
        attrs["effective_stabilizers"] = _gf2_matrix(
            context, definition.effective_stabilizers)
        attrs["decomposition"] = _gf2_matrix(context, definition.decomposition)
        attrs["effective_metachecks"] = _gf2_matrix(
            context, definition.effective_metachecks)
        attrs["kept_from_effective"] = _gf2_matrix(
            context, definition.kept_from_effective)
        if definition.metachecks:
            families = {}
            if definition.metachecks.x is not None:
                families["x"] = _gf2_matrix(context, definition.metachecks.x)
            if definition.metachecks.z is not None:
                families["z"] = _gf2_matrix(context, definition.metachecks.z)
            if definition.metachecks.gauge is not None:
                families["gauge"] = _gf2_matrix(context,
                                                definition.metachecks.gauge)
            attrs["metachecks"] = mlir_ir.DictAttr.get(
                families,
                context=context,
            )
        if definition.gauge_measurements is not None:
            attrs["gauge_measurements"] = mlir_ir.DictAttr.get(
                {
                    "operators":
                        _gf2_matrix(context,
                                    definition.gauge_measurements.operators),
                    "stabilizer_map":
                        _gf2_matrix(
                            context,
                            definition.gauge_measurements.stabilizer_map),
                },
                context=context,
            )
        if definition.dynamic_phases:
            phases = []
            for phase, action in zip(definition.dynamic_phases,
                                     definition.transition_actions):
                values = {
                    "name":
                        mlir_ir.StringAttr.get(phase.name, context=context),
                    "measured_gauges":
                        _gf2_matrix(context, phase.measured_gauges),
                    "instantaneous_stabilizers":
                        _gf2_matrix(context, phase.instantaneous_stabilizers),
                }
                if phase.temporal_recovery is not None:
                    values["temporal_recovery"] = _gf2_matrix(
                        context, phase.temporal_recovery)
                if phase.input_epoch is not None:
                    values["input_epoch"] = mlir_ir.StringAttr.get(
                        phase.input_epoch, context=context)
                if phase.output_epoch is not None:
                    values["output_epoch"] = mlir_ir.StringAttr.get(
                        phase.output_epoch, context=context)
                if phase.logical_map:
                    values["logical_map"] = _metadata(context,
                                                      phase.logical_map)
                values["logical_action"] = _gf2_matrix(
                    context, GF2Matrix.from_rows(action.matrix))
                phases.append(mlir_ir.DictAttr.get(values, context=context))
            attrs["dynamic_phases"] = mlir_ir.ArrayAttr.get(phases,
                                                            context=context)
        if definition.temporal_recovery is not None:
            attrs["temporal_recovery"] = _gf2_matrix(
                context, definition.temporal_recovery)
            attrs["temporal_recovery_targets"] = _gf2_matrix(
                context, definition.temporal_recovery_targets)
        if definition.record_logicals is not None:
            attrs["record_logicals"] = mlir_ir.DictAttr.get(
                {
                    "names":
                        mlir_ir.ArrayAttr.get([
                            mlir_ir.StringAttr.get(name, context=context)
                            for name in definition.record_logicals.names
                        ],
                                              context=context),
                    "x":
                        _gf2_matrix(context, definition.record_logicals.x),
                    "z":
                        _gf2_matrix(context, definition.record_logicals.z),
                    "gauge_pair_indices":
                        mlir_ir.DenseI64ArrayAttr.get(
                            definition.record_logicals.gauge_pair_indices,
                            context=context,
                        ),
                },
                context=context,
            )
        if definition.period_closure is not None:
            attrs["period_closure"] = _gf2_matrix(context,
                                                  definition.period_closure)
        if definition.evidence:
            attrs["evidence"] = mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(str(item), context=context)
                    for item in definition.evidence
                ],
                context=context,
            )
        profile_metadata = dict(definition.metadata)
        profile_metadata["effective_stabilizer_labels"] = (
            definition.effective_stabilizer_labels)
        profile_metadata.update(_distance_metadata(definition.distance))
        attrs["metadata"] = _metadata(context, profile_metadata)
        equivalent = _equivalent_declaration(
            transaction,
            "fabric.code_profile",
            attrs,
        )
        if equivalent is not None:
            handle = DefinitionHandle(_operation_symbol(equivalent),
                                      CodeProfile, "p2s")
            transaction.bind(definition, handle)
            return handle
        symbol = transaction.unique_symbol(definition.name)
        attrs["sym_name"] = mlir_ir.StringAttr.get(symbol, context=context)
        _insert(transaction, "fabric.code_profile", attrs)
        handle = DefinitionHandle(symbol, CodeProfile, "p2s")
        transaction.bind(definition, handle)
        return handle

    if isinstance(definition, Encoding):
        code_handle = materialize_qec(transaction, definition.code)
        profile_handle = materialize_qec(transaction, definition.profile)
        epoch_schema_handle = materialize_qec(transaction,
                                              definition.epoch_schema)
        existing = transaction.lookup(definition)
        if existing is not None:
            return existing
        hierarchy_handle = (None if definition.hierarchy is None else
                            transaction.materialize(definition.hierarchy))
        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(definition.name, context=context),
            "code":
                mlir_ir.FlatSymbolRefAttr.get(code_handle.symbol,
                                              context=context),
            "profile":
                mlir_ir.FlatSymbolRefAttr.get(profile_handle.symbol,
                                              context=context),
            "block":
                mlir_ir.StringAttr.get(definition.block, context=context),
            "logical_ports":
                mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.StringAttr.get(port, context=context)
                        for port in definition.logical_ports
                    ],
                    context=context,
                ),
            "epoch_schema":
                mlir_ir.FlatSymbolRefAttr.get(epoch_schema_handle.symbol,
                                              context=context),
            "initial_epoch":
                mlir_ir.FlatSymbolRefAttr.get(definition.initial_epoch.name,
                                              context=context),
        }
        if definition.layout:
            attrs["layout"] = _metadata(context, definition.layout)
        if hierarchy_handle is not None:
            attrs["hierarchy"] = mlir_ir.FlatSymbolRefAttr.get(
                hierarchy_handle.symbol, context=context)
        if definition.metadata:
            attrs["metadata"] = _metadata(context, definition.metadata)
        initial_epoch_attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(definition.initial_epoch.name,
                                       context=context),
            "encoding":
                mlir_ir.FlatSymbolRefAttr.get(definition.name, context=context),
            "schema":
                mlir_ir.FlatSymbolRefAttr.get(epoch_schema_handle.symbol,
                                              context=context),
            "phase":
                mlir_ir.StringAttr.get(definition.initial_epoch.phase,
                                       context=context),
            "index":
                _i64(context, definition.initial_epoch.index),
        }
        # ``initial_epoch`` is a generated identity-bearing child symbol. Its
        # phase/schema payload is fixed by the remaining encoding attributes,
        # so it is intentionally excluded from the structural comparison.
        equivalent = _equivalent_declaration(
            transaction,
            "fabric.encoding",
            attrs,
            ignored=("sym_name", "initial_epoch"),
        )
        if equivalent is not None:
            symbol = _operation_symbol(equivalent)
            initial_epoch_symbol = str(
                getattr(
                    equivalent.attributes["initial_epoch"],
                    "value",
                    equivalent.attributes["initial_epoch"],
                )).strip('"@')
            handle = DefinitionHandle(symbol, Encoding, "p2s")
            transaction.bind(definition, handle)
            transaction.bind(
                definition.initial_epoch,
                DefinitionHandle(
                    initial_epoch_symbol,
                    EncodingEpoch,
                    "p2s",
                ),
            )
            return handle
        symbol = transaction.unique_symbol(definition.name)
        initial_epoch_symbol = transaction.unique_symbol(
            definition.initial_epoch.name)
        attrs["sym_name"] = mlir_ir.StringAttr.get(symbol, context=context)
        attrs["initial_epoch"] = mlir_ir.FlatSymbolRefAttr.get(
            initial_epoch_symbol, context=context)
        _insert(transaction, "fabric.encoding", attrs)
        handle = DefinitionHandle(symbol, Encoding, "p2s")
        transaction.bind(definition, handle)
        initial_epoch_attrs["sym_name"] = mlir_ir.StringAttr.get(
            initial_epoch_symbol, context=context)
        initial_epoch_attrs["encoding"] = mlir_ir.FlatSymbolRefAttr.get(
            symbol, context=context)
        _insert(
            transaction,
            "fabric.encoding_epoch",
            initial_epoch_attrs,
        )
        transaction.bind(
            definition.initial_epoch,
            DefinitionHandle(initial_epoch_symbol, EncodingEpoch, "p2s"),
        )
        if definition.flat_projection is not None:
            materialize_qec(transaction, definition.flat_projection)
        return handle

    if isinstance(definition, EncodingHierarchy):
        code_handle = materialize_qec(transaction, definition.code)
        outer_handle = materialize_qec(transaction, definition.outer)
        child_handle = materialize_qec(transaction, definition.child)
        flat_handle = materialize_qec(transaction, definition.flat_encoding)
        existing = transaction.lookup(definition)
        if existing is not None:
            return existing

        def port_array(values):
            return mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(f"{child}:{port}", context=context)
                    for child, port in values
                ],
                context=context,
            )

        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(definition.name, context=context),
            "code":
                mlir_ir.FlatSymbolRefAttr.get(code_handle.symbol,
                                              context=context),
            "outer":
                mlir_ir.FlatSymbolRefAttr.get(outer_handle.symbol,
                                              context=context),
            "child":
                mlir_ir.FlatSymbolRefAttr.get(child_handle.symbol,
                                              context=context),
            "multiplicity":
                _i64(context, definition.multiplicity),
            "carrier_map":
                mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.StringAttr.get(f"{outer}:{child}:{port}",
                                               context=context)
                        for outer, child, port in definition.carrier_map
                    ],
                    context=context,
                ),
            "flat_encoding":
                mlir_ir.FlatSymbolRefAttr.get(flat_handle.symbol,
                                              context=context),
            "depth":
                _i64(context, definition.depth),
        }
        if definition.exposed_ports:
            attrs["exposed_ports"] = port_array(definition.exposed_ports)
        if definition.gauge_ports:
            attrs["gauge_ports"] = port_array(definition.gauge_ports)
        if definition.fixed_ports:
            attrs["fixed_ports"] = mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.DictAttr.get(
                        {
                            "child":
                                _i64(context, child),
                            "port":
                                _i64(context, port),
                            "basis":
                                mlir_ir.StringAttr.get(disposition.basis,
                                                       context=context),
                            "eigenvalue":
                                _i64(context, disposition.eigenvalue),
                            "evidence":
                                mlir_ir.StringAttr.get(disposition.evidence,
                                                       context=context),
                        },
                        context=context,
                    ) for child, port, disposition in definition.fixed_ports
                ],
                context=context,
            )
        equivalent = _equivalent_declaration(transaction,
                                             "fabric.encoding_hierarchy", attrs)
        if equivalent is not None:
            handle = DefinitionHandle(_operation_symbol(equivalent),
                                      EncodingHierarchy, "p2s")
            transaction.bind(definition, handle)
            return handle
        symbol = transaction.unique_symbol(definition.name)
        attrs["sym_name"] = mlir_ir.StringAttr.get(symbol, context=context)
        _insert(transaction, "fabric.encoding_hierarchy", attrs)
        handle = DefinitionHandle(symbol, EncodingHierarchy, "p2s")
        transaction.bind(definition, handle)
        return handle

    if isinstance(definition, EncodingProjection):
        source = materialize_qec(transaction, definition.source)
        destination = materialize_qec(transaction, definition.destination)
        existing = transaction.lookup(definition)
        if existing is not None:
            return existing
        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(definition.name, context=context),
            "source":
                mlir_ir.FlatSymbolRefAttr.get(source.symbol, context=context),
            "destination":
                mlir_ir.FlatSymbolRefAttr.get(destination.symbol,
                                              context=context),
            "carrier_map":
                mlir_ir.DenseI64ArrayAttr.get(definition.carrier_map, context),
            "logical_map":
                mlir_ir.DenseI64ArrayAttr.get(definition.logical_map, context),
            "evidence":
                mlir_ir.StringAttr.get(definition.evidence, context=context),
        }
        equivalent = _equivalent_declaration(transaction,
                                             "fabric.encoding_projection",
                                             attrs)
        if equivalent is not None:
            handle = DefinitionHandle(_operation_symbol(equivalent),
                                      EncodingProjection, "p2s")
            transaction.bind(definition, handle)
            return handle
        symbol = transaction.unique_symbol(definition.name)
        attrs["sym_name"] = mlir_ir.StringAttr.get(symbol, context=context)
        _insert(transaction, "fabric.encoding_projection", attrs)
        handle = DefinitionHandle(symbol, EncodingProjection, "p2s")
        transaction.bind(definition, handle)
        return handle

    if isinstance(definition, PatchTransform):
        source = materialize_qec(transaction, definition.source)
        destination = materialize_qec(transaction, definition.destination)
        existing = transaction.lookup(definition)
        if existing is not None:
            return existing

        def roles(value):
            return mlir_ir.DictAttr.get(
                {
                    name:
                        mlir_ir.DenseI64ArrayAttr.get(getattr(value, name),
                                                      context) for name in (
                                                          "active",
                                                          "measured",
                                                          "reset",
                                                          "scratch",
                                                          "dormant",
                                                      )
                },
                context=context,
            )

        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(definition.name, context=context),
            "source":
                mlir_ir.FlatSymbolRefAttr.get(source.symbol, context=context),
            "destination":
                mlir_ir.FlatSymbolRefAttr.get(destination.symbol,
                                              context=context),
            "frame_partitions":
                mlir_ir.DictAttr.get(
                    {
                        name: _i64(context, count)
                        for name, count in definition.frame.partitions.items()
                    },
                    context=context,
                ),
            "source_support":
                mlir_ir.DenseI64ArrayAttr.get(definition.source_support,
                                              context),
            "destination_support":
                mlir_ir.DenseI64ArrayAttr.get(definition.destination_support,
                                              context),
            "source_roles":
                roles(definition.source_roles),
            "destination_roles":
                roles(definition.destination_roles),
            "logical_map":
                mlir_ir.DenseI64ArrayAttr.get(definition.logical_map, context),
            "evidence":
                mlir_ir.StringAttr.get(definition.evidence, context=context),
        }
        equivalent = _equivalent_declaration(transaction,
                                             "fabric.patch_transform", attrs)
        if equivalent is not None:
            handle = DefinitionHandle(_operation_symbol(equivalent),
                                      PatchTransform, "p2s")
            transaction.bind(definition, handle)
            return handle
        symbol = transaction.unique_symbol(definition.name)
        attrs["sym_name"] = mlir_ir.StringAttr.get(symbol, context=context)
        _insert(transaction, "fabric.patch_transform", attrs)
        handle = DefinitionHandle(symbol, PatchTransform, "p2s")
        transaction.bind(definition, handle)
        return handle

    raise TypeError(f"unsupported QEC definition {definition!r}")
