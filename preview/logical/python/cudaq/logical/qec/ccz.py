# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Generated multi-block CCZ-state consumption for native RPP/MPP targets.

The implementation is deliberately narrow: one local logical space, one
common encoding, native Pauli-product rotations, and a typed product-
measurement instrument. It does not imply a general surgery or bridge path.
"""

from __future__ import annotations

from inspect import Parameter, Signature

from .. import std
from cudaq.logical.algebra import X, Y, Z, pi
from cudaq.logical.gadgets import patch
from cudaq.logical.ops import cond, discard, mpp, rotate, unpack_resource
from cudaq.logical.protocols import protocol
from cudaq.logical.qec.lowering import GeneratedQECArtifact, QECLowering
from cudaq.logical.types import resource


@protocol(
    implements=std.transport(std.CCZ_STATE),
    metadata={
        "role": "typed CCZ-state delivery provenance",
        "boundary": "factory-to-compute",
    },
)
def ccz_state_delivery(
    state: resource[std.CCZ_STATE],) -> resource[std.CCZ_STATE]:
    """Typed delivery boundary whose exact route is selected by the device."""

    return state


def _native_names(resource_class):
    actions = {
        action.name if hasattr(action, "name") else action
        for action in resource_class.native_actions
    }
    instruments = {
        instrument.operation for instrument in resource_class.native_instruments
    }
    return actions, instruments


def _rotate_values(values, terms, angle):
    """Apply one RPP and replace exactly the participating patch owners."""

    factors = {"X": X, "Y": Y, "Z": Z}
    product = None
    for block, logical, pauli in terms:
        factor = factors[pauli](values[block][logical])
        product = factor if product is None else product @ factor
    originals = []
    for factor in product.factors:
        owner = factor.operand.patch
        if all(owner is not item for item in originals):
            originals.append(owner)
    successors = rotate(product, angle=angle)
    replacement = {
        id(original): successor
        for original, successor in zip(originals, successors)
    }
    return [replacement.get(id(value), value) for value in values]


def _hadamard(values, operand):
    """Exact ``H = RY(pi/2) RZ(pi)`` up to global phase."""

    values = _rotate_values(values, ((*operand, "Z"),), pi)
    return _rotate_values(values, ((*operand, "Y"),), pi / 2)


def _cz(values, left, right):
    """Exact CZ from commuting Z, Z, and ZZ product rotations."""

    values = _rotate_values(values, ((*left, "Z"),), pi / 2)
    values = _rotate_values(values, ((*right, "Z"),), pi / 2)
    return _rotate_values(
        values,
        ((*left, "Z"), (*right, "Z")),
        -pi / 2,
    )


def _cx(values, control, target):
    values = _hadamard(values, target)
    values = _cz(values, control, target)
    return _hadamard(values, target)


def _measure_z(values, operand):
    block, logical = operand
    successor, outcome = mpp(Z(values[block][logical]))
    updated = list(values)
    updated[block] = successor
    return updated, outcome


def _x(values, operand):
    """Exact logical X as ``R_X(pi)`` up to global phase."""

    return _rotate_values(values, ((*operand, "X"),), pi)


def _validate_stream(site, context):
    if site.resource_stream_owner != context.device.logical.name:
        raise ValueError(
            "CCZ action site resource stream belongs to a different logical machine"
        )
    streams = tuple(stream for stream in context.device.logical.streams
                    if stream.name == site.resource_stream)
    if len(streams) != 1:
        raise ValueError(
            "CCZ action site must reference one declared logical resource stream"
        )
    stream = streams[0]
    if stream.produces is not std.CCZ_STATE:
        raise ValueError("CCZ action site resource stream has the wrong kind")
    if not stream.is_physically_complete:
        raise ValueError(
            "CCZ resource stream requires a backed factory or explicit external supply"
        )
    if stream.produced_by is None:
        raise ValueError(
            "CCZ resource stream has no production-protocol provenance")
    if stream.transfer is None:
        raise ValueError(
            "CCZ resource stream has no transfer-protocol provenance")
    if std.CCZ_STATE not in stream.transfer._resource_input_kinds():
        raise ValueError("CCZ transfer protocol does not consume CCZ_STATE")
    return stream


def _layout(site, context):
    if site.kind != "resource_action" or site.objective != std.ccz.name:
        raise ValueError(
            "CCZ-state compiler received a non-CCZ resource action")
    if site.resource_kind != std.CCZ_STATE.name:
        raise ValueError(
            "CCZ action requires the typed CCZ_STATE resource kind")
    if len(context.placements) != 3:
        raise ValueError(
            "CCZ-state consumption requires exactly three logical operands")
    if len({placement.space for placement in context.placements}) != 1:
        raise NotImplementedError(
            "CCZ-state consumption currently requires one local logical space; "
            "no cross-region bridge is implied")
    encodings = tuple(placement.encoding for placement in context.placements)
    if any(encoding is None for encoding in encodings) or len(
        {encoding.name for encoding in encodings}) != 1:
        details = tuple((
            placement.placement,
            placement.block,
            None if placement.encoding is None else placement.encoding.name,
        ) for placement in context.placements)
        raise NotImplementedError(
            "CCZ-state consumption requires one common concrete encoding; "
            f"received {details!r}")

    block_names = tuple(
        dict.fromkeys(placement.block or placement.placement
                      for placement in context.placements))
    block_index = {name: index for index, name in enumerate(block_names)}
    operands = tuple((
        block_index[placement.block or placement.placement],
        placement.logical_index,
    ) for placement in context.placements)
    if any(logical is None for _block, logical in operands):
        raise ValueError(
            "CCZ operands require explicit packed logical-port witnesses")
    if len(set(operands)) != 3:
        raise ValueError(
            "CCZ operands must map to three distinct logical ports")

    witness_by_block = {
        block.block: block for block in context.qec_selection.blocks
    }
    site_placements = set(site.placements)
    for block_name in block_names:
        witness = witness_by_block[block_name]
        selected = {
            owner.placement
            for owner in witness.owners
            if owner.placement in site_placements
        }
        occupants = {owner.placement for owner in witness.owners}
        if occupants != selected:
            unrelated = sorted(occupants - selected)
            raise NotImplementedError(
                "CCZ-state consumption fails closed when an involved packed "
                f"block contains unrelated live logical occupants: {unrelated!r}"
            )
    groups = tuple(
        tuple(logical
              for block, logical in operands
              if block == block_index[block_name])
        for block_name in block_names)
    return block_names, operands, groups, encodings[0]


def _generated_protocol(context, block_names, operands, groups, encoding):
    annotation = patch[encoding]
    parameters = tuple(
        Parameter(
            f"block{index}",
            Parameter.POSITIONAL_OR_KEYWORD,
            annotation=annotation,
        ) for index in range(len(block_names))) + (Parameter(
            "state",
            Parameter.POSITIONAL_OR_KEYWORD,
            annotation=resource[std.CCZ_STATE],
        ),)
    result_annotation = (annotation if len(block_names) == 1 else tuple[tuple(
        annotation for _name in block_names)])

    def generated(*arguments):
        data = list(arguments[:-1])
        state = arguments[-1]
        data, payloads = unpack_resource(
            state,
            like=tuple(data),
            encoding=encoding,
            logical_ports=groups,
        )
        data = list(data)
        payloads = list(payloads)

        for operand in operands:
            payloads_and_data = [*payloads, *data]
            control = operand
            target = (len(payloads) + operand[0], operand[1])
            payloads_and_data = _cx(payloads_and_data, control, target)
            payloads = payloads_and_data[:len(payloads)]
            data = payloads_and_data[len(payloads):]

        outcomes = []
        for operand in operands:
            data, outcome = _measure_z(data, operand)
            outcomes.append(outcome)
        discard(tuple(data), reason="CCZ teleportation data blocks measured")

        for index, outcome in enumerate(outcomes):
            other = tuple(item for item in range(3) if item != index)

            def correct(*live, index=index, other=other):
                values = _x(list(live), operands[index])
                return tuple(_cz(values, operands[other[0]],
                                 operands[other[1]]))

            payloads = list(
                cond(
                    outcome,
                    carries=tuple(payloads),
                    then=correct,
                    else_=lambda *live: tuple(live),
                ))
        return payloads[0] if len(payloads) == 1 else tuple(payloads)

    generated.__name__ = generated.__qualname__ = (
        f"{context.lowering.name}_{len(block_names)}block_protocol")
    generated.__module__ = __name__
    generated.__signature__ = Signature(parameters,
                                        return_annotation=result_annotation)
    generated.__annotations__ = {
        **{
            parameter.name: parameter.annotation for parameter in parameters
        },
        "return": result_annotation,
    }
    return protocol(
        generated,
        implements=std.ccz,
        name=generated.__name__,
        metadata={
            "construction": "CCZ gate teleportation from CCZ|+++>",
            "correction_identity": "CCZ X_i CCZ = X_i CZ_jk",
            "target_primitives": "native RPP and typed MPP",
        },
    )


def compiler(
    code_or_encoding,
    *,
    plugin="cudaq.logical.ccz_state",
    version="1.0.0",
    name=None,
):
    """Create the versioned multi-block CCZ-state consumption compiler."""

    def provider(site, context):
        _validate_stream(site, context)
        block_names, operands, groups, encoding = _layout(site, context)
        if context.physical is None:
            raise ValueError("CCZ-state consumption requires a physical target")
        capabilities = tuple(
            _native_names(resource_class)
            for resource_class in context.physical.resource_classes)
        if not any("rpp" in actions for actions, _instruments in capabilities):
            raise NotImplementedError(
                "CCZ-state consumption requires an explicitly advertised native RPP action"
            )
        if not any("measure_product" in instruments
                   for _actions, instruments in capabilities):
            raise NotImplementedError(
                "CCZ-state consumption requires a typed native MPP instrument")
        definition = _generated_protocol(context, block_names, operands, groups,
                                         encoding)
        return GeneratedQECArtifact(
            definition,
            {
                "block_count":
                    len(block_names),
                "block_layout":
                    "+".join(str(len(group)) for group in groups),
                "_payload_block_ids":
                    block_names,
                "logical_port_map":
                    ";".join(f"{block}:{logical}" for block, logical in operands
                            ),
                "protocol":
                    "ccz_gate_teleportation",
            },
        )

    provider.__name__ = name or f"{code_or_encoding.name}_ccz_state_compiler"
    return QECLowering(
        provider,
        objective_family="resource_action",
        objective=std.ccz,
        codes=(code_or_encoding,),
        # The selected stream producer and transfer protocols are serialized
        # by P1 and are the dependency closure.  Do not substitute a builtin
        # transfer for the actual selected stream provenance.
        dependencies=(),
        plugin=plugin,
        version=version,
        name=provider.__name__,
        metadata={
            "construction": "CCZ gate teleportation",
            "supported_layouts": "1|2+1|1+1+1",
            "target_primitives": "native RPP and typed MPP",
        },
    )


__all__ = ["ccz_state_delivery", "compiler"]
